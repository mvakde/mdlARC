"""First-contact ledgers: a periodic scan compresses whole objects by color and time.

Only Python's standard library and NumPy are used. All solving conventions are
fixed or encoded by the annotated pilot on the input edge; no puzzle state is used.
"""
from __future__ import annotations

import json
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SIDES = ("top", "right", "bottom", "left")


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= s <= 30 for s in g.shape), "grid dimensions")
    _check(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _check(np.all((0 <= g) & (g <= 9)), "colors outside 0..9")
    return g.astype(np.int8, copy=False)


def _canonical(g, side):
    if side == "top":
        return g
    if side == "bottom":
        return g[::-1]
    if side == "left":
        return g.T
    return g.T[::-1]


def _orient(g, side):
    if side == "top":
        return g.copy()
    if side == "bottom":
        return g[::-1].copy()
    if side == "left":
        return g.T.copy()
    return g[::-1].T.copy()


def _components(g, bg, excluded=()):
    seen = np.isin(g, (bg,) + tuple(excluded))
    result = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c]:
                continue
            color = int(g[r, c])
            seen[r, c] = True
            cells, stack = [], [(r, c)]
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS:
                    a, b = rr + dr, cc + dc
                    if (0 <= a < g.shape[0] and 0 <= b < g.shape[1]
                            and not seen[a, b] and int(g[a, b]) == color):
                        seen[a, b] = True
                        stack.append((a, b))
            rows, cols = zip(*cells)
            result.append(dict(color=color, cells=tuple(cells),
                               box=(min(rows), max(rows), min(cols), max(cols))))
    return result


def _perimeter(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _event(cells, lines):
    rows = {r for r, _ in cells}
    return next((k for k, d in enumerate(lines) if d in rows), None)


def _inspect(input_grid):
    raw = _grid(input_grid)
    values, sizes = np.unique(raw, return_counts=True)
    bg = int(values[np.argmax(sizes)])
    _check(int(sizes.max()) > raw.size / 2, "background must occupy over half the input")
    pilots = []
    for marker in values:
        marker = int(marker)
        if marker == bg:
            continue
        for side in _SIDES:
            g = _canonical(raw, side)
            points = np.argwhere(g == marker)
            if not len(points) or not np.all(points[:, 0] == 0):
                continue
            positions = sorted(int(p[1]) for p in points)
            start, length = positions[0], len(positions)
            if positions != list(range(start, start + length)):
                continue
            if length == 1 and start in (0, g.shape[1] - 1):
                continue
            if g.shape[0] < 2:
                continue
            keys = {int(g[1, c]) for c in range(start, start + length)}
            for key in keys:
                if key not in (bg, marker) and int(np.sum(g == key)) == 1:
                    pilots.append((side, marker, key, start, length))
    _check(len(pilots) == 1, "exactly one annotated edge bar is required")
    side, marker, special, start, length = pilots[0]
    g = _canonical(raw, side)
    lines = tuple(range(4, g.shape[0], 6))
    _check(bool(lines), "at least one special scan line is required")
    parts = _components(g, bg, (marker, special))
    frames, objects = [], []
    for part in parts:
        r0, r1, c0, c1 = part["box"]
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if (w >= 5 and w % 2 == 1 and h >= 2 * len(lines) + 3
                and set(part["cells"]) == _perimeter(part["box"])):
            _check(np.all(g[r0 + 1:r1, c0 + 1:c1] == bg), "frame interior is not empty")
            _check(c1 < start or c0 >= start + length, "frame intersects ruler corridor")
            frames.append(part)
        else:
            objects.append(part)
    _check(1 <= len(frames) <= 3, "expected one through three receipt frames")
    _check(len({f["color"] for f in frames}) == len(frames), "duplicate frame category")
    # Frames reserve their full rectangles, not merely their perimeter pixels.
    for o in objects:
        for r, c in o["cells"]:
            _check(r >= 2, "objects may not intrude into the pilot's two edge rows")
            for f in frames:
                r0, r1, c0, c1 = f["box"]
                _check(not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1),
                       "object touches or lies inside a frame")
        o["event"] = _event(o["cells"], lines)
    labels = np.full(g.shape, -1, dtype=int)
    for i, p in enumerate(parts):
        for r, c in p["cells"]:
            labels[r, c] = i
    for r, c in np.argwhere(labels >= 0):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                a, b = int(r + dr), int(c + dc)
                if 0 <= a < len(g) and 0 <= b < g.shape[1] and labels[a, b] >= 0:
                    _check(labels[r, c] == labels[a, b], "separate objects/frames touch")
    counts = {f["color"]: [0] * len(lines) for f in frames}
    for o in objects:
        if o["color"] in counts and o["event"] is not None:
            counts[o["color"]][o["event"]] += 1
    for f in frames:
        capacity = (f["box"][3] - f["box"][2] - 2) // 2
        _check(max(counts[f["color"]]) <= capacity, "receipt row exceeds capacity")
    return dict(raw=raw, grid=g, bg=bg, side=side, marker=marker, special=special,
                start=start, length=length, lines=lines, frames=frames,
                objects=objects, counts=counts)


def _draw(g, info, counts):
    bg, start, length = info["bg"], info["start"], info["length"]
    result = np.full_like(g, bg)
    for f in info["frames"]:
        for r, c in f["cells"]:
            result[r, c] = f["color"]
        r0, r1, c0, c1 = f["box"]
        for k, count in enumerate(counts[f["color"]]):
            for j in range(count):
                c = c1 - 2 * (j + 1) if c1 < start else c0 + 2 * (j + 1)
                result[r0 + 2 + 2 * k, c] = f["color"]
    for d in range(0, len(g), 2):
        result[d, start:start + length] = info["special"] if d % 6 == 4 else info["marker"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Segment ORIGINAL objects, assign first special contact, then write receipts."""
    info = _inspect(input_grid)
    return _orient(_draw(info["grid"], info, info["counts"]), info["side"])


def _grow_shape(rng, area, allowed, root):
    cells = {root}
    while len(cells) < area:
        candidates = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _NEIGHBORS}
                            & allowed - cells)
        if not candidates:
            return None
        cells.add(candidates[int(rng.integers(len(candidates)))])
    return cells


def _normalize(cells):
    a, b = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r - a, c - b) for r, c in cells)), a


def _hit_shape(rng, area, line):
    # Exactly one pixel on the scan line, with independently grown arms nearby.
    allowed = {(r, c) for r in range(-2, 3) for c in range(-3, 4)
               if r != 0 or c == 0}
    cells = _grow_shape(rng, area, allowed, (0, 0))
    shape, low = _normalize(cells)
    return shape, line + low


def _bridge_shape(rng, area, line):
    before = int(rng.integers(0, 3))
    after = int(rng.integers(0, min(3, area - 7 - before + 1)))
    cells = {(r, 0) for r in range(-before, 7 + after)}
    allowed = {(r, c) for r in range(-before, 7 + after) for c in range(-3, 4)
               if r not in (0, 6) or c == 0}
    while len(cells) < area:
        options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _NEIGHBORS}
                         & allowed - cells)
        cells.add(options[int(rng.integers(len(options)))])
    shape, low = _normalize(cells)
    return shape, line + low


def _random_shape(rng, area):
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(r, 0) for r in range(area)}
    elif kind == 1:
        width = int(rng.integers(1, min(5, area) + 1))
        cells = {(i // width, i % width) for i in range(area)}
    else:
        allowed = {(r, c) for r in range(-4, 5) for c in range(-3, 4)}
        cells = _grow_shape(rng, area, allowed, (0, 0))
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return _normalize(cells)[0]


def _layout(rng, depth, colors, anchor_width, requested, layout_mode=None):
    n = len(colors)
    minima = [max(anchor_width if i == 0 else 5,
                  2 * max(requested[c]) + 3) for i, c in enumerate(colors)]
    widths = [int(rng.choice(tuple(range(w, 10, 2)))) for w in minima]
    while sum(widths) + n + 9 > 30:
        choices = [i for i, w in enumerate(widths) if w > minima[i]]
        if not choices:
            return None
        i = choices[int(rng.integers(len(choices)))]
        widths[i] -= 2
    minimum = sum(widths) + n + 9
    span = int(rng.integers(minimum, 31))
    mode = layout_mode or (str(rng.choice(("left", "right", "split"))) if n > 1
                           else str(rng.choice(("left", "right"))))
    left_count = n if mode == "left" else 0 if mode == "right" else int(rng.integers(1, n))
    origins = [None] * n
    x = 0
    for i in range(left_count):
        origins[i] = x
        x += widths[i] + 1
    low = x + 1 if left_count else 0
    x = span
    for i in range(n - 1, left_count - 1, -1):
        x -= widths[i]
        origins[i] = x
        x -= 1
    high = x - 1 if left_count < n else span - 1
    length = int(rng.integers(1, min(4, high - low + 1)))
    start = int(rng.integers(max(1, low), min(high - length + 2, span - length)))
    epochs = len(range(4, depth, 6))
    frames = []
    for color, c, w in zip(colors, origins, widths):
        height = int(rng.integers(2 * epochs + 3, min(depth, 2 * epochs + 12) + 1))
        r = int(rng.integers(depth - height + 1))
        box = (r, r + height - 1, c, c + w - 1)
        frames.append(dict(color=color, box=box, cells=tuple(sorted(_perimeter(box)))))
    return span, start, length, frames, (low, high)


def _put_object(rng, g, blocked, field, color, area, event, lines, kind):
    for attempt in range(35):
        fixed_row = None
        if kind == "bridge":
            shape, fixed_row = _bridge_shape(rng, area, lines[event])
        elif kind == "one_pixel":
            shape, fixed_row = _hit_shape(rng, area, lines[event])
        else:
            shape = _random_shape(rng, area) if attempt < 25 else ((0, 0),)
        h, w = 1 + max(r for r, c in shape), 1 + max(c for r, c in shape)
        if h > len(g) - 2 or w > g.shape[1]:
            continue
        row_choices = [fixed_row] if fixed_row is not None else range(2, len(g) - h + 1)
        positions = []
        for r in row_choices:
            if r < 2 or r + h > len(g):
                continue
            if _event([(r + dr, dc) for dr, dc in shape], lines) != event:
                continue
            for c in range(g.shape[1] - w + 1):
                if all(not blocked[r + dr, c + dc] for dr, dc in shape):
                    positions.append((r, c))
        if not positions:
            continue
        # Usually use the open field; sometimes use gaps beyond frame corners.
        central = [(r, c) for r, c in positions if field[0] <= c and c + w - 1 <= field[1]]
        if central and rng.random() < .8:
            positions = central
        r0, c0 = positions[int(rng.integers(len(positions)))]
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        for r, c in cells:
            g[r, c] = color
            blocked[max(0, r - 1):min(len(g), r + 2),
                    max(0, c - 1):min(g.shape[1], c + 2)] = True
        return dict(color=color, cells=tuple(cells), event=event)
    return None


def _make_pair(rng, side, colors, bg, marker, special, specs, anchor_width, depth, layout_mode=None):
    lines = tuple(range(4, depth, 6))
    requested = {c: [0] * len(lines) for c in colors}
    for color, area, event, kind in specs:
        if color in requested and event is not None:
            requested[color][event] += 1
    layout = _layout(rng, depth, colors, anchor_width, requested, layout_mode)
    if layout is None:
        return None
    span, start, length, frames, field = layout
    g = np.full((depth, span), bg, dtype=np.int8)
    blocked = np.zeros_like(g, dtype=bool)
    blocked[:2] = True
    for f in frames:
        for r, c in f["cells"]:
            g[r, c] = f["color"]
        r0, r1, c0, c1 = f["box"]
        blocked[max(0, r0 - 1):min(depth, r1 + 2),
                max(0, c0 - 1):min(span, c1 + 2)] = True
    g[0, start:start + length] = marker
    g[1, start + int(rng.integers(length))] = special
    capacities = {f["color"]: (f["box"][3] - f["box"][2] - 2) // 2 for f in frames}
    if any(max(v) > capacities[c] for c, v in requested.items()):
        return None
    objects = []
    # Pack required large witnesses before optional shapes.
    for color, area, event, kind in sorted(specs, key=lambda x: -x[1]):
        obj = _put_object(rng, g, blocked, field, color, area, event, lines, kind)
        if obj is None:
            return None
        objects.append(obj)
    if np.sum(g == bg) <= g.size / 2:
        return None
    info = dict(bg=bg, marker=marker, special=special, start=start, length=length, frames=frames)
    out = _draw(g, info, requested)
    return {"input": _orient(g, side), "output": _orient(out, side)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    anchor, switched, novel = (int(c) for c in rng.choice(10, 3, replace=False))
    witness_area = int(rng.integers(10, 15))
    part_area = int(rng.integers(4, min(8, witness_area - 3)))
    base = int(rng.integers(4))
    # Opposite teaching directions make the scan and receipt origins edge-relative.
    teaching_sides = (_SIDES[base], _SIDES[(base + 2) % 4])
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt + nq):
        train = index < nt
        local = index if train else index - nt
        n = 2 if index < 2 else int(rng.integers(1, 4))
        side = teaching_sides[index] if index < 2 else (
            _SIDES[(base + 1 + 2 * (local % 2)) % 4] if not train else str(rng.choice(_SIDES)))
        depth = int(rng.integers(5, 17)) if not train and rng.random() < .28 else int(rng.integers(17, 31))
        epochs = len(range(4, depth, 6))
        forbidden = {anchor, switched, novel} if index < 2 else {novel} if train else set()
        pool = [c for c in range(10) if c not in forbidden]
        bg, marker, special = (int(c) for c in rng.choice(pool, 3, replace=False))
        available = [c for c in range(10) if c not in (bg, marker, special)
                     and (not train or c != novel)]
        if index == 0:
            colors = [anchor, switched]
        elif index == 1:
            colors = [anchor, int(rng.choice([c for c in available if c not in (anchor, switched)]))]
        else:
            colors = [int(c) for c in rng.choice(available, n, replace=False)]
            if not train and local == 0 and novel not in colors:
                # Reserve the test's new category from the other role colors too.
                if novel in (bg, marker, special):
                    replacement = next(c for c in range(10) if c not in colors + [bg, marker, special, novel])
                    if bg == novel:
                        bg = replacement
                    elif marker == novel:
                        marker = replacement
                    else:
                        special = replacement
                colors[0] = novel
        distractor_pool = [c for c in range(10) if c not in colors + [bg, marker, special]]
        specs = []
        if index == 0:
            specs = [(anchor, witness_area, 0, "bridge"),
                     (switched, int(rng.integers(3, 8)), epochs - 1, "one_pixel"),
                     (switched, int(rng.integers(1, 4)), None, "random"),
                     (int(rng.choice(distractor_pool)), 3, 1, "one_pixel")]
        elif index == 1:
            specs = [(anchor, part_area, 0, "one_pixel"),
                     (anchor, witness_area - part_area, 0, "one_pixel"),
                     (colors[1], int(rng.integers(2, 7)), epochs - 1, "one_pixel"),
                     (switched, 3, 1, "one_pixel")]
        else:
            # Counts are bounded per scan event, independent of shape areas.
            occupancy = [0] * epochs
            for color in colors:
                local_counts = [0] * epochs
                for _ in range(int(rng.integers(0, 5))):
                    event = None if rng.random() < .23 else int(rng.integers(epochs))
                    if event is not None and (occupancy[event] >= 3 or local_counts[event] >= (1 if n == 3 else 2 if n == 2 else 3)):
                        continue
                    if event is not None:
                        occupancy[event] += 1
                        local_counts[event] += 1
                    specs.append((color, int(rng.integers(1, 13)), event, "random"))
            for _ in range(int(rng.integers(0, 4))):
                event = int(rng.integers(epochs))
                if occupancy[event] < 3:
                    occupancy[event] += 1
                    specs.append((int(rng.choice(distractor_pool)), int(rng.integers(1, 10)), event, "random"))
            if not specs:
                specs = [(int(rng.choice(distractor_pool)), 2, None, "random")]
            if depth < 17:
                # Shorter scans need fewer free objects, not a different rule.
                specs = specs[:3 if depth < 11 else 6]
        accepted = None
        for _ in range(100):
            pair = _make_pair(rng, side, colors, bg, marker, special, specs,
                              7 if index == 0 else 9 if index == 1 else 5, depth,
                              "left" if index == 0 else "right" if index == 1 else None)
            if pair is not None:
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    if index == 1:
                        evidence = _evidence([_inspect(p["input"]) for p in puzzle["train"]]
                                             + [_inspect(pair["input"])])
                        if not all(evidence[k] for k in ("remote", "corridor", "strip_changes")):
                            continue
                    accepted = pair
                    seen.add(key)
                    break
        _check(accepted is not None, "bounded receipt packing failed")
        puzzle["train" if train else "test"].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pixel_expected(info):
    """A grid-derived validation reconstruction, assigning each pixel directly."""
    g = info["grid"]
    counts = {}
    for f in info["frames"]:
        histogram = [0] * len(info["lines"])
        for obj in info["objects"]:
            contacts = [k for k, d in enumerate(info["lines"])
                        if any(r == d for r, _ in obj["cells"])]
            if obj["color"] == f["color"] and contacts:
                histogram[min(contacts)] += 1
        counts[f["color"]] = histogram
    result = np.full_like(g, info["bg"])
    for r in range(len(g)):
        for c in range(g.shape[1]):
            if info["start"] <= c < info["start"] + info["length"] and r % 2 == 0:
                result[r, c] = info["special"] if r % 6 == 4 else info["marker"]
            for f in info["frames"]:
                r0, r1, c0, c1 = f["box"]
                if r0 <= r <= r1 and c0 <= c <= c1:
                    if r in (r0, r1) or c in (c0, c1):
                        result[r, c] = f["color"]
                    elif r >= r0 + 2 and (r - r0) % 2 == 0:
                        k = (r - r0 - 2) // 2
                        distance = c1 - c if c1 < info["start"] else c - c0
                        if (k < len(info["lines"]) and distance % 2 == 0
                                and 2 <= distance <= 2 * counts[f["color"]][k]):
                            result[r, c] = f["color"]
    return _orient(result, info["side"])


def _evidence(details):
    records = []
    flags = dict(bridge=False, ignored=False, remote=False, corridor=False,
                 unmatched=False, later=False, multiple=False, hit_pixels=False,
                 strip_changes=False)
    for d in details:
        for color in d["counts"]:
            selected = [o for o in d["objects"] if o["color"] == color and o["event"] is not None]
            area = sum(len(o["cells"]) for o in selected)
            pixels = sum(r in d["lines"] for o in selected for r, c in o["cells"])
            records.append((color, sum(d["counts"][color]), area, pixels))
            flags["hit_pixels"] |= pixels != len(selected)
            flags["later"] |= any(d["counts"][color][1:])
            flags["multiple"] |= max(d["counts"][color]) >= 2
        for o in d["objects"]:
            matched = o["color"] in d["counts"]
            contacts = {r for r, c in o["cells"] if r in d["lines"]}
            flags["bridge"] |= matched and len(contacts) >= 2
            flags["ignored"] |= matched and not contacts
            flags["unmatched"] |= not matched and bool(contacts)
            flags["remote"] |= matched and bool(contacts) and all(
                c < d["start"] or c >= d["start"] + d["length"] for r, c in o["cells"])
            flags["corridor"] |= matched and any(
                d["start"] <= c < d["start"] + d["length"] for r, c in o["cells"])
        erased = np.full_like(d["grid"], d["bg"])
        for o in d["objects"]:
            for r, c in o["cells"]:
                if not d["start"] <= c < d["start"] + d["length"]:
                    erased[r, c] = o["color"]
        after_counts = {c: [0] * len(d["lines"]) for c in d["counts"]}
        for o in _components(erased, d["bg"]):
            event = _event(o["cells"], d["lines"])
            if o["color"] in after_counts and event is not None:
                after_counts[o["color"]][event] += 1
        flags["strip_changes"] |= after_counts != d["counts"]
    flags["area_collision"] = any(c == cc and a == aa and n != nn
                                 for c, n, a, p in records for cc, nn, aa, pp in records)
    flags["pixel_collision"] = any(c == cc and p == pp and n != nn
                                  for c, n, a, p in records for cc, nn, aa, pp in records)
    return flags


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "output dimensions changed")
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect(inp)
            _check(np.array_equal(out, _pixel_expected(d)), "incorrect ruler, first-contact tally, or erasure")
            _check(np.array_equal(out, solve(inp)), "solve mismatch")
            details[split].append(d)
    flags = _evidence(details["train"])
    for key in ("bridge", "ignored", "remote", "corridor", "strip_changes", "unmatched", "later", "multiple",
                "hit_pixels", "area_collision", "pixel_collision"):
        _check(flags[key], "missing training evidence: " + key)
    _check(len({d["side"] for d in details["train"]}) >= 2, "training must teach opposite directions")
    faces = {"left" if f["box"][3] < d["start"] else "right"
             for d in details["train"] for f in d["frames"]
             if any(0 < n < (f["box"][3] - f["box"][2] - 2) // 2
                    for n in d["counts"][f["color"]])}
    _check(faces == {"left", "right"}, "both ruler-facing tally ends must be demonstrated")
    train_colors = {f["color"] for d in details["train"] for f in d["frames"]}
    discards = {o["color"] for d in details["train"] for o in d["objects"] if o["color"] not in d["counts"]}
    _check(bool(train_colors & discards), "no demonstration of a color changing category role")
    _check(any(f["color"] not in train_colors for d in details["test"] for f in d["frames"]),
           "test must introduce a local category")


def _signature(puzzle):
    return tuple((s, pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
                 for s in ("train", "test") for pair in puzzle[s])


def self_test():
    """Bounded 42-seed semantic, diversity, API, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_signature(p) for p in puzzles}) == 42, "seed diversity")
    details = [_inspect(pair["input"]) for p in puzzles for s in p for pair in p[s]]
    _check({d["side"] for d in details} == set(_SIDES), "four scan directions")
    _check({len(d["frames"]) for d in details} == {1, 2, 3}, "category counts")
    _check(len({d["raw"].shape for d in details}) >= 35, "canvas variation")
    _check({d["bg"] for d in details} == set(range(10)), "background palette roles")
    _check({d["special"] for d in details} == set(range(10)), "special palette roles")
    _check({len(d["lines"]) for d in details} == {1, 2, 3, 4, 5}, "scan epoch counts")
    masks = set()
    for d in details:
        for o in d["objects"]:
            masks.add(_normalize(o["cells"])[0])
    _check(len(masks) >= 100, "object shape variation")
    _check(any(_evidence([d])["corridor"] for d in details), "objects crossing erased corridor")
    _check(any(max(d["counts"][c]) == 0 for d in details for c in d["counts"]), "zero ledgers")
    _check(any(any(f["box"][3] < d["start"] for f in d["frames"])
               and any(f["box"][2] >= d["start"] + d["length"] for f in d["frames"])
               for d in details), "receipts on both sides of ruler")
    _check(_signature(generate_puzzle(618571388)) == _signature(generate_puzzle(618571388)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(42 + nt + nq, nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Rotation and arbitrary palette permutation must commute with solving.
    rng = np.random.default_rng(104)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for turn in range(4):
            perm = rng.permutation(10)
            inp = perm[np.rot90(pair["input"], turn)]
            expected = perm[np.rot90(pair["output"], turn)]
            _check(np.array_equal(solve(inp), expected), "rotation/palette equivariance")
    # A hand-specified member beyond sampled bar/receipt widths. The last
    # special contact is on the opposite boundary; a long object is credited
    # only at its first contact, and three separate objects share one row.
    inp = np.zeros((17, 27), dtype=np.int8)
    for r, c in _perimeter((1, 12, 1, 11)):
        inp[r, c] = 3
    inp[0, 13:17] = 1
    inp[1, 14] = 2
    inp[3:12, 17] = 3
    inp[4, 20] = inp[4, 23] = inp[16, 20] = inp[13, 17] = 3
    inp[10, 23] = 4
    expected = np.zeros_like(inp)
    for r, c in _perimeter((1, 12, 1, 11)):
        expected[r, c] = 3
    for r in range(0, 17, 2):
        expected[r, 13:17] = 2 if r % 6 == 4 else 1
    expected[3, [9, 7, 5]] = 3
    expected[7, 9] = 3
    _check(np.array_equal(solve(inp), expected), "hand-specified boundary/width regression")
    empty = inp.copy()
    for o in _inspect(inp)["objects"]:
        for r, c in o["cells"]:
            empty[r, c] = 0
    no_counts = expected.copy()
    no_counts[3, [9, 7, 5]] = no_counts[7, 9] = 0
    _check(np.array_equal(solve(empty), no_counts), "empty field is a valid zero ledger")
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in puzzles[0]}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": len(details), "canvas_shapes": len({d["raw"].shape for d in details}),
            "object_masks": len(masks), "independent_semantic_review": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
