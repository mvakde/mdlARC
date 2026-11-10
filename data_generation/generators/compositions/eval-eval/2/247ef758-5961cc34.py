"""Coordinate-addressed broadcast relays. Standard library and NumPy only.

solve reads a persistent catalog and an otherwise empty framed field. A hit
wakes all Cartesian copies of that catalog identity, not just the struck copy.
No seed, training data, or construction records are used by solve.
"""
from __future__ import annotations

from collections import deque
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape):
        raise ValueError("Grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("Grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _connected(cells):
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in DIRS:
            p = (r + dr, c + dc)
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _port_side(cells, port):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
    sides = [i for i, ok in enumerate((port[0] == r0, port[1] == c1,
                                      port[0] == r1, port[1] == c0)) if ok]
    if len(sides) != 1:
        raise ValueError("A port must be on exactly one extreme side, away from corners")
    side = sides[0]
    dr, dc = DIRS[side]
    if (port[0] - dr, port[1] - dc) not in cells - {port}:
        raise ValueError("Port needs an inward body neighbor")
    return side


def _instantiate(templates):
    objects = []
    for t in templates:
        for r in t["rows"]:
            for c in t["cols"]:
                objects.append({"group": t["color"], "center": (r, c),
                                "cells": {(r + y, c + x) for y, x in t["offsets"]},
                                "port": (r + t["port"][0], c + t["port"][1]),
                                "direction": t["direction"]})
    return objects


def _owners(objects, bounds):
    lo_r, hi_r, lo_c, hi_c = bounds
    owner = {}
    for i, obj in enumerate(objects):
        for p in obj["cells"]:
            if not (lo_r <= p[0] <= hi_r and lo_c <= p[1] <= hi_c):
                raise ValueError("All candidate footprint cells must fit inside the frame")
            if p in owner:
                raise ValueError("Candidate occupied footprints must be disjoint")
            owner[p] = i
    return owner


def _decode(a):
    h, w = a.shape
    bg = int(a[0, 0])
    layouts = []
    for d in range(3, w - 6):
        frame = int(a[0, d])
        if frame == bg or not np.all(a[:, d] == frame):
            continue
        if not all(a[r, c] == frame for r, c in
                   ((0, d + 1), (0, w - 1), (h - 1, d + 1), (h - 1, w - 1))):
            continue
        if not np.array_equal(a[0, d + 2:-1], a[-1, d + 2:-1]):
            continue
        if not np.array_equal(a[1:-1, d + 1], a[1:-1, -1]):
            continue
        layouts.append((d, frame))
    if len(layouts) != 1:
        raise ValueError("A unique catalog divider and paired frame are required")
    d, frame = layouts[0]
    bounds = (1, h - 2, d + 2, w - 2)
    field = a[1:-1, d + 2:-1]
    cells = {(int(r) + 1, int(c) + d + 2) for r, c in np.argwhere(field != bg)}
    if len(cells) < 3:
        raise ValueError("Missing launcher")
    cap_cells = {p for p in cells if a[p] == frame}
    if len(cap_cells) != 1:
        raise ValueError("Launcher needs exactly one frame-colored cap")
    cap = next(iter(cap_cells))
    tail = cells - {cap}
    signals = {int(a[p]) for p in tail}
    if len(signals) != 1 or len(tail) < 2:
        raise ValueError("Launcher tail must have one color and at least two cells")
    signal = next(iter(signals))
    if signal in (bg, frame):
        raise ValueError("Signal, background, and scaffold colors are distinct")
    launches = []
    for direction, (dr, dc) in enumerate(DIRS):
        line = [(cap[0] - k * dr, cap[1] - k * dc) for k in range(1, len(tail) + 1)]
        start = line[-1]
        at_edge = (start[0] == bounds[1] if direction == 0 else
                   start[1] == bounds[2] if direction == 1 else
                   start[0] == bounds[0] if direction == 2 else start[1] == bounds[3])
        if set(line) == tail and at_edge:
            launches.append(direction)
    if len(launches) != 1:
        raise ValueError("Launcher must run straight inward from a field edge")
    direction = launches[0]

    stage = a[:, :d]
    colors = sorted(set(map(int, np.unique(stage))) - {bg, frame})
    if not colors or signal in colors:
        raise ValueError("Catalog identities must differ from signal")
    templates = []
    assigned_ports = set()
    for color in colors:
        body = {tuple(map(int, p)) for p in np.argwhere(stage == color)}
        neighbors = {(r + dr, c + dc) for r, c in body for dr, dc in DIRS
                     if 0 <= r + dr < h and 0 <= c + dc < d
                     and a[r + dr, c + dc] == frame}
        if len(neighbors) != 1:
            raise ValueError("Each catalog template has exactly one scaffold-colored port")
        port = next(iter(neighbors))
        if port in assigned_ports:
            raise ValueError("Catalog ports cannot be shared")
        assigned_ports.add(port)
        footprint = body | {port}
        if not _connected(footprint):
            raise ValueError("Catalog footprints must be four-connected")
        r0, c0 = min(r for r, c in footprint), min(c for r, c in footprint)
        r1, c1 = max(r for r, c in footprint), max(c for r, c in footprint)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh % 2 != 1 or bw % 2 != 1 or min(bh, bw) < 3:
            raise ValueError("Catalog boxes require odd dimensions of at least three")
        side = _port_side(footprint, port)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        rows = [r for r in range(1, h - 1) if a[r, d + 1] == color]
        cols = [c for c in range(d + 2, w - 1) if a[0, c] == color]
        templates.append({"color": color, "rows": rows, "cols": cols,
                          "offsets": {(r - center[0], c - center[1]) for r, c in footprint},
                          "port": (port[0] - center[0], port[1] - center[1]),
                          "direction": side, "shape": (bh, bw)})
    if {tuple(map(int, p)) for p in np.argwhere(stage == frame)} != assigned_ports:
        raise ValueError("Unassigned scaffold cells in catalog")
    edge_colors = set(map(int, np.concatenate((a[0, d + 2:-1], a[1:-1, d + 1]))))
    if not edge_colors <= {frame} | set(colors):
        raise ValueError("Coordinate marks must name catalog templates")
    objects = _instantiate(templates)
    owner = _owners(objects, bounds)
    if set(owner) & cells:
        raise ValueError("Latent footprints cannot overlap the launcher")
    return {"bg": bg, "frame": frame, "signal": signal, "d": d, "bounds": bounds,
            "tail": tail, "cap": cap, "direction": direction,
            "templates": templates, "objects": objects, "owner": owner}


def _walk(origin, direction, owner, bounds):
    """Pixel traversal used by the public input-only solver."""
    dr, dc = DIRS[direction]
    r, c = origin
    trail = set()
    while True:
        r, c = r + dr, c + dc
        if not (bounds[0] <= r <= bounds[1] and bounds[2] <= c <= bounds[3]):
            return trail, None
        if (r, c) in owner:
            return trail, owner[(r, c)]
        trail.add((r, c))


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    p = _decode(a)
    out = a.copy()
    trace = p["tail"] | {p["cap"]}
    pending = deque([(p["cap"], p["direction"])])
    awake = set()
    groups = {}
    for obj in p["objects"]:
        groups.setdefault(obj["group"], []).append(obj)
    while pending:
        origin, direction = pending.popleft()
        trail, hit = _walk(origin, direction, p["owner"], p["bounds"])
        trace |= trail
        if hit is None:
            continue
        group = p["objects"][hit]["group"]
        if group in awake:
            continue
        awake.add(group)
        for obj in groups[group]:
            trace |= obj["cells"]
            pending.append((obj["port"], obj["direction"]))
    for r, c in trace:
        out[r, c] = p["signal"]
    return out


def _project(origin, direction, objects, bounds):
    """Construction/audit: nearest positive projection, rather than walking."""
    r, c = origin
    dr, dc = DIRS[direction]
    limit = (r - bounds[0] + 1 if direction == 0 else bounds[3] - c + 1
             if direction == 1 else bounds[1] - r + 1 if direction == 2 else c - bounds[2] + 1)
    best, hit = limit, None
    for i, obj in enumerate(objects):
        for rr, cc in obj["cells"]:
            distance = (rr - r) * dr + (cc - c) * dc
            if (rr - r) * dc == (cc - c) * dr and 0 < distance < best:
                best, hit = distance, i
    return {(r + k * dr, c + k * dc) for k in range(1, best)}, hit


def _closure(p, mode="broadcast"):
    objects = p["objects"]
    edges = [_project(o["port"], o["direction"], objects, p["bounds"]) for o in objects]
    trail, first = _project(p["cap"], p["direction"], objects, p["bounds"])
    trace = p["tail"] | {p["cap"]} | trail
    todo = deque([] if first is None else [first])
    reached = set()
    if mode == "all":
        todo = deque(range(len(objects)))
    while todo:
        hit = todo.popleft()
        additions = ([hit] if mode == "local" else
                     [i for i, o in enumerate(objects) if o["group"] == objects[hit]["group"]])
        for i in additions:
            if i in reached:
                continue
            reached.add(i)
            trace |= objects[i]["cells"] | edges[i][0]
            if edges[i][1] is not None:
                todo.append(edges[i][1])
    return trace, reached, edges, first


def _paint(a, p, trace):
    out = a.copy()
    for cell in trace:
        out[cell] = p["signal"]
    return out


def _audit(a):
    p = _decode(a)
    trace, reached, edges, first = _closure(p)
    out = _paint(a, p, trace)
    local_trace, local, _, _ = _closure(p, "local")
    all_trace, _, _, _ = _closure(p, "all")
    active = {p["objects"][i]["group"] for i in reached}
    local_groups = {p["objects"][i]["group"] for i in local}
    candidate_groups = {o["group"] for o in p["objects"]}
    products = [(len(t["rows"]), len(t["cols"])) for t in p["templates"]]
    by_color = {t["color"]: t for t in p["templates"]}
    reduced = dict(p, objects=[o for o in p["objects"] if o["center"] ==
                              (min(by_color[o["group"]]["rows"]), min(by_color[o["group"]]["cols"]))])
    reduced_trace = _closure(reduced)[0]
    zipped = dict(p, objects=[o for o in p["objects"] if o["center"] in
                             set(zip(by_color[o["group"]]["rows"], by_color[o["group"]]["cols"]))])
    zipped_trace = _closure(zipped)[0]
    holes = 0
    for i in reached:
        obj = p["objects"][i]
        r0, r1 = min(r for r, c in obj["cells"]), max(r for r, c in obj["cells"])
        c0, c1 = min(c for r, c in obj["cells"]), max(c for r, c in obj["cells"])
        holes += sum((r, c) not in obj["cells"] and (r, c) not in trace
                     for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    return p, out, {"groups": len(candidate_groups), "copies": len(p["objects"]),
                    "awake": len(active), "reached_copies": len(reached),
                    "cascade": bool(active - local_groups),
                    "local_difference": trace != local_trace,
                    "all_difference": trace != all_trace,
                    "single_difference": trace != reduced_trace,
                    "zip_difference": trace != zipped_trace,
                    "holes": holes, "products": products,
                    "direction": p["direction"], "first": first,
                    "source_shapes": [t["shape"] for t in p["templates"]],
                    "active_groups": active, "trace": trace, "reached": reached,
                    "edges": edges}


def _sample_template(rng, color, dimensions=None, nonrect=False):
    h, w = dimensions or [(3, 3), (3, 5), (5, 3), (5, 5)][int(rng.integers(4))]
    kind = int(rng.integers(1 if nonrect else 0, 5))
    mask = np.ones((h, w), bool)
    if kind == 1:
        mask[1:-1, 1:-1] = False
        # Give an outline a genuine inward-supported port while preserving
        # its empty center where dimensions permit it.
        side = int(rng.integers(4))
        port = [(0, w // 2), (h // 2, w - 1),
                (h - 1, w // 2), (h // 2, 0)][side]
        dr, dc = DIRS[side]
        mask[port[0] - dr, port[1] - dc] = True
        if h == w == 3:
            mask[int(rng.choice([0, 2])), int(rng.choice([0, 2]))] = False
    elif kind == 2:
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    elif kind >= 3:
        # Many connected, asymmetric silhouettes, including edge notches.
        target = int(rng.integers(max(5, h + w - 1), h * w))
        for flat in rng.permutation(h * w):
            r, c = divmod(int(flat), w)
            if int(mask.sum()) <= target:
                break
            mask[r, c] = False
            cells = set(map(tuple, np.argwhere(mask)))
            tight = mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
            possible = [(r0, c0) for r0, c0 in cells if
                        sum((r0 == 0, c0 == w - 1, r0 == h - 1, c0 == 0)) == 1]
            if not tight or not _connected(cells) or not possible:
                mask[r, c] = True
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    ports = []
    for port in sorted(cells):
        try:
            side = _port_side(cells, port)
        except ValueError:
            continue
        ports.append((port, side))
    if not ports:
        # Deterministic bounded repair keeps the requested dimensions.
        mask[:] = True
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
        cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
        ports = [((0, w // 2), 0), ((h // 2, w - 1), 1),
                 ((h - 1, w // 2), 2), ((h // 2, 0), 3)]
    port, side = ports[int(rng.integers(len(ports)))]
    return {"color": color, "shape": (h, w), "mask": mask,
            "offsets": {(r - h // 2, c - w // 2) for r, c in cells},
            "port": (port[0] - h // 2, port[1] - w // 2),
            "direction": side, "rows": [], "cols": []}


def _axis(rng, lo, hi, count, spacing, forbidden):
    options = [v for v in range(lo, hi + 1) if v not in forbidden]
    if len(options) < count:
        return None
    for _ in range(40):
        values = sorted(map(int, rng.choice(options, count, replace=False)))
        if all(b - a >= spacing for a, b in zip(values, values[1:])):
            return values
    return None


def _place(rng, templates, products, bounds):
    occupied, used_rows, used_cols = set(), set(), set()
    order = sorted(range(len(templates)), key=lambda i: -products[i][0] * products[i][1] * len(templates[i]["offsets"]))
    for i in order:
        t = templates[i]
        bh, bw = t["shape"]
        for _ in range(120):
            rows = _axis(rng, bounds[0] + bh // 2, bounds[1] - bh // 2,
                         products[i][0], bh + 1, used_rows)
            cols = _axis(rng, bounds[2] + bw // 2, bounds[3] - bw // 2,
                         products[i][1], bw + 1, used_cols)
            if rows is None or cols is None:
                continue
            cells = {(r + y, c + x) for r in rows for c in cols for y, x in t["offsets"]}
            if cells & occupied:
                continue
            t["rows"], t["cols"] = rows, cols
            occupied |= cells
            used_rows.update(rows)
            used_cols.update(cols)
            break
        else:
            return False
    return True


def _launcher_options(bounds, owner, rng):
    options = []
    lo_r, hi_r, lo_c, hi_c = bounds
    for direction, (dr, dc) in enumerate(DIRS):
        for line in range(lo_c + 1, hi_c) if direction % 2 == 0 else range(lo_r + 1, hi_r):
            start = ((hi_r, line) if direction == 0 else (line, lo_c) if direction == 1
                     else (lo_r, line) if direction == 2 else (line, hi_c))
            length = int(rng.integers(2, 4))
            tail = {(start[0] + k * dr, start[1] + k * dc) for k in range(length)}
            cap = (start[0] + length * dr, start[1] + length * dc)
            if not (tail | {cap}) & set(owner):
                options.append((tail, cap, direction))
    rng.shuffle(options)
    return options


def _assemble(rng, templates, products, palette, witness=None, test=False):
    bg, frame, signal = palette[:3]
    min_h = sum(t["shape"][0] for t in templates) + len(templates) + 1
    sw = max(t["shape"][1] for t in templates) + 2
    for _ in range(180):
        h = int(rng.integers(max(min_h, 20 if test else 17), 31))
        inner_w = int(rng.integers(16, 28 - sw))
        w = sw + 3 + inner_w
        bounds = (1, h - 2, sw + 2, w - 2)
        if not _place(rng, templates, products, bounds):
            continue
        objects = _instantiate(templates)
        owner = _owners(objects, bounds)
        for tail, cap, direction in _launcher_options(bounds, owner, rng):
            p = {"objects": objects, "bounds": bounds, "tail": tail,
                 "cap": cap, "direction": direction, "signal": signal}
            trace, reached, _, _ = _closure(p)
            active = {objects[i]["group"] for i in reached}
            if not active:
                continue
            local_trace, local, _, _ = _closure(p, "local")
            local_groups = {objects[i]["group"] for i in local}
            if witness == "cascade" and not (active - local_groups and len(active) < len(templates)):
                continue
            if witness == "replicate" and not any(t["color"] in active and len(t["rows"]) * len(t["cols"]) > 1 for t in templates):
                continue
            if test and (len(active) < (3 if len(templates) == 4 else 2) or
                         trace == local_trace or
                         any(a * b > 4 and t["color"] not in active for t, (a, b) in zip(templates, products))):
                continue
            inp = np.full((h, w), bg, dtype=np.int8)
            inp[:, sw] = frame
            inp[[0, -1], sw + 1:] = frame
            inp[:, [sw + 1, -1]] = frame
            slack = h - min_h
            gaps = list(map(int, rng.multinomial(slack, np.ones(len(templates) + 1) / (len(templates) + 1))))
            top = 1 + gaps[0]
            for index in rng.permutation(len(templates)):
                t = templates[int(index)]
                bh, bw = t["shape"]
                left = int(rng.integers(1, sw - bw))
                center = (top + bh // 2, left + bw // 2)
                for y, x in t["offsets"]:
                    inp[center[0] + y, center[1] + x] = t["color"]
                inp[center[0] + t["port"][0], center[1] + t["port"][1]] = frame
                top += bh + 1 + gaps[int(index) + 1]
                for r in t["rows"]:
                    inp[r, sw + 1] = inp[r, -1] = t["color"]
                for c in t["cols"]:
                    inp[0, c] = inp[-1, c] = t["color"]
            for cell in tail:
                inp[cell] = signal
            inp[cap] = frame
            out = _paint(inp, p, trace)
            parsed, expected, f = _audit(inp)
            if not np.array_equal(expected, out):
                raise AssertionError("Construction/decoding disagreement")
            if witness == "cascade" and not (f["cascade"] and f["zip_difference"] and
                                               f["all_difference"] and f["holes"]):
                continue
            if witness == "replicate" and not (f["single_difference"] and f["local_difference"] and f["holes"]):
                continue
            return {"input": inp, "output": out}
    raise RuntimeError("Unable to pack the fixed relay specification within bounded search")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed % (1 << 64))
    base_palette = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    # Test shift is more groups OR an increased Cartesian product, never a new rule.
    shift = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = "cascade" if split == "train" and index == 0 else "replicate" if split == "train" and index == 1 else None
            if witness == "cascade":
                products = [(2, 2), (1, 1), (1, 1)]
            elif witness == "replicate":
                products = [(2, 1), (1, 2)]
            elif split == "test" and shift == 0:
                products = [(2, 2), (1, 1), (1, 1), (1, 1)]
            elif split == "test":
                products = [tuple([(2, 3), (3, 2)][int(rng.integers(2))]), (1, 1), (1, 1)]
            else:
                group_count = int(rng.integers(2, 4))
                options = [(1, 1), (1, 2), (2, 1), (2, 2)]
                products = [options[int(rng.integers(4))]] + [(1, 1)] * (group_count - 1)
                if rng.random() < .4:
                    products[-1] = options[int(rng.integers(3))]
            palette = list(np.roll(base_palette, index + 1)) if vary_palette else base_palette
            if vary_palette and (split == "test" or index >= 2):
                palette = list(map(int, rng.permutation(10)))
            palette = list(map(int, palette))
            for _ in range(12):
                templates = [_sample_template(rng, palette[3 + j], (3, 3) if a * b >= 4 else None,
                                              nonrect=j == 0) for j, (a, b) in enumerate(products)]
                pair = _assemble(rng, templates, products, palette, witness, split == "test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            p, expected, f = _audit(a)
            assert np.array_equal(expected, b), "Incorrect broadcast closure or ray geometry"
            assert np.array_equal(solve(a), b), "Pixel traversal disagrees with geometric projection"
            assert np.array_equal(a[:, :p["d"] + 2], b[:, :p["d"] + 2]), "Catalog/divider/left frame must survive"
            assert np.array_equal(a[[0, -1]], b[[0, -1]]) and np.array_equal(a[:, -1], b[:, -1])
            assert set(map(int, np.unique(b[1:-1, p["d"] + 2:-1]))) <= {p["bg"], p["signal"]}
            for group in f["active_groups"]:
                indices = {i for i, o in enumerate(p["objects"]) if o["group"] == group}
                assert indices <= f["reached"], "A broadcast must activate the full Cartesian group"
            for i in f["reached"]:
                hit = f["edges"][i][1]
                assert hit is None or hit in f["reached"], "Activation set must be a fixed point"
            assert b[p["cap"]] == p["signal"]
            facts[split].append(f)
    train = facts["train"]
    assert any(f["cascade"] and f["zip_difference"] and f["all_difference"] for f in train), "Missing remote broadcast/Cartesian/filtering witness"
    assert any(f["single_difference"] and f["local_difference"] for f in train), "Missing replication witness"
    assert any(f["holes"] for f in train), "Transparency must be visible"
    assert any((2, 2) in f["products"] for f in train)
    # Generator's shift is witnessed by visible input features, not hidden metadata.
    max_groups = max(f["groups"] for f in train)
    max_product = max(a * b for f in train for a, b in f["products"])
    assert all(f["groups"] > max_groups or max(a * b for a, b in f["products"]) > max_product
               for f in facts["test"]), "Missing compositional train/test shift"
    for pair, f in zip(puzzle["test"], facts["test"]):
        p = _decode(_grid(pair["input"]))
        assert f["awake"] >= (3 if f["groups"] == 4 else 2) and f["local_difference"]
        assert all(t["color"] in f["active_groups"] for t in p["templates"]
                   if len(t["rows"]) * len(t["cols"]) > 4), "Expanded group must participate"


def self_test():
    signatures, shapes, masks, products, directions, group_counts = set(), set(), set(), set(), set(), set()
    within = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s]))
        local_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                p, _, f = _audit(a)
                shapes.add(a.shape)
                local_shapes.add(a.shape)
                products.update(f["products"])
                directions.add(p["direction"])
                group_counts.add(f["groups"])
                for t in p["templates"]:
                    masks.add((t["shape"], tuple(sorted(t["offsets"])), t["port"]))
        within += len(local_shapes) >= 3
    assert len(signatures) == 42 and len(shapes) >= 25 and len(masks) >= 50
    assert within >= 35 and directions == set(range(4))
    assert {(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)} <= products
    assert {2, 3, 4} <= group_counts
    p = generate_puzzle(213881667, 2, 1)
    assert tuple((_key(x["input"]), _key(x["output"])) for s in p for x in p[s]) == tuple(
        (_key(x["input"]), _key(x["output"])) for s in p for x in generate_puzzle(213881667, 2, 1)[s])
    for split in p:
        for pair in p[split]:
            permutation = np.random.default_rng(571).permutation(10)
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a corrupted output")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            _counts(a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal example counts accepted")
    _edge_case_tests()


def _edge_case_tests():
    """Literal geometry checks for cycles, misses and an incomplete group."""
    a = np.zeros((17, 27), dtype=np.int8)
    a[:, 7] = 1
    a[[0, -1], 8:] = 1
    a[:, [8, -1]] = 1
    # Permanent catalog: A 3x5 right port, B 5x3 offset left port.
    a[1:4, 1:6] = 3
    a[2, 5] = 1
    a[6:11, 1:4] = 4
    a[7, 1] = 1
    a[6, [8, 26]] = 3
    a[[0, 16], 12] = 3
    a[7, [8, 26]] = 4
    a[[0, 16], 20] = 4
    a[1:3, 12] = 2
    a[3, 12] = 1
    # The two ports face each other along row 6. Each group emits once.
    expected = a.copy()
    expected[1:5, 12] = 2
    expected[5:8, 10:15] = 2
    expected[5:10, 19:22] = 2
    expected[6, 15:19] = 2
    assert np.array_equal(solve(a), expected)
    assert np.count_nonzero(expected[1:-1, 9:-1] == 2) == 38
    # A new boundary column misses both latent footprints.
    miss = a.copy()
    miss[1:4, 12] = 0
    miss[1:3, 24] = 2
    miss[3, 24] = 1
    expected = miss.copy()
    expected[1:16, 24] = 2
    assert np.array_equal(solve(miss), expected)
    # B loses its row axis: it is no longer present to stop A's ray.
    incomplete = a.copy()
    incomplete[7, [8, 26]] = 1
    expected = incomplete.copy()
    expected[1:5, 12] = 2
    expected[5:8, 10:15] = 2
    expected[6, 15:26] = 2
    assert np.array_equal(solve(incomplete), expected)


if __name__ == "__main__":
    self_test()
    print("42-seed broadcast-relay self-test passed")
