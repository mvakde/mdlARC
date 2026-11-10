"""Registered carriers and a finite cable on their shared exterior envelope.

All conventions are visible in the input. Only stdlib and NumPy are used.
The rail is implicit; no parent generator is imported or executed here.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
AUDIT_SEEDS = (0, 600099169, 608900318, 541162495, 64369060, 745869407,
               81411295, 50610148, 403832544, 532411152, 508049646,
               129466551, 123392331, 176845757, 89815406, 500953974,
               293924069, 102299136, 735798582, 39403872, 697140241,
               762545273, 512580277, 879931141, 798723219, 858803433,
               20011142, 232535326, 351498046, 362359853, 89196047,
               358629320, 506231208, 308469257, 687995626, 373374498,
               849023843, 924064955, 789111227, 376598590, 935330472, 33067947)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((g >= 0) & (g <= 9))), "invalid colors")
    return g


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(cells, directions=DIR4):
    unseen, result = set(cells), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        reached, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    reached.add(q)
                    stack.append(q)
        result.append(reached)
    return result


def _local(p, anchor, forward, outward):
    r, c = p[0] - anchor[0], p[1] - anchor[1]
    return r * forward[0] + c * forward[1], r * outward[0] + c * outward[1]


def _global(p, anchor, forward, outward):
    u, v = p
    return (anchor[0] + u * forward[0] + v * outward[0],
            anchor[1] + u * forward[1] + v * outward[1])


def _route(obstacles, length):
    """The least height profile with unit clearance and slopes -1, 0, +1.

    Each material cell casts a 45-degree clearance cone. Their upper envelope,
    together with the rail's baseline v=1, gives one route cell per axial step.
    Obstacles beyond the finite cable endpoint still influence its approach.
    """
    _require(bool(obstacles), "no registered material")
    return [(u, max(1, max(v + 1 - abs(u - x) for x, v in obstacles)))
            for u in range(1, length + 1)]


def _decode(input_grid):
    g = _grid(input_grid)
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > g.size / 2, "background must occupy a majority")
    h, w = g.shape
    foreground = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    boundary = {p for p in foreground if p[0] in (0, h - 1) or p[1] in (0, w - 1)}
    _require(len(boundary) == 1, "exactly one input boundary beacon is required")
    anchor = next(iter(boundary))
    r, c = anchor
    _require((r in (0, h - 1)) != (c in (0, w - 1)), "corner beacon is ambiguous")
    forward = ((1, 0) if r == 0 else (-1, 0)) if r in (0, h - 1) else (
        (0, 1) if c == 0 else (0, -1))
    marker = int(g[anchor])
    candidates = []
    for outward in ((forward[1], -forward[0]), (-forward[1], forward[0])):
        start = _global((1, 1), anchor, forward, outward)
        if not _inside(start, g.shape) or int(g[start]) in (bg, marker):
            continue
        cable_color = int(g[start])
        slack = {tuple(map(int, p)) for p in np.argwhere(g == cable_color)}
        local_slack = {_local(p, anchor, forward, outward) for p in slack}
        if (len(_components(slack, DIR8)) == 1 and
                all(1 <= u <= 3 and v >= 1 for u, v in local_slack)):
            candidates.append((outward, cable_color, slack, local_slack))
    _require(len(candidates) == 1, "cable must visibly identify one winding side")
    outward, cable_color, slack, local_slack = candidates[0]
    _require(2 <= len(slack) <= 36, "invalid unary cable length")
    remaining = foreground - slack - {anchor}
    carriers = []
    claimed = set()
    for cells in _components(remaining):
        marks = [p for p in sorted(cells) if int(g[p]) == marker]
        _require(len(cells) >= 2 and len(marks) == 1, "carrier needs one registration cell")
        source = {_local(p, anchor, forward, outward): int(g[p]) for p in cells}
        _require(all(u >= 5 for u, v in source), "carriers must follow the slack region")
        registration = _local(marks[0], anchor, forward, outward)
        target = {(u, v - registration[1]): color for (u, v), color in source.items()}
        _require(all(_inside(_global(p, anchor, forward, outward), g.shape) for p in target),
                 "registered carriers must fit completely")
        _require(not claimed.intersection(target), "registered carriers overlap")
        claimed.update(target)
        carriers.append({"cells": cells, "source": source, "target": target,
                         "registration": registration, "shift": -registration[1]})
    _require(1 <= len(carriers) <= 4, "invalid carrier count")
    _require(all(v <= u - 1 for u, v in claimed), "obstacle would displace the fixed launch")
    path = _route(claimed, len(slack))
    _require(path[0] == (1, 1), "launch must be fixed")
    _require(not set(path).intersection(claimed), "cable enters registered material")
    return {"grid": g, "background": bg, "anchor": anchor, "forward": forward,
            "outward": outward, "marker": marker, "cable_color": cable_color,
            "slack": slack, "local_slack": local_slack, "length": len(slack),
            "carriers": carriers, "obstacles": claimed, "path": path}


def _draw(data, path=None, keep_slack=False):
    out = np.full(data["grid"].shape, data["background"], dtype=int)
    out[data["anchor"]] = data["marker"]
    for carrier in data["carriers"]:
        for p, color in carrier["target"].items():
            q = _global(p, data["anchor"], data["forward"], data["outward"])
            _require(out[q] == data["background"], "target collision")
            out[q] = color
    if keep_slack:
        points = data["slack"]
    else:
        points = [_global(p, data["anchor"], data["forward"], data["outward"])
                  for p in (data["path"] if path is None else path)]
    for p in points:
        if _inside(p, out.shape):
            _require(out[p] == data["background"], "cable collision")
            out[p] = data["cable_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, register material, and tighten the finite cable from input only."""
    return _draw(_decode(input_grid))


def _bbox_route(data):
    boxes = set()
    for carrier in data["carriers"]:
        points = carrier["target"]
        a, b = min(u for u, v in points), max(u for u, v in points)
        c, d = min(v for u, v in points), max(v for u, v in points)
        boxes.update((u, v) for u in range(a, b + 1) for v in range(c, d + 1))
    return _route(boxes, data["length"])


def _features(pair):
    d = _decode(pair["input"])
    visible = [p for p in d["path"] if _inside(
        _global(p, d["anchor"], d["forward"], d["outward"]), d["grid"].shape)]
    source = set().union(*(set(c["source"]) for c in d["carriers"]))
    source_path = _route(source, d["length"])
    visible_source = [p for p in source_path if _inside(
        _global(p, d["anchor"], d["forward"], d["outward"]), d["grid"].shape)]
    intervals = sorted((min(u for u, v in c["target"]), max(u for u, v in c["target"]))
                       for c in d["carriers"])
    gaps = {u for (_, end), (start, _) in zip(intervals, intervals[1:])
            for u in range(end + 1, start)}
    bridge = any(u in gaps and v > 1 for u, v in visible)
    normalized = []
    for c in d["carriers"]:
        mark = c["registration"][0]
        normalized.append(tuple(sorted((u - mark, v) for u, v in c["target"])))
    contributors = 0
    for carrier in d["carriers"]:
        others = d["obstacles"] - set(carrier["target"])
        alternative = _route(others, d["length"]) if others else [(u, 1) for u in range(1, d["length"] + 1)]
        contributors += alternative != d["path"]
    return {"data": d, "visible": visible, "clipped": len(visible) < d["length"],
            "source_changes_cable": visible != visible_source,
            "bbox_changes_cable": d["path"] != _bbox_route(d),
            "diagonal_slack": len(_components(d["local_slack"], DIR4)) > 1,
            "bridge": bridge, "shapes": normalized, "contributors": contributors,
            "full": len(visible) == d["length"] and d["path"][-1][1] == 1
                    and d["length"] > max(u for u, v in d["obstacles"])}


def _sweep_route(obstacles, length):
    """Constraint propagation for validation of the analytic clearance formula."""
    n = max(length, max(u for u, v in obstacles)) + 1
    heights = [1] * n
    for u, v in obstacles:
        heights[u] = max(heights[u], v + 1)
    for u in range(1, n):
        heights[u] = max(heights[u], heights[u - 1] - 1)
    for u in range(n - 2, -1, -1):
        heights[u] = max(heights[u], heights[u + 1] - 1)
    return [(u, heights[u]) for u in range(1, length + 1)]


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, fs = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _features(pair)
            d = f["data"]
            _require(np.array_equal(_draw(d), out), "input-only solution mismatch")
            _require(not np.array_equal(inp, out), "identity example")
            _require(d["path"] == _sweep_route(d["obstacles"], d["length"]),
                     "cable is not the minimal clearance profile")
            _require(all(abs(bv - av) <= 1 for (_, av), (_, bv) in zip(d["path"], d["path"][1:])),
                     "cable loses eight-connectivity")
            _require(out[d["anchor"]] == inp[d["anchor"]], "beacon moved")
            _require(int(np.sum(out == d["cable_color"])) == len(f["visible"]),
                     "length or clipping is wrong")
            for carrier in d["carriers"]:
                # The exact colored offsets and axial coordinates must survive.
                mark_u, mark_v = carrier["registration"]
                _require(carrier["target"][(mark_u, 0)] == d["marker"], "registration failed")
                _require(carrier["target"] == {(u, v - mark_v): color
                         for (u, v), color in carrier["source"].items()}, "carrier distorted")
                for p, color in carrier["target"].items():
                    q = _global(p, d["anchor"], d["forward"], d["outward"])
                    _require(out[q] == color, "carrier color not preserved")
            fs[split].append(f)
    training = fs["train"]
    # These are curriculum witnesses, not requirements on every family input.
    _require(any(f["full"] and f["diagonal_slack"] and f["bridge"] and f["contributors"] >= 2 and
                 f["source_changes_cable"] and f["bbox_changes_cable"] for f in training),
             "training lacks the joint contour and unary-count witness")
    _require(len({f["data"]["length"] for f in training}) >= 2, "only one demonstrated cable length")
    _require(any(any(c["shift"] for c in f["data"]["carriers"]) for f in training), "no registration movement")
    _require(any(not np.array_equal(_draw(f["data"], keep_slack=True),
                                   np.asarray(p["output"])) for f, p in zip(training, puzzle["train"])),
             "omitting B does not change training")
    if any(f["clipped"] for f in fs["test"]):
        _require(any(f["clipped"] for f in training), "clipping lacks a demonstration")


def _shape(rng, kind=None, compact=False, outward_mark=False):
    """Four-connected solids, frames, branches, and genuine stepped outlines."""
    kind = int(rng.integers(6)) if kind is None else kind
    width = int(rng.integers(1 if compact else 2, 4 if compact else 6))
    height = int(rng.integers(2, 6))
    if kind == 0:
        cells = {(u, v) for u in range(width) for v in range(height)}
    elif kind == 1:
        width = max(3, width)
        cells = {(u, v) for u in range(width) for v in range(height)
                 if u in (0, width - 1) or v in (0, height - 1)}
    elif kind == 2:
        cells = {(u, 0) for u in range(width)} | {(width - 1, v) for v in range(height)}
    elif kind == 3:
        width = max(3, width)
        cells = {(u, 0) for u in range(width)}
        for u in range(width):
            cells.update((u, v) for v in range(int(rng.integers(1, height + 1))))
    elif kind == 4:
        width = max(3, width)
        cells = {(u, v) for u in range(width) for v in range(u + 1)}
    else:
        cells = {(u, 0) for u in range(width)}
        for u in sorted(set([0, width - 1, int(rng.integers(width))])):
            sign = int(rng.choice((-1, 1)))
            cells.update((u, sign * v) for v in range(1, int(rng.integers(2, height + 1))))
    if rng.random() < .5 and kind != 4:
        cells = {(width - 1 - u, v) for u, v in cells}
    points = sorted(cells)
    mark = points[int(rng.integers(len(points)))]
    if kind == 4:
        mark = (0, 0)
    elif outward_mark:
        minimum = min(v for u, v in cells)
        choices = sorted(p for p in cells if p[1] == minimum)
        mark = choices[int(rng.integers(len(choices)))]
    return {(u - mark[0], v - mark[1]) for u, v in cells}


def _slack(rng, length, diagonal):
    depth = (length + 2) // 3 + 3 + int(rng.integers(0, 3))
    cells = {(1, 1), (2, 2)} if diagonal else {(1, 1)}
    style = int(rng.integers(3))
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in DIR8
                           if 1 <= u + du <= 3 and 1 <= v + dv <= depth} - cells)
        if diagonal:
            frontier = [p for p in frontier if p[1] >= 2 and p != (1, 2)]
        _require(bool(frontier), "slack growth exhausted")
        if style == 0:
            weights = np.array([1 / p[1] ** 2 for p in frontier])
        elif style == 1:
            weights = np.array([(1 + sum((u + du, v + dv) in cells for du, dv in DIR8)) ** 2
                                for u, v in frontier], dtype=float)
        else:
            weights = np.ones(len(frontier))
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _make_pair(rng, palette, count, lesson, turns, reflected, gap_range):
    bg, marker, cable = map(int, palette[:3])
    bodies = list(map(int, palette[3:]))
    shapes = [_shape(rng, 4 if lesson == "full" and j == 0 else None, count >= 3,
                     outward_mark=lesson == "full")
              for j in range(count)]
    x = int(rng.integers(5, 8))
    targets = []
    for shape in shapes:
        start = x - min(u for u, v in shape)
        body = int(rng.choice(bodies))
        target = {}
        for u, v in sorted(shape):
            color = marker if (u, v) == (0, 0) else body
            if color != marker and rng.random() < .12:
                color = int(rng.choice(bodies))
            target[(u + start, v)] = color
        targets.append(target)
        x = max(u for u, v in target) + 1 + int(rng.integers(*gap_range))
    far = max(u for t in targets for u, v in t)
    peak = max(v for t in targets for u, v in t)
    if far > 27:
        return None
    if lesson == "full":
        length = far + max(1, peak) + 1
        if length > 28:
            return None
        width = int(rng.integers(length + 1, min(30, length + 5) + 1))
    else:
        width = int(rng.integers(far + 2, min(30, far + 7) + 1))
        if lesson == "clip":
            length = width + int(rng.integers(1, min(5, 37 - width)))
        elif lesson == "prefix":
            length = int(rng.integers(3, min(11, far + 1)))
        else:
            length = int(rng.integers(3, width))
    slack = _slack(rng, length, lesson == "full" or rng.random() < .35)
    minimum = min(v for t in targets for u, v in t)
    outward_extent = max(peak, max(v for u, v in slack))
    guide = max(2, 1 - minimum) + int(rng.integers(1, 6))
    height = guide + outward_extent + int(rng.integers(2, 6))
    if height > 30:
        return None
    if lesson != "clip" and rng.random() < .18:
        height = width = max(height, width)
    inp = np.full((height, width), bg, dtype=int)
    inp[guide, 0] = marker
    for u, v in slack:
        inp[guide + v, u] = cable
    for j, target in enumerate(targets):
        lo = 1 - guide - min(v for u, v in target)
        hi = height - 2 - guide - max(v for u, v in target)
        options = [k for k in range(lo, hi + 1) if k or lesson != "full"]
        if not options:
            return None
        shift = int(rng.choice(options))
        for (u, v), color in target.items():
            inp[guide + v + shift, u] = color
    if reflected:
        inp = np.flipud(inp).copy()
    inp = np.rot90(inp, turns).copy()
    try:
        out = solve(inp)
        f = _features({"input": inp, "output": out})
    except AssertionError:
        return None
    if np.array_equal(inp, out) or lesson == "clip" and not f["clipped"]:
        return None
    if lesson == "full" and not (f["full"] and f["bridge"] and f["contributors"] >= 2 and
                                 f["source_changes_cable"] and f["bbox_changes_cable"]):
        return None
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample one coherent family; curriculum parameters never affect the solver."""
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    stable_palette, clipping = rng.random() < .3, rng.random() < .45
    regime, base_turns = int(rng.integers(3)), int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    full_length = None
    for split, n in (("train", a), ("test", b)):
        for index in range(n):
            lesson = "full" if split == "train" and index == 0 else (
                "clip" if clipping and index == (1 if split == "train" else 0) else
                "prefix" if split == "train" and index == 1 else "random")
            for attempt in range(300):
                colors = palette if stable_palette else rng.permutation(10)
                if split == "train":
                    count = 2 if index == 0 else int(rng.integers(1, 3))
                elif regime == 1:
                    count = int(rng.integers(3, 5))
                elif regime == 2:
                    count = int(rng.integers(2, 4))
                else:
                    count = int(rng.integers(1, 4))
                turns = (base_turns if split == "train" else (base_turns + 1) % 4) if regime == 0 else int(rng.integers(4))
                # In the spacing regime, tests separate objects farther apart;
                # training's full witness teaches a shared bridge across a gap.
                gaps = (3, 5) if split == "test" and regime == 2 else (1, 3)
                pair = _make_pair(rng, colors, count, lesson, turns, rng.random() < .5, gaps)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                length = int(np.sum(pair["input"] == _decode(pair["input"])["cable_color"]))
                if split == "train" and index == 1 and length == full_length:
                    continue
                if split == "train" and index == 0:
                    full_length = length
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded procedural sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    payload = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded 42-seed semantic, curriculum, variation and metamorphic audit."""
    signatures, dims, silhouettes, lengths, backgrounds = set(), set(), set(), set(), set()
    counts, directions, clipping_edges = set(), set(), set()
    bridge_pairs = bbox_pairs = source_pairs = diagonal_pairs = clipped_pairs = 0
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                _require(np.array_equal(solve(before), pair["output"]), "solve failed")
                _require(np.array_equal(before, pair["input"]), "solver mutated input")
                f, shape = _features(pair), before.shape
                d = f["data"]
                dims.add(shape)
                silhouettes.update(f["shapes"])
                lengths.add(d["length"])
                backgrounds.add(d["background"])
                counts.add(len(d["carriers"]))
                directions.add(d["forward"])
                bridge_pairs += f["bridge"]
                bbox_pairs += f["bbox_changes_cable"]
                source_pairs += f["source_changes_cable"]
                diagonal_pairs += f["diagonal_slack"]
                clipped_pairs += f["clipped"]
                for p in d["path"]:
                    r, c = _global(p, d["anchor"], d["forward"], d["outward"])
                    for edge, outside in (("top", r < 0), ("bottom", r >= shape[0]),
                                          ("left", c < 0), ("right", c >= shape[1])):
                        if outside:
                            clipping_edges.add(edge)
    _require(len(signatures) == 42, "puzzles repeat")
    _require(len(dims) >= 50 and len(silhouettes) >= 60 and len(lengths) >= 15, "narrow structural variation")
    _require(backgrounds == set(range(10)), "palette role coverage")
    _require(counts == {1, 2, 3, 4} and len(directions) == 4, "count or direction coverage")
    _require(clipping_edges == {"top", "bottom", "left", "right"}, "clipping edge coverage")
    _require(min(bridge_pairs, bbox_pairs, source_pairs, diagonal_pairs) >= 42, "parent interaction coverage")
    # Every legal explicit count combination is exercised by verify.py; retain
    # boundary calls here and reject types that Python would otherwise coerce.
    for a, b in ((2, 1), (10, 4), (10, 1), (5, 4)):
        p = generate_puzzle(909, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(2718, 2, 1)
    _require(_fingerprint(p) == _fingerprint(generate_puzzle(2718, 2, 1)), "not deterministic")
    rng = np.random.default_rng(811)
    original = p["train"][0]
    d = _decode(original["input"])
    carrier = d["carriers"][0]
    relocated = original["input"].copy()
    for q in carrier["cells"]:
        relocated[q] = d["background"]
    for delta in (-1, 1):
        shifted = {(r + delta * d["outward"][0], c + delta * d["outward"][1]): int(d["grid"][r, c])
                   for r, c in carrier["cells"]}
        if all(_inside(q, relocated.shape) and q[0] not in (0, relocated.shape[0] - 1)
               and q[1] not in (0, relocated.shape[1] - 1) and relocated[q] == d["background"]
               for q in shifted):
            for q, color in shifted.items():
                relocated[q] = color
            break
    else:
        raise AssertionError("relocation metamorphism has no legal offset")
    _require(np.array_equal(solve(relocated), original["output"]), "source offset leaked into cable")
    for _ in range(30):
        new_slack = _slack(rng, d["length"], False)
        points = {_global(q, d["anchor"], d["forward"], d["outward"]) for q in new_slack}
        if points != d["slack"] and all(_inside(q, d["grid"].shape) and
               q[0] not in (0, d["grid"].shape[0] - 1) and q[1] not in (0, d["grid"].shape[1] - 1)
               for q in points):
            reshaped = d["grid"].copy()
            for q in d["slack"]:
                reshaped[q] = d["background"]
            for q in points:
                reshaped[q] = d["cable_color"]
            _require(np.array_equal(solve(reshaped), original["output"]), "slack shape leaked into cable")
            break
    else:
        raise AssertionError("equal-length slack metamorphism exhausted")
    for turns in (1, 2, 3):
        q = {s: [] for s in p}
        for s in p:
            for pair in p[s]:
                colors = rng.permutation(10)
                q[s].append({k: colors[np.fliplr(np.rot90(v, turns))] for k, v in pair.items()})
        validate_generated_puzzle(q)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    d = _decode(bad["test"][0]["input"])
    bad["test"][0]["output"][d["anchor"]] = d["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted beacon accepted")
    # Explicit nonconvex joint profile: two humps joined by a high bridge.
    material = {(5, 2), (6, 4), (10, 3)}
    expected = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5),
                (7, 4), (8, 3), (9, 3), (10, 4), (11, 3), (12, 2), (13, 1)]
    _require(_route(material, 13) == expected == _sweep_route(material, 13), "shared-envelope regression")
    # A fully recoverable carrier registers against the lateral canvas edge.
    # Its cable leaves the canvas over the peak and then re-enters; omitted
    # route cells still consume length. This is solver coverage, not a sampled
    # training/test regime (the sampler clips axial tails).
    edge = np.zeros((10, 13), dtype=int)
    edge[4, 0] = 2
    for u, v in [(1, 1)] + [(u, v) for v in range(1, 4) for u in (2, 3)] + [(1, 4), (2, 4), (3, 4)]:
        edge[4 + v, u] = 4
    edge[3:9, 6] = 3
    edge[3, 6] = 2
    ed = _decode(edge)
    eo = solve(edge)
    _require(ed["length"] == 10 and int(np.sum(eo == 4)) == 9,
             "lateral clipping must consume the off-grid cell")
    _require(eo[8, 8] == 4, "cable must re-enter after lateral clipping")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "grid_shapes": len(dims), "registration_relative_shapes": len(silhouettes),
            "lengths": sorted(lengths), "carrier_counts": sorted(counts),
            "bridge_pairs": bridge_pairs, "bbox_counterexamples": bbox_pairs,
            "source_contour_counterexamples": source_pairs, "diagonal_slack_pairs": diagonal_pairs,
            "clipped_pairs": clipped_pairs, "clipping_edges": sorted(clipping_edges)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
