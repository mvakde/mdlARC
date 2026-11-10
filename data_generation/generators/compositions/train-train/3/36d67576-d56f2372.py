"""Transfer a reference into ambiguous sockets; extract its unique bilateral result.

All reasoning in solve is input-only.  Construction and validation do not use
the seed or construction metadata to interpret an input.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np


_OPS = ("id", "r90", "r180", "r270", "lr", "tb", "diag", "anti")


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(x) for x in a.flat)


def _sig(mapping):
    return tuple(sorted((r, c, int(v)) for (r, c), v in mapping.items()))


def _normal(mapping):
    top = min(r for r, _ in mapping)
    left = min(c for _, c in mapping)
    return {(r - top, c - left): v for (r, c), v in mapping.items()}, (top, left)


def _point(p, op):
    r, c = p
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-r, c), (c, r), (-c, -r))[op]


def _variants(retained, details):
    """Normalize by the retained cells, never by the full colored footprint."""
    result = []
    for op in range(8):
        moved = {_point(p, op): v for p, v in retained.items()}
        moved, (r0, c0) = _normal(moved)
        ink = {}
        for p, v in details.items():
            r, c = _point(p, op)
            ink[r - r0, c - c0] = v
        result.append((moved, ink, op))
    return result


def _options(retained, details, socket):
    target, (r0, c0) = _normal(socket)
    found = {}
    for stencil, ink, op in _variants(retained, details):
        if stencil == target:
            placed = {(r + r0, c + c0): v for (r, c), v in ink.items()}
            signature = _sig(placed)
            if signature not in found:
                found[signature] = (placed, [])
            found[signature][1].append(op)
    return list(found.values())


def _neighbors(p):
    r, c = p
    return ((r + dr, c + dc) for dr in (-1, 0, 1)
            for dc in (-1, 0, 1) if dr or dc)


def _connected(points):
    points = set(points)
    if not points:
        return False
    seen = {min(points)}
    stack = list(seen)
    while stack:
        for p in _neighbors(stack.pop()):
            if p in points and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == points


def _box(points):
    return (min(r for r, _ in points), min(c for _, c in points),
            max(r for r, _ in points) + 1, max(c for _, c in points) + 1)


def _crop(mapping, background):
    r0, c0, r1, c1 = _box(mapping)
    result = np.full((r1 - r0, c1 - c0), background, dtype=int)
    for (r, c), color in mapping.items():
        result[r - r0, c - c0] = color
    return result


def _symmetric(points, axis="lr"):
    r0, c0, r1, c1 = _box(points)
    if axis == "lr":
        return {(r, c0 + c1 - 1 - c) for r, c in points} == set(points)
    return {(r0 + r1 - 1 - r, c) for r, c in points} == set(points)


def _components(grid, background):
    remaining = {tuple(int(x) for x in p) for p in np.argwhere(grid != background)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        comp, stack = {first}, [first]
        while stack:
            for p in _neighbors(stack.pop()):
                if p in remaining:
                    remaining.remove(p)
                    comp.add(p)
                    stack.append(p)
        result.append({p: int(grid[p]) for p in comp})
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=False)


def _scene(value):
    grid = _grid(value)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if frequencies[bg] <= grid.size // 2:
        raise ValueError("Background must occupy more than half the input")
    objects = _components(grid, bg)
    if not 3 <= len(objects) <= 5:
        raise ValueError("Expected a reference and two to four candidates")
    color_sets = [set(obj.values()) for obj in objects]
    prototypes = [i for i, colors in enumerate(color_sets) if len(colors) == 4]
    if len(prototypes) != 1:
        raise ValueError("Expected one four-color reference")
    pi = prototypes[0]
    shared = set.intersection(*color_sets)
    if len(shared) != 2:
        raise ValueError("Expected two common socket colors")
    markers = [c for c in shared if all(list(o.values()).count(c) == 1 for o in objects)]
    if len(markers) != 1:
        raise ValueError("Expected a unique one-cell marker role")
    marker = markers[0]
    scaffold = next(iter(shared - {marker}))
    if any(list(o.values()).count(scaffold) < 3 for o in objects):
        raise ValueError("Socket scaffold too small")
    others = [o for i, o in enumerate(objects) if i != pi]
    bodies = [set(o.values()) - shared for o in others]
    if any(len(x) != 1 for x in bodies) or any(x != bodies[0] for x in bodies):
        raise ValueError("Candidates must share a separate body color")
    body = next(iter(bodies[0]))
    prototype = objects[pi]
    inks = set(prototype.values()) - shared
    if body in inks or len(inks) != 2:
        raise ValueError("Reference ink and body colors must differ")
    retained = {p: v for p, v in prototype.items() if v in shared}
    details = {p: v for p, v in prototype.items() if v in inks}
    candidates = []
    successes = []
    for obj in others:
        socket = {p: v for p, v in obj.items() if v in shared}
        options = _options(retained, details, socket)
        if not options:
            raise ValueError("Candidate socket does not match the reference")
        legal = []
        for ink, ops in options:
            if all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                   and grid[r, c] == bg for r, c in ink):
                completed = dict(obj)
                completed.update(ink)
                legal.append((ink, ops, completed))
                if _symmetric(completed):
                    successes.append((len(candidates), ink, completed))
        candidates.append({"visible": obj, "socket": socket, "options": options,
                           "legal": legal})
    return {"grid": grid, "bg": bg, "scaffold": scaffold, "marker": marker,
            "body": body, "inks": inks, "retained": retained, "details": details,
            "prototype": prototype, "candidates": candidates, "successes": successes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Find the unique colored stamp yielding a left-right symmetric silhouette."""
    scene = _scene(input_grid)
    if len(scene["successes"]) != 1:
        raise ValueError("Expected exactly one candidate and colored completion")
    return _crop(scene["successes"][0][2], scene["bg"])


def _scaffold(rng, mode, sc, mk):
    h = int(rng.choice((3, 5)))
    w = int(rng.integers(3, 6))
    if mode == "bar":
        h, w = 1, int(rng.integers(4, 7))
        points = {(0, c) for c in range(w)}
        marker = (0, 0)
    elif mode == "cross":
        h = w = int(rng.choice((3, 5)))
        points = {(h // 2, c) for c in range(w)} | {(r, w // 2) for r in range(h)}
        marker = (h // 2, w // 2)
    elif mode == "rails":
        points = {(r, c) for r in range(h) for c in range(w)
                  if r in (0, h - 1) or c in (0, w - 1)}
        marker = (h // 2, 0)
    elif mode == "fork":
        points = {(r, 0) for r in range(h)}
        for r in range(h // 2 + 1):
            length = w if r == 0 else int(rng.integers(1, w + 1))
            points.update((rr, c) for rr in (r, h - 1 - r) for c in range(length))
        marker = (h // 2, 0)
    else:  # an open C: top-bottom ambiguity, but left and right differ
        points = {(r, 0) for r in range(h)}
        points |= {(r, c) for r in (0, h - 1) for c in range(w)}
        marker = (h // 2, 0)
    return {p: mk if p == marker else sc for p in points}


def _motif(rng, mode, colors, detail_range):
    bg, sc, mk, body, a, b = colors
    retained = _scaffold(rng, mode, sc, mk)
    h = max(r for r, _ in retained) + 1
    w = max(c for _, c in retained) + 1
    frontier = sorted({p for q in retained for p in _neighbors(q)
                       if p not in retained and -1 <= p[0] <= h and -1 <= p[1] <= w})
    internal = [p for p in frontier if 0 <= p[0] < h and 0 <= p[1] < w]
    if len(internal) >= detail_range[0] and rng.random() < 0.28:
        frontier = internal
    amount = int(rng.integers(detail_range[0], min(detail_range[1], len(frontier)) + 1))
    chosen = [frontier[int(i)] for i in rng.choice(len(frontier), amount, replace=False)]
    details = {p: (a if i == 0 else b if i == 1 else int(rng.choice((a, b))))
               for i, p in enumerate(chosen)}
    # The reference itself can face any direction.  Every coordinate is visible.
    variant = _variants(retained, details)[int(rng.integers(8))]
    return variant[0], variant[1]


def _reflect(p, axis, axis2):
    r, c = p
    return (r, axis2 - c) if axis == "lr" else (axis2 - r, c)


def _closure(points, axis, axis2):
    return set(points) | {_reflect(p, axis, axis2) for p in points}


def _body_pattern(rng, required, forbidden, axis, axis2):
    """A symmetric silhouette, with optional outlines, bands, or irregular arms."""
    base = _closure(required, axis, axis2)
    banned = _closure(forbidden, axis, axis2) if forbidden else set()
    if base & banned:
        return None
    r0, c0, r1, c1 = _box(base)
    pr, pc = int(rng.integers(2)), int(rng.integers(2))
    r0, c0, r1, c1 = r0 - pr, c0 - pc, r1 + pr, c1 + pc
    if r1 - r0 > 9 or c1 - c0 > 13:
        return None
    kind = str(rng.choice(("irregular", "bands", "outline", "arms")))
    selected_rows = {r for r in range(r0, r1) if rng.random() < 0.38}
    selected_cols = {c for c in range(c0, c1) if rng.random() < 0.28}
    fill = set(base)
    seen = set()
    for r in range(r0, r1):
        for c in range(c0, c1):
            p = (r, c)
            if p in seen:
                continue
            orbit = {p, _reflect(p, axis, axis2)}
            seen.update(orbit)
            if orbit & banned:
                continue
            if kind == "bands":
                take = r in selected_rows or rng.random() < 0.12
            elif kind == "outline":
                take = r in (r0, r1 - 1) or c in (c0, c1 - 1) or rng.random() < 0.08
            elif kind == "arms":
                take = r in selected_rows or c in selected_cols or rng.random() < 0.06
            else:
                take = rng.random() < 0.46
            if take:
                fill.update(orbit)
    return fill


def _local_legal(visible, options):
    legal = []
    for ink, ops in options:
        if set(ink).isdisjoint(visible):
            completed = dict(visible)
            completed.update(ink)
            legal.append((ink, ops, completed))
    return legal


def _object(visible_points, retained, body):
    visible = {p: body for p in visible_points}
    visible.update(retained)
    return visible


def _target(rng, retained, details, body):
    variants = _variants(retained, details)
    for _ in range(280):
        socket, chosen, _ = variants[int(rng.integers(8))]
        options = _options(retained, details, socket)
        alternatives = [d for d, _ in options if set(d) != set(chosen)]
        if not alternatives:
            continue
        alternative = alternatives[int(rng.integers(len(alternatives)))]
        axis2 = max(c for _, c in socket) + int(rng.integers(-4, 5))
        # Keep a genuinely competing stamp empty: occupied-body rejection alone
        # must not resolve the selected socket's orientation.
        forbidden = set(alternative) - set(chosen)
        full = _body_pattern(rng, set(socket) | set(chosen), forbidden, "lr", axis2)
        if full is None:
            continue
        visible_points = full - set(chosen)
        if (not set(socket) <= visible_points or not _connected(visible_points)
                or len(visible_points - set(socket)) < 3
                or _symmetric(visible_points) or _symmetric(full, "tb")):
            continue
        visible = _object(visible_points, socket, body)
        legal = _local_legal(visible, options)
        symmetric = [x for x in legal if _symmetric(x[2])]
        if len(legal) >= 2 and len(symmetric) == 1 and symmetric[0][0] == chosen:
            completed = symmetric[0][2]
            crop = _crop(completed, -1)
            if np.array_equal(crop, crop[:, ::-1]) or -1 not in crop:
                continue
            return visible, socket, options, completed
    return None


def _decoy(rng, retained, details, body, kind, target):
    variants = _variants(retained, details)
    for _ in range(240):
        if kind == "near":
            visible, socket, options, _ = target
            visible = dict(visible)
            r0, c0, r1, c1 = _box(visible)
            for _ in range(int(rng.integers(1, 5))):
                p = (int(rng.integers(r0, r1)), int(rng.integers(c0, c1)))
                if p not in socket:
                    if p in visible:
                        del visible[p]
                    else:
                        visible[p] = body
        else:
            socket, chosen, _ = variants[int(rng.integers(8))]
            options = _options(retained, details, socket)
            axis = "lr" if kind == "visible" else "tb"
            coords = [p[1 if axis == "lr" else 0] for p in socket]
            axis2 = min(coords) + max(coords) + int(rng.integers(-4, 5))
            required = set(socket) if kind == "visible" else set(socket) | set(chosen)
            forbidden = set(chosen) if kind == "visible" else set()
            full = _body_pattern(rng, required, forbidden, axis, axis2)
            if full is None:
                continue
            points = full if kind == "visible" else full - set(chosen)
            if not set(socket) <= points:
                continue
            visible = _object(points, socket, body)
        if not _connected(visible) or len(set(visible) - set(socket)) < 3:
            continue
        legal = _local_legal(visible, options)
        if not legal or any(_symmetric(x[2]) for x in legal):
            continue
        if kind == "horizontal" and not any(_symmetric(x[2], "tb") for x in legal):
            continue
        if kind == "visible" and not _symmetric(visible):
            continue
        # Reserve the full possible footprint, including erased external cells.
        footprint = set(visible) | {p for ink, _ in options for p in ink}
        r0, c0, r1, c1 = _box(footprint)
        if r1 - r0 <= 9 and c1 - c0 <= 13:
            return visible, socket, options, None
    return None


def _separate(a, b):
    r0, c0, r1, c1 = a
    s0, d0, s1, d1 = b
    return r1 + 1 <= s0 or s1 + 1 <= r0 or c1 + 1 <= d0 or d1 + 1 <= c0


def _pack(rng, objects, bg):
    # Each tuple carries the visible map and the entire reserved footprint.
    normalized = []
    for visible, footprint in objects:
        r0, c0, r1, c1 = _box(footprint)
        normalized.append(({(r - r0, c - c0): v for (r, c), v in visible.items()},
                           r1 - r0, c1 - c0))
    min_h = max(12, max(h for _, h, _ in normalized))
    min_w = max(12, max(w for _, _, w in normalized))
    area = sum(h * w for _, h, w in normalized)
    for _ in range(36):
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
        if h * w < area * 1.30:
            continue
        boxes = []
        placed = []
        order = sorted(range(len(normalized)), key=lambda i: -normalized[i][1] * normalized[i][2])
        for i in order:
            visible, oh, ow = normalized[i]
            found = False
            for flat in rng.permutation((h - oh + 1) * (w - ow + 1)):
                r, c = divmod(int(flat), w - ow + 1)
                box = (r, c, r + oh, c + ow)
                if all(_separate(box, b) for b in boxes):
                    boxes.append(box)
                    placed.append((visible, r, c))
                    found = True
                    break
            if not found:
                break
        if len(placed) == len(objects):
            grid = np.full((h, w), bg, dtype=int)
            for visible, r0, c0 in placed:
                for (r, c), color in visible.items():
                    grid[r0 + r, c0 + c] = color
            if np.count_nonzero(grid == bg) > grid.size // 2:
                return grid
    # Bounded fallback, still with randomized object order and within-slot anchors.
    grid = np.full((30, 30), bg, dtype=int)
    for (visible, h, w), slot in zip(normalized, rng.permutation(6)):
        r0 = (int(slot) // 2) * 10 + int(rng.integers(10 - h))
        c0 = (int(slot) % 2) * 15 + int(rng.integers(15 - w))
        for (r, c), color in visible.items():
            grid[r0 + r, c0 + c] = color
    return grid if np.count_nonzero(grid == bg) > grid.size // 2 else None


def _pair(rng, mode, count, horizontal, detail_range):
    for _ in range(110):
        colors = tuple(int(x) for x in rng.choice(10, 6, replace=False))
        bg, sc, mk, body, a, b = colors
        retained, details = _motif(rng, mode, colors, detail_range)
        target = _target(rng, retained, details, body)
        if target is None:
            continue
        decoys = [_decoy(rng, retained, details, body, "visible", target)]
        if horizontal:
            decoys.append(_decoy(rng, retained, details, body, "horizontal", target))
        while len(decoys) < count - 1:
            decoys.append(_decoy(rng, retained, details, body, "near", target))
        if any(x is None for x in decoys):
            continue
        prototype = dict(retained)
        prototype.update(details)
        objects = [(prototype, set(prototype))]
        for visible, socket, options, _ in [target] + decoys:
            footprint = set(visible) | {p for ink, _ in options for p in ink}
            r0, c0, r1, c1 = _box(footprint)
            if r1 - r0 > 9 or c1 - c0 > 13:
                break
            objects.append((visible, footprint))
        else:
            grid = _pack(rng, objects, bg)
            if grid is None:
                continue
            output = _crop(target[3], bg)
            # Construction's chosen completion is checked against visible evidence.
            if np.array_equal(solve(grid), output):
                return {"input": grid, "output": output}
    raise RuntimeError("Could not construct a bilateral socket scene")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} is outside its legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train" and i == 0:
                mode, n, horizontal, detail_range = "open", int(rng.integers(2, 4)), False, (3, 4)
            elif split == "train" and i == 1:
                mode, n, horizontal, detail_range = "cross", 3, True, (3, 5)
            else:
                mode = str(rng.choice(("bar", "cross", "rails", "fork", "open")))
                n = int(rng.integers(3 if split == "test" else 2, 5))
                horizontal = n >= 3 and (split == "test" or rng.random() < 0.5)
                detail_range = (6, 8) if split == "test" else (3, 5)
            for _ in range(10):
                pair = _pair(rng, mode, n, horizontal, detail_range)
                key = _key(pair["input"])
                scene = _scene(pair["input"])
                roles = (scene["bg"], scene["scaffold"], scene["marker"], scene["body"],
                         tuple(sorted(scene["inks"])))
                if key not in seen and roles != previous_roles:
                    seen.add(key)
                    previous_roles = roles
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    horizontal_training = False
    roles = set()
    detail_counts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(grid) not in seen
            seen.add(_key(grid))
            s = _scene(grid)
            detail_counts[split].append(len(s["details"]))
            assert all(any(q in s["retained"] for q in _neighbors(p)) for p in s["details"])
            assert len(s["successes"]) == 1
            index, ink, completed = s["successes"][0]
            selected = s["candidates"][index]
            assert len(selected["legal"]) >= 2, "Symmetry must resolve real alignment ambiguity"
            assert not _symmetric(selected["visible"]), "Raw symmetry must miss the answer"
            assert not _symmetric(completed, "tb"), "Axis must matter"
            assert np.array_equal(output, _crop(completed, s["bg"]))
            assert not np.array_equal(output, output[:, ::-1]), "Test shape, not literal color symmetry"
            assert s["bg"] in output and s["inks"] <= set(int(x) for x in output.flat)
            assert any(_symmetric(c["visible"]) for c in s["candidates"]), "Need a raw symmetric decoy"
            assert len(ink) == len(s["details"]) and len(ink) >= 3
            assert _connected(completed)
            footprints = [set(s["prototype"])]
            for candidate in s["candidates"]:
                assert candidate["legal"]
                assert len(candidate["options"]) >= 2
                points = set(candidate["visible"])
                points |= {p for d, _ in candidate["options"] for p in d}
                assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in points)
                footprints.append(points)
                if split == "train":
                    horizontal_training |= any(_symmetric(full, "tb") and not _symmetric(full)
                                               for _, _, full in candidate["legal"])
            boxes = [_box(points) for points in footprints]
            assert all(_separate(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
            roles.add((s["bg"], s["scaffold"], s["marker"], s["body"], tuple(sorted(s["inks"]))))
    assert horizontal_training, "Training must distinguish the two reflection axes"
    assert len(roles) >= 2
    assert all(3 <= n <= 5 for n in detail_counts["train"])
    assert all(6 <= n <= 8 for n in detail_counts["test"])
    assert max(detail_counts["train"]) < min(detail_counts["test"])


def self_test():
    """Bounded semantic and diversity checks on 42 seeds plus count boundaries."""
    shapes, outputs, stencils, palettes, signatures, counts = set(), set(), set(), set(), set(), set()
    metrics = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(_key(p[key]) == _key(q[key]) for split in puzzle
                       for p, q in zip(puzzle[split], other[split]) for key in ("input", "output"))
        signatures.add(tuple(_key(p["input"]) for split in puzzle for p in puzzle[split]))
        for split in puzzle:
            for pair in puzzle[split]:
                s = _scene(pair["input"])
                i, ink, full = s["successes"][0]
                cand = s["candidates"][i]
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                norm, _ = _normal(s["retained"])
                stencils.add(tuple(sorted(norm)))
                palettes.add((s["bg"], s["scaffold"], s["marker"], s["body"]))
                counts.add(len(s["candidates"]))
                metrics["even_axis"] += pair["output"].shape[1] % 2 == 0
                metrics["odd_axis"] += pair["output"].shape[1] % 2 == 1
                metrics["external_reference"] += not all(
                    _box(s["retained"])[0] <= r < _box(s["retained"])[2]
                    and _box(s["retained"])[1] <= c < _box(s["retained"])[3]
                    for r, c in s["details"])
                metrics["internal_reference"] += all(
                    _box(s["retained"])[0] <= r < _box(s["retained"])[2]
                    and _box(s["retained"])[1] <= c < _box(s["retained"])[3]
                    for r, c in s["details"])
                metrics["many_matches"] += len(cand["options"]) >= 4
                metrics["many_legal"] += len(cand["legal"]) >= 3
                metrics["expanding_crop"] += _box(cand["visible"]) != _box(full)
                metrics["same_size_decoy"] += any(
                    _crop(c["visible"], s["bg"]).shape == pair["output"].shape
                    for j, c in enumerate(s["candidates"]) if j != i)
                metrics["larger_decoy"] += any(
                    _crop(c["visible"], s["bg"]).size > pair["output"].size
                    for j, c in enumerate(s["candidates"]) if j != i)
                metrics["target_largest"] += all(
                    _crop(c["visible"], s["bg"]).size < pair["output"].size
                    for j, c in enumerate(s["candidates"]) if j != i)
                metrics["nonzero_background"] += s["bg"] != 0
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(outputs) >= 15 and len(stencils) >= 25 and len(palettes) >= 100
    assert counts == {2, 3, 4}
    assert all(metrics[name] > 0 for name in metrics)
    # Palette equivariance checks both ink identities and input-only operation.
    sample = generate_puzzle(512, 2, 1)["test"][0]
    permutation = np.array([8, 3, 0, 9, 5, 1, 7, 4, 2, 6])
    assert np.array_equal(solve(permutation[sample["input"]]), permutation[sample["output"]])
    scene = _scene(sample["input"])
    a, b = sorted(scene["inks"])
    changed = sample["input"].copy()
    changed[sample["input"] == a] = b
    changed[sample["input"] == b] = a
    changed_output = solve(changed)
    expected = sample["output"].copy()
    expected[sample["output"] == a] = b
    expected[sample["output"] == b] = a
    assert np.array_equal(changed_output, expected)
    assert not np.array_equal(changed_output, sample["output"])
    for a, b in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(900 + a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = generate_puzzle(917, 2, 1)
    damaged["test"][0]["output"] = np.zeros((1, 1), dtype=int)
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "output_shapes": len(outputs),
            "stencil_geometries": len(stencils), "palettes": len(palettes),
            "candidate_counts": sorted(counts), "coverage": dict(metrics)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
