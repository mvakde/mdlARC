"""ARC family: floating splitters and grounded absorbers.

Only NumPy and the standard library are used. No files, examples, seeds, or
generation metadata are consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or not np.all((0 <= a) & (a <= 9)):
        raise ValueError("grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    h, w = allowed.shape
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc] and not reached[rr, cc]:
                reached[rr, cc] = True
                stack.append((rr, cc))
    return reached


def _components(mask):
    unseen = mask.copy()
    result = []
    for r, c in np.argwhere(mask):
        if unseen[r, c]:
            component = _flood(unseen, (int(r), int(c)))
            unseen[component] = False
            result.append(component)
    return result


def _roles(a):
    """The unique singleton is the seed; a short rod extends away from it."""
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4 or sum(counts == counts.max()) != 1:
        raise ValueError("exactly four colors and a unique dominant background required")
    bg = int(colors[counts.argmax()])
    singleton = colors[counts == 1]
    if len(singleton) != 1:
        raise ValueError("exactly one singleton marker required")
    marker = int(singleton[0])
    source = tuple(map(int, np.argwhere(a == marker)[0]))
    candidates = []
    for guide in map(int, colors):
        if guide in (bg, marker):
            continue
        n = int(np.count_nonzero(a == guide))
        if not 2 <= n <= 4:
            continue
        for dr, dc in _N4:
            positions = [(source[0] + j * dr, source[1] + j * dc) for j in range(1, n + 1)]
            if all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] and a[r, c] == guide
                   for r, c in positions):
                st = next(int(v) for v in colors if v not in (bg, marker, guide))
                if np.count_nonzero(a == st) >= 5 and counts.max() > np.count_nonzero(a == st):
                    candidates.append((bg, st, guide, marker, source, (dr, dc), positions[-1]))
    if len(candidates) != 1:
        raise ValueError("the launch pin must determine one guide color and direction")
    return candidates[0]


def _topology(a, roles):
    _, st, _, _, source, _, _ = roles
    selected = _flood(a != st, source)
    # The padding makes every canvas edge part of the same outside.
    complement = np.pad(~selected, 1, constant_values=True)
    outside = _flood(complement, (0, 0))[1:-1, 1:-1]
    floating = (a == st) & ~outside
    return selected, outside, floating


def _run(a, mode="correct", details=False):
    """Monotone ray work list; all collision queries use immutable input.

    Fault modes are used only for observable teaching checks, never for labels.
    They intentionally omit or simplify one part of the specified mechanism.
    """
    roles = _roles(a)
    bg, st, guide, marker, source, (dr, dc), tip = roles
    tr, tc = -dc, dr
    h, w = a.shape
    selected, outside, floating = _topology(a, roles)
    out = a.copy()
    pending = [(tip[0] + dr, tip[1] + dc)]
    visited, events, painted = set(), {}, set()
    split_layers = set()

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w

    def paint(r, c):
        if inside(r, c) and selected[r, c] and a[r, c] == bg:
            out[r, c] = guide
            painted.add((r, c))

    while pending:
        r, c = pending.pop()
        if not inside(r, c) or (r, c) in visited:
            continue
        visited.add((r, c))
        if a[r, c] != st:
            if selected[r, c]:
                paint(r, c)
                pending.append((r + dr, c + dc))
            continue
        is_floating = bool(floating[r, c])
        scatter = is_floating
        if mode == "all_absorb":
            scatter = False
        elif mode == "all_split":
            scatter = True
        elif mode == "transparent_islands" and is_floating:
            pending.append((r + dr, c + dc))
            continue
        left = right = 0
        while inside(r + (left - 1) * tr, c + (left - 1) * tc) and a[r + (left - 1) * tr, c + (left - 1) * tc] == st:
            left -= 1
        while inside(r + (right + 1) * tr, c + (right + 1) * tc) and a[r + (right + 1) * tr, c + (right + 1) * tc] == st:
            right += 1
        endpoints = ((r + left * tr, c + left * tc), (r + right * tr, c + right * tc))
        key = (endpoints, is_floating)
        events.setdefault(key, set()).add((r, c))
        if not scatter:
            continue
        depth = r * dr + c * dc
        if mode == "one_split_per_layer" and depth in split_layers and len(events[key]) == 1:
            continue
        split_layers.add(depth)
        if mode == "ignore_singletons" and left == right:
            continue
        if mode == "whole_object":
            component = _flood(a == st, (r, c))
            transverse = [int((rr - r) * tr + (cc - c) * tc) for rr, cc in np.argwhere(component)]
            left, right = min(transverse), max(transverse)
        for k in range(left - 1, right + 2):
            paint(r - dr + k * tr, c - dc + k * tc)
        for k in (left - 1, right + 1):
            rr, cc = r + k * tr, c + k * tc
            # Even faulty all-split never overwrites structures or crosses rooms.
            if inside(rr, cc) and selected[rr, cc]:
                pending.append((rr, cc))
    if not details:
        return out
    split_events = [key for key in events if key[1]]
    absorb_events = [key for key in events if not key[1]]
    layers = {}
    for (ends, is_float), hits in events.items():
        if is_float:
            r, c = ends[0]
            layers.setdefault(r * dr + c * dc, 0)
            layers[r * dr + c * dc] += 1
    free_components = _components(a != st)
    struct_components = _components(a == st)
    regions = sorted(int(m.sum()) for m in free_components)
    feature = {
        "shape": tuple(map(int, a.shape)), "palette": tuple(roles[:4]),
        "direction": (dr, dc), "source": source,
        "source_on_edge": source[0] in (0, h - 1) or source[1] in (0, w - 1),
        "pin_length": int(np.count_nonzero(a == guide)),
        "compartments": len(regions), "selected_area": int(selected.sum()),
        "selects_smaller": int(selected.sum()) < max(regions),
        "islands": sum(bool(np.any(m & floating)) for m in struct_components),
        "grounded_components": sum(bool(np.any(m & outside)) for m in struct_components),
        "split_events": len(split_events), "absorb_events": len(absorb_events),
        "simultaneous": max(layers.values(), default=0),
        "multi_hit": sum(len(events[key]) > 1 for key in split_events),
        "singleton_hit": any(ends[0] == ends[1] for ends, _ in split_events),
        "changed_cells": len(painted), "selected": selected, "floating": floating,
        "events": events,
    }
    return out, feature


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, topology, and the complete ray tree from the input alone."""
    return _run(_grid(input_grid))


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _halo(cells, h, w, radius=1):
    return {(r + dr, c + dc) for r, c in cells
            for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _put_detached(a, cells):
    h, w = a.shape
    if not cells or any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in cells):
        return False
    if any(a[r, c] != 0 for r, c in _halo(cells, h, w)):
        return False
    for r, c in cells:
        a[r, c] = 1
    return True


def _motif(rng):
    """Connected, hole-free polyominoes, independently sized on both axes."""
    h, w = int(rng.integers(1, 5)), int(rng.integers(1, 7))
    kind = int(rng.integers(5))
    if kind == 0:
        cells = {(0, c) for c in range(w)}
    elif kind == 1:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 2:
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif kind == 3:
        cells = {(r, c) for r in range(h) for c in range(min(w, r + 1))}
    else:
        cells = {(r, w // 2) for r in range(h)} | {(0, c) for c in range(w)}
    return cells


def _extras(rng, a, budget):
    h, w = a.shape
    for _ in range(budget):
        shape = _motif(rng)
        rh, cw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
        if rh + 2 > h or cw + 2 > w:
            continue
        r0, c0 = int(rng.integers(1, h - rh)), int(rng.integers(1, w - cw))
        _put_detached(a, {(r0 + r, c0 + c) for r, c in shape})


def _normalized(a):
    """Internal construction convention: 0 background, 1 wall, 2 ray, 3 seed."""
    return _run(a, details=True)


def _core(rng, witness, large=False):
    """Readable causal witnesses with independently variable distances and shape.

    Witness 0: parallel floating collisions and a broad subsequent multi-hit.
    Witness 1: identical local bars, but one is grounded by a remote thin stem.
    Both contain a bent floating object, teaching runs rather than bounding boxes.
    """
    chamber = 17 if witness == 1 else int(rng.integers(18, 27))
    extra = 12 if witness == 1 else (int(rng.integers(3, min(6, 30 - chamber))) if chamber <= 26 and rng.random() < .65 else 0)
    width = chamber + 1 + extra if extra else chamber
    c = int(rng.integers(8, chamber - 8))
    start = int(rng.integers(0, 2 if witness == 1 else 4))
    n = int(rng.integers(2, 4 if witness == 1 else 5))
    r1 = start + n + int(rng.integers(2, 3 if witness == 1 else 4))
    r2 = r1 + int(rng.integers(3, 5))
    r3 = r2 + int(rng.integers(5, 7))
    height = 27 if witness == 1 else min(30, r3 + int(rng.integers(5, 8)))
    if large:
        height = 30
    a = np.zeros((height, width), np.int8)
    if extra:
        # Full or stepped divider, with a spare, excluded compartment.
        previous = chamber
        for r in range(height):
            col = chamber + (int(r >= height // 2) if witness != 1 and rng.random() < .25 else 0)
            col = min(col, width - 3)
            a[r, min(previous, col):max(previous, col) + 1] = 1
            previous = col
    a[start, c] = 3
    a[start + 1:start + n + 1, c] = 2
    left, right = c - int(rng.integers(1, 3)), c + int(rng.integers(1, 3))
    a[r1, left:right + 1] = 1
    pivots = (left - 1, right + 1)
    elbow_side = -1 if witness % 2 == 0 else 1
    for side, p in zip((-1, 1), pivots):
        # The non-elbow floating bar can be a singleton: same local run rule.
        half = 0 if witness == 0 and side != elbow_side else 1
        a[r2, p - half:p + half + 1] = 1
        if witness == 1 and side != elbow_side:
            a[r2:, p + side] = 1
        elif side == elbow_side:
            a[r2 + 1, p + side] = 1
            lo, hi = sorted((p + side, p + 3 * side))
            a[r2 + 2, lo:hi + 1] = 1
    # Derive the actual forward rays before placing a later spanning splitter.
    before, _ = _normalized(a)
    columns = np.flatnonzero(before[r3] == 2)
    lo, hi = int(columns.min()), int(columns.max())
    # All surviving rays meet this island together; it has free space on each side.
    a[r3, lo:hi + 1] = 1
    # A later grounded object can extinguish another independent descendant.
    p = lo - 1 if rng.random() < .5 else hi + 1
    aa, bb = max(0, p - 1), min(chamber - 1, p + 1)
    a[r3 + 3, aa:bb + 1] = 1
    tether = aa if p < c else bb
    a[r3 + 3:, tether] = 1
    _extras(rng, a, int(rng.integers(1, 8)))
    if witness == 1:
        # A third compartment below the selected one leaves a larger, empty
        # far-side room. The source, rather than area, must select the chamber.
        a[r3 + 4, :chamber] = 1
        a[:, chamber + 1:] = 0
    return a


def _free_scene(rng, large=False):
    """A different scene grammar: spatial obstacle fields, no event-row template."""
    h = 30 if large else int(rng.integers(14, 26))
    w = int(rng.integers(13, 31))
    a = np.zeros((h, w), np.int8)
    n = int(rng.integers(2, 5))
    r0, c0 = int(rng.integers(0, h - n - 6)), int(rng.integers(3, w - 3))
    a[r0, c0] = 3
    a[r0 + 1:r0 + n + 1, c0] = 2
    # One or two dividers may select a small, central, or large compartment.
    for _ in range(int(rng.integers(0, 3))):
        candidates = [c for c in range(2, w - 2) if abs(c - c0) >= 4]
        if candidates:
            x = int(rng.choice(candidates))
            a[:, x] = 1
    # Rooted intrusions from any edge. Clearance prevents accidental glyphs.
    for _ in range(int(rng.integers(1, 6))):
        edge = int(rng.integers(4))
        length = int(rng.integers(2, max(3, (h if edge < 2 else w) // 2)))
        if edge < 2:
            c = int(rng.integers(w))
            rows = range(length) if edge == 0 else range(h - length, h)
            cells = {(r, c) for r in rows}
        else:
            r = int(rng.integers(h))
            cols = range(length) if edge == 2 else range(w - length, w)
            cells = {(r, c) for c in cols}
        if all(a[r, c] in (0, 1) for r, c in _halo(cells, h, w)):
            for r, c in cells:
                a[r, c] = 1
    # Force one plausible forward collision, then distribute unrelated shapes.
    r = int(rng.integers(r0 + n + 2, h - 2))
    length = int(rng.integers(1, min(8, w - 4)))
    left = max(1, min(w - length - 1, c0 - int(rng.integers(length))))
    _put_detached(a, {(r, c) for c in range(left, left + length)})
    _extras(rng, a, int(rng.integers(6, 19)))
    return a


def _materialize(a, palette, rotation, reflect):
    b = np.asarray(palette, np.int8)[a]
    b = np.rot90(b, rotation)
    if reflect:
        b = np.fliplr(b)
    return b.copy()


def _differences(a, expected):
    return {mode: int(np.count_nonzero(_run(a, mode) != expected)) for mode in
            ("all_absorb", "all_split", "transparent_islands", "whole_object",
             "one_split_per_layer", "ignore_singletons")}


def _make_pair(rng, index, split, palette, rotation, seen, covered, large):
    for attempt in range(64):
        core = split == "train" and index < 2 or attempt >= 40
        role = _core(rng, index % 2, large) if core else _free_scene(rng, large)
        inp = _materialize(role, palette, rotation, bool(rng.integers(2)))
        if _key(inp) in seen:
            continue
        try:
            out, f = _run(inp, details=True)
        except ValueError:
            continue
        if f["changed_cells"] < 3 or f["split_events"] < 1:
            continue
        diffs = _differences(inp, out)
        if split == "train" and index < 2:
            if not (diffs["all_absorb"] and diffs["all_split"] and diffs["transparent_islands"] and diffs["whole_object"]):
                continue
            if index == 0 and not (diffs["one_split_per_layer"] and diffs["ignore_singletons"] and f["multi_hit"]):
                continue
            if index == 1 and not f["selects_smaller"]:
                continue
        if split == "test" and any(diffs[m] and m not in covered for m in diffs):
            continue
        return {"input": inp, "output": out}, diffs
    raise AssertionError("bounded construction budget exhausted")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    palette_mode, orientation_mode = int(rng.integers(3)), int(rng.integers(3))
    base_rotation = int(rng.integers(4))
    scale_shift = bool(rng.integers(2))
    if not scale_shift:
        orientation_mode = 2
    puzzle, seen, covered = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            if palette_mode == 0 or index == 0 and split == "train":
                colors = palette
            elif palette_mode == 1:
                remaining = [c for c in range(10) if c not in (palette[0], palette[3])]
                st, guide = map(int, rng.choice(remaining, 2, replace=False))
                colors = (palette[0], st, guide, palette[3])
            else:
                colors = tuple(map(int, rng.choice(10, 4, replace=False)))
            if orientation_mode == 0:
                rotation = base_rotation
            elif orientation_mode == 1:
                rotation = (base_rotation + index + int(split == "test")) % 4
            else:
                rotation = (base_rotation + (2 * (index % 2) if split == "train" else 1 + 2 * (index % 2))) % 4
            pair, diffs = _make_pair(rng, index, split, colors, rotation, seen, covered,
                                     scale_shift and split == "test")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                covered.update(m for m, n in diffs.items() if n)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, covered, train_features = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, f = _run(a, details=True)
            assert np.array_equal(out, expected), "wrong topology-sensitive ray tree"
            bg, st, guide, marker = f["palette"]
            assert np.all(out[a != bg] == a[a != bg]), "source or structure overwritten"
            assert np.all(out[out != a] == guide), "new cells must use the local guide color"
            assert not np.any((out != a) & ~f["selected"]), "light escaped the selected compartment"
            assert f["changed_cells"] >= 3 and f["split_events"] >= 1
            diffs = _differences(a, out)
            if split == "train":
                covered.update(mode for mode, n in diffs.items() if n)
                train_features.append(f)
            else:
                assert all(not n or mode in covered for mode, n in diffs.items()), "unexposed shortcut in training"
    assert {"all_absorb", "all_split", "transparent_islands", "whole_object",
            "one_split_per_layer", "ignore_singletons"} <= covered
    assert any(f["multi_hit"] for f in train_features)
    assert any(f["absorb_events"] for f in train_features)
    assert any(f["selects_smaller"] for f in train_features), "training must reject largest-compartment selection"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _layer_reference(a):
    """Supplementary check: Boolean region growth and simultaneous row updates.

    It shares role recognition, but not floods, ray work lists, or run queries
    with the production solver. It is authored validation, not external review.
    """
    bg, st, guide, marker, _, direction, _ = _roles(a)
    if direction == (1, 0):
        canonical, restore = a.copy(), lambda b: b
    elif direction == (-1, 0):
        canonical, restore = np.flipud(a).copy(), np.flipud
    elif direction == (0, 1):
        canonical, restore = a.T.copy(), lambda b: b.T
    else:
        canonical, restore = np.fliplr(a).T.copy(), lambda b: np.fliplr(b.T)
    h, w = canonical.shape

    def spread(allowed, initial):
        reached = initial & allowed
        while True:
            p = np.pad(reached, 1)
            following = reached | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            following &= allowed
            if np.array_equal(following, reached):
                return reached
            reached = following

    target = spread(canonical != st, canonical == marker)
    complement = np.pad(~target, 1, constant_values=True)
    border = np.zeros_like(complement)
    border[[0, -1], :], border[:, [0, -1]] = True, True
    grounded = spread(complement, border)[1:-1, 1:-1]
    tip_r, tip_c = map(int, np.argwhere(canonical == guide)[-1])
    active, out = {tip_c}, canonical.copy()
    for r in range(tip_r + 1, h):
        following, hit_runs = set(), set()
        for c in active:
            if canonical[r, c] == st:
                if grounded[r, c]:
                    continue
                indices = np.flatnonzero(canonical[r] != st)
                before, after = indices[indices < c], indices[indices > c]
                lo = int(before[-1]) + 1 if len(before) else 0
                hi = int(after[0]) - 1 if len(after) else w - 1
                hit_runs.add((lo, hi))
            elif target[r, c]:
                following.add(c)
                if canonical[r, c] == bg:
                    out[r, c] = guide
        for lo, hi in hit_runs:
            for c in range(max(0, lo - 1), min(w, hi + 2)):
                if target[r - 1, c] and canonical[r - 1, c] == bg:
                    out[r - 1, c] = guide
            for c in (lo - 1, hi + 1):
                if 0 <= c < w and target[r, c]:
                    following.add(c)
                    if canonical[r, c] == bg:
                        out[r, c] = guide
        active = following
    return restore(out).copy()


def _hand_regressions():
    # A manual two-event example: one floating bar splits, then a rooted stem
    # absorbs its left branch. The exact expected pixels are stated directly.
    a = np.zeros((12, 13), np.int8)
    a[1, 6], a[2:4, 6] = 3, 2
    a[5, 5:8] = 1
    a[8, 3:5], a[8:, 3] = 1, 1
    expected = a.copy()
    expected[4, 4:9] = 2
    expected[5:8, 4] = 2
    expected[5:, 8] = 2
    assert np.array_equal(solve(a), expected)
    # Cut a remote tether: the unchanged local collision now scatters.
    floating = a.copy()
    floating[10:, 3] = 0
    assert solve(floating)[7, 2] == 2 and expected[7, 2] == 0
    # Joining a floating bar to the edge suppresses all its descendants.
    grounded = a.copy()
    grounded[5:, 7] = 1
    grounded_out = solve(grounded)
    assert grounded_out[4, 6] == 2 and grounded_out[4, 4] == 0
    assert grounded_out[6, 8] == 0
    # Input-only behavior under every spatial symmetry and a role permutation.
    for k in range(4):
        for reflect in (False, True):
            aa, oo = np.rot90(a, k), np.rot90(expected, k)
            if reflect:
                aa, oo = np.fliplr(aa), np.fliplr(oo)
            assert np.array_equal(solve(aa), oo)
    mapping = np.array([7, 0, 9, 4, 1, 2, 3, 5, 6, 8], np.int8)
    assert np.array_equal(solve(mapping[a]), mapping[expected])
    # Nothing added to an excluded compartment changes the ray drawing.
    divided = np.pad(a, ((0, 0), (0, 4)), constant_values=0)
    divided[:, 13] = 1
    baseline = solve(divided)
    divided[3:5, 15:17] = 1
    altered = solve(divided)
    assert np.array_equal(altered[:, :14], baseline[:, :14])
    assert np.array_equal(altered[:, 14:], divided[:, 14:])
    before = a.copy()
    solve(a)
    assert np.array_equal(a, before)


def self_test():
    """Bounded: stream 42 seeds; retain hashes and scalar feature sets only."""
    _hand_regressions()
    signatures, shapes, directions, palettes, structures = set(), set(), set(), set(), set()
    counts = {name: set() for name in ("islands", "grounded_components", "split_events",
                                      "absorb_events", "compartments", "pin_length")}
    small_selected = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = np.asarray(pair["input"])
                out, f = _run(a, details=True)
                assert np.array_equal(out, pair["output"])
                assert np.array_equal(_layer_reference(a), out)
                shapes.add(f["shape"])
                directions.add(f["direction"])
                palettes.add(f["palette"])
                role = np.zeros_like(a)
                for j, color in enumerate(f["palette"]):
                    role[a == color] = j
                structures.add(hashlib.sha256(role.tobytes() + bytes(a.shape)).hexdigest())
                for name in counts:
                    counts[name].add(f[name])
                small_selected += f["selects_smaller"]
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 60 and len(palettes) >= 40 and len(directions) == 4
    assert all(len(values) >= 3 for values in counts.values())
    assert small_selected > 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, nq)
            validate_generated_puzzle(p)
    assert _digest(generate_puzzle(123, 7, 3)) == _digest(generate_puzzle(123, 7, 3))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(321, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
