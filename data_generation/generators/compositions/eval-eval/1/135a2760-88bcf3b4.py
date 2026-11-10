"""Mend the repeating bead word of a folded cord, then cinch it around a peg.

All semantics are visible in the input. No parent generators or data are loaded.
Only Python's standard library and NumPy are required. Running this module prints
a compact self-test report; it never writes samples or execution traces.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 312376601, 394795717, 233972072, 97691235, 341157589,
              411107276, 918369071, 756151685, 456952031, 836575370,
              249769015, 974673661, 522515350, 319816014, 449475447,
              331690998, 948604808, 294011613, 134627350, 26155893,
              118029254, 113511463, 613738572, 458095733, 765739215,
              278209210, 423298066, 854848058, 931618674, 515336775,
              991764294, 461762825, 338174579, 404674311, 740341485,
              759882924, 412138320, 236498814, 451280329, 412795657,
              966032398)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _coords(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _local(p, end, f, l):
    r, c = p[0] - end[0], p[1] - end[1]
    return r * f[0] + c * f[1], r * l[0] + c * l[1]


def _global(p, end, f, l):
    u, v = p
    return end[0] + u * f[0] + v * l[0], end[1] + u * f[1] + v * l[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _connected(cells, diagonal=False):
    if not cells:
        return False
    steps = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b] if diagonal else DIR4
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in steps:
            q = r + dr, c + dc
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(cells)


def _ordered_chain(cells, start):
    """The union of both bead colors is an induced, four-neighbor simple path."""
    if start not in cells or len(cells) < 8:
        raise ValueError("missing cord attachment")
    neighbors = {p: sorted((p[0] + dr, p[1] + dc) for dr, dc in DIR4
                           if (p[0] + dr, p[1] + dc) in cells) for p in cells}
    if len(neighbors[start]) != 1 or sorted(map(len, neighbors.values())) != [1, 1] + [2] * (len(cells) - 2):
        raise ValueError("cord is not an unbranched four-neighbor chain")
    order, prev, here = [], None, start
    while here is not None:
        order.append(here)
        onward = [q for q in neighbors[here] if q != prev]
        prev, here = here, (onward[0] if onward else None)
        if len(order) > len(cells):
            raise ValueError("cord cycle")
    if len(order) != len(cells):
        raise ValueError("disconnected cord")
    return order


def _repair(word):
    """Unique nearest periodic word, with >=4 observations of every phase.

    Redundant periods are allowed if they give the same answer. Neither the
    sampled period nor the sampled error positions are used by this decoder.
    """
    seq = np.asarray(word, dtype=np.int8)
    n = len(seq)
    candidates = []
    for p in range(1, min(5, n // 4) + 1):
        unit = []
        for phase in range(p):
            vals, counts = np.unique(seq[phase::p], return_counts=True)
            if int(counts.max()) * 2 <= int(counts.sum()):
                break
            unit.append(int(vals[int(counts.argmax())]))
        if len(unit) != p or len(set(unit)) != 2:
            continue
        clean = np.array([unit[i % p] for i in range(n)], dtype=np.int8)
        distance = int(np.count_nonzero(seq != clean))
        if distance <= max(1, n // 10):
            candidates.append((distance, p, clean))
    if not candidates:
        raise ValueError("no supported binary bead period")
    candidates.sort(key=lambda x: (x[0], x[1]))
    score, period, answer = candidates[0]
    if any(score == d and not np.array_equal(answer, other) for d, p, other in candidates):
        raise ValueError("ambiguous bead repair")
    return answer, period, score


def _taut_route(obstacle, length):
    """Discrete convex clearance envelope, in support-relative coordinates.

    Each peg cell (t,v) contributes v+1-|u-t|. The side opposite the support
    takes the upper envelope. Parallel approaches hold v=1 until the first
    exposed peak; thereafter departure follows the envelope immediately.
    """
    if not obstacle or min(u for u, v in obstacle) < 1:
        raise ValueError("peg must be in front of the attachment")
    peak = max(v for u, v in obstacle)
    if peak < 0 or (peak != 0 and max(v - u for u, v in obstacle) != 0):
        raise ValueError("unsupported incoming tangent")
    first_peak = min(u for u, v in obstacle if v == peak)
    n = max(length, max(u for u, v in obstacle) + 4)
    path = [(u, 1 if peak == 0 and u <= first_peak else
             max(v + 1 - abs(u - t) for t, v in obstacle)) for u in range(n)]
    slopes = [b[1] - a[1] for a, b in zip(path, path[1:])]
    if path[0] != (0, 1) or any(s not in (-1, 0, 1) for s in slopes) or slopes != sorted(slopes, reverse=True):
        raise ValueError("peg does not have a supported convex exposed contour")
    if set(path) & obstacle:
        raise ValueError("route intersects peg")
    return path[:length]


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if any(int(a[p]) != bg for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("input corners must identify one background")
    colors = [int(c) for c in np.unique(a) if int(c) != bg]
    if len(colors) != 4:
        raise ValueError("expected support, peg, and two bead colors")
    cells = {c: _coords(a, c) for c in colors}
    candidates = []
    for sc in colors:
        support = cells[sc]
        if len(support) < 3:
            continue
        rows, cols = {r for r, c in support}, {c for r, c in support}
        if len(rows) == 1:
            positive = (0, 1)
        elif len(cols) == 1:
            positive = (1, 0)
        else:
            continue
        if not _connected(support):
            continue
        for end, f in ((max(support), positive), (min(support), (-positive[0], -positive[1]))):
            for l in ((f[1], -f[0]), (-f[1], f[0])):
                for beads in combinations([c for c in colors if c != sc], 2):
                    oc, = [c for c in colors if c not in (sc, *beads)]
                    rope = cells[beads[0]] | cells[beads[1]]
                    attachment = _global((0, 1), end, f, l)
                    local_rope = {_local(p, end, f, l) for p in rope}
                    if any(u > 0 or v < 1 or (v == 1 and u != 0) for u, v in local_rope):
                        continue
                    obstacle = {_local(p, end, f, l) for p in cells[oc]}
                    try:
                        if not _connected(obstacle, diagonal=True):
                            continue
                        order = _ordered_chain(rope, attachment)
                        sequence = np.array([a[p] for p in order], dtype=np.int8)
                        clean, period, errors = _repair(sequence)
                        route = _taut_route(obstacle, len(order))
                    except ValueError:
                        continue
                    path = [_global(p, end, f, l) for p in route]
                    candidates.append(dict(bg=bg, support=support, obstacle=cells[oc],
                                           peg_local=obstacle, rope=rope, order=order,
                                           sequence=sequence, clean=clean, period=period,
                                           errors=errors, path=path, route=route,
                                           endpoint=end, forward=f, lateral=l,
                                           colors=(bg, sc, oc, *beads)))
    if len(candidates) != 1:
        raise ValueError("input must have one unique support/cord/peg interpretation")
    return a, candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only inference, including role assignment, graph order and repair."""
    a, d = _parse(input_grid)
    out = a.copy()
    for p in d["rope"]:
        out[p] = d["bg"]
    for p, color in zip(d["path"], d["clean"]):
        if _inside(p, a.shape):
            if out[p] != d["bg"]:
                raise ValueError("taut cord collides with fixed material")
            out[p] = color
    return out


def _sample_peg(rng, approach, contour_witness=False):
    # Actual silhouettes, not bounding boxes. Unsupported exposed profiles are
    # rejected before rendering. This is a finite retry, with a point fallback.
    for attempt in range(48):
        kind = int(rng.integers(3 if contour_witness else 0, 6)) if attempt < 47 else (3 if contour_witness else 0)
        h, w = int(rng.integers(1, 5)), int(rng.integers(1, 4))
        if kind == 0:
            raw = {(0, 0)}
        elif kind == 1:
            raw = {(u, 0) for u in range(h)} if rng.random() < .5 else {(0, v) for v in range(w)}
        elif kind == 2:
            raw = {(u, v) for u in range(h) for v in range(w)}
        elif kind == 3:
            radius = int(rng.integers(1, 3))
            raw = {(u, v) for u in range(2 * radius + 1)
                   for v in range(-radius, radius + 1) if abs(u - radius) + abs(v) <= radius}
        elif kind == 4:
            ramp = int(rng.integers(1, 4))
            raw = {(u, v) for u in range(ramp + 1) for v in range(ramp - u + 1)}
        else:
            plateau, ramp = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            raw = {(u, ramp - max(0, u - plateau + 1) - t)
                   for u in range(plateau + ramp) for t in range(int(rng.integers(1, 3)))}
        if rng.random() < .5:
            last = max(u for u, v in raw)
            raw = {(last - u, v) for u, v in raw}
        near = int(rng.integers(2, 5))
        offset = -max(v for u, v in raw) if approach == "parallel" else near - max(v - u for u, v in raw)
        peg = {(u + near, v + offset) for u, v in raw}
        try:
            actual_route = _taut_route(peg, 24)
            if contour_witness:
                box = {(u, v) for u in range(min(u for u, v in peg), max(u for u, v in peg) + 1)
                       for v in range(min(v for u, v in peg), max(v for u, v in peg) + 1)}
                if actual_route == _taut_route(box, 24):
                    continue
        except ValueError:
            continue
        return peg
    raise RuntimeError("bounded peg construction exhausted")


def _sample_word(rng, n, period, beads, visible_prefix=None):
    for attempt in range(160 + (1 << period)):
        bits = (rng.integers(0, 2, size=period) if attempt < 160 else
                np.array([((attempt - 160) >> k) & 1 for k in range(period)]))
        if len(set(map(int, bits))) != 2:
            continue
        clean = np.array([beads[int(bits[i % period])] for i in range(n)], dtype=np.int8)
        noisy = clean.copy()
        # First-copy damage makes copying a supposedly perfect master fail.
        positions = [int(rng.integers(min(period, n if visible_prefix is None else visible_prefix)))]
        if n >= 20 and attempt < 160 and rng.random() < .55:
            other = int(rng.integers(period, n))
            if other % period != positions[0] % period:
                positions.append(other)
        for i in positions:
            noisy[i] = beads[1] if clean[i] == beads[0] else beads[0]
        try:
            decoded, recovered, errors = _repair(noisy)
        except ValueError:
            continue
        if recovered == period and np.array_equal(decoded, clean):
            return noisy, clean
    raise RuntimeError("bounded bead-word construction exhausted")


def _slack(rng, length, depth_limit):
    """Parameterized open meanders; no nonconsecutive orthogonal contacts."""
    for attempt in range(80):
        order = [(0, 1)]
        u, v, direction = 0, 1, 1
        while len(order) < length:
            target = int(rng.integers(max(v + 3, 6), 11)) if direction == 1 else int(rng.integers(2, max(3, v - 2)))
            while v != target and len(order) < length:
                v += direction
                order.append((u, v))
            if len(order) == length:
                break
            gap = int(rng.integers(2, 4))
            for _ in range(gap):
                if len(order) == length:
                    break
                u -= 1
                order.append((u, v))
            direction *= -1
        if -min(u for u, v in order) <= depth_limit:
            assert _ordered_chain(set(order), (0, 1)) == order
            return order
    # A wide, two-cell-spaced meander guarantees <=4 depth through 24 cells.
    order = [(0, v) for v in range(1, 11)]
    order += [(-1, 10), (-2, 10)] + [(-2, v) for v in range(9, 1, -1)]
    order += [(-3, 2), (-4, 2)] + [(-4, v) for v in range(3, 11)]
    order = order[:length]
    if -min(u for u, v in order) > depth_limit:
        raise RuntimeError("slack depth budget too small")
    return order


def _render(rng, support, slack, peg, sequence, clean, colors, clipped):
    bg, sc, oc, b0, b1 = colors
    route = _taut_route(peg, len(slack))
    all_input = support | set(slack) | peg
    included = all_input if clipped else all_input | set(route)
    u0, u1 = min(u for u, v in included) - 1, max(u for u, v in included) + 1
    v0, v1 = min(v for u, v in included) - 1, max(v for u, v in included) + 1
    # One compulsory input-background border; optional additional padding.
    budget_u, budget_v = 30 - (u1 - u0 + 1), 30 - (v1 - v0 + 1)
    if min(budget_u, budget_v) < 0:
        raise ValueError("scene exceeds canvas")
    for _ in range(int(rng.integers(0, min(3, budget_u) + 1))):
        if clipped or rng.random() < .5:
            u0 -= 1
        else:
            u1 += 1
    for _ in range(int(rng.integers(0, min(3, budget_v) + 1))):
        if rng.random() < .5:
            v0 -= 1
        else:
            v1 += 1
    a = np.full((u1 - u0 + 1, v1 - v0 + 1), bg, dtype=np.int8)
    at = lambda p: (p[0] - u0, p[1] - v0)
    for p in support:
        a[at(p)] = sc
    for p in peg:
        a[at(p)] = oc
    for p, color in zip(slack, sequence):
        a[at(p)] = color
    out = a.copy()
    for p in slack:
        out[at(p)] = bg
    for p, color in zip(route, clean):
        q = at(p)
        if _inside(q, out.shape):
            assert out[q] == bg
            out[q] = color
    return a, out


def _make_pair(rng, n, period, colors, approach, clipped, symmetry, contour_witness=False):
    for attempt in range(128):
        peg = _sample_peg(rng, approach, contour_witness)
        support_length = int(rng.integers(4, min(7, 29 - n) + 1))
        support = {(-i, 0) for i in range(support_length)}
        depth_limit = 28 - n if not clipped else 12
        slack = _slack(rng, n, depth_limit)
        visible_prefix = max(u for u, v in peg) + 2 if clipped else n
        sequence, clean = _sample_word(rng, n, period, colors[-2:], visible_prefix)
        try:
            a, out = _render(rng, support, slack, peg, sequence, clean, colors, clipped)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("bounded canvas-fit construction exhausted")
    for name, grid in (("input", a), ("output", out)):
        transformed = np.rot90(grid, symmetry % 4)
        if symmetry >= 4:
            transformed = np.fliplr(transformed)
        if name == "input":
            a = transformed.copy()
        else:
            out = transformed.copy()
    return {"input": a, "output": out}


def _int_in_range(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _int_in_range(num_train, 2, 10, "num_train")
    nq = _int_in_range(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    train_cap = int(rng.integers(18, 21))
    backgrounds = list(map(int, rng.permutation(10)))
    lessons = list(map(int, rng.permutation(nt)))
    base_symmetry = int(rng.integers(8))
    puzzle, seen = {"train": [], "test": []}, set()
    for serial in range(nt + nq):
        split = "train" if serial < nt else "test"
        lesson = lessons.index(serial) if split == "train" else -1
        period = (2 + lesson if lesson in (0, 1) else int(rng.integers(2, 5))) if split == "train" else int(rng.integers(3, 6))
        n = int(rng.integers(max(12, 4 * period), train_cap + 1)) if split == "train" else int(rng.integers(max(20, train_cap + 1, 4 * period), 25))
        approach = ("parallel" if lesson == 0 else "diagonal" if lesson == 1 else str(rng.choice(["parallel", "diagonal"])))
        clipped = False if lesson == 0 else True if lesson == 1 else bool(rng.integers(2))
        symmetry = (base_symmetry + lesson) % 8 if lesson in (0, 1) else int(rng.integers(8))
        bg = backgrounds[serial % 10]
        palette = [bg, *map(int, rng.choice([c for c in range(10) if c != bg], 4, replace=False))]
        for retry in range(12):
            pair = _make_pair(rng, n, period, palette, approach, clipped, symmetry, lesson == 0)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    a, d = _parse(pair["input"])
    local = [_local(p, d["endpoint"], d["forward"], d["lateral"]) for p in d["order"]]
    directions = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(local, local[1:])]
    peg = d["peg_local"]
    lo_u, lo_v = min(u for u, v in peg), min(v for u, v in peg)
    box = {(u, v) for u in range(lo_u, max(u for u, v in peg) + 1)
           for v in range(lo_v, max(v for u, v in peg) + 1)}
    try:
        box_route = _taut_route(box, len(local))
    except ValueError:
        # A bounding box can destroy the very tangent from the fixed anchor.
        box_route = None
    return {"shape": tuple(a.shape), "length": len(local), "period": d["period"],
            "errors": d["errors"], "background": d["bg"],
            "axis": "vertical" if d["forward"][0] else "horizontal",
            "forward": d["forward"], "side": d["lateral"],
            "clipped": sum(not _inside(p, a.shape) for p in d["path"]),
            "approach": "parallel" if max(v for u, v in peg) == 0 else "diagonal",
            "turns": sum(x != y for x, y in zip(directions, directions[1:])),
            "slack_shape": tuple(local),
            "peg_shape": tuple(sorted((u - lo_u, v - lo_v) for u, v in peg)),
            "contour_witness": box_route is None or any(q != r and _inside(p, a.shape)
                                                        for q, r, p in zip(d["route"], box_route, d["path"])),
            "partial_cycle": len(local) % d["period"] != 0}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            _, d = _parse(a)
            assert np.array_equal(solve(a), out), "input-only solution disagrees"
            assert 12 <= len(d["order"]) <= 24 and 2 <= d["period"] <= 5
            assert 1 <= d["errors"] <= max(1, len(d["order"]) // 10)
            assert len(d["support"]) >= 4
            assert d["order"][0] == d["path"][0]
            assert set(d["path"]) & d["rope"] == {d["order"][0]}
            assert any(d["sequence"][i] != d["clean"][i] for i in range(d["period"]))
            fixed = d["support"] | d["obstacle"]
            assert all(a[p] == out[p] for p in fixed)
            mutable = d["rope"] | {p for p in d["path"] if _inside(p, a.shape)}
            assert all(tuple(map(int, p)) in mutable for p in np.argwhere(a != out))
            visible = [(i, p) for i, p in enumerate(d["path"]) if _inside(p, a.shape)]
            assert any(d["sequence"][i] != d["clean"][i] for i, p in visible), "A must affect a visible answer cell"
            # Remove A: move the noisy colors. Remove B: mend in place.
            no_a, no_b = a.copy(), a.copy()
            for p in d["rope"]:
                no_a[p] = d["bg"]
            for i, p in visible:
                no_a[p] = d["sequence"][i]
            for p, color in zip(d["order"], d["clean"]):
                no_b[p] = color
            assert not np.array_equal(no_a, out) and not np.array_equal(no_b, out)
            # Ignoring the peg and continuing straight also changes geometry.
            assert any(_local(p, d["endpoint"], d["forward"], d["lateral"])[1] != 1 for i, p in visible)
            metrics[split].append(measure_pair(pair))
    training = metrics["train"]
    assert {m["approach"] for m in training} == {"parallel", "diagonal"}
    assert any(m["clipped"] for m in training) and any(not m["clipped"] for m in training)
    assert any(m["contour_witness"] and not m["clipped"] for m in training)
    assert {2, 3} <= {m["period"] for m in training}
    assert len({m["background"] for m in training}) >= 2
    assert min(m["length"] for m in metrics["test"]) > max(m["length"] for m in training)


def self_test():
    """Bounded 42-seed audit plus semantic and API regressions; no files saved."""
    shapes, slack, pegs, signatures = set(), set(), set(), set()
    periods, lengths, directions, backgrounds, turns, errors = set(), set(), set(), set(), set(), set()
    clips = {"train": 0, "test": 0}
    partial, pairs = 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                pairs += 1
                shapes.add(m["shape"])
                slack.add(m["slack_shape"])
                pegs.add(m["peg_shape"])
                periods.add(m["period"])
                lengths.add(m["length"])
                directions.add((m["forward"], m["side"]))
                backgrounds.add(m["background"])
                turns.add(m["turns"])
                errors.add(m["errors"])
                clips[split] += bool(m["clipped"])
                partial += m["partial_cycle"]
                for role in ("input", "output"):
                    digest.update(bytes(pair[role].shape) + pair[role].tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and periods == {2, 3, 4, 5}
    assert len(shapes) >= 35 and len(slack) >= 70 and len(pegs) >= 15
    assert len(lengths) >= 10 and len(directions) == 8 and len(backgrounds) == 10
    assert errors == {1, 2} and len(turns) >= 5 and 0 < partial < pairs
    assert clips["train"] >= 42 and clips["test"] > 0
    # Exact routing departure and parallel approach regressions.
    assert _taut_route({(2, 0)}, 6) == [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]
    assert _taut_route({(u, v) for u in (2, 3) for v in (0, 1, 2)}, 6) == [(0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1)]
    # Even a clipped period-five point-peg case must expose some damage.
    rng = np.random.default_rng(881)
    for _ in range(12):
        noisy, clean = _sample_word(rng, 20, 5, (1, 2), visible_prefix=4)
        assert np.any(noisy[:4] != clean[:4])
    # A phase-three word keeps phase one at a contour turn at index four.
    clean = np.array([1, 2, 2] * 4, dtype=np.int8)
    noisy = clean.copy()
    noisy[1] = 1
    peg = {(u, v) for u in range(3, 6) for v in range(4)}
    a, expected = _render(rng, {(-i, 0) for i in range(4)}, _slack(rng, 12, 12),
                          peg, noisy, clean, (0, 8, 3, 1, 2), False)
    _, d = _parse(a)
    assert d["route"][4] == (4, 4) and solve(a)[d["path"][4]] == 2
    assert np.array_equal(solve(a), expected)
    p = generate_puzzle(731, 2, 1)
    q = generate_puzzle(731, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Solving must commute with every square symmetry and a palette permutation.
    a, b = p["train"][0]["input"], p["train"][0]["output"]
    palette = np.array([9, 3, 0, 7, 2, 8, 1, 4, 6, 5], dtype=np.int8)
    for k in range(8):
        x, y = np.rot90(a, k % 4), np.rot90(b, k % 4)
        if k >= 4:
            x, y = np.fliplr(x), np.fliplr(y)
        assert np.array_equal(solve(palette[x]), palette[y])
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(17, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # The validator must notice fixed-object damage, rather than accepting any
    # answer that still has plausible colors and dimensions.
    _, d = _parse(p["test"][0]["input"])
    p["test"][0]["output"][min(d["support"])] = d["bg"]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged support accepted")
    return {"seeds": 42, "pairs": pairs, "puzzles": len(signatures),
            "shapes": len(shapes), "slack_shapes": len(slack), "peg_shapes": len(pegs),
            "periods": sorted(periods), "lengths": sorted(lengths),
            "forward_side_combinations": len(directions), "backgrounds": len(backgrounds),
            "turn_counts": sorted(turns), "error_counts": sorted(errors),
            "clipped_pairs": clips, "partial_cycles": partial,
            "independent_semantic_oracle": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
