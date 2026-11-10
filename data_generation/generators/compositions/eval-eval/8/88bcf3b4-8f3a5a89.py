"""Shoreline tension: a finite strand follows an octagonal external cap.

Only the input determines roles, compartment, cap, length and erasures.
No parent module, file, seed or generation metadata is used by solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _flood(mask, start):
    reached = np.zeros(mask.shape, dtype=bool)
    if not mask[start]:
        return reached
    reached[start] = True
    todo = [start]
    h, w = mask.shape
    while todo:
        r, c = todo.pop()
        for dr, dc in D4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and mask[x, y] and not reached[x, y]:
                reached[x, y] = True
                todo.append((x, y))
    return reached


def _components(mask, directions=D4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append(cells)
    return result


def _point(anchor, forward, lateral, u, v):
    return (anchor[0] + u * forward[0] + v * lateral[0],
            anchor[1] + u * forward[1] + v * lateral[1])


def _local(p, anchor, forward, lateral):
    dr, dc = p[0] - anchor[0], p[1] - anchor[1]
    return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 5, "Five visible roles are required"
    bg = int(colors[counts.argmax()])
    assert np.count_nonzero(counts == counts.max()) == 1, "Background must be dominant"
    singletons = colors[counts == 1]
    assert len(singletons) == 1, "The anchor is the unique singleton color"
    marker = int(singletons[0])
    anchor = tuple(map(int, np.argwhere(a == marker)[0]))
    remaining = sorted(set(map(int, colors)) - {bg, marker})
    candidates = []
    for rod in remaining:
        support = set(map(tuple, np.argwhere(a == rod).tolist()))
        if not 3 <= len(support) <= 7:
            continue
        for f in D4:
            if support != {_point(anchor, f, (f[1], -f[0]), -k, 0)
                           for k in range(1, len(support) + 1)}:
                continue
            for l in ((f[1], -f[0]), (-f[1], f[0])):
                attachment = _point(anchor, f, l, 0, 1)
                if not _inside(attachment, a.shape):
                    continue
                strand = int(a[attachment])
                if strand not in remaining or strand == rod:
                    continue
                cells = set(map(tuple, np.argwhere(a == strand).tolist()))
                local = {_local(p, anchor, f, l) for p in cells}
                if (len(cells) >= 2 and all(u <= 0 and v >= 1 for u, v in local)
                        and len(_components(a == strand, D8)) == 1):
                    structure = next(c for c in remaining if c not in (rod, strand))
                    candidates.append((bg, marker, rod, strand, structure, anchor, f, l))
    assert len(candidates) == 1, "Ambiguous rod/strand/shore interpretation"
    return candidates[0]


def _cap(profile, length):
    """Least cap with successive slopes +1, then 0, then -1.

    profile[t-1] is the farthest exterior cell on the negative-lateral bank
    in forward column t. These two extrema define its digital convex cap.
    """
    peak = max(1, max(b + 1 for b in profile))
    departure = max(1, max(t + b + 1 for t, b in enumerate(profile, 1)))
    return [(u, min(u + 1, peak, departure - u)) for u in range(length)]


def _analyze(a):
    bg, marker, rod, strand, structure, anchor, f, l = _roles(a)
    selected = _flood(a != structure, anchor)
    # Padded complement distinguishes exterior-connected shore from islands.
    exterior = _flood(np.pad(~selected, 1, constant_values=True), (0, 0))
    coords = [_local((r, c), anchor, f, l)
              for r in range(a.shape[0]) for c in range(a.shape[1])]
    umin, umax = min(u for u, v in coords), max(u for u, v in coords)
    vmin, vmax = min(v for u, v in coords), max(v for u, v in coords)
    assert umax >= 2
    profile = []
    for u in range(umax + 1):
        assert selected[_point(anchor, f, l, u, vmax)], "The selected room must reach its lateral edge"
        rows = []
        for v in range(vmin, vmax + 1):
            r, c = _point(anchor, f, l, u, v)
            if exterior[r + 1, c + 1]:
                rows.append(v)
        # Taking the external silhouette, rather than the first selected cell,
        # also handles shore-connected overhangs. Internal islands are not E.
        bank = max(rows, default=vmin - 1)
        if u == 0:
            assert bank < 0, "The attachment must start beyond the bank"
        else:
            profile.append(bank)
    assert max(profile) >= 0, "At least one shore bulge must influence the strand"
    assert all(b <= t for t, b in enumerate(profile, 1)), "Unsupported incoming tangent"
    length = int(np.count_nonzero(a == strand))
    local_path = _cap(profile, length)
    assert local_path[0] == (0, 1)
    slopes = [q[1] - p[1] for p, q in zip(local_path, local_path[1:])]
    assert all(s in (-1, 0, 1) for s in slopes) and slopes == sorted(slopes, reverse=True)
    path = [_point(anchor, f, l, u, v) for u, v in local_path]
    visible = [p for p in path if _inside(p, a.shape)]
    assert len(visible) == min(length, umax + 1), "Only the forward canvas edge may clip"
    assert all(selected[p] and a[p] not in (structure, rod, marker) for p in visible)
    islands, erased, kept = [], [], []
    for cells in _components(a == structure):
        touches = any(_inside((r + dr, c + dc), a.shape)
                      and selected[r + dr, c + dc]
                      for r, c in cells for dr, dc in D4)
        (kept if touches else erased).append(cells)
        if all(not exterior[r + 1, c + 1] for r, c in cells):
            islands.append(cells)
    assert all(c in kept for c in islands)
    return {"roles": (bg, marker, rod, strand, structure), "anchor": anchor,
            "forward": f, "lateral": l, "bounds": (umin, umax, vmin, vmax),
            "selected": selected, "exterior": exterior, "profile": profile,
            "length": length, "path": path, "visible": visible,
            "islands": islands, "erased": erased, "kept": kept}


def _answer(a, d):
    bg, marker, rod, strand, structure = d["roles"]
    out = a.copy()
    out[a == strand] = bg
    for cells in d["erased"]:
        for p in cells:
            out[p] = bg
    for p in d["visible"]:
        out[p] = strand
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _answer(a, _analyze(a))


def _island_blind_path(a, d):
    """Counterfactual: incorrectly use every structural cell as an obstacle."""
    st = d["roles"][4]
    profile = list(d["profile"])
    for p in map(tuple, np.argwhere(a == st).tolist()):
        u, v = _local(p, d["anchor"], d["forward"], d["lateral"])
        if 1 <= u <= len(profile):
            profile[u - 1] = max(profile[u - 1], v)
    return [_point(d["anchor"], d["forward"], d["lateral"], u, v)
            for u, v in _cap(profile, d["length"])]


def _features(a, d=None):
    d = _analyze(a) if d is None else d
    ext = d["exterior"]
    off_rim = sum(not ext[r:r + 3, c:c + 3].any() for r, c in d["visible"])
    naive = {p for p in _island_blind_path(a, d) if _inside(p, a.shape)}
    bg, marker, rod, strand, structure = d["roles"]
    profile = d["profile"]
    bay_cells = 0
    for p in d["visible"]:
        u, v = _local(p, d["anchor"], d["forward"], d["lateral"])
        if 2 <= u < len(profile):
            between_teeth = profile[u - 1] < min(max(profile[:u - 1]), max(profile[u:]))
            r, c = p
            bay_cells += int(between_teeth and not ext[r:r + 3, c:c + 3].any())
    teeth = sum(b >= 0 and (i == 0 or profile[i - 1] < 0) for i, b in enumerate(profile))
    return {"shape": tuple(a.shape), "length": d["length"],
            "support_length": int(np.count_nonzero(a == rod)),
            "forward": d["forward"], "lateral": d["lateral"],
            "islands": len(d["islands"]), "erased": len(d["erased"]),
            "clipped": len(d["visible"]) < d["length"], "off_rim": off_rim,
            "bay_bridge_cells": bay_cells, "shore_teeth": teeth,
            "island_matters": naive != set(d["visible"]),
            "diagonal_slack": len(_components(a == strand, D4)) > 1,
            "peak": max(d["profile"]), "profile": tuple(d["profile"]),
            "compartments": len(_components(a != structure)),
            "selected_smaller": int(d["selected"].sum()) < int((a != structure).sum()) / 2,
            "roles": d["roles"]}


def _slack(rng, length, back, depth, diagonal=False, style=0):
    cells = {(0, 1)}
    if diagonal:
        cells.add((-1, 2))
        allowed = {(u, v) for u in range(-back, 0) for v in range(2, depth + 1)}
    else:
        allowed = {(u, v) for u in range(-back, 1) for v in range(1, depth + 1)}
    assert len(allowed | {(0, 1)}) >= length
    dirs = D4 if style in (0, 1) and not diagonal else D8
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells for du, dv in dirs}
                          & allowed - cells)
        assert frontier
        scores = []
        for u, v in frontier:
            adjacent = sum((u + du, v + dv) in cells for du, dv in D8)
            scores.append((adjacent + 1) ** 2 if style == 0 else
                          1 / (adjacent + 1) ** 2 if style == 2 else
                          1 + (v if style == 3 else -u))
        weights = np.array(scores, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _island_shape(rng):
    h, w = map(int, rng.integers(1, 4, size=2))
    cells = {(r, c) for r in range(h) for c in range(w)}
    if rng.random() < .4:
        cells = {(r, c) for r, c in cells if r == 0 or c == 0}
    return cells


def _construct(rng, lesson, split, colors, generous=False):
    """Return a role scene, then validate all semantics from its visible array."""
    h = 30 if generous else int(rng.integers(20, 31))
    w = 30 if generous else int(rng.integers(20, 31))
    rod_length = int(rng.integers(3, 8))
    ac = rod_length + int(rng.integers(0, 3))
    upper = w - ac - 1
    if upper < 12:
        return None
    baseline = int(rng.integers(3, h - 13))
    gap = int(rng.integers(2, 4))
    ar = baseline + gap
    depth = h - ar - 1
    shore = np.full(w, baseline, dtype=int)
    peak = int(rng.integers(0, 4))
    # Pack actual disjoint teeth, rather than counting possibly merged draws.
    # Both demonstrations are useful even for the legal minimum of two pairs.
    count = (2 if lesson in ("visible", "clipped") else int(rng.integers(3, min(5, upper // 3) + 1))
             if split == "test" else int(rng.integers(1, 3)))
    first_allowed = max(2, peak) if lesson == "visible" else 2
    minimum_gap = 3 if lesson == "visible" else 2
    widths = list(map(int, rng.integers(1, 4, size=count)))
    available = upper - first_allowed
    while sum(widths) + minimum_gap * (count - 1) > available:
        choices = [i for i, value in enumerate(widths) if value > 1]
        if not choices:
            return None
        widths[int(rng.choice(choices))] -= 1
    spare = available - sum(widths) - minimum_gap * (count - 1)
    extras = rng.multinomial(spare, np.full(count + 1, 1 / (count + 1)))
    cursor = first_allowed + int(extras[0])
    teeth = []
    for i, width in enumerate(widths):
        tip = (peak if lesson == "visible" else int(rng.integers(0, 2)) if lesson == "clipped" and i == 0
               else int(rng.integers(2, 4)) if lesson == "clipped" else int(rng.integers(0, min(3, cursor) + 1)))
        teeth.append((cursor, tip, width))
        cursor += width + minimum_gap + int(extras[i + 1])
    for u, tip, width in teeth:
        for t in range(u, u + width):
            # Stepped and flat teeth both stay within the supported tangent.
            bottom = ar + max(0, tip - (t - u if rng.random() < .5 else 0))
            shore[ac + t] = max(shore[ac + t], bottom)
    role = np.zeros((h, w), dtype=np.int8)
    thick = int(rng.integers(1, 3))
    for c, bottom in enumerate(shore):
        role[baseline - thick + 1:bottom + 1, c] = 4
    profile = (shore[ac + 1:] - ar).tolist()
    cap = dict(_cap(profile, upper + 1))
    max_cap = max(cap.values())
    # Islands lie well inside the selected room, beyond the taut line. They
    # would raise the cap if exterior topology were ignored.
    island_goal = int(rng.integers(1, 4)) if lesson == "visible" else int(rng.integers(0, 4))
    for _ in range(island_goal):
        shape = _island_shape(rng)
        sh = max(r for r, c in shape) + 1
        sw = max(c for r, c in shape) + 1
        positions = [(u, v) for u in range(max_cap + 3, upper - sw + 1)
                     for v in range(max_cap + 2, depth - sh)
                     if v + sh - 1 <= u]
        rng.shuffle(positions)
        for u, v in positions:
            cells = {(ar + v + dr, ac + u + dc) for dr, dc in shape}
            halo = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if all(_inside(p, role.shape) and role[p] == 0 for p in halo):
                for p in cells:
                    role[p] = 4
                break
    # Disconnected structures above the divider must be erased; some touch an edge.
    distractor_goal = int(rng.integers(1, 4)) if lesson == "visible" else int(rng.integers(0, 4))
    for _ in range(distractor_goal):
        for attempt in range(8):
            rr = int(rng.integers(0, baseline - thick))
            cc = int(rng.integers(0, w - 2))
            ww = int(rng.integers(1, 4))
            cells = {(rr, c) for c in range(cc, cc + ww)}
            halo = {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if all(not _inside(p, role.shape) or role[p] == 0 for p in halo):
                for p in cells:
                    role[p] = 4
                break
    if lesson == "clipped":
        length = upper + int(rng.integers(2, 6))
    elif lesson == "visible":
        minimum_length = max(max_cap + 4, teeth[-1][0] + teeth[-1][2] + 1)
        length = int(rng.integers(minimum_length, upper + 2))
    else:
        length = int(rng.integers(2, upper + 6))
    diagonal = lesson == "visible" or bool(rng.integers(2))
    capacity = ac * (depth - 1) + 1 if diagonal else (ac + 1) * depth
    if length > capacity:
        return None
    slack = _slack(rng, length, ac, depth, diagonal, int(rng.integers(0, 5)))
    role[ar, ac - rod_length:ac] = 2
    role[ar, ac] = 1
    for u, v in slack:
        assert role[ar + v, ac + u] == 0
        role[ar + v, ac + u] = 3
    a = np.asarray(colors, dtype=np.int8)[role]
    a = np.rot90(a, int(rng.integers(4))).copy()
    if rng.integers(2):
        a = np.fliplr(a).copy()
    try:
        d = _analyze(a)
    except AssertionError:
        return None
    feats = _features(a, d)
    if lesson == "visible" and not (feats["island_matters"] and feats["bay_bridge_cells"]
            and feats["erased"] and not feats["clipped"] and feats["diagonal_slack"]):
        return None
    if lesson == "clipped" and not feats["clipped"]:
        return None
    return {"input": a, "output": _answer(a, d)}


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    palette_mode = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = ("visible" if index == 0 else "clipped") if split == "train" and index < 2 else None
            colors = palette
            if palette_mode == 1:
                colors = (palette[0],) + tuple(map(int, rng.choice([c for c in range(10) if c != palette[0]], 4, replace=False)))
            elif palette_mode == 2:
                colors = tuple(map(int, rng.choice(10, 5, replace=False)))
            for attempt in range(120):
                pair = _construct(rng, lesson, split, colors, generous=attempt >= 80)
                distinct_budget = (pair is not None and (split != "train" or index != 1
                                   or int(np.count_nonzero(pair["input"] == _roles(pair["input"])[3]))
                                   != _analyze(puzzle["train"][0]["input"])["length"]))
                if pair is not None and distinct_budget and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded distinct-scene construction exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            d = _analyze(a)
            bg, marker, rod, strand, structure = d["roles"]
            assert np.array_equal(out, solve(a)), "Output differs from input-only rule"
            assert not np.array_equal(a, out)
            assert np.array_equal(a[a == rod], out[a == rod]) and out[d["anchor"]] == marker
            assert int((out == strand).sum()) == len(d["visible"])
            assert int((a == strand).sum()) == d["length"]
            assert len(d["visible"]) == len(set(d["visible"]))
            for cells in d["kept"]:
                assert all(out[p] == structure for p in cells)
            for cells in d["erased"]:
                assert all(out[p] == bg for p in cells)
            for u, b in enumerate(d["profile"], 1):
                assert _cap(d["profile"], u + 1)[u][1] >= b + 1
            assert len(_components(a != structure)) >= 2
            if split == "train":
                train_features.append(_features(a, d))
    assert any(f["island_matters"] and f["bay_bridge_cells"] and f["erased"]
               and f["diagonal_slack"] and not f["clipped"] for f in train_features)
    assert any(f["clipped"] for f in train_features)
    assert len({f["length"] for f in train_features}) >= 2


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _metamorphic_checks(puzzle):
    witness = next(p for p in puzzle["train"] if _features(p["input"])["island_matters"]
                   and not _features(p["input"])["clipped"])
    a, out = witness["input"], witness["output"]
    original = a.copy()
    assert np.array_equal(solve(a), out) and np.array_equal(a, original)
    for rotation in range(4):
        for reflect in (False, True):
            x, y = np.rot90(a, rotation), np.rot90(out, rotation)
            if reflect:
                x, y = np.fliplr(x), np.fliplr(y)
            assert np.array_equal(solve(x), y)
    rng = np.random.default_rng(505)
    perm = rng.permutation(10)
    assert np.array_equal(solve(perm[a]), perm[out])
    d = _analyze(a)
    bg, marker, rod, strand, structure = d["roles"]
    back, depth = -d["bounds"][0], d["bounds"][3]
    # Rearranging the slack without changing its length leaves the answer fixed.
    rearranged = a.copy()
    rearranged[a == strand] = bg
    for u, v in _slack(rng, d["length"], back, depth, False, 0):
        rearranged[_point(d["anchor"], d["forward"], d["lateral"], u, v)] = strand
    assert np.array_equal(solve(rearranged), out)
    shorter = a.copy()
    shorter[a == strand] = bg
    for u, v in _slack(rng, d["length"] - 1, back, depth, False, 2):
        shorter[_point(d["anchor"], d["forward"], d["lateral"], u, v)] = strand
    short_out = solve(shorter)
    assert int((short_out == strand).sum()) == d["length"] - 1
    assert set(map(tuple, np.argwhere(short_out == strand).tolist())) == set(d["visible"][:-1])
    # Removing an enclosed island changes only that preserved object's cells.
    for island in d["islands"]:
        removed, expected = a.copy(), out.copy()
        for p in island:
            removed[p] = expected[p] = bg
        assert np.array_equal(solve(removed), expected)
    # Connect an influential island to the outer shore. Its geometry now pulls
    # the strand, although its old cells, color and count have not changed.
    changed_route = False
    for island in d["islands"]:
        local = sorted(_local(p, d["anchor"], d["forward"], d["lateral"]) for p in island)
        u = local[0][0]
        top = min(v for t, v in local if t == u)
        # Remove other islands so a newly pulled route cannot collide with a
        # still-ignored island; collisions are outside this family's grammar.
        isolated = a.copy()
        for other in d["islands"]:
            if other != island:
                for p in other:
                    isolated[p] = bg
        isolated_out = solve(isolated)
        attached = isolated.copy()
        for v in range(d["profile"][u - 1] + 1, top):
            attached[_point(d["anchor"], d["forward"], d["lateral"], u, v)] = structure
        answer = solve(attached)
        if not np.array_equal(answer == strand, isolated_out == strand):
            changed_route = True
            break
    assert changed_route, "An exterior connection must sometimes change tension"
    corrupted = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")


def self_test():
    """Bounded streaming checks over 42 seeds; no files or traces are written."""
    shapes, signatures, profiles, lengths, orientations, palettes = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _features(a)
                digest.update(bytes(a.shape) + a.tobytes())
                shapes.add(tuple(a.shape))
                profiles.add(f["profile"])
                lengths.add(f["length"])
                orientations.add((f["forward"], f["lateral"]))
                palettes.add(f["roles"])
                totals[split + "_pairs"] += 1
                for name in ("clipped", "island_matters", "diagonal_slack", "selected_smaller"):
                    totals[name] += int(f[name])
                totals["off_rim_cells"] += f["off_rim"]
                totals["bay_bridge_cells"] += f["bay_bridge_cells"]
                totals[split + "_teeth_" + str(f["shore_teeth"])] += 1
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 40 and len(profiles) >= 150
    assert len(lengths) >= 15 and len(orientations) == 8 and len(palettes) >= 30
    assert all(totals[k] > 0 for k in ("clipped", "island_matters", "diagonal_slack", "selected_smaller"))
    assert _puzzles_equal(generate_puzzle(19, 2, 1), generate_puzzle(19, 2, 1))
    assert not _puzzles_equal(generate_puzzle(19, 2, 1), generate_puzzle(20, 2, 1))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(77, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    _metamorphic_checks(generate_puzzle(0))
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "shore_profiles": len(profiles), "lengths": len(lengths),
            "orientations": len(orientations), "palettes": len(palettes),
            "totals": dict(totals), "independent_semantic_verification": False}
