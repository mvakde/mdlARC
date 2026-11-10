"""Rhythm inspectors: periodic rail repair determines eligibility and destination.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _plus(p, d):
    return p[0] + d[0], p[1] + d[1]


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalise(inp):
    choices = []
    for k in range(4):
        g = np.rot90(inp, k)
        if g.shape[1] < 3:
            continue
        token = int(g[0, 0])
        if np.all(g[:, 0] == token) and np.count_nonzero(g == token) == len(g):
            bg = int(g[0, 1])
            if bg != token and np.all(g[:, 1] == bg):
                choices.append((g.copy(), k, token, bg))
    if len(choices) != 1:
        raise ValueError("one unique token edge with a clear approach is required")
    return choices[0]


def _segments(path):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    ends = [0] + [i for i in range(1, len(path) - 1)
                  if directions[i] != directions[i - 1]] + [len(path) - 1]
    result = []
    for start, end in zip(ends, ends[1:]):
        d = directions[start]
        normal = (-d[1], d[0])
        indexes = list(range(start + 2, end - 1))
        result.append(dict(start=start, end=end, direction=d, indexes=indexes,
                           flanks=([_plus(path[i], normal) for i in indexes],
                                   [_plus(path[i], (-normal[0], -normal[1]))
                                    for i in indexes])))
    return result


def _walls(path, shape):
    cells = set(path)
    return {q for p in path for d in DIRS for q in [_plus(p, d)]
            if _inside(shape, q) and q[1] >= 2 and q not in cells}


def _trace(g, bg):
    """Topology finds geometric tracks; recovered color words select movers."""
    h, w = g.shape
    examined, tracks = set(), []
    for r in range(1, h - 1):
        start = (r, 2)
        if (start in examined or int(g[start]) != bg
                or int(g[r - 1, 2]) == bg or int(g[r + 1, 2]) == bg):
            continue
        queue, comp = deque([start]), {start}
        while queue:
            p = queue.popleft()
            for d in DIRS:
                q = _plus(p, d)
                if (_inside(g.shape, q) and q[1] >= 2 and q not in comp
                        and int(g[q]) == bg):
                    comp.add(q)
                    queue.append(q)
        examined.update(comp)
        neighbors = {p: [_plus(p, d) for d in DIRS if _plus(p, d) in comp]
                     for p in comp}
        endpoints = [p for p in comp if len(neighbors[p]) == 1]
        # The broad exterior component is not a one-cell track.
        if (len(comp) < 12 or len(endpoints) != 2 or start not in endpoints
                or any(len(v) not in (1, 2) for v in neighbors.values())):
            continue
        if {p for p in comp if p[1] == 2} != {start}:
            raise ValueError("a track must have exactly one entry")
        path, prev, current = [start], None, start
        while True:
            onward = [q for q in neighbors[current] if q != prev]
            if not onward:
                break
            if len(onward) != 1:
                raise ValueError("branching track")
            prev, current = current, onward[0]
            path.append(current)
        if len(path) != len(comp) or path[1] != (r, 3):
            raise ValueError("invalid path or entrance direction")
        if any(p[0] in (0, h - 1) or p[1] == w - 1 for p in path[:-1]):
            raise ValueError("only a terminal may meet an exit edge")
        runs = _segments(path)
        if any(len(s["indexes"]) < 8 for s in runs):
            raise ValueError("too little periodic evidence on a straight run")
        walls = _walls(path, g.shape)
        if any(int(g[q]) == bg for q in walls):
            raise ValueError("track walls must be geometrically complete")
        tracks.append(dict(path=path, runs=runs, walls=walls,
                           start=(r, 0), terminal=path[-1]))
    if not tracks:
        raise ValueError("no readable tracks")
    occupied = set()
    for t in tracks:
        footprint = set(t["path"]) | t["walls"]
        if footprint & occupied:
            raise ValueError("track footprints overlap")
        occupied.update(footprint)
    visible = {(int(r), int(c)) for r, c in np.argwhere(g[:, 2:] != bg)}
    visible = {(r, c + 2) for r, c in visible}
    if visible != set().union(*(t["walls"] for t in tracks)):
        raise ValueError("unexplained guide cells")
    return tracks


def _recover_word(sequence):
    seq = np.asarray(sequence, dtype=np.int8)
    if seq.ndim != 1 or len(seq) < 8 or len(np.unique(seq)) != 2:
        raise ValueError("a rail must be a nonconstant binary word")
    colors = np.unique(seq)
    cap, candidates = max(1, len(seq) // 8), []
    for p in range(1, len(seq) // 4 + 1):
        unit = np.empty(p, dtype=np.int8)
        error = 0
        for phase in range(p):
            values = seq[phase::p]
            counts = [int(np.count_nonzero(values == c)) for c in colors]
            if counts[0] == counts[1]:
                break
            unit[phase] = colors[int(counts[1] > counts[0])]
            error += min(counts)
        else:
            if error <= cap and len(np.unique(unit)) == 2:
                clean = unit[np.arange(len(seq)) % p]
                candidates.append((clean, p))
    if not candidates:
        raise ValueError("no supported periodic explanation")
    clean, p = candidates[0]
    if any(not np.array_equal(clean, other) for other, _ in candidates):
        raise ValueError("competing periodic repairs")
    return clean.copy(), p


def _analyse(inp):
    g, rotation, token, bg = _normalise(_array(inp))
    tracks = _trace(g, bg)
    clean = g.copy()
    motif_cells, rail_colors = set(), set()
    for t in tracks:
        t["eligible"], t["errors"], t["mismatched_runs"] = True, [], []
        for j, run in enumerate(t["runs"]):
            original, restored, periods = [], [], []
            for flanks in run["flanks"]:
                if any(not _inside(g.shape, q) for q in flanks):
                    raise ValueError("a sampled rail leaves the grid")
                seq = np.array([g[q] for q in flanks], dtype=np.int8)
                if token in seq or bg in seq:
                    raise ValueError("invalid rail palette")
                word, period = _recover_word(seq)
                original.append(seq)
                restored.append(word)
                periods.append(period)
                rail_colors.update(map(int, np.unique(seq)))
                for q, value in zip(flanks, word):
                    if q in motif_cells:
                        raise ValueError("two runs share a patterned cell")
                    motif_cells.add(q)
                    clean[q] = value
            if set(map(int, np.unique(original[0]))) != set(map(int, np.unique(original[1]))):
                raise ValueError("paired rails must share their two-color palette")
            matched = np.array_equal(restored[0], restored[1])
            if not matched:
                t["eligible"] = False
                t["mismatched_runs"].append(j)
            changed = (original[0] != restored[0]) | (original[1] != restored[1])
            t["errors"].extend(i for i, yes in zip(run["indexes"], changed) if yes)
            run.update(original=original, restored=restored, periods=periods,
                       matched=matched, changed=changed)
        t["errors"].sort()
        t["destination"] = t["path"][t["errors"][0]] if t["errors"] else t["terminal"]
    connector_cells = set().union(*(t["walls"] for t in tracks)) - motif_cells
    connectors = {int(g[q]) for q in connector_cells}
    if len(connectors) != 1 or connectors & (rail_colors | {bg, token}):
        raise ValueError("connector pieces must have their own uniform role")
    return dict(grid=g, clean=clean, rotation=rotation, token=token, bg=bg,
                connector=next(iter(connectors)), tracks=tracks, motif_cells=motif_cells)


def _answer(a, *, ignore_matching=False, terminal_only=False):
    out = a["clean"].copy()
    for t in a["tracks"]:
        if t["eligible"] or ignore_matching:
            out[t["start"]] = a["bg"]
            out[t["terminal"] if terminal_only else t["destination"]] = a["token"]
    return np.rot90(out, -a["rotation"]).copy()


def _without_periodic_inference(a):
    """Diagnostic ablation: compare raw words and perform terminal transport."""
    out = a["grid"].copy()
    for t in a["tracks"]:
        if all(np.array_equal(*s["original"]) for s in t["runs"]):
            out[t["start"]] = a["bg"]
            out[t["terminal"]] = a["token"]
    return np.rot90(out, -a["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, paths, periodic words, errors and destinations from input."""
    return _answer(_analyse(input_grid))


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _plus(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        degree = sum(_plus(p, d) in cells for d in DIRS)
        if degree != (1 if i in (0, len(path) - 1) else 2):
            return None
    # A near return can share a wall without touching the centerline. That
    # wall cannot carry two independently chosen straight-run words.
    observations = [q for s in _segments(path) for flank in s["flanks"] for q in flank]
    if len(observations) != len(set(observations)):
        return None
    return path


def _random_path(rng, shape, bends, closed):
    h, w = shape
    points, direction = [(int(rng.integers(1, h - 1)), 2)], (0, 1)
    for j in range(bends + 1):
        p = points[-1]
        if j == bends and not closed and direction != (0, -1):
            target = ((h - 1 if direction[0] > 0 else 0), p[1]) if direction[0] else (p[0], w - 1)
            distance = abs(target[0] - p[0]) + abs(target[1] - p[1])
            if distance < 11:
                return None
        else:
            targets = []
            for length in range(11, 30):
                q = (p[0] + length * direction[0], p[1] + length * direction[1])
                if 1 <= q[0] <= h - 2 and 4 <= q[1] <= w - 2:
                    targets.append(q)
            if not targets:
                return None
            target = targets[int(rng.integers(len(targets)))]
        points.append(target)
        direction = ((-direction[1], direction[0]) if rng.integers(2)
                     else (direction[1], -direction[0]))
    path = _polyline(points)
    if path is None or any(p[1] == 2 for p in path[1:]):
        return None
    return path


def _pack(rng, shape, count, bends, first_closed, second_closed=None):
    paths, footprints = [], set()
    for i in range(count):
        nturns = bends if i == 0 else (1 if bends and rng.random() < .18 else 0)
        closed = first_closed if i == 0 else (
            second_closed if i == 1 and second_closed is not None else bool(rng.integers(2)))
        for _ in range(160):
            path = _random_path(rng, shape, nturns, closed)
            if path is None:
                continue
            footprint = set(path) | _walls(path, shape)
            if footprint & footprints:
                continue
            paths.append(path)
            footprints.update(footprint)
            break
        else:
            return None
    return paths


def _bent_demo(rng, shape, rotation):
    """Variable nested elbows; both concepts are exposed even with two demos."""
    h, w = shape
    r1 = int(rng.integers(11, h - 5))
    r2 = int(rng.integers(r1 + 3, h - 1))
    c1 = int(rng.integers(13, w - 6))
    c2 = int(rng.integers(c1 + 3, w - 1))
    end2 = int(rng.integers(0, r2 - 10))
    paths = [_polyline([(r1, 2), (r1, c1), (0, c1)]),
             _polyline([(r2, 2), (r2, c2), (end2, c2)])]
    # The damaged outgoing run is traversed against raw row-major order.
    if rotation in (2, 3):
        paths = [[(h - 1 - r, c) for r, c in p] for p in paths]
    return paths


def _fallback_paths(rng, count, bends):
    shape = (30, 30)
    if bends >= 3:
        assert count == 2
        r = int(rng.integers(1, 7))
        far = int(rng.integers(18, 27))
        near = int(rng.integers(4, far - 10))
        points = [(r, 2), (r, far), (r + 11, far), (r + 11, near)]
        if bends == 3:
            points.append((29 if rng.integers(2) else r + 22, near))
        else:
            points += [(r + 22, near), (r + 22, 29)]
        paths = [_polyline(points), _polyline([(r + 4, 2), (r + 4, 13)])]
        if rng.integers(2):
            paths = [[(29 - rr, c) for rr, c in p] for p in paths]
    elif bends:
        points = [(1, 2), (1, 18), (14, 18)]
        points += [(14, 29)] if bends == 2 else [(29, 18)]
        # For one bend use a single vertical run, not a redundant waypoint.
        if bends == 1:
            points = [(1, 2), (1, 18), (29, 18)]
        rows = [5, 9, 20, 24]
        paths = [_polyline(points)] + [_polyline([(r, 2), (r, 13)])
                                        for r in rows[:count - 1]]
    else:
        paths = [_polyline([(1 + 4 * i, 2), (1 + 4 * i, 29 if i % 2 == 0 else 24)])
                 for i in range(count)]
    return shape, paths


def _primitive_words(p):
    words = []
    for code in range(1, (1 << p) - 1):
        bits = np.array([(code >> j) & 1 for j in range(p)], dtype=np.int8)
        if not any(p % q == 0 and np.array_equal(bits, bits[np.arange(p) % q])
                   for q in range(1, p)):
            words.append(bits)
    return words


def _flip(clean, period, positions):
    noisy = clean.copy()
    colors = np.unique(clean)
    for i in positions:
        noisy[i] = colors[1] if noisy[i] == colors[0] else colors[0]
    recovered, p = _recover_word(noisy)
    if p != period or not np.array_equal(recovered, clean):
        raise ValueError("corruption has an ambiguous or unintended explanation")
    return noisy


def _words(rng, n, palette, matched, policy):
    """Procedural binary units and bounded noise; alternating fallback is safe."""
    for attempt in range(65):
        p = int(rng.integers(2, min(6, n // 4) + 1)) if attempt < 64 else 2
        words = _primitive_words(p)
        unit = words[int(rng.integers(len(words)))]
        seq = np.array(palette, dtype=np.int8)[unit[np.arange(n) % p]]
        other_p = p
        if matched:
            other = seq.copy()
        elif attempt < 64 and rng.random() < .5:
            other_p = int(rng.integers(2, min(6, n // 4) + 1))
            other_words = _primitive_words(other_p)
            second = other_words[int(rng.integers(len(other_words)))]
            other = np.array(palette, dtype=np.int8)[second[np.arange(n) % other_p]]
            if np.array_equal(other, seq):
                continue
        else:
            other = np.where(seq == palette[0], palette[1], palette[0]).astype(np.int8)
        if policy == "concealed":
            i = int(rng.integers(n))
            positions = [[i], [i]]
        elif policy == "delayed":
            i = int(rng.integers(0, n - 1))
            j = int(rng.integers(i + 1, n))
            positions = [[i], [j]]
        elif policy == "clean":
            positions = [[], []]
        else:
            positions = []
            for period in (p, other_p):
                cap = min(3, period, max(1, n // 8))
                minimum = 1 if policy == "damaged" else 0
                amount = int(rng.integers(minimum, cap + 1))
                phases = rng.choice(period, size=amount, replace=False)
                positions.append([int(rng.choice(np.arange(int(ph), n, period))) for ph in phases])
        try:
            noisy = [_flip(seq, p, positions[0]), _flip(other, other_p, positions[1])]
        except ValueError:
            continue
        return noisy, [seq, other]
    raise RuntimeError("bounded word construction exhausted")


def _decorate(rng, shape, paths, roles, mode):
    bg, token, connector, palette = roles
    noisy = np.full(shape, bg, dtype=np.int8)
    noisy[:, 0] = token
    clean = noisy.copy()
    walls, sampled, path_walls = set(), set(), []
    for path in paths:
        own = _walls(path, shape)
        path_walls.append(own)
        assert not (own | set(path)) & walls
        walls.update(own | set(path))
        for q in own:
            noisy[q] = clean[q] = connector
    track_infos = []
    for i, path in enumerate(paths):
        runs = _segments(path)
        active = i != 1 and (i == 0 or mode == "straight_demo" and i == 2 or rng.random() < .65)
        mismatch = (len(runs) - 1 if mode == "bent_demo" else int(rng.integers(len(runs))))
        force_segment = len(runs) - 1 if mode == "bent_demo" and i == 0 else int(rng.integers(len(runs)))
        errors = []
        for j, run in enumerate(runs):
            colors = tuple(map(int, rng.choice(palette, 2, replace=False)))
            matched = active or j != mismatch
            if mode == "straight_demo" and i == 0:
                policy = "concealed"
            elif mode == "straight_demo" and i == 2:
                policy = "clean"
            elif mode == "bent_demo" and i == 0:
                policy = "delayed" if j == force_segment else "clean"
            elif active and i == 0 and j == force_segment:
                policy = "damaged"
            else:
                policy = "random"
            observed, restored = _words(rng, len(run["indexes"]), colors, matched, policy)
            for side in range(2):
                for q, before, after in zip(run["flanks"][side], observed[side], restored[side]):
                    assert q not in sampled and q in path_walls[i]
                    sampled.add(q)
                    noisy[q], clean[q] = before, after
            changed = (observed[0] != restored[0]) | (observed[1] != restored[1])
            errors.extend(k for k, yes in zip(run["indexes"], changed) if yes)
        track_infos.append((active, path, sorted(errors)))
    expected = clean.copy()
    for active, path, errors in track_infos:
        if active:
            expected[path[0][0], 0] = bg
            expected[path[errors[0]] if errors else path[-1]] = token
    return noisy, expected


def _roles(rng, base, mixing):
    colors = list(map(int, rng.permutation(10))) if mixing else base
    size = int(rng.integers(2, 8))
    return colors[0], colors[1], colors[2], colors[3:3 + size]


def _make_pair(rng, mode, roles, rotation, count, bends):
    for attempt in range(50):
        if mode == "bent_demo":
            shape = (int(rng.integers(24, 31)), int(rng.integers(26, 31)))
            paths = _bent_demo(rng, shape, rotation)
        else:
            lower_h = 26 if bends >= 3 else 18
            lower_w = 27 if bends >= 3 else (21 if mode == "test" else 16)
            shape = (int(rng.integers(lower_h, 31)), int(rng.integers(lower_w, 31)))
            closed = bool(rng.integers(2))
            paths = _pack(rng, shape, count, bends, closed,
                          not closed if mode == "straight_demo" else None)
            # In the straight demo index 2 is the clean active track. Give it
            # the opposite terminal type to index 0; index 1 is ineligible.
            if mode == "straight_demo" and paths is not None:
                # Repack with statuses ordered [damaged active, clean active,
                # inactive], then put the inactive geometry at index 1.
                paths[1], paths[2] = paths[2], paths[1]
        if paths is None:
            continue
        try:
            inp, out = _decorate(rng, shape, paths, roles, mode)
            inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
            if not np.array_equal(solve(inp), out):
                raise AssertionError("construction and input reader disagree")
        except ValueError:
            continue
        return dict(input=inp, output=out)
    # Geometry fallback is procedural and follows the same exact rail rules.
    if mode == "bent_demo":
        shape = (30, 30)
        paths = _bent_demo(rng, shape, rotation)
    else:
        shape, paths = _fallback_paths(rng, count, bends)
        if mode == "straight_demo":
            paths[1], paths[2] = paths[2], paths[1]
    inp, out = _decorate(rng, shape, paths, roles, mode)
    pair = dict(input=np.rot90(inp, rotation).copy(), output=np.rot90(out, rotation).copy())
    assert np.array_equal(solve(pair["input"]), pair["output"])
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = list(map(int, rng.permutation(10)))
    mixing, mixed_rotation = bool(rng.integers(2)), bool(rng.integers(3))
    initial_rotation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            mode = ("straight_demo" if index == 0 else "bent_demo" if index == 1 else "train") if split == "train" else "test"
            ntracks = int(rng.integers(3, 6)) if mode == "straight_demo" else (2 if mode == "bent_demo" else int(rng.integers(2, 6)))
            bends = 0 if mode == "straight_demo" else 1 if mode == "bent_demo" else int(rng.integers(0, 5 if split == "test" else 3))
            if bends >= 3:
                ntracks = 2
            rotation = int(rng.integers(4)) if mixed_rotation else initial_rotation
            for _ in range(30):
                roles = _roles(rng, base, mixing)
                pair = _make_pair(rng, mode, roles, rotation, ntracks, bends)
                if _key(pair["input"]) not in seen:
                    # Normalize dimensions so count/palette changes do not
                    # hide accidental identical geometry in the two demos.
                    if mode == "bent_demo" and _analyse(pair["input"])["grid"].shape == _analyse(puzzle["train"][0]["input"])["grid"].shape:
                        continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-example construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    tracks = a["tracks"]
    concealed = any(t["eligible"] and any(
        bool(changed) and int(left) == int(right)
        for s in t["runs"] for changed, left, right in zip(s["changed"], *s["original"]))
                    for t in tracks)
    delayed = any(t["eligible"] and t["errors"] and len(t["runs"]) > 1
                  and t["errors"][0] > t["runs"][0]["end"] for t in tracks)
    downstream = any(not t["eligible"] and t["runs"][0]["matched"]
                     and t["mismatched_runs"] for t in tracks)
    clean_active = any(t["eligible"] and not t["errors"] for t in tracks)
    open_end = lambda t: (t["terminal"][0] in (0, a["grid"].shape[0] - 1)
                          or t["terminal"][1] == a["grid"].shape[1] - 1)
    return dict(concealed=concealed, delayed=delayed, downstream=downstream,
                clean_active=clean_active, open=any(t["eligible"] and open_end(t) for t in tracks),
                closed=any(t["eligible"] and not open_end(t) for t in tracks))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyse(inp)
            tracks = a["tracks"]
            assert 2 <= len(tracks) <= 5
            assert any(t["eligible"] and t["errors"] for t in tracks)
            assert any(not t["eligible"] for t in tracks)
            expected = _answer(a)
            assert np.array_equal(expected, out), "incorrect rail restoration or inspector destination"
            normalized_output = np.rot90(out, a["rotation"])
            repairs = int(np.count_nonzero(a["grid"] != a["clean"]))
            movers = [t for t in tracks if t["eligible"]]
            assert np.count_nonzero(inp != out) == repairs + 2 * len(movers)
            assert np.count_nonzero(inp == a["token"]) == np.count_nonzero(out == a["token"])
            destinations = [t["destination"] for t in movers]
            assert len(set(destinations)) == len(destinations)
            for t in tracks:
                assert int(normalized_output[t["start"]]) == (a["bg"] if t["eligible"] else a["token"])
                if t["eligible"]:
                    assert int(a["grid"][t["destination"]]) == a["bg"]
                    assert int(normalized_output[t["destination"]]) == a["token"]
                for run in t["runs"]:
                    for flanks, restored in zip(run["flanks"], run["restored"]):
                        assert np.array_equal([normalized_output[q] for q in flanks], restored)
            # Parent omissions must affect actual token placement, not merely
            # our descriptive metadata or the already-required rail changes.
            assert not np.array_equal(_answer(a, ignore_matching=True) == a["token"], out == a["token"])
            assert not np.array_equal(_answer(a, terminal_only=True) == a["token"], out == a["token"])
            assert not np.array_equal(_without_periodic_inference(a) == a["token"], out == a["token"])
            if split == "train":
                training.append(a)
    first, second = training[:2]
    f1, f2 = _features(first), _features(second)
    assert f1["concealed"] and f1["clean_active"] and f1["open"] and f1["closed"]
    assert f2["delayed"] and f2["downstream"] and f2["open"]
    assert len(first["tracks"]) != len(second["tracks"])
    assert first["grid"].shape != second["grid"].shape
    # The second demo distinguishes route order from raster order after rotation.
    def raw_point(p, shape, k):
        r, c = p
        for _ in range(k % 4):
            r, c = shape[1] - 1 - c, r
            shape = (shape[1], shape[0])
        return r, c
    against_raster = False
    for t in second["tracks"]:
        if t["eligible"] and len(t["errors"]) >= 2:
            points = [raw_point(t["path"][i], second["grid"].shape, -second["rotation"]) for i in t["errors"]]
            against_raster |= points[0] != min(points)
    assert against_raster, "training must distinguish path order from grid order"


def self_test():
    """Bounded 42-seed measurements, count contract, equivariance and mutations."""
    shapes, layouts, structures, signatures = set(), set(), set(), set()
    counts, turns, periods, rotations, tokens, partials = set(), set(), set(), set(), set(), 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                a = _analyse(pair["input"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                shapes.add(pair["input"].shape)
                counts.add(len(a["tracks"]))
                rotations.add(a["rotation"])
                tokens.add(a["token"])
                layouts.add(tuple((tuple(len(s["indexes"]) for s in t["runs"]), t["eligible"],
                                   tuple(t["errors"])) for t in a["tracks"]))
                roles = {}
                structures.add(tuple(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat))
                for t in a["tracks"]:
                    turns.add(len(t["runs"]) - 1)
                    for s in t["runs"]:
                        periods.update(s["periods"])
                        partials += any(len(s["indexes"]) % p for p in s["periods"])
        if seed < 4:
            # Arbitrary literal relabeling and all rotations preserve semantics.
            pair = puzzle["train"][1]
            permutation = np.random.default_rng(seed + 912).permutation(10)
            for k in range(4):
                inp = np.rot90(permutation[pair["input"]], k).copy()
                out = np.rot90(permutation[pair["output"]], k).copy()
                assert np.array_equal(solve(inp), out)
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 50 and len(layouts) >= 180
    assert counts == {2, 3, 4, 5} and turns == {0, 1, 2, 3, 4}
    assert {2, 3, 4, 5} <= periods and rotations == {0, 1, 2, 3}
    assert len(tokens) >= 8 and partials >= 100
    # Solving is broader than the discriminative multi-track sampler. A lone
    # clean track still follows the visibly encoded terminal convention.
    path = _polyline([(4, 2), (4, 16)])
    inp, expected = _decorate(np.random.default_rng(831), (10, 18), [path],
                              (0, 1, 2, [3, 4]), "straight_demo")
    a = _analyse(inp)
    clean_single = a["clean"].copy()
    clean_expected = clean_single.copy()
    clean_expected[4, 0], clean_expected[4, 16] = 0, 1
    assert np.array_equal(solve(clean_single), clean_expected)
    assert np.array_equal(solve(inp), expected)
    # Tied phases and two distinct equally sparse periodic explanations must
    # be rejected, rather than resolved by period or literal-color priority.
    ambiguous = np.zeros(24, dtype=np.int8)
    ambiguous[[1, 4, 7, 9, 19]] = 1
    for sequence in (np.array([0, 1, 0, 1, 1, 0, 1, 0]), ambiguous):
        try:
            _recover_word(sequence)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous periodic evidence was accepted")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(9100 + 10 * n + m, n, m)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == m
    p, q = generate_puzzle(73, 3, 2), generate_puzzle(73, 3, 2)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
               for k in ("input", "output"))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for mutation in ("terminal", "all_tracks", "no_repairs"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
        pair = bad["test"][0]
        a = _analyse(pair["input"])
        if mutation == "terminal":
            pair["output"] = _answer(a, terminal_only=True)
        elif mutation == "all_tracks":
            pair["output"] = _answer(a, ignore_matching=True)
        else:
            out = np.rot90(pair["output"], a["rotation"]).copy()
            for pos in a["motif_cells"]:
                out[pos] = a["grid"][pos]
            pair["output"] = np.rot90(out, -a["rotation"]).copy()
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong output was accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                shapes=len(shapes), layouts=len(layouts), color_normalized_structures=len(structures),
                track_counts=sorted(counts), turn_counts=sorted(turns), periods=sorted(periods),
                entry_rotations=sorted(rotations), token_literals=sorted(tokens),
                partial_runs=partials, independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
