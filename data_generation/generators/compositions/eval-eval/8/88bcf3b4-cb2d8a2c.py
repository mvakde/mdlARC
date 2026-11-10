"""Finite slack cords threaded through clearance-coded, turning elbow guides.

No parent module, seed, training pair, or generation state is used by solve.
Coordinates below are physical (row, column). The initial attachment is cell 1
of the cord; corner cells are counted once, and off-grid cells still cost length.
"""
from __future__ import annotations

import json

import numpy as np


DIR4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


class _SceneSearchExhausted(RuntimeError):
    """No sampled presentation fitted; this does not change the family rule."""


def _add(p, v, distance=1):
    return p[0] + distance * v[0], p[1] + distance * v[1]


def _dot(p, v):
    return p[0] * v[0] + p[1] * v[1]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer 1..30 by 1..30 grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    unseen = set(cells)
    groups = []
    neighbors = DIR8 if diagonal else DIR4
    while unseen:
        first = min(unseen)
        todo, group = [first], {first}
        unseen.remove(first)
        while todo:
            p = todo.pop()
            for d in neighbors:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _line(cells):
    """Return the ordered cells of a contiguous axis-aligned line, or None."""
    if len(cells) < 2:
        return None
    ordered = sorted(cells)
    a, b = ordered[0], ordered[-1]
    if a[0] != b[0] and a[1] != b[1]:
        return None
    if abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1 != len(cells):
        return None
    return ordered


def _decode(input_grid):
    """Infer the four roles through the capped stem and the slack attachment."""
    inp = _grid(input_grid)
    histogram = np.bincount(inp.ravel(), minlength=10)
    if np.count_nonzero(histogram == histogram.max()) != 1:
        raise ValueError("Input must have a unique modal background")
    bg = int(histogram.argmax())
    foreground = sorted(set(map(int, np.unique(inp))) - {bg})
    if len(foreground) != 3:
        raise ValueError("Expected background, cord, permanent material and cap/markers")
    candidates = []
    for cord in foreground:
        slack = {tuple(map(int, p)) for p in np.argwhere(inp == cord)}
        if len(slack) < 2 or len(_components(slack, True)) != 1:
            continue
        other = [c for c in foreground if c != cord]
        structures = _components({tuple(map(int, p)) for p in
                                  np.argwhere((inp != bg) & (inp != cord))})
        lines = [_line(cells) for cells in structures]
        if any(line is None for line in lines) or not 2 <= len(lines) <= 5:
            continue
        for permanent, marker in (other, other[::-1]):
            for index, anchor in enumerate(lines):
                if len(anchor) < 4:
                    continue
                caps = [p for p in anchor if inp[p] == marker]
                if len(caps) != 1 or caps[0] not in (anchor[0], anchor[-1]):
                    continue
                cap = caps[0]
                rear = anchor[-1] if cap == anchor[0] else anchor[0]
                delta = _sub(cap, rear)
                heading = tuple(int(np.sign(x)) for x in delta)
                attachment = _add(cap, heading)
                if attachment not in slack:
                    continue
                lateral = heading[1], -heading[0]
                local = [(_dot(_sub(p, attachment), heading),
                          _dot(_sub(p, attachment), lateral))
                         for p in slack if p != attachment]
                if any(u > 0 for u, v in local):
                    continue
                if not (all(v >= 1 for u, v in local)
                        or all(v <= -1 for u, v in local)):
                    continue
                guides = []
                for j, line in enumerate(lines):
                    if j == index:
                        continue
                    marks = {p for p in line if inp[p] == marker}
                    n = len(marks)
                    if not 1 <= n <= 4 or len(line) < 2 * n:
                        break
                    matches = [end for end in (0, 1)
                               if marks == set((line if end == 0 else line[::-1])[::2][:n])]
                    if len(matches) != 1:
                        break
                    bad, good = ((line[0], line[-1]) if matches[0] == 0
                                 else (line[-1], line[0]))
                    guides.append({"cells": set(line), "markers": marks, "n": n,
                                   "bad": bad, "good": good})
                else:
                    candidates.append({"input": inp, "bg": bg, "cord": cord,
                                       "permanent": permanent, "marker": marker,
                                       "slack": slack, "anchor": set(anchor),
                                       "cap": cap, "attachment": attachment,
                                       "heading": heading, "guides": guides})
    if len(candidates) != 1:
        raise ValueError("Capped anchor/cord/guide role assignment must be unique")
    return candidates[0]


def _ray_hit(current, heading, guide):
    good, bad = guide["good"], guide["bad"]
    axis = tuple(int(np.sign(x)) for x in _sub(good, bad))
    if _dot(axis, heading):
        return None  # Parallel bars are not intercepted by this forward ray.
    t = _dot(_sub(good, current), heading)
    if t <= 0:
        return None
    cross = _add(current, heading, t)
    if cross not in guide["cells"]:
        return None
    return t, axis


def _segment(start, end):
    """Unit steps after start, including end; a zero-length segment is empty."""
    delta = _sub(end, start)
    if delta[0] and delta[1]:
        raise ValueError("Route segments must be axis aligned")
    distance = abs(delta[0]) + abs(delta[1])
    step = tuple(int(np.sign(x)) for x in delta)
    return [_add(start, step, k) for k in range(1, distance + 1)]


def _walk(scene, length=None, clearance_override=None, keep_heading=False):
    """Trace the finite conceptual strand, without reference to canvas bounds.

    Override arguments are used only for documented shortcut checks, never by
    solve or construction. Events record full-wrap completion cell counts.
    """
    budget = len(scene["slack"]) if length is None else int(length)
    current, heading = scene["attachment"], scene["heading"]
    path, events, visited = [current], [], set()
    while len(path) < budget:
        hits = [(hit[0], j, hit[1]) for j, guide in enumerate(scene["guides"])
                if j not in visited and (hit := _ray_hit(current, heading, guide)) is not None]
        if not hits:
            path.extend(_add(current, heading, k) for k in range(1, budget - len(path) + 1))
            break
        hits.sort()
        if len(hits) > 1 and hits[0][0] == hits[1][0]:
            raise ValueError("Two guides are tied on the forward ray")
        distance, j, lateral = hits[0]
        guide = scene["guides"][j]
        d = guide["n"] + 1 if clearance_override is None else clearance_override
        if distance <= d:
            raise ValueError("A guide must allow at least one approach step before its turn")
        q1 = _add(current, heading, distance - d)
        q2 = _add(_add(guide["good"], heading, -d), lateral, d)
        q3 = _add(q2, heading, 2 * d)
        # Every lateral turn must proceed toward the unmarked endpoint.
        if _dot(_sub(q2, q1), lateral) <= 0:
            raise ValueError("Invalid elbow geometry")
        full = _segment(current, q1) + _segment(q1, q2) + _segment(q2, q3)
        complete_at = len(path) + len(full)
        first_turn_at = len(path) + len(_segment(current, q1))
        events.append({"guide": j, "n": guide["n"], "clearance": d,
                       "incoming": heading, "outgoing": lateral,
                       "corners": (q1, q2, q3), "complete_at": complete_at,
                       "turn_at": first_turn_at, "complete": complete_at <= budget})
        path.extend(full[:budget - len(path)])
        if len(path) == budget:
            break
        visited.add(j)
        current = q3
        heading = heading if keep_heading else lateral
    return path, events


def _paint(scene, path, events):
    out = scene["input"].copy()
    for p in scene["slack"]:
        out[p] = scene["bg"]
    for event in events:
        if event["complete"]:
            for p in scene["guides"][event["guide"]]["markers"]:
                out[p] = scene["permanent"]
    for p in path:
        if _inside(p, out.shape):
            out[p] = scene["cord"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the input's anchor, cord budget, guide instructions and geometry."""
    scene = _decode(input_grid)
    path, events = _walk(scene)
    structural = scene["anchor"] | set().union(*(g["cells"] for g in scene["guides"]))
    if len(set(path)) != len(path) or set(path) & structural:
        raise ValueError("Cord must neither cross itself nor enter fixed material")
    return _paint(scene, path, events)


def _bbox(cells):
    return [min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells)]


def _slack(rng, length, stem_size, diagonal, style):
    """One local eight-connected object, behind and beside its attachment."""
    width = int(rng.integers(4, 11))
    depth = max(stem_size, (length - 1 + width - 1) // width)
    depth += int(rng.integers(0, 3))
    top = -1 if diagonal else 0
    cells = {(0, 0), (-1, 1)} if diagonal else {(0, 0), (0, 1)}
    if style == 0:  # Compact raster fill, independent of the eventual route.
        for u in range(top, -depth - 1, -1):
            for v in range(1, width + 1):
                cells.add((u, v))
                if len(cells) == length:
                    return cells
    elif style == 1:  # Bent filament with diagonal links and a wider reach.
        u, v = min(cells - {(0, 0)})
        while len(cells) < length:
            choices = [p for p in ((u - 1, v + 1), (u - 1, v), (u, v + 1))
                       if -depth <= p[0] <= top and p[1] <= width]
            if not choices:
                break
            u, v = choices[int(rng.integers(len(choices)))]
            cells.add((u, v))
    while len(cells) < length:
        frontier = sorted({_add(p, d) for p in cells for d in DIR8
                           if -depth <= _add(p, d)[0] <= top
                           and 1 <= _add(p, d)[1] <= width} - cells)
        if not frontier:
            raise ValueError("Slack region too small")
        # Sparse growth or dense lobes. Sorting makes reproducibility explicit.
        weights = np.array([sum(_add(p, d) in cells for d in DIR8) + 1
                            for p in frontier], dtype=float)
        if style == 2:
            weights = 1 / weights ** 2
        elif style == 3:
            weights = weights ** 2
        else:
            weights[:] = 1
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _plan(rng, counts, approach_band, repeat_first=False, compact=False):
    """Sample physical guides; construction does not use the input decoder."""
    current, heading = (0, 0), (1, 0)
    path, guides, events = [current], [], []
    previous_sign = None
    for index, n in enumerate(counts):
        d = n + 1
        gap = approach_band[0] if compact else int(rng.integers(approach_band[0], approach_band[1] + 1))
        sign = (-1 if index % 2 == 0 else 1) if compact else int(rng.choice((-1, 1)))
        if repeat_first and index == 1:
            sign = previous_sign
        previous_sign = sign
        lateral = sign * heading[1], -sign * heading[0]
        if compact:
            # A wide second guide must not extend back into the source slack.
            good_offset = 3 if index == 1 and counts[0] == 1 and n > 1 else 0
            span = max(2 * n, good_offset + 1)
        else:
            good_offset = int(rng.integers(0, 4))
            span = int(rng.integers(max(2 * n, good_offset + 2), max(2 * n, good_offset + 2) + 5))
        bad_offset = good_offset - span + 1
        center = _add(current, heading, gap + d)
        good, bad = _add(center, lateral, good_offset), _add(center, lateral, bad_offset)
        cells = {_add(bad, lateral, k) for k in range(span)}
        marks = {_add(bad, lateral, 2 * k) for k in range(n)}
        q1 = _add(current, heading, gap)
        q2 = _add(q1, lateral, good_offset + d)
        q3 = _add(q2, heading, 2 * d)
        new_path = _segment(current, q1) + _segment(q1, q2) + _segment(q2, q3)
        events.append({"guide": len(guides), "complete_at": len(path) + len(new_path),
                       "turn_at": len(path) + gap, "corners": (q1, q2, q3)})
        guides.append({"cells": cells, "markers": marks, "n": n, "good": good, "bad": bad})
        path.extend(new_path)
        current, heading = q3, lateral
    return path, guides, events, heading


def _make_pair(rng, colors, counts, lesson, rotation, reflect, approach_band=(1, 3), clip=False,
               repeat_first=False):
    bg, cord, permanent, marker = colors
    for attempt in range(500):
        compact = attempt >= 400
        route, guides, events, tail_heading = _plan(rng, counts, approach_band, repeat_first, compact)
        if len(set(route)) != len(route):
            continue
        if lesson == "partial":
            event = events[-1]
            # Clear the bar's axial level, but stop just before its far clearance
            # corner. This contradicts normalizing at the first turn or crossing.
            low = event["complete_at"] - (counts[-1] + 1) + 1
            high = event["complete_at"] - 1
            length = high if compact else int(rng.integers(low, high + 1))
        elif lesson == "early":
            # A partial first wrap while later guides remain untouched.
            event = events[0]
            length = event["turn_at"] + 1 if compact else int(rng.integers(event["turn_at"] + 1, event["complete_at"]))
        else:
            length = len(route) + (1 if compact else int(rng.integers(1, 9 if clip else 6)))
        if length > len(route):
            tail_start = route[-1]
            route.extend(_add(tail_start, tail_heading, k)
                         for k in range(1, length - len(route) + 1))
        prefix = route[:length]
        stem_size = 4 if compact else int(rng.integers(4, 8))
        anchor = {(-k, 0) for k in range(1, stem_size + 1)}
        if compact:
            w = 10 if len(counts) >= 3 and approach_band[0] >= 4 else 6
            slack = {(0, 0)} | {(-1 - k // w, 1 + k % w) for k in range(length - 1)}
        else:
            slack = _slack(rng, length, stem_size, lesson == "complete", int(rng.integers(0, 5)))
        structural = anchor | set().union(*(g["cells"] for g in guides))
        if structural & slack or structural & set(route):
            continue
        # A capped stem and each guide must remain separate straight components.
        groups = _components(structural)
        if len(groups) != len(guides) + 1:
            continue
        # Avoid misleading near contacts between separate physical guides.
        if any(any(_add(p, d) in other["cells"] for p in g["cells"] for d in DIR8)
               for j, g in enumerate(guides) for other in guides[j + 1:]):
            continue
        if any(any(_add(p, d) in g["cells"] for p in slack | anchor for d in DIR8)
               for g in guides):
            continue
        conceptual = {"attachment": (0, 0), "heading": (1, 0), "slack": slack, "guides": guides}
        inferred_route, inferred_events = _walk(conceptual, len(route))
        if inferred_route != route or [e["guide"] for e in inferred_events] != list(range(len(guides))):
            continue  # A future guide intercepted an earlier ray: resample geometry.
        occupied = structural | slack
        bounds = _bbox(occupied | set(prefix))
        input_bounds = _bbox(occupied)
        clip_edge = None
        if clip:
            choices = [j for j in range(4) if
                       (bounds[j] < input_bounds[j] if j % 2 == 0 else bounds[j] > input_bounds[j])]
            if not choices:
                continue
            clip_edge = int(rng.choice(choices))
            bounds[clip_edge] = input_bounds[clip_edge]
        height, width = bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1
        if max(height, width) > 30:
            continue
        for axis, size in enumerate((height, width)):
            extra = min(30 - size, int(rng.integers(0, 6)))
            edges = [2 * axis + j for j in (0, 1) if 2 * axis + j != clip_edge]
            for _ in range(extra):
                j = int(rng.choice(edges))
                bounds[j] += -1 if j % 2 == 0 else 1
        height, width = bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1
        if rng.random() < .25:
            axis = 0 if height < width else 1
            edges = [2 * axis + j for j in (0, 1) if 2 * axis + j != clip_edge]
            j = int(rng.choice(edges))
            bounds[j] += (-1 if j % 2 == 0 else 1) * abs(height - width)
            height = width = max(height, width)
        offset = -bounds[0], -bounds[2]

        def place(p):
            return _add(p, offset)

        inp = np.full((height, width), bg, dtype=np.int8)
        for p in anchor:
            inp[place(p)] = marker if p == (-1, 0) else permanent
        for p in slack:
            inp[place(p)] = cord
        for g in guides:
            for p in g["cells"]:
                inp[place(p)] = marker if p in g["markers"] else permanent
        out = inp.copy()
        for p in slack:
            out[place(p)] = bg
        for g, event in zip(guides, events):
            if event["complete_at"] <= length:
                for p in g["markers"]:
                    out[place(p)] = permanent
        for p in prefix:
            q = place(p)
            if _inside(q, out.shape):
                out[q] = cord
        if reflect:
            inp, out = inp[:, ::-1], out[:, ::-1]
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            scene = _decode(inp)
            if len(scene["guides"]) != len(counts) or not np.array_equal(solve(inp), out):
                continue
            path, decoded_events = _walk(scene)
            if lesson == "complete" and len(_components(scene["slack"])) == 1:
                continue
            if clip != any(not _inside(p, inp.shape) for p in path):
                continue
            # Teach actual complete state changes, not only a count in an unreached bar.
            if lesson == "complete" and (len(decoded_events) != len(counts)
                                         or not all(e["complete"] for e in decoded_events)):
                continue
            if lesson == "partial" and (not any(e["complete"] for e in decoded_events)
                                        or decoded_events[-1]["complete"]):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise _SceneSearchExhausted("Bounded elbow-scene search exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _sample_counts(rng, count, maximum, require_maximum=False):
    ns = list(map(int, rng.choice((1, maximum), size=count)))
    protected = int(rng.integers(count)) if require_maximum else None
    if protected is not None:
        ns[protected] = maximum
    # Four wide elbows plus a large slack mass seldom fit an uncrossed 30-cell
    # canvas. Limit the total clearance burden, not the guide count or grammar.
    limit = 8 if count == 4 else 9 if count == 3 else 8
    while sum(ns) > limit:
        choices = [j for j, n in enumerate(ns) if n > 1 and j != protected]
        ns[int(rng.choice(choices))] = 1
    return ns


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    pairwise_palette = bool(rng.integers(2))
    maximum = int(rng.integers(2, 5))
    regime = int(rng.integers(3))  # Held-out heading, guide count, or approach distance.
    base_rotation = int(rng.integers(4))
    train_rotations = [base_rotation] if regime == 0 else list(range(4))
    test_rotations = [(base_rotation + int(rng.choice((1, 2, 3)))) % 4] if regime == 0 else list(range(4))
    lesson_counts = [1, maximum]
    rng.shuffle(lesson_counts)
    puzzle, seen = {"train": [], "test": []}, set()

    def add(split, counts, lesson, clip=False, repeat_first=False):
        rotations = train_rotations if split == "train" else test_rotations
        band = (4, 6) if regime == 2 and split == "test" else (1, 3)
        feasible_counts = list(counts)
        for _ in range(30):
            colors = tuple(map(int, rng.choice(10, 4, replace=False))) if pairwise_palette else palette
            try:
                pair = _make_pair(rng, colors, feasible_counts, lesson, int(rng.choice(rotations)),
                                  bool(rng.integers(2)), band, clip, repeat_first)
            except _SceneSearchExhausted:
                # Redraw presentation parameters, retaining taught counts and
                # the selected train/test shift. Core two-guide lessons remain.
                if len(feasible_counts) >= 3:
                    count = 3 if band[0] >= 4 else len(feasible_counts)
                    feasible_counts = [maximum] + [1] * (count - 1)
                continue
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                return
        raise AssertionError("Unique input search exhausted")

    # Two variable geometric lessons suffice even at the legal minimum count.
    repeated_lesson = int(rng.integers(2))
    add("train", lesson_counts, "complete", repeat_first=repeated_lesson == 0)
    add("train", lesson_counts[::-1], "partial", clip=bool(rng.random() < .3),
        repeat_first=repeated_lesson == 1)
    for j in range(2, num_train):
        guide_count = int(rng.integers(1, 3 if regime == 1 else 5))
        counts = _sample_counts(rng, guide_count, maximum)
        lesson = str(rng.choice(("complete", "partial", "early"))) if guide_count >= 2 else str(rng.choice(("complete", "early")))
        # A clipping lesson is required before test examples may use clipping.
        clip = j == 2 or bool(rng.random() < .2)
        if clip and lesson == "early":
            lesson = "complete"
        add("train", counts, lesson, clip)
    clipping_taught = any(measure_pair(p)["clipped"] for p in puzzle["train"])
    for j in range(num_test):
        guide_count = int(rng.integers(3, 5) if regime == 1 else rng.integers(1, 5))
        counts = _sample_counts(rng, guide_count, maximum, require_maximum=True)
        # Non-unit clearance acts on every query. A fully completed, clipped
        # query can saturate at the boundary, so extra length may be invisible.
        lesson = str(rng.choice(("complete", "partial"))) if guide_count >= 2 else "complete"
        add("test", counts, lesson, clip=clipping_taught and bool(rng.random() < .4))
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(j)] for j in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    scene = _decode(pair["input"])
    path, events = _walk(scene)
    visible = {p for p in path if _inside(p, scene["input"].shape)}
    signs = [int(np.sign(e["incoming"][0] * e["outgoing"][1]
                         - e["incoming"][1] * e["outgoing"][0])) for e in events]
    return {"shape": tuple(scene["input"].shape), "heading": scene["heading"],
            "palette": (scene["bg"], scene["cord"], scene["permanent"], scene["marker"]),
            "guides": len(scene["guides"]), "length": len(scene["slack"]),
            "counts": tuple(g["n"] for g in scene["guides"]),
            "completed_counts": tuple(e["n"] for e in events if e["complete"]),
            "complete": sum(e["complete"] for e in events),
            "partial": bool(events and not events[-1]["complete"]),
            "clipped": len(visible) < len(path), "lost": len(path) - len(visible),
            "diagonal_slack": len(_components(scene["slack"])) > 1,
            "turn_signs": tuple(signs),
            "approaches": tuple(abs(e["corners"][0][0] - (scene["attachment"] if i == 0 else events[i - 1]["corners"][2])[0])
                                + abs(e["corners"][0][1] - (scene["attachment"] if i == 0 else events[i - 1]["corners"][2])[1])
                                for i, e in enumerate(events))}


def _alternative(scene, **options):
    try:
        path, events = _walk(scene, **options)
        return _paint(scene, path, events)
    except ValueError:
        # A shortcut can make a later elbow physically impossible. That also
        # contradicts the observed answer; it is not a new family fallback.
        return None


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _decode(inp)
            path, events = _walk(scene)
            assert len(path) == len(scene["slack"]) and len(path) == len(set(path))
            assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
            assert path[0] == scene["attachment"]
            structural = scene["anchor"] | set().union(*(g["cells"] for g in scene["guides"]))
            assert not structural & set(path)
            assert np.array_equal(solve(inp), out), "Input-only solve mismatch"
            assert all(out[p] == inp[p] for p in scene["anchor"]), "Anchor/cap changed"
            completed = {e["guide"] for e in events if e["complete"]}
            for j, guide in enumerate(scene["guides"]):
                for p in guide["cells"]:
                    expected = scene["permanent"] if j in completed else int(inp[p])
                    assert out[p] == expected, "Wrong completion-dependent guide state"
            visible = {p for p in path if _inside(p, inp.shape)}
            assert set(map(tuple, np.argwhere(out == scene["cord"]))) == visible
            assert out[scene["attachment"]] == scene["cord"]
            changed_material = {tuple(map(int, p)) for p in np.argwhere((inp != out) & (inp != scene["bg"]))}
            assert changed_material <= scene["slack"] | set().union(*(g["markers"] for g in scene["guides"]))
            m = measure_pair(pair)
            (training if split == "train" else testing).append((m, scene, out))
    trained_counts = {n for m, scene, out in training for n in m["completed_counts"]}
    assert 1 in trained_counts and len(trained_counts) >= 2
    assert {n for m, scene, out in testing for n in m["counts"]} <= trained_counts
    assert any(m["complete"] >= 2 and not m["clipped"] and m["diagonal_slack"] for m, scene, out in training)
    assert any(m["partial"] and m["complete"] >= 1 for m, scene, out in training)
    assert any(any(a == b for a, b in zip(m["turn_signs"], m["turn_signs"][1:]))
               for m, scene, out in training), "Missing repeated-turn demonstration"
    critical_partial = False
    for m, scene, out in training:
        path, events = _walk(scene)
        if events and not events[-1]["complete"] and any(e["complete"] for e in events):
            last = events[-1]
            remaining = last["complete_at"] - len(path)
            critical_partial |= 0 < remaining < last["clearance"]
    assert critical_partial, "Partial lesson must clear the bar level but not finish the wrap"
    assert any(not np.array_equal(_alternative(scene, length=len(scene["slack"]) + 120), out)
               for m, scene, out in training), "Training does not expose finite cord length"
    assert any(not np.array_equal(_alternative(scene, clearance_override=2), out)
               for m, scene, out in training), "Training does not expose unary clearance"
    assert any(not np.array_equal(_alternative(scene, keep_heading=True), out)
               for m, scene, out in training), "Training does not expose heading changes"
    if any(m["clipped"] for m, scene, out in testing):
        assert any(m["clipped"] for m, scene, out in training), "Clipping was not demonstrated"


def _fixture(length, width=21):
    """Hand-specified two-elbow regression: output bends are stated literally."""
    a = np.zeros((18, width), dtype=np.int8)
    a[1:4, 5], a[4, 5] = 8, 2
    slack = [(5, 5)] + [(r, c) for r in range(4, -1, -1) for c in range(6, 11)]
    for p in slack[:length]:
        a[p] = 3
    a[10, 3:8] = (2, 8, 2, 8, 8)
    a[11:17, 15] = (8, 8, 8, 8, 8, 2)
    bends = [(5, 5), (7, 5), (7, 10), (13, 10), (13, 13), (9, 13), (9, 17)]
    literal_path = [bends[0]]
    for start, end in zip(bends, bends[1:]):
        literal_path.extend(_segment(start, end))
    if length > len(literal_path):
        literal_path.extend((9 - k, 17) for k in range(1, length - len(literal_path) + 1))
    b = a.copy()
    b[b == 3] = 0
    if length >= 14:
        b[10, 3:8] = 8
    if length >= 25:
        b[11:17, 15] = 8
    for p in literal_path[:length]:
        if _inside(p, b.shape):
            b[p] = 3
    return {"input": a, "output": b}


def self_test():
    """Bounded semantic regressions and measured structural diversity, 42 seeds."""
    for length in (10, 14, 20, 24, 25, 26):
        for width in (16, 21):
            pair = _fixture(length, width)
            for angle in range(4):
                for reflection in (False, True):
                    permutation = np.array([7, 0, 5, 9, 1, 2, 8, 6, 4, 3], dtype=np.int8)
                    a, b = pair["input"], pair["output"]
                    if reflection:
                        a, b = a[:, ::-1], b[:, ::-1]
                    a, b = permutation[np.rot90(a, angle)], permutation[np.rot90(b, angle)]
                    before = a.copy()
                    assert np.array_equal(solve(a), b), (length, width, angle, reflection)
                    assert np.array_equal(a, before), "solve mutated its input"
    # Same slack shape and geometry, different physical budget: the second guide
    # changes state precisely at cell 25, including an off-grid release corner.
    assert solve(_fixture(24, 16)["input"])[16, 15] == 2
    assert solve(_fixture(25, 16)["input"])[16, 15] == 8
    base = _fixture(25)
    scene = _decode(base["input"])
    assert not np.array_equal(_alternative(scene, clearance_override=2), base["output"])
    assert not np.array_equal(_alternative(scene, keep_heading=True), base["output"])
    # Rearrange a slack mass without changing its count; the output is invariant.
    rearranged = base["input"].copy()
    rearranged[rearranged == 3] = 0
    for p in [(5, 5)] + [(r, c) for c in range(6, 12) for r in range(4, 0, -1)]:
        rearranged[p] = 3
    assert np.array_equal(solve(rearranged), base["output"])
    metrics, puzzles, shapes, normalized, role_colors = [], [], set(), set(), [set() for _ in range(4)]
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                metrics.append((split, m))
                shapes.add(m["shape"])
                semantic = np.empty_like(pair["input"])
                for role, color in enumerate(m["palette"]):
                    semantic[pair["input"] == color] = role
                    role_colors[role].add(color)
                normalized.add(_key(semantic))
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 50 and len(normalized) >= 180
    assert all(len(colors) >= 8 for colors in role_colors)
    for split in ("train", "test"):
        ms = [m for s, m in metrics if s == split]
        assert {m["heading"] for m in ms} == set(DIR4)
        assert {n for m in ms for n in m["counts"]} == {1, 2, 3, 4}
        assert {m["guides"] for m in ms} == {1, 2, 3, 4}
        assert any(m["clipped"] for m in ms) and any(not m["clipped"] for m in ms)
        assert any(m["partial"] for m in ms)
        assert any(m["diagonal_slack"] for m in ms)
        assert any(m["shape"][0] == m["shape"][1] for m in ms)
        assert {a for m in ms for a in m["turn_signs"]} == {-1, 1}
    transitions = {(a, b) for split, m in metrics for a, b in zip(m["turn_signs"], m["turn_signs"][1:])}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    first = generate_puzzle(617, 5, 3)
    again = generate_puzzle(617, 5, 3)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    # Regression for a wide, long-approach four-guide request that formerly
    # exhausted the scene search. It must preserve the requested pair counts.
    recovered = generate_puzzle(1006, 10, 4)
    assert len(recovered["train"]) == 10 and len(recovered["test"]) == 4
    # The compact three-guide spine fits the long-approach shift even at n=4.
    route, _, events, _ = _plan(np.random.default_rng(1), [4, 1, 1], (4, 6), compact=True)
    assert [e["corners"] for e in events] == [((4, 0), (4, 5), (14, 5)),
        ((14, 9), (16, 9), (16, 13)), ((20, 13), (20, 15), (24, 15))]
    assert len(route) == 40
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(81, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal explicit counts accepted")
    damaged = generate_puzzle(19, 2, 1)
    scene = _decode(damaged["test"][0]["input"])
    damaged["test"][0]["output"][scene["cap"]] = scene["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a damaged anchor")
    return {"seeds": 42, "pairs": len(metrics), "shapes": len(shapes),
            "color_normalized_inputs": len(normalized),
            "guide_counts": sorted({m["guides"] for s, m in metrics}),
            "marker_counts": sorted({n for s, m in metrics for n in m["counts"]}),
            "clipped_pairs": sum(m["clipped"] for s, m in metrics),
            "partial_pairs": sum(m["partial"] for s, m in metrics)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
