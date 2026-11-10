"""Clockwork lasso: finite periodic emissions folded around one obstacle.

The specification, not the seed, fixes phase, winding and collision priority.
No parent generator, data file, learned state or external dependency is used.
Experiments are streaming and self_test stores no puzzle or execution trace.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 284077189, 504840030, 397955129, 458175641, 609624938,
              336506168, 446805591, 125847534, 708565852, 512539155,
              170051397, 508051500, 39834126, 689657584, 599258443,
              99261882, 248789202, 775463698, 681858808, 163820880,
              205238617, 407540705, 987928960, 384676054, 819739828,
              916393310, 74039791, 485133312, 995248117, 381361067,
              741610838, 837433513, 80156559, 864932403, 400193979,
              391692934, 71133926, 289775088, 11444369, 377021255, 976904399)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _transform(a, flip, turn):
    return np.rot90(np.fliplr(a) if flip else a, turn)


def _undo(a, flip, turn):
    a = np.rot90(a, -turn)
    return (np.fliplr(a) if flip else a).copy()


def _connected(cells, directions=DIR8):
    if not cells:
        return False
    seen, todo = {min(cells)}, [min(cells)]
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            q = r + dr, c + dc
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(cells)


def _components(a, background):
    unseen = {tuple(map(int, p)) for p in np.argwhere(a != background)}
    result = []
    while unseen:
        p = min(unseen)
        color = int(a[p])
        cells, todo = {p}, [p]
        unseen.remove(p)
        while todo:
            r, c = todo.pop()
            for dr, dc in DIR8:
                q = r + dr, c + dc
                if q in unseen and a[q] == color:
                    unseen.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append((color, cells))
    return result


def _octagon(cells):
    """Recognize a filled rectangle cut by four diagonal half-planes."""
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    bounds = (min(r + c for r, c in cells), min(r - c for r, c in cells),
              max(r + c for r, c in cells), max(r - c for r, c in cells))
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r + c >= bounds[0] and r - c >= bounds[1]
                and r + c <= bounds[2] and r - c <= bounds[3]}
    return cells == expected and _connected(cells)


def _loop(obstacle):
    """One-cell Chebyshev clearance, with diagonal steps on chamfered sides.

    Dilated rows are intervals. Top/bottom rows and the two extreme cells of
    intermediate rows give an ordered, simple exterior loop. The family uses
    filled digital octagons, so side changes never exceed one cell per row.
    """
    inflated = {(r + dr, c + dc) for r, c in obstacle
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    rows = {}
    for r, c in inflated:
        rows.setdefault(r, []).append(c)
    top, bottom = min(rows), max(rows)
    _require(set(rows) == set(range(top, bottom + 1)), "disconnected contour")
    bounds = {r: (min(cs), max(cs)) for r, cs in rows.items()}
    _require(all(len(rows[r]) == bounds[r][1] - bounds[r][0] + 1 for r in rows),
             "contour row has a hole")
    _require(all(abs(bounds[r][i] - bounds[r - 1][i]) <= 1
                 for r in range(top + 1, bottom + 1) for i in (0, 1)),
             "unsupported contour slope")
    route = [(top, c) for c in range(bounds[top][0], bounds[top][1] + 1)]
    route += [(r, bounds[r][1]) for r in range(top + 1, bottom + 1)]
    route += [(bottom, c) for c in range(bounds[bottom][1] - 1, bounds[bottom][0] - 1, -1)]
    route += [(r, bounds[r][0]) for r in range(bottom - 1, top, -1)]
    _require(len(route) == len(set(route)) and not set(route) & obstacle,
             "self-intersecting contour")
    _require(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
                 for a, b in zip(route, route[1:] + route[:1])), "broken contour")
    return route


def _canonical_candidate(a, bg, support):
    ar, sc = max(r for r, c in support), next(iter(support))[1]
    if support != {(r, sc) for r in range(ar - len(support) + 1, ar + 1)}:
        return None
    occupied = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    source = {(r, c) for r, c in occupied if r == ar and c < sc}
    slack = {(r, c) for r, c in occupied if r <= ar and c > sc}
    obstacle = {(r, c) for r, c in occupied if r > ar}
    if not source or len(slack) < 3 or not obstacle:
        return None
    if occupied != source | slack | obstacle | support:
        return None
    anchor = ar, sc + 1
    if {p for p in slack if p[0] == ar} != {anchor} or not _connected(slack):
        return None
    colors = [{int(a[p]) for p in cells} for cells in (support, slack, obstacle)]
    if any(len(cs) != 1 for cs in colors) or len(set.union(*colors)) != 3:
        return None
    if not _octagon(obstacle):
        return None
    try:
        route = _loop(obstacle)
    except ValueError:
        return None
    if route[0] != anchor:
        return None
    runs, c = [], sc - 1
    while c >= 0:
        if a[ar, c] == bg:
            c -= 1
            continue
        color, near, length = int(a[ar, c]), sc - c, 0
        while c >= 0 and a[ar, c] == color:
            length += 1
            c -= 1
        runs.append((length, color, near))
    if not 1 <= len(runs) <= 4 or not all(2 <= p <= 7 for p, color, near in runs):
        return None
    if any(color in set.union(*colors) for p, color, near in runs):
        return None
    if set(route) & (source | support):
        return None
    return dict(canonical=a, background=bg, support=support, source=source,
                slack=slack, obstacle=obstacle, anchor=anchor, runs=runs,
                route=route, length=len(slack), material_color=next(iter(colors[1])))


def _analyze(value):
    a = _grid(value)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    _require(int(counts.max()) > a.size // 2, "background must occupy over half the input")
    bg = int(counts.argmax())
    candidates = []
    for flip in (False, True):
        for turn in range(4):
            view = _transform(a, flip, turn)
            for color, cells in _components(view, bg):
                if 3 <= len(cells) <= 10 and len({c for r, c in cells}) == 1:
                    info = _canonical_candidate(view, bg, cells)
                    if info is not None:
                        info.update(flip=flip, turn=turn)
                        candidates.append(info)
    _require(len(candidates) == 1, "input must have exactly one lasso interpretation")
    return candidates[0]


def _emissions(route, length, runs, alternative=None):
    """Ranked union of the finite pulse positions; inputs are plain values."""
    if alternative == "straight":
        r, c = route[0]
        route = [(r, c + t) for t in range(max(1, length))]
    perimeter = len(route)
    if alternative == "no_wrap":
        length = min(length, perimeter)
    if alternative == "infinite":
        length = perimeter * math.lcm(*(p for p, color, near in runs))
    if alternative == "dense":
        return {route[t % perimeter]: runs[0][1] for t in range(length)}
    if alternative in ("last_event", "first_event"):
        result = {}
        for t in range(length):
            active = [run for run in runs if t % run[0] == 0]
            if active:
                if alternative == "first_event":
                    result.setdefault(route[t % perimeter], active[0][1])
                else:
                    result[route[t % perimeter]] = active[0][1]
        return result
    ordered = list(runs)
    if alternative == "far":
        ordered.reverse()
    elif alternative == "short":
        ordered.sort(key=lambda b: b[0])
    elif alternative == "long":
        ordered.sort(key=lambda b: -b[0])
    result = {}
    for period, color, near in ordered:
        if alternative == "period_plus":
            period += 1
        elif alternative == "period_minus":
            period -= 1
        phase = near - 1 if alternative == "source_phase" else 0
        for t in range(phase, length, period):
            result.setdefault(route[t % perimeter], color)
    return result


def _predict_info(info, alternative=None):
    a, bg = info["canonical"], info["background"]
    out = a.copy()
    for p in info["slack"]:
        out[p] = bg
    route, length = info["route"], info["length"]
    if alternative == "box":
        obstacle = info["obstacle"]
        r0, r1 = min(r for r, c in obstacle), max(r for r, c in obstacle)
        c0, c1 = min(c for r, c in obstacle), max(c for r, c in obstacle)
        route = _loop({(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)})
        start = route.index(info["anchor"])
        route = route[start:] + route[:start]
    elif alternative == "reverse":
        route = route[:1] + route[:0:-1]
    elif alternative == "visible_clock":
        route = [(r, c) for r, c in route if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]]
    elif alternative == "support_budget":
        length = len(info["support"])
    elif alternative == "four_budget":
        reached, todo = {info["anchor"]}, [info["anchor"]]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIR4:
                q = r + dr, c + dc
                if q in info["slack"] and q not in reached:
                    reached.add(q)
                    todo.append(q)
        length = len(reached)
    for p, color in _emissions(route, length, info["runs"], alternative).items():
        if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]:
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer geometry and all counters from this input alone."""
    info = _analyze(input_grid)
    return _undo(_predict_info(info), info["flip"], info["turn"])


def _shape(rng, large=False, irregular=False):
    h, w = map(int, rng.integers(7 if large else 2, 10 if large else 7, size=2))
    if not large and not irregular and rng.random() < .18:
        h = 1
        if rng.random() < .4:
            w = 1
    cuts = [0, 0, 0, 0]
    if min(h, w) >= 3:
        cap = min(3, (h - 1) // 2, (w - 1) // 2)
        cuts = list(map(int, rng.integers(0, cap + 1, size=4)))
        if irregular and not any(cuts):
            cuts[int(rng.integers(4))] = 1
    tl, tr, br, bl = cuts
    body = {(r, c) for r in range(h) for c in range(w)
            if r + c >= tl and r + w - 1 - c >= tr
            and h - 1 - r + w - 1 - c >= br and h - 1 - r + c >= bl}
    top_left = min(c for r, c in body if r == 0)
    return {(r + 1, c + 1 - top_left) for r, c in body}


def _slack(rng, length, bridge):
    width = int(rng.integers(4, 10))
    depth = (length - 2) // width + 1 + int(rng.integers(0, 3))
    offset = 1 if bridge else 0
    start = -1, offset
    cells, frontier = {start}, set()
    style = int(rng.integers(3))

    def extend(p):
        for dr, dc in DIR8:
            q = p[0] + dr, p[1] + dc
            if -depth <= q[0] <= -1 and offset <= q[1] < offset + width and q not in cells:
                frontier.add(q)

    extend(start)
    while len(cells) < length - 1:
        choices = sorted(frontier)
        _require(bool(choices), "slack region exhausted")
        degrees = np.array([1 + sum((r + dr, c + dc) in cells for dr, dc in DIR8)
                            for r, c in choices], dtype=float)
        weights = degrees ** (2 if style == 0 else -2 if style == 1 else 0)
        p = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        frontier.remove(p)
        cells.add(p)
        extend(p)
    return cells | {(0, 0)}


def _lap_interaction(route, length, runs):
    """Both directions of temporal interference must appear in lessons."""
    perimeter = len(route)
    first, earlier, takeover, protected = {}, {}, False, False
    for t in range(length):
        hit = next((k for k, b in enumerate(runs) if t % b[0] == 0), None)
        if hit is None:
            continue
        j = t % perimeter
        if j in earlier and hit != earlier[j] and t >= perimeter:
            takeover |= hit < earlier[j]
            protected |= hit > earlier[j]
        earlier[j] = min(hit, earlier.get(j, hit))
        if t < perimeter:
            first[j] = hit
    return takeover, protected


def _construct_pair(rng, periods, pulse_colors, lesson, shift, orientation, clip):
    """At most 256 small trials; only the successful pair is retained."""
    for attempt in range(256):
        obstacle = _shape(rng, large=shift == "scale", irregular=lesson)
        route = _loop(obstacle)
        perimeter = len(route)
        runs = [(int(p), int(c), 1) for p, c in zip(periods, pulse_colors)]
        if lesson:
            if perimeter <= math.lcm(*periods) + 1:
                continue
            length = perimeter + int(rng.integers(6, min(14, perimeter - 1)))
        elif shift == "early":
            length = int(rng.integers(max(7, perimeter // 2), perimeter))
        elif shift == "laps":
            length = 2 * perimeter + int(rng.integers(1, max(3, perimeter // 2)))
        else:
            length = int(rng.integers(perimeter + 1, min(perimeter + 13, 2 * perimeter)))
        emissions = _emissions(route, length, runs)
        if not emissions or len(emissions) == perimeter:
            continue  # Keep some unpainted contour cells: periodicity stays visible.
        if lesson:
            if not all(_lap_interaction(route, length, runs)):
                continue
            if any(emissions == _emissions(route, length, runs, alt)
                   for alt in ("no_wrap", "infinite", "last_event", "period_plus", "period_minus")):
                continue
            if len(set(emissions.values())) != len(set(pulse_colors)):
                continue
        slack = _slack(rng, length, bridge=lesson or rng.random() < .6)
        support_length = int(rng.integers(3, 9))
        support = {(-k, -1) for k in range(support_length)}
        right = max(c for r, c in obstacle | slack | set(route))
        # Choose gaps only after reserving the obstacle, material and margins.
        max_span = 27 - right
        spare = max_span - sum(periods)
        if spare < 1:
            continue
        gaps = [0] * len(periods)
        gaps[0] = int(rng.integers(1, min(3, spare) + 1))
        spare -= gaps[0]
        for j in range(1, len(gaps)):
            low = int(pulse_colors[j] == pulse_colors[j - 1])
            if spare < low:
                break
            gaps[j] = int(rng.integers(low, min(2, spare) + 1))
            spare -= gaps[j]
        else:
            source, actual_runs, cursor = {}, [], -2
            for period, color, gap in zip(periods, pulse_colors, gaps):
                cursor -= gap
                actual_runs.append((int(period), int(color), -1 - cursor))
                for c in range(cursor - period + 1, cursor + 1):
                    source[0, c] = int(color)
                cursor -= period
            cells = support | slack | obstacle | set(source)
            # All input is retained. Clipping may omit one exterior contour
            # row/column; the clocks still count those off-canvas positions.
            reserved = cells if clip else cells | set(route)
            r0, r1 = min(r for r, c in reserved), max(r for r, c in reserved)
            c0, c1 = min(c for r, c in reserved), max(c for r, c in reserved)
            top, left = int(rng.integers(1, 3)), int(rng.integers(0, 2))
            bottom = 0 if clip else int(rng.integers(0, 3))
            right_margin = 0 if clip else int(rng.integers(0, 2))
            h, w = r1 - r0 + 1 + top + bottom, c1 - c0 + 1 + left + right_margin
            if max(h, w) > 30:
                continue
            available = [c for c in range(10) if c not in pulse_colors]
            bg, support_color, material, obstacle_color = map(int, rng.choice(available, 4, replace=False))
            inp = np.full((h, w), bg, dtype=np.int8)
            dr, dc = top - r0, left - c0

            def paint(points, color):
                for r, c in points:
                    inp[r + dr, c + dc] = color

            paint(support, support_color)
            paint(slack, material)
            paint(obstacle, obstacle_color)
            for p, color in source.items():
                paint([p], color)
            if np.count_nonzero(inp == bg) <= inp.size // 2:
                continue
            # Construction uses repeated chronological writes far-to-near.
            # solve instead forms nearest-first sets reconstructed from input.
            out = inp.copy()
            for r, c in slack:
                out[r + dr, c + dc] = bg
            for period, color, near in reversed(actual_runs):
                for t in range(0, length, period):
                    r, c = route[t % perimeter]
                    r, c = r + dr, c + dc
                    if 0 <= r < h and 0 <= c < w:
                        out[r, c] = color
            flip, turn = orientation
            pair = {"input": _transform(inp, flip, turn).copy(),
                    "output": _transform(out, flip, turn).copy()}
            try:
                info = _analyze(pair["input"])
            except ValueError:
                continue
            canonical_out = _transform(pair["output"], info["flip"], info["turn"])
            if not any(canonical_out[r, c] == info["background"]
                       for r, c in info["route"]
                       if 0 <= r < canonical_out.shape[0] and 0 <= c < canonical_out.shape[1]):
                continue
            # Check the actual visible witnesses, after clipping, not labels
            # sampled before the geometry was laid out.
            required = ("dense", "straight")
            if lesson:
                required += ("no_wrap", "infinite", "last_event", "first_event", "box", "reverse",
                             "four_budget", "support_budget", "period_plus", "period_minus",
                             "source_phase")
                if clip:
                    required += ("visible_clock",)
            if any(np.array_equal(canonical_out, _predict_info(info, alt)) for alt in required):
                continue
            if clip and not any(not (0 <= r < h and 0 <= c < w)
                                for r, c in info["route"]):
                continue
            return pair
    raise RuntimeError("bounded lasso construction exhausted")


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= value <= high, name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct pairs; all conventions are input-visible and fixed."""
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    pair_periods = ((2, 3), (2, 5), (3, 4))[int(rng.integers(3))]
    colors = list(map(int, rng.choice(10, 4, replace=False)))
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    shift = str(rng.choice(["early", "laps", "scale"]))
    base_flip, base_turn = bool(rng.integers(2)), int(rng.integers(4))
    orientation_shift = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = split == "train" and index in slots
            if lesson:
                rank = slots.index(index)
                periods = list(pair_periods[::1 if rank == 0 else -1])
                payload = colors[:2][::1 if rank == 0 else -1]
            else:
                n = int(rng.choice([1, 2, 3, 4], p=[.24, .44, .24, .08]))
                periods = list(map(int, rng.integers(2, 8, size=n)))
                # Four runs have to fit the same 30-cell canvas.
                while sum(periods) > 16:
                    j = int(np.argmax(periods))
                    periods[j] -= 1
                payload = list(map(int, rng.choice(10, n, replace=False)))
            pair_shift = shift if split == "test" else "ordinary"
            # Test shift always changes a measured budget/geometry range.
            # Orientation is an optional additional recombination, never latent.
            if orientation_shift:
                orientation = (base_flip, (base_turn + (split == "test")) % 4)
            else:
                orientation = bool(rng.integers(2)), int(rng.integers(4))
            clip = (lesson and slots.index(index) == 1) or (not lesson and rng.random() < .35)
            for retry in range(16):
                pair = _construct_pair(rng, periods, payload, lesson, pair_shift, orientation, clip)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


RIVALS = ("dense", "straight", "no_wrap", "infinite", "last_event", "box", "reverse",
          "four_budget", "support_budget", "period_plus", "period_minus", "source_phase",
          "short", "long", "far", "visible_clock", "first_event")


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape must be preserved")
    info = _analyze(inp)
    actual = _transform(out, info["flip"], info["turn"])
    a, bg = info["canonical"], info["background"]
    _require(np.array_equal(actual, _predict_info(info)), "input-only solve mismatch")
    _require(not np.array_equal(inp, out), "identity pair")
    perimeter, length = len(info["route"]), info["length"]
    expected_colors, visible, claims = {}, set(), Counter()
    # Check each contour cell using visit numbers j+kP, independently of
    # emission enumeration. Off-grid visits still use their original times.
    for j, p in enumerate(info["route"]):
        if not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]):
            continue
        visible.add(p)
        winner = next((k for k, (period, color, near) in enumerate(info["runs"])
                       if any((j + lap * perimeter) % period == 0
                              for lap in range(max(0, (length - 1 - j) // perimeter + 1)))), None)
        color = bg if winner is None else info["runs"][winner][1]
        _require(int(actual[p]) == color, "wrong phase, visit budget or collision")
        expected_colors[p] = color
        if winner is not None:
            claims[winner] += 1
    mutable = info["slack"] | visible
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if (r, c) not in mutable:
                _require(actual[r, c] == a[r, c], "preserved cell changed")
    for p in info["slack"] - visible:
        _require(actual[p] == bg, "slack material not erased")
    _require(int(actual[info["anchor"]]) == info["runs"][0][1], "anchor phase/priority")
    _require(any(color == bg for color in expected_colors.values()), "no visible pulse gaps")
    _require(not np.any(actual == info["material_color"]), "material color must be consumed")
    refuted = {alt for alt in RIVALS if not np.array_equal(actual, _predict_info(info, alt))}
    _require({"dense", "straight"} <= refuted, "both inherited mechanisms must affect each pair")
    info.update(refuted=refuted, claims=claims,
                clipped=len(visible) < perimeter,
                irregular=info["obstacle"] != {(r, c)
                    for r in range(min(r for r, c in info["obstacle"]), max(r for r, c in info["obstacle"]) + 1)
                    for c in range(min(c for r, c in info["obstacle"]), max(c for r, c in info["obstacle"]) + 1)})
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists required")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training, test = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            (training if split == "train" else test).append(info)
    _require(set.union(*(i["refuted"] for i in training)) == set(RIVALS), "teaching gap against a documented rival")
    _require(any(not i["clipped"] for i in training) and any(i["clipped"] for i in training),
             "teach intact and boundary-clipped contours")
    _require(any(i["irregular"] and "box" in i["refuted"] for i in training), "teach actual silhouette")
    _require(any(not _connected(i["slack"], DIR4) and "four_budget" in i["refuted"] for i in training),
             "teach eight-connected count")
    precedence = set()
    for i in training:
        if len(i["runs"]) == 2 and len(i["claims"]) == 2:
            precedence.add((i["runs"][0][1], i["runs"][1][1]))
    _require(any((b, a) in precedence for a, b in precedence), "teach a color-priority reversal")
    _require(any(all(_lap_interaction(i["route"], i["length"], i["runs"]))
                 and {"no_wrap", "last_event", "infinite"} <= i["refuted"] for i in training),
             "teach finite cross-lap interference")
    _require(all(len(i["route"]) < i["length"] < 2 * len(i["route"]) for i in training),
             "training must show a partial second lap")
    for i in test:
        height = max(r for r, c in i["obstacle"]) - min(r for r, c in i["obstacle"]) + 1
        width = max(c for r, c in i["obstacle"]) - min(c for r, c in i["obstacle"]) + 1
        _require(i["length"] < len(i["route"]) or i["length"] > 2 * len(i["route"])
                 or (height >= 7 and width >= 7), "missing test shift")


def _hand_checks():
    # One-point obstacle has exactly this eight-cell loop, not a seed-derived answer.
    route = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    _require(_loop({(1, 1)}) == route, "point contour regression")
    # 2 wins over 3 even when the 3-pulse arrives on a later visit (j=4,t=12).
    expected = {route[0]: 2, route[2]: 2, route[4]: 2, route[6]: 2,
                route[3]: 3, route[1]: 3}
    _require(_emissions(route, 14, [(2, 2, 1), (3, 3, 4)]) == expected, "fold/priority regression")
    # Chamfer removes a corner from the island; its clearance loop follows a diagonal.
    body = {(r, c) for r in range(1, 4) for c in range(1, 4)} - {(1, 1)}
    loop = _loop(body)
    _require(loop[0] == (0, 1) and (0, 0) not in loop and (1, 0) in loop,
             "actual-silhouette regression")


def self_test():
    """42 seeds, one puzzle in memory, meaningful geometry/ablation checks.

    Return compact aggregate measurements for optional reports. No filesystem
    writes, puzzle bank, unbounded retry, model prediction or trace logging.
    """
    _hand_checks()
    shapes, silhouettes, perimeters, budgets, run_counts = set(), set(), set(), set(), set()
    orientations, backgrounds, fingerprints, shifts = set(), set(), set(), Counter()
    counters = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _check_pair(pair)
                for role in ("input", "output"):
                    a = _grid(pair[role])
                    digest.update(bytes(a.shape) + a.tobytes())
                shapes.add(pair["input"].shape)
                r0 = min(r for r, c in info["obstacle"])
                c0 = min(c for r, c in info["obstacle"])
                silhouettes.add(tuple(sorted((r - r0, c - c0) for r, c in info["obstacle"])))
                perimeters.add(len(info["route"]))
                budgets.add(info["length"])
                run_counts.add(len(info["runs"]))
                orientations.add((info["flip"], info["turn"]))
                backgrounds.add(info["background"])
                counters["pairs"] += 1
                counters["clipped"] += info["clipped"]
                counters["irregular"] += info["irregular"]
                counters["multi_lap"] += info["length"] > len(info["route"])
                local.append((split, info))
                # Full D4 covariance checks on one sample per puzzle.
                if split == "test":
                    for flip, turn in ((False, 1), (True, 0)):
                        a = _transform(pair["input"], flip, turn)
                        b = _transform(pair["output"], flip, turn)
                        _require(np.array_equal(solve(a), b), "rotation/reflection covariance")
        fingerprints.add(digest.hexdigest())
        train_ratios = [i["length"] / len(i["route"]) for s, i in local if s == "train"]
        test_info = local[-1][1]
        ratio = test_info["length"] / len(test_info["route"])
        shifts["early" if ratio < 1 else "laps" if ratio > 2 else "scale"] += 1
        # This is a geometric/budget shift, not a hidden change of convention.
        _require(all(1 < ratio < 2 for ratio in train_ratios), "training lap range")
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 45 and len(silhouettes) >= 35, "structural variation")
    _require(len(perimeters) >= 15 and len(budgets) >= 25, "perimeter/budget variation")
    _require(run_counts == {1, 2, 3, 4} and len(orientations) == 8, "run/orientation variation")
    _require(len(backgrounds) == 10 and set(shifts) == {"early", "laps", "scale"}, "palette/shift variation")
    # All legal explicit count combinations, plus type and boundary rejections.
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1447, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
            legal_counts += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(7, 2, 1), generate_puzzle(7, 2, 1)
    _require(all(np.array_equal(p[r], q[r]) for s in a for p, q in zip(a[s], b[s])
                 for r in ("input", "output")), "determinism")
    # A changed obstacle/support output cell must be rejected.
    bad = generate_puzzle(19, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, **dict(counters), "shapes": len(shapes),
            "silhouettes": len(silhouettes), "perimeters": sorted(perimeters),
            "length_budgets": [min(budgets), max(budgets)], "source_counts": sorted(run_counts),
            "orientations": len(orientations), "backgrounds": len(backgrounds),
            "shifts": dict(shifts), "legal_count_combinations": legal_counts,
            "semantic_verification": "generator-owned invariants and explicit rival rules"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
