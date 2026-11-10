"""A moving ribbon printer pays for the ink pixels in each cyclic slice.

All roles and all motion parameters are reconstructed from input geometry.
There are no parent imports, stored puzzles, or seed-dependent solver rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "illegal grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "illegal color")
    return a.astype(int, copy=False)


def _key(value):
    a = np.asarray(value, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.add(p)
        groups.append(group)
    return groups


def _frame_candidates(a, background):
    """Recognize complete monochrome rectangular contours, not solid bars."""
    frames = []
    for color in np.unique(a):
        color = int(color)
        if color == background:
            continue
        for cells in _components(a == color):
            ys, xs = zip(*cells)
            t, b, l, r = min(ys), max(ys), min(xs), max(xs)
            if b - t < 2 or r - l < 2:
                continue
            border = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
                      if y in (t, b) or x in (l, r)}
            if cells != border:
                continue
            inside = set(map(int, np.unique(a[t + 1:b, l + 1:r])))
            if color not in inside and background not in inside and len(inside) <= 2:
                frames.append(dict(t=int(t), b=int(b), l=int(l), r=int(r), frame=color))
    return frames


def _geometry(frame, direction):
    t, b, l, r = (frame[k] for k in ("t", "b", "l", "r"))
    dr, dc = DIRECTIONS[direction]
    origin = ((t, l), (t, r), (b, r), (b, l))[direction]
    height, width = (b - t + 1, r - l + 1) if dr else (r - l + 1, b - t + 1)
    return origin, height, width, (dr, dc), (dc, -dr)


def _point(v, row, col):
    y, x = v["origin"]
    dr, dc = v["delta"]
    er, ec = v["across"]
    return y - row * dr + col * er, x - row * dc + col * ec


def _slice(v, row, offset=0, width=None):
    width = v["width"] if width is None else width
    return [_point(v, row, j) for j in range(offset, offset + width)]


def _in_bounds(points, shape):
    return all(0 <= y < shape[0] and 0 <= x < shape[1] for y, x in points)


def _read_printer(a, frame, direction, background):
    origin, height, width, delta, across = _geometry(frame, direction)
    v = dict(frame, origin=origin, height=height, width=width,
             delta=delta, across=across, direction=direction)
    cap = _slice(v, -1)
    _require(_in_bounds(cap, a.shape), "cap outside grid")
    paper = int(a[cap[0]])
    _require(paper not in (background, frame["frame"]) and
             all(int(a[p]) == paper for p in cap), "no leading bar")
    # Full face bars must have exactly the frame's transverse extent.
    for j in (-1, width):
        p = _point(v, -1, j)
        if _in_bounds([p], a.shape):
            _require(int(a[p]) != paper, "overlong leading bar")
    distance, destination = None, None
    for d in range(1, 30):
        points = _slice(v, -1 - d)
        if not _in_bounds(points, a.shape):
            break
        values = [int(a[p]) for p in points]
        if all(c == paper for c in values):
            distance, destination = d, points
            for j in (-1, width):
                p = _point(v, -1 - d, j)
                if _in_bounds([p], a.shape):
                    _require(int(a[p]) != paper, "overlong destination bar")
            break
        _require(all(c == background for c in values), "obstructed corridor")
    _require(distance is not None, "no destination")
    rear = _slice(v, height)
    _require(_in_bounds(rear, a.shape), "no rear word")
    filled = [j for j, p in enumerate(rear) if int(a[p]) != background]
    z = len(filled)
    offset = (width - z) // 2
    _require(1 <= z <= width - 2 and (width - z) % 2 == 0 and
             filled == list(range(offset, offset + z)), "rear word is not centered")
    near_to_far, trail = [], []
    for i in range(height, 30 + height):
        points = _slice(v, i, offset, z)
        if not _in_bounds(points, a.shape):
            break
        values = [int(a[p]) for p in points]
        if all(c == background for c in values):
            break
        _require(all(c not in (background, frame["frame"]) for c in values), "partial rear slice")
        near_to_far.append(values)
        trail.extend(points)
    _require(bool(near_to_far), "empty rear word")
    counter = [_point(v, i, j) for i in range(1, height - 1) for j in range(1, width - 1)]
    alphabet = {paper} | {int(a[p]) for p in counter + trail}
    _require(paper in alphabet and len(alphabet) == 2, "counter/ribbon need ink and paper")
    ink = next(c for c in alphabet if c != paper)
    _require(ink not in (background, frame["frame"]), "conflicting roles")
    values = [int(a[p]) for p in counter]
    spent = values.count(paper)
    _require(values == [paper] * spent + [ink] * (len(values) - spent), "nonprefix counter")
    word = np.array(near_to_far[::-1], dtype=int)
    costs = np.count_nonzero(word == ink, axis=1)
    _require(np.all(costs > 0), "all-paper slices are excluded")
    body = {_point(v, i, j) for i in range(height) for j in range(width)}
    claimed = body | set(cap) | set(destination) | set(trail)
    footprint = {_point(v, i, j) for i in range(-1 - distance, height + len(word))
                 for j in range(width)}
    _require(_in_bounds(footprint, a.shape), "footprint outside grid")
    _require(all(int(a[p]) == background for p in footprint - claimed), "extra material in footprint")
    v.update(paper=paper, ink=ink, distance=distance, cap=cap, destination=destination,
             counter=counter, spent=spent, capacity=len(counter), body=body,
             ribbon_width=z, offset=offset, word=word, costs=costs,
             trail=trail, claimed=claimed, footprint=footprint)
    return v


def _analyze(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    _require(np.all(a[0] == background) and np.all(a[-1] == background) and
             np.all(a[:, 0] == background) and np.all(a[:, -1] == background), "background border required")
    candidates = []
    for f in _frame_candidates(a, background):
        for direction in range(4):
            try:
                v = _read_printer(a, f, direction, background)
                v["collar"] = {(y + dy, x + dx) for y, x in v["footprint"]
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                candidates.append(v)
            except AssertionError:
                pass
    _require(bool(candidates), "no valid printers")
    foreground = set(map(tuple, np.argwhere(a != background)))
    by_pixel = {p: [] for p in foreground}
    for i, v in enumerate(candidates):
        for p in v["claimed"]:
            by_pixel[p].append(i)
    explanations = []

    def cover(claimed, occupied, selected, roles):
        if len(explanations) >= 2:
            return
        uncovered = foreground - claimed
        if not uncovered:
            explanations.append([candidates[i] for i in selected])
            return
        compatible = [i for i, v in enumerate(candidates)
                      if (roles is None or roles == (v["paper"], v["ink"]))
                      and not claimed.intersection(v["claimed"])
                      and not occupied.intersection(v["collar"])]
        compatible = set(compatible)
        p = min(uncovered, key=lambda p: (len(compatible.intersection(by_pixel[p])), p))
        for i in by_pixel[p]:
            if i in compatible:
                v = candidates[i]
                cover(claimed | v["claimed"], occupied | v["footprint"], selected + [i],
                      (v["paper"], v["ink"]))

    # Complete-scene coverage prevents incidental ink contours being mistaken
    # for extra printers. Local candidates do not dictate the global reading.
    cover(set(), set(), [], None)
    _require(len(explanations) == 1, "input needs one complete geometric explanation")
    return a, background, explanations[0]


def _travel(v):
    """Maximum affordable prefix of repeated slices; never skip a slice."""
    costs = np.resize(v["costs"], v["distance"])
    cumulative = np.cumsum(costs)
    budget = v["capacity"] - v["spent"]
    movement = int(np.count_nonzero(cumulative <= budget))
    used = int(cumulative[movement - 1]) if movement else 0
    return movement, used


def _transform(a, background, vehicles, override=None):
    out = a.copy()
    for v in vehicles:
        for p in v["body"] | set(v["cap"]):
            out[p] = background
    for v in vehicles:
        movement, used = _travel(v) if override is None else override(v)
        moved = dict(v)
        dy, dx = v["delta"]
        y, x = v["origin"]
        moved["origin"] = (y + dy * movement, x + dx * movement)
        for i in range(v["height"]):
            for j in range(v["width"]):
                out[_point(moved, i, j)] = v["frame"]
        for index, p in enumerate([_point(moved, i, j) for i in range(1, v["height"] - 1)
                                   for j in range(1, v["width"] - 1)]):
            out[p] = v["paper"] if index < v["spent"] + used else v["ink"]
        for p in _slice(moved, -1):
            out[p] = v["paper"]
        for step in range(movement):
            values = v["word"][step % len(v["word"])]
            for p, color in zip(_slice(v, v["height"] - 1 - step, v["offset"], v["ribbon_width"]), values):
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, background, vehicles = _analyze(input_grid)
    return _transform(a, background, vehicles)


def _primitive_period(word):
    return next(p for p in range(1, len(word) + 1)
                if len(word) % p == 0 and np.array_equal(word, np.resize(word[:p], word.shape)))


def _sample_word(rng, width, period, paper, ink):
    if width == 1:
        return np.full((period, 1), ink, dtype=int)
    word = np.full((period, width), paper, dtype=int)
    style = int(rng.integers(4))
    for row in range(period):
        amount = 1 if style == 0 else width if style == 1 else int(rng.integers(1, width + 1))
        word[row, rng.choice(width, size=amount, replace=False)] = ink
    if period > 1 and _primitive_period(word) != period:
        word[:] = paper
        word[:, 0] = ink
        word[-1, -1] = ink
    return word


def _local_pair(rng, palette, width, period, *, lesson=None, compact=False, active_cycle=False):
    """Construction labels use sampled stencils and an explicit step simulation."""
    bg, frame, paper, ink = palette
    if lesson in ("atomic", "density"):
        width, period = 2, 4
        a = np.array([ink, paper], dtype=int)
        if rng.integers(2):
            a = a[::-1]
        b = np.array([ink, ink], dtype=int)
        c = a[::-1]
        word = np.array([a, b, c, a])
        cross = int(rng.choice((4, 6)))
        depth = int(rng.integers(2, 5))
        fuel, distance, copies = 7 if lesson == "atomic" else 6, int(rng.integers(7, 11)), 1
    elif lesson == "arrival":
        width, period = 3, 1
        word = np.full((1, 3), ink, dtype=int)
        word[0, int(rng.integers(3))] = paper
        distance = int(rng.integers(2, 5))
        fuel = 2 * distance + int(rng.integers(1, 4))
        cross = int(rng.choice((3, 5)))
        depth = int(rng.integers(max(2, (fuel + 1 + cross - 1) // cross), 6))
        copies = int(rng.integers(1, 4))
    else:
        word = _sample_word(rng, width, period, paper, ink)
        copies = 1 if period >= 4 or compact else int(rng.integers(1, 3))
        cross = width + 2 * int(rng.integers(0, 2 if compact else 3))
        depth = int(rng.integers(1, 4 if compact else 7))
        minimum_charge = sum(int(np.count_nonzero(word[j % len(word)] == ink)) for j in range(period + 1))
        if active_cycle:
            minimum_depth = (minimum_charge + cross - 1) // cross
            depth = int(rng.integers(minimum_depth, 7))
        maximum = min(6 if compact else 14, 28 - len(word) * copies - depth - 3)
        minimum_distance = period + 1 if active_cycle else 1
        _require(maximum >= minimum_distance, "local geometry too large")
        distance = int(rng.integers(minimum_distance, maximum + 1))
        capacity = cross * depth
        mode = int(rng.integers(5))
        if mode == 0:
            fuel = 0
        elif mode == 1:
            fuel = capacity
        elif mode == 2:
            total = sum(int(np.count_nonzero(word[i % len(word)] == ink)) for i in range(distance))
            fuel = min(total, capacity)
        else:
            fuel = int(rng.integers(0, capacity + 1))
        if active_cycle:
            total = sum(int(np.count_nonzero(word[j % len(word)] == ink)) for j in range(distance))
            fuel = (capacity if mode == 1 else total if mode == 2 and total <= capacity
                    else int(rng.integers(minimum_charge, capacity + 1)))
    height, outer_width = depth + 2, cross + 2
    visible = np.tile(word, (copies, 1))
    length = len(visible)
    total_height = distance + 1 + height + length
    _require(total_height <= 28 and outer_width <= 28, "printer exceeds workspace")
    inp = np.full((total_height, outer_width), bg, dtype=int)
    top = distance + 1
    inp[0, :] = paper
    inp[top - 1, :] = paper
    inp[top:top + height, :] = frame
    initial = np.array([paper] * (depth * cross - fuel) + [ink] * fuel).reshape(depth, cross)
    inp[top + 1:top + height - 1, 1:-1] = initial
    offset = (outer_width - width) // 2
    inp[top + height:top + height + length, offset:offset + width] = visible[::-1]
    # This simulation advances one full slice at a time, updating real cells.
    output_counter = initial.copy()
    steps, printed = 0, []
    while steps < distance:
        row = word[steps % len(word)]
        charge = int(np.count_nonzero(row == ink))
        pending = np.flatnonzero(output_counter.ravel() == ink)
        if len(pending) < charge:
            break
        output_counter.flat[pending[:charge]] = paper
        printed.append(row.copy())
        steps += 1
    out = inp.copy()
    out[top - 1:top + height, :] = bg
    new_top = top - steps
    out[new_top:new_top + height, :] = frame
    out[new_top + 1:new_top + height - 1, 1:-1] = output_counter
    out[new_top - 1, :] = paper
    for j, row in enumerate(printed):
        out[top + height - 1 - j, offset:offset + width] = row
    return dict(input=inp, output=out)


def _free_positions(occupied, shape):
    h, w = shape
    H, W = occupied.shape
    if h > H or w > W:
        return []
    p = np.pad(occupied, 1)
    blocked = np.zeros_like(occupied)
    for dy in range(3):
        for dx in range(3):
            blocked |= p[dy:dy + H, dx:dx + W]
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    candidates = np.argwhere(sums == 0)
    return candidates[(candidates[:, 0] >= 1) & (candidates[:, 1] >= 1) &
                      (candidates[:, 0] + h < H) & (candidates[:, 1] + w < W)]


def _scene(rng, palette, directions, regime, *, lesson=None, testing=False):
    if lesson:
        width, period = (2, 4) if lesson == "atomic" else (3, 1)
    elif testing:
        width = 3 if regime == "composition" else int(rng.integers(4, 6)) if regime == "wider" else int(rng.integers(2, 4))
        period = 4 if regime == "composition" else 5 if regime == "longer" else int(rng.integers(2, 5))
    else:
        width = int(rng.integers(1, 4))
        period = 1 if regime == "composition" and width == 3 else int(rng.integers(1, 5))
    direction = directions[0] if lesson == "atomic" else directions[1] if lesson == "arrival" else int(rng.choice(directions))
    primary = _local_pair(rng, palette, width, period, lesson=lesson, active_cycle=testing)
    primary = {k: np.rot90(g, -direction).copy() for k, g in primary.items()}
    ph, pw = primary["input"].shape
    H = int(rng.integers(max(ph + 2, 12), 31))
    W = int(rng.integers(max(pw + 2, 12), 31))
    if lesson == "atomic":
        # Two budgets for the same cost schedule teach atomicity and refute
        # average-density division. Space is still packed by actual footprints.
        H, W = max(H, 24), max(W, 24)
    bg, frame, paper, ink = palette
    inp = np.full((H, W), bg, dtype=int)
    out = inp.copy()
    occupied = np.zeros((H, W), dtype=bool)
    requested = int(rng.choice((1, 2, 3, 4), p=(.25, .40, .25, .10)))
    if lesson == "atomic":
        requested = max(2, requested)
    for index in range(requested):
        for attempt in range(48):
            if index == 0:
                patch = primary
            elif lesson == "atomic" and index == 1:
                patch = _local_pair(rng, palette, 2, 4, lesson="density")
                patch = {k: np.rot90(g, -direction).copy() for k, g in patch.items()}
            else:
                z = int(rng.integers(1, 4 if testing else 3))
                p = int(rng.integers(1, 5))
                local_palette = list(palette)
                if rng.random() < .35:
                    local_palette[1] = int(rng.choice([c for c in range(10) if c not in (bg, paper, ink)]))
                patch = _local_pair(rng, local_palette, z, p, compact=True)
                d = int(rng.choice(directions))
                patch = {k: np.rot90(g, -d).copy() for k, g in patch.items()}
            h, w = patch["input"].shape
            positions = _free_positions(occupied, (h, w))
            if len(positions) == 0:
                continue
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            inp[y:y + h, x:x + w] = patch["input"]
            out[y:y + h, x:x + w] = patch["output"]
            occupied[y:y + h, x:x + w] = True
            break
        else:
            if lesson == "atomic" and index == 1:
                raise RuntimeError("required contrast printer did not fit")
            _require(index > 0, "primary cannot be placed")
    # Crop only empty outer margins; retain at least one visible background row.
    ys, xs = np.nonzero(occupied)
    t = int(rng.integers(0, int(ys.min())))
    l = int(rng.integers(0, int(xs.min())))
    b = int(rng.integers(int(ys.max()) + 2, H + 1))
    r = int(rng.integers(int(xs.max()) + 2, W + 1))
    return dict(input=inp[t:b, l:r].copy(), output=out[t:b, l:r].copy())


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("composition", "wider", "longer")))
    first = int(rng.integers(4))
    second = int(rng.choice([d for d in range(4) if d != first]))
    directions = (first, second)
    palette = tuple(map(int, rng.choice(10, size=4, replace=False)))
    shared = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = ("atomic", "arrival")[index] if split == "train" and index < 2 else None
            for attempt in range(64):
                local_palette = palette if shared else tuple(map(int, rng.choice(10, size=4, replace=False)))
                try:
                    pair = _scene(rng, local_palette, directions, regime, lesson=lesson, testing=split == "test")
                except RuntimeError:
                    continue
                try:
                    _analyze(pair["input"])
                except AssertionError:
                    continue
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction could not produce a unique valid input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation(a, bg, vehicles, kind):
    if kind == "unit_cost":
        return _transform(a, bg, vehicles, lambda v: (min(v["distance"], v["capacity"] - v["spent"]),
                                                     min(v["distance"], v["capacity"] - v["spent"])))
    if kind == "no_budget":
        # Only the endpoint/ribbon is needed to refute this shortcut.
        return _transform(a, bg, vehicles, lambda v: (v["distance"], 0))
    if kind == "average_density":
        def average(v):
            fuel = v["capacity"] - v["spent"]
            m = min(v["distance"], fuel * len(v["costs"]) // int(sum(v["costs"])))
            charge = sum(int(v["costs"][j % len(v["costs"])]) for j in range(m))
            return m, min(charge, fuel)
        return _transform(a, bg, vehicles, average)
    altered = []
    for v in vehicles:
        w = dict(v)
        if kind == "reverse_word":
            w["word"] = v["word"][::-1]
        elif kind == "constant_slice":
            w["word"] = v["word"][:1]
        elif kind == "prefix_period":
            # An arbitrary nondividing prefix period is not a cyclic word.
            n = len(v["word"])
            p = next((p for p in range(1, n + 1)
                      if all(np.array_equal(v["word"][i], v["word"][i % p]) for i in range(n))), n)
            w["word"] = v["word"][:p]
        else:
            raise ValueError(kind)
        w["costs"] = np.count_nonzero(w["word"] == w["ink"], axis=1)
        altered.append(w)
    return _transform(a, bg, altered)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnesses, training, testing = set(), set(), [], []
    alternatives = {"unit_cost", "no_budget", "reverse_word", "constant_slice", "prefix_period", "average_density"}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, bg, vs = _analyze(pair["input"])
            b = _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            _require(np.array_equal(_transform(a, bg, vs), b), "wrong printer result")
            changed = set(map(tuple, np.argwhere(a != b)))
            _require(changed <= set().union(*(v["footprint"] for v in vs)), "unrelated cells changed")
            for v in vs:
                movement, used = _travel(v)
                fuel = v["capacity"] - v["spent"]
                _require(0 <= used <= fuel and movement <= v["distance"], "resource or distance exceeded")
                _require(used == sum(int(v["costs"][j % len(v["costs"])]) for j in range(movement)), "ink conservation")
                if movement < v["distance"]:
                    _require(fuel - used < int(v["costs"][movement % len(v["costs"])]), "stopped too early")
                moved = dict(v)
                y, x = v["origin"]
                dy, dx = v["delta"]
                moved["origin"] = (y + movement * dy, x + movement * dx)
                new_counter = [_point(moved, i, j) for i in range(1, v["height"] - 1)
                               for j in range(1, v["width"] - 1)]
                remaining_pixels = sum(int(b[p]) == v["ink"] for p in new_counter)
                new_ribbon = [p for j in range(movement) for p in
                              _slice(v, v["height"] - 1 - j, v["offset"], v["ribbon_width"])]
                printed_pixels = sum(int(b[p]) == v["ink"] for p in new_ribbon)
                _require(fuel - remaining_pixels == printed_pixels == used, "pixel ink conservation")
                for j in range(movement):
                    actual = [int(b[p]) for p in _slice(v, v["height"] - 1 - j, v["offset"], v["ribbon_width"])]
                    _require(actual == list(v["word"][j % len(v["word"])]), "wrong printed phase")
                for p in v["destination"] + v["trail"]:
                    _require(int(b[p]) == int(a[p]), "authoritative seed/destination changed")
                (training if split == "train" else testing).append(v)
            if split == "train":
                for kind in alternatives:
                    if not np.array_equal(_ablation(a, bg, vs, kind), b):
                        witnesses.add(kind)
    _require(witnesses == alternatives, "training misses parent/phase ablation")
    _require(any(0 < v["capacity"] - v["spent"] - _travel(v)[1] <
                 int(v["costs"][_travel(v)[0] % len(v["costs"])]) and _travel(v)[0] < v["distance"]
                 for v in training), "teach atomic stopping with leftover ink")
    _require(any(_travel(v)[0] == v["distance"] and _travel(v)[1] > v["distance"] and
                 v["capacity"] - v["spent"] > _travel(v)[1] for v in training), "teach arrival with surplus ink")
    _require(len({v["direction"] for v in training if _travel(v)[0]}) >= 2, "teach rotated counters")
    _require({v["direction"] for v in testing} <= {v["direction"] for v in training}, "untaught test direction")
    for pair in puzzle["test"]:
        _, _, vs = _analyze(pair["input"])
        _require(any(_primitive_period(v["word"]) > 1 and _travel(v)[0] > _primitive_period(v["word"])
                     for v in vs), "test must print beyond a complete nonconstant cycle")


def _features(puzzle):
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a, bg, vs = _analyze(pair["input"])
            objects = []
            for v in vs:
                m, used = _travel(v)
                fuel = v["capacity"] - v["spent"]
                stop = "arrival" if m == v["distance"] else "exhausted" if fuel == 0 else "budget"
                objects.append(dict(direction=v["direction"], width=v["ribbon_width"],
                                    period=_primitive_period(v["word"]), seed_length=len(v["word"]),
                                    depth=v["height"] - 2, cross=v["width"] - 2,
                                    capacity=v["capacity"], spent=v["spent"], fuel=fuel,
                                    distance=v["distance"], movement=m, used=used,
                                    residual=fuel - used, stop=stop,
                                    costs=tuple(map(int, v["costs"])), frame=v["frame"],
                                    fresh=v["spent"] == 0))
            records.append(dict(split=split, shape=a.shape, count=len(vs), background=bg,
                                paper=vs[0]["paper"], ink=vs[0]["ink"], objects=objects))
    return records


def self_test():
    """Bounded checks, including 42 seeds and pixel-derived structural diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in _features(p)]
    objects = [v for r in records for v in r["objects"]]
    _require(len({tuple(_key(pair["input"]) for split in p.values() for pair in split) for p in puzzles}) == 42,
             "seeds failed to vary puzzles")
    repeat = generate_puzzle(0)
    _require(all(_key(x[k]) == _key(y[k]) for s in repeat for x, y in zip(repeat[s], puzzles[0][s])
                 for k in ("input", "output")), "nondeterministic generation")
    _require(len({r["shape"] for r in records}) >= 45, "insufficient canvas diversity")
    _require({r["count"] for r in records} >= {1, 2, 3}, "insufficient object count diversity")
    _require({v["direction"] for v in objects} == set(range(4)), "missing direction")
    _require({v["width"] for v in objects} >= {1, 2, 3, 4, 5}, "missing stencil widths")
    _require({v["period"] for v in objects} >= {1, 2, 3, 4, 5}, "missing primitive periods")
    _require(len({(v["depth"], v["cross"], v["distance"], v["seed_length"]) for v in objects}) >= 100,
             "geometric diversity collapsed")
    _require(any(v["fresh"] for v in objects) and any(v["fuel"] == 0 for v in objects), "missing uniform counters")
    _require(any(v["movement"] == v["distance"] and v["residual"] == 0 for v in objects), "missing exact arrival")
    _require(any(v["seed_length"] > v["period"] for v in objects), "missing multiple visible copies")
    _require(any(v["depth"] == 1 for v in objects) and any(v["cross"] == 1 for v in objects), "missing thin reservoirs")
    _require(len({(r["background"], r["paper"], r["ink"]) for r in records}) >= 30, "palette role diversity")
    _require(any(len({v["frame"] for v in r["objects"]}) > 1 for r in records), "missing per-printer frames")
    _require(any(len({v["direction"] for v in r["objects"]}) > 1 for r in records), "missing mixed directions")
    shifts = set()
    for p in puzzles:
        rs = _features(p)
        train = [v for r in rs if r["split"] == "train" for v in r["objects"]]
        test = [v for r in rs if r["split"] == "test" for v in r["objects"]]
        if max(v["width"] for v in test) > max(v["width"] for v in train):
            shifts.add("wider")
        if max(v["period"] for v in test) > max(v["period"] for v in train):
            shifts.add("longer")
        if any(v["width"] == 3 and v["period"] == 4 for v in test) and not any(
                v["width"] >= 3 and v["period"] > 1 for v in train):
            shifts.add("composition")
    _require(shifts == {"wider", "longer", "composition"}, "missing train/test shifts")
    # Rigid rotation covariance is required; reflection of the counter order is not.
    for turns in range(4):
        for pair in puzzles[0]["train"] + puzzles[0]["test"]:
            before = np.rot90(pair["input"], turns)
            untouched = before.copy()
            _require(np.array_equal(solve(before), np.rot90(pair["output"], turns)), "rotation covariance")
            _require(np.array_equal(before, untouched), "solve mutated its argument")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1234, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Reject a plausible wrong charge/phase output and a generic pixel mutation.
    damaged = {s: [{k: np.array(g, copy=True) for k, g in pair.items()} for pair in puzzles[0][s]]
               for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    # Descriptive evidence is compact and can be regenerated without hidden metadata.
    report = dict(seeds=42, pairs=len(records), printers=len(objects),
                  shapes=len({r["shape"] for r in records}),
                  counts=sorted({r["count"] for r in records}),
                  widths=sorted({v["width"] for v in objects}),
                  periods=sorted({v["period"] for v in objects}),
                  directions=sorted({v["direction"] for v in objects}),
                  shifts=sorted(shifts), independent_solver=False,
                  checks="input parsing, exact labels, ink conservation, stopping, preservation, ablations, rotations")
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test_report.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
