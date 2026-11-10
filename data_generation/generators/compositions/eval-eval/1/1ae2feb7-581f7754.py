"""Pulse-lock docking: a competing periodic boundary ruler registers rigid objects.

Only standard library and NumPy are used. Generation paints from construction
records; solve reconstructs geometry, pulses, handles and translations from input.
No file I/O, parent imports, saved examples, or mutable semantic state is used.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_TEACH_PERIODS = ((2, 3), (2, 5), (2, 7), (3, 4), (3, 5), (4, 6))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a


def _orient(a, turn, flip):
    return np.rot90(np.fliplr(a) if flip else a, turn).copy()


def _unorient(a, turn, flip):
    b = np.rot90(a, -turn)
    return (np.fliplr(b) if flip else b).copy()


def _components(a, background):
    todo = {tuple(map(int, p)) for p in np.argwhere(a != background)}
    answer = []
    while todo:
        p = min(todo)
        todo.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        answer.append(sorted(cells))
    return answer


def _decode(input_grid):
    grid = _array(input_grid)
    counts = np.bincount(grid.astype(int).ravel(), minlength=10)
    background = int(counts.argmax())
    _require(counts[background] > grid.size / 2, "background must occupy over half the input")
    candidates = []
    for flip in (False, True):
        for turn in range(4):
            a = _orient(grid, turn, flip)
            if a.shape[0] < 3:
                continue
            for col in range(1, a.shape[1] - 1):
                spine = int(a[0, col])
                if spine == background or not np.all(a[:, col] == spine):
                    continue
                if np.count_nonzero(a == spine) != a.shape[0]:
                    continue
                if (np.any(a[1:, :col] != background) or
                        np.any(a[:2, col + 1:] != background) or
                        not np.any(a[0, :col] != background) or
                        not np.any(a[2:, col + 1:] != background)):
                    continue
                runs, cursor = [], col - 1
                while cursor >= 0:
                    color = int(a[0, cursor])
                    if color == background:
                        cursor -= 1
                        continue
                    near, length = col - cursor, 0
                    while cursor >= 0 and a[0, cursor] == color:
                        length += 1
                        cursor -= 1
                    runs.append((length, color, near))
                objects = [[(r + 2, c + col + 1) for r, c in points]
                           for points in _components(a[2:, col + 1:], background)]
                colors = {color for _, color, _ in runs}
                if not all(any(int(a[p]) in colors for p in obj) for obj in objects):
                    continue
                candidates.append({"a": a, "background": background, "spine": spine,
                                   "col": col, "runs": runs, "objects": objects,
                                   "turn": turn, "flip": flip,
                                   "distance": a.shape[1] - col - 1})
    _require(len(candidates) == 1, "need one unambiguous spine/ruler interpretation")
    return candidates[0]


def _ruler(info, priority="near", period_delta=0, phase=0):
    # Cell-wise arithmetic; construction instead uses far-to-near slice painting.
    result = np.full(info["distance"], info["background"], dtype=np.int8)
    runs = info["runs"]
    for t in range(len(result)):
        firing = [run for run in runs if (t - phase) % max(1, run[0] + period_delta) == 0]
        if not firing:
            continue
        if priority == "far":
            winner = firing[-1]
        elif priority == "short":
            winner = min(firing, key=lambda run: (run[0], run[2]))
        elif priority == "long":
            winner = min(firing, key=lambda run: (-run[0], run[2]))
        else:
            winner = firing[0]
        result[t] = winner[1]
    return result


def _object_record(info, points, rail, only_first=False, lock_choice="first"):
    a, col = info["a"], info["col"]
    ruler_colors = {color for _, color, _ in info["runs"]}
    left = min(c for _, c in points)
    handles = sorted((c - left, r, int(a[r, c])) for r, c in points
                     if int(a[r, c]) in ruler_colors)
    _require(handles, "object has no registration marks")
    used = handles[:1] if only_first else handles
    locks = [t for t in range(len(rail))
             if all(t + c < len(rail) and rail[t + c] == color for c, _, color in used)]
    _require(locks, "object has no simultaneous registration position")
    if lock_choice == "last":
        target = locks[-1]
    elif lock_choice == "nearest":
        source = left - col - 1
        target = min(locks, key=lambda t: (abs(t - source), t))
    else:
        target = locks[0]
    delta = col + 1 + target - left
    return {"points": points, "handles": handles, "target": target, "locks": locks,
            "delta": delta, "left": left,
            "clipped": sum(c + delta >= a.shape[1] for _, c in points)}


def _prediction(input_grid, *, first_handle=False, keep_sources=False, marker_only=False,
                priority="near", period_delta=0, phase=0, lock_choice="first"):
    info = _decode(input_grid)
    a, col, bg = info["a"], info["col"], info["background"]
    rail = _ruler(info, priority, period_delta, phase)
    out = a.copy()
    out[0, col + 1:] = rail
    if not keep_sources:
        out[1:, col + 1:] = bg
    occupied = set()
    for points in info["objects"]:
        obj = _object_record(info, points, rail, only_first=first_handle, lock_choice=lock_choice)
        marks = {(r, obj["left"] + c) for c, r, _ in obj["handles"]}
        for r, c in points:
            tc = c + obj["delta"] if not marker_only or (r, c) in marks else c
            if tc >= a.shape[1]:
                continue
            _require(tc > col and (r, tc) not in occupied, "contradictory target footprints")
            occupied.add((r, tc))
            out[r, tc] = a[r, c]
    return _unorient(out, info["turn"], info["flip"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the boundary ruler and dock each object at its first full lock."""
    return _prediction(input_grid)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _signature(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def _handle_options(rail, background, count, delayed, clipping, alternate=None):
    """Enumerate small offset/color motifs, not puzzle or object templates."""
    d, options, seen = len(rail), [], set()
    multi_color = len(set(map(int, rail)) - {background}) > 1
    for rest in itertools.combinations(range(1, min(8, d)), count - 1):
        base = (0,) + rest
        for padding in range(3):
            cols = tuple(c + padding for c in base)
            room = d - max(cols)
            if room <= 0:
                continue
            for at in range(room):
                colors = tuple(int(rail[at + c]) for c in cols)
                key = cols, colors
                if background in colors or key in seen:
                    continue
                seen.add(key)
                if count > 1 and multi_color and len(set(colors)) < 2:
                    continue
                matches = np.ones(room, dtype=bool)
                prefix_target = None
                for index, (c, color) in enumerate(zip(cols, colors)):
                    matches &= rail[c:c + room] == color
                    if index == count - 2:
                        prefix_target = int(np.flatnonzero(matches)[0])
                target = int(np.flatnonzero(matches)[0])
                first = next(t for t in range(d - cols[0]) if rail[t + cols[0]] == colors[0])
                if delayed and not target > first:
                    continue
                if clipping and target < 2:
                    continue
                if count == 3 and delayed and target <= prefix_target:
                    continue  # the new third handle must actually postpone docking
                if alternate is not None:
                    other = np.ones(room, dtype=bool)
                    for c, color in zip(cols, colors):
                        other &= alternate[c:c + room] == color
                    positions = np.flatnonzero(other)
                    if len(positions) and int(positions[0]) == target:
                        continue
                options.append((cols, colors, target))
    return options


def _shape(rng, width, height, handle_cols, body_colors, handle_colors, singleton=False):
    if singleton:
        return [(0, 0, handle_colors[0])]
    kind = int(rng.integers(5))
    if kind == 0:
        points = {(r, c) for r in range(height) for c in range(width)}
    elif kind == 1 and height >= 3 and width >= 3:
        points = {(r, c) for r in range(height) for c in range(width)
                  if r in (0, height - 1) or c in (0, width - 1)}
    else:
        spine_row, branch_col = int(rng.integers(height)), int(rng.integers(width))
        points = {(spine_row, c) for c in range(width)} | {(r, branch_col) for r in range(height)}
        # A finite frontier grows connected, often indented/branched carriers.
        budget = int(rng.integers(len(points), max(len(points) + 1, height * width + 1)))
        for _ in range(budget - len(points)):
            frontier = sorted({(r + dr, c + dc) for r, c in points for dr, dc in _DIRS
                               if 0 <= r + dr < height and 0 <= c + dc < width} - points)
            if not frontier:
                break
            points.add(frontier[int(rng.integers(len(frontier)))])
    handles = []
    for col, color in zip(handle_cols, handle_colors):
        rows = [r for r in range(height) if (r, col) in points]
        row = int(rng.choice(rows))
        handles.append((row, col, color))
    body = int(rng.choice(body_colors))
    colors = {p: body for p in points}
    for r, c, color in handles:
        colors[r, c] = color
    others = sorted(points - {(r, c) for r, c, _ in handles})
    if others and rng.random() < .45:
        for index in rng.choice(len(others), min(len(others), int(rng.integers(1, 4))), replace=False):
            colors[others[int(index)]] = int(rng.choice(body_colors))
    return [(r, c, colors[r, c]) for r, c in sorted(points)]


def _run_plan(rng, witness, parameters, test):
    if witness is not None:
        a, b = parameters["periods"]
        colors = parameters["witness_colors"]
        lengths = [a, b] if witness == 0 else [b, a]
        colors = list(colors if witness == 0 else colors[::-1])
        minimum = int(np.lcm(a, b)) + 1
    else:
        count = int(rng.choice([1, 2, 3, 4], p=[.16, .43, .29, .12]))
        if test:
            count = max(2, count)
        lengths = list(map(int, rng.integers(1, 8, size=count)))
        # Numerical packing cap; broad mixtures remain possible within 30 cells.
        while sum(lengths) > 13:
            i = int(np.argmax(lengths))
            lengths[i] -= 1
        colors = None
        minimum = max(10, max(lengths) + 2)
    gaps = list(map(int, rng.integers(0, 3, size=len(lengths))))
    if witness is not None:
        gaps[0] = int(rng.integers(1, 4))  # phase must ignore a visible source gap
    minimum_source = sum(lengths) + sum(gaps)
    upper = min(23 if test else 19, 29 - minimum_source)
    if upper < minimum:
        gaps = [1 if witness is not None and i == 0 else 0 for i in range(len(lengths))]
        minimum_source = sum(lengths) + sum(gaps)
        upper = min(23 if test else 19, 29 - minimum_source)
    _require(upper >= minimum, "run plan does not fit")
    d = int(rng.integers(minimum, upper + 1))
    instruction = minimum_source + int(rng.integers(0, min(3, 29 - minimum_source - d) + 1))
    return lengths, colors, gaps, instruction, d


def _scene(rng, parameters, witness, test, transform):
    lengths, run_colors, gaps, instruction, d = _run_plan(rng, witness, parameters, test)
    palette = list(parameters["palette"] if parameters["stable_palette"] else map(int, rng.permutation(10)))
    if run_colors is not None:
        rest = [c for c in palette if c not in run_colors]
        bg, spine = rest[:2]
    else:
        bg, spine = palette[:2]
        available = palette[2:]
        run_colors = available[:len(lengths)]
        # Separate repeated colors are legal; keep an actual separating gap.
        if len(lengths) >= 3 and rng.random() < .3:
            run_colors[-1] = run_colors[0]
    body_colors = [c for c in palette if c not in {bg, spine, *run_colors}]
    for i in range(1, len(lengths)):
        if run_colors[i] == run_colors[i - 1] and gaps[i] == 0:
            return None
    rail = np.full(d, bg, np.int8)
    for length, color in reversed(list(zip(lengths, run_colors))):
        rail[::length] = color
    far_rail = np.full(d, bg, np.int8)
    for length, color in zip(lengths, run_colors):
        far_rail[::length] = color
    if test and len(set(map(int, rail)) - {bg}) < 2:
        return None
    object_count = int(rng.choice([1, 2, 3, 4], p=[.20, .39, .30, .11]))
    if witness == 0:
        object_count = max(2, object_count)
    requested_clip = (witness != 0 and parameters["clipping"] and
                      rng.random() < (.6 if test else .22))
    records = []
    cache = {}
    for i in range(object_count):
        tie_witness = witness == 0 and i == 1
        k = 3 if test and i == 0 else 2 if witness is not None and i == 0 else int(rng.integers(1, 3))
        if tie_witness:
            k = 1
        delayed = i == 0 and (witness is not None or test)
        clipping = bool(requested_clip and i == 0)
        key = k, delayed, clipping
        if key not in cache:
            alternate = far_rail if witness == 1 and i == 0 else None
            cache[key] = _handle_options(rail, bg, *key, alternate)
        options = cache[key]
        if not options:
            return None
        if tie_witness:
            cols, colors, target = (0,), (run_colors[0],), 0
        else:
            cols, colors, target = options[int(rng.integers(len(options)))]
        singleton = k == 1 and cols == (0,) and not clipping and rng.random() < .16
        height = 1 if singleton else int(rng.integers(2, 6))
        if clipping:
            width = d - target + int(rng.integers(1, min(target, 3) + 1))
        else:
            width = int(rng.integers(max(cols) + 1, min(d - target, max(cols) + 4) + 1))
            if not singleton:
                width = max(2, width)
        if singleton:
            width = 1
        elif tie_witness:
            width = int(rng.integers(2, min(4, d - lengths[0]) + 1))
        if width > d or not clipping and target + width > d:
            return None
        shape = _shape(rng, width, height, cols, body_colors, colors, singleton)
        origins = list(range(d - width + 1))
        moving = [x for x in origins if x != target]
        if moving and (i == 0 or rng.random() < .85):
            origins = moving
        source = int(rng.choice(origins))
        if tie_witness:
            source = max(t for t in range(d - width + 1) if rail[t] == colors[0])
            _require(source > 0, "tie witness needs a later accepted starting lock")
        records.append({"shape": shape, "height": height, "source": source,
                        "target": target, "handles": k, "clip": clipping})
    row_gaps = list(map(int, rng.integers(1, 4, size=max(0, object_count - 1))))
    height = 2 + sum(o["height"] for o in records) + sum(row_gaps) + int(rng.integers(0, 4))
    if height > 30:
        return None
    inp = np.full((height, instruction + 1 + d), bg, np.int8)
    inp[:, instruction] = spine
    cursor = instruction
    for length, color, gap in zip(lengths, run_colors, gaps):
        cursor -= gap
        inp[0, cursor - length:cursor] = color
        cursor -= length
    out = inp.copy()
    out[0, instruction + 1:] = rail
    row = 2
    for i, record in enumerate(records):
        for r, c, color in record["shape"]:
            inp[row + r, instruction + 1 + record["source"] + c] = color
            tc = instruction + 1 + record["target"] + c
            if tc < out.shape[1]:
                out[row + r, tc] = color
        row += record["height"] + (row_gaps[i] if i < len(row_gaps) else 0)
    if np.mean(inp == bg) <= .5:
        return None
    turn, flip = transform
    pair = {"input": _orient(inp, turn, flip), "output": _orient(out, turn, flip)}
    # Construction metadata is discarded; semantic checks use only returned grids.
    try:
        _pair_analysis(pair)
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    parameters = {"palette": palette, "witness_colors": palette[2:4],
                  "stable_palette": bool(rng.random() < .35),
                  "periods": _TEACH_PERIODS[int(rng.integers(len(_TEACH_PERIODS)))],
                  "clipping": bool(rng.random() < .65)}
    lesson_order = list(map(int, rng.permutation(2)))
    base_turn, base_flip = int(rng.integers(4)), bool(rng.integers(2))
    orientation_mode = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = lesson_order[index] if split == "train" and index < 2 else None
            if orientation_mode == 0:
                transform = base_turn, base_flip
            elif orientation_mode == 1:
                transform = (base_turn + (split == "test")) % 4, base_flip
            else:
                transform = int(rng.integers(4)), bool(rng.integers(2))
            for _ in range(100):
                pair = _scene(rng, parameters, witness, split == "test", transform)
                if pair is not None and _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_analysis(pair):
    inp, out = _array(pair["input"]), _array(pair["output"])
    _require(inp.shape == out.shape, "dimensions must be preserved")
    info = _decode(inp)
    a, col, bg = info["a"], info["col"], info["background"]
    actual = _orient(out, info["turn"], info["flip"])
    rail = _ruler(info)
    _require(np.array_equal(actual[:, :col + 1], a[:, :col + 1]), "source ruler/spine changed")
    _require(np.array_equal(actual[0, col + 1:], rail), "wrong pulse arithmetic/priority")
    expected = np.full_like(a, bg)
    expected[:, :col + 1] = a[:, :col + 1]
    expected[0, col + 1:] = rail
    occupied, records = set(), []
    for points in info["objects"]:
        record = _object_record(info, points, rail)
        # Check the whole colored footprint, erasure, only orthogonal movement,
        # all-handle lock, and the absence of an earlier acceptable position.
        for r, c in points:
            tc = c + record["delta"]
            if tc < a.shape[1]:
                _require(tc > col and (r, tc) not in occupied, "overlapping destinations")
                occupied.add((r, tc))
                expected[r, tc] = a[r, c]
        first = _object_record(info, points, rail, only_first=True)["target"]
        record["delayed"] = record["target"] > first
        source_start = record["left"] - col - 1
        record["later_source"] = source_start in record["locks"] and source_start > record["target"]
        if len(record["handles"]) >= 3:
            prefix = record["handles"][:-1]
            first_prefix = next(t for t in range(len(rail))
                                if all(t + c < len(rail) and rail[t + c] == color
                                       for c, _, color in prefix))
            record["last_needed"] = record["target"] > first_prefix
        else:
            record["last_needed"] = False
        try:
            other_target = _object_record(info, points, _ruler(info, "far"))["target"]
            record["priority_sensitive"] = other_target != record["target"]
        except AssertionError:
            record["priority_sensitive"] = True
        records.append(record)
    _require(np.array_equal(actual, expected), "rigidity, erasure, first lock, or clipping")
    _require(np.array_equal(solve(inp), out), "public solver mismatch")
    return info, records


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, teaching, train_handles, test_handles = set(), [], set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            key = _signature(_array(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info, records = _pair_analysis(pair)
            handle_set = train_handles if split == "train" else test_handles
            handle_set.update(len(o["handles"]) for o in records)
            if split == "train":
                teaching.append((info, records))
            else:
                _require(any(o["last_needed"] for o in records), "third handle must add a test constraint")
    near_short, near_long, color_orders, delayed, full_move, gap = False, False, set(), False, False, False
    for info, records in teaching:
        runs, rail = info["runs"], _ruler(info)
        gap |= runs[0][2] > 1
        for i, near in enumerate(runs):
            for far in runs[i + 1:]:
                # Recurrence, a post-origin collision, and a visible far pulse.
                collision = any(t > 0 and t % near[0] == t % far[0] == 0 and rail[t] == near[1]
                                for t in range(len(rail)))
                far_visible = any(t > 0 and t % far[0] == 0 and rail[t] == far[1]
                                  for t in range(len(rail)))
                if near[1] != far[1] and collision and far_visible:
                    near_short |= near[0] < far[0]
                    near_long |= near[0] > far[0]
                    color_orders.add((near[1], far[1]))
        delayed |= any(o["delayed"] and o["delta"] != 0 for o in records)
        full_move |= any(o["delta"] != 0 and not o["clipped"] and
                         len(o["points"]) > len(o["handles"]) for o in records)
    _require(near_short and near_long and any((b, a) in color_orders for a, b in color_orders),
             "training must distinguish proximity from period/color priority")
    _require(gap and delayed and full_move, "missing phase / simultaneous lock / rigid payload lesson")
    _require(any(o["priority_sensitive"] for _, records in teaching for o in records),
             "pulse competition must affect an object's destination, not only the ruler")
    _require(any(o["later_source"] for _, records in teaching for o in records),
             "training must distinguish the first lock from the closest or last lock")
    _require(max(train_handles) == 2 and 3 in test_handles, "expected two-to-three-handle shift")
    # These explicit rivals must fail on at least one complete training output.
    for kwargs in ({"first_handle": True}, {"keep_sources": True}, {"marker_only": True},
                   {"priority": "short"}, {"priority": "long"}, {"priority": "far"},
                   {"period_delta": 1}, {"period_delta": -1}, {"phase": 1},
                   {"lock_choice": "nearest"}, {"lock_choice": "last"}):
        distinguish = False
        for pair in puzzle["train"]:
            try:
                rival = _prediction(pair["input"], **kwargs)
                distinguish |= not np.array_equal(rival, pair["output"])
            except AssertionError:
                distinguish = True  # no lock/contradictory footprints also refutes it
        _require(distinguish, "training admits an implemented shortcut: " + str(kwargs))


def _manual_regression():
    # Hand-computed: near A=3, far B=2 gives A0,B2,A3,B4,A6,B8,A9,B10.
    # The two horizontal handles A,B at offsets 0,1 first lock at t=3.
    inp = np.zeros((7, 18), np.int8)
    inp[:, 5] = 9
    inp[0, :5] = [2, 2, 1, 1, 1]
    inp[3, 13:15] = [1, 2]
    inp[4, 13:16] = [4, 4, 4]
    out = np.zeros_like(inp)
    out[:, :6] = inp[:, :6]
    out[0, 6:] = [1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0]
    out[3, 9:11] = [1, 2]
    out[4, 9:12] = [4, 4, 4]
    for turn in range(4):
        for flip in (False, True):
            assert np.array_equal(solve(_orient(inp, turn, flip)), _orient(out, turn, flip))
    # Triple A0,A3,B5: the first two allow t=0, but all three first lock at t=3.
    tri = inp.copy()
    tri[3:5, 6:] = 0
    tri[3, 12:18] = [1, 4, 4, 1, 4, 2]
    tri[4, 12:18] = 4
    expected = out.copy()
    expected[3:5, 6:] = 0
    expected[3, 9:15] = [1, 4, 4, 1, 4, 2]
    expected[4, 9:15] = 4
    # Twelve-cell payload extends beyond the outer edge; both handles survive.
    clip = inp.copy()
    clip[3:5, 6:] = 0
    clip[3, 6:8] = [1, 2]
    clip[4, 6:] = 4
    clipped = out.copy()
    clipped[3:5, 6:] = 0
    clipped[3, 9:11] = [1, 2]
    clipped[4, 9:] = 4
    for source, target in ((tri, expected), (clip, clipped)):
        for turn in range(4):
            for flip in (False, True):
                assert np.array_equal(solve(_orient(source, turn, flip)), _orient(target, turn, flip))
    return inp, out


def self_test():
    """Bounded streaming suite; returns compact aggregate evidence, writes nothing."""
    _manual_regression()
    fingerprints, shapes, structures, periods, object_counts = set(), set(), set(), set(), set()
    directions, backgrounds, sizes, handle_counts, locks = set(), set(), set(), set(), set()
    clipped = delayed = multi_lock = moved = unchanged = third_needed = coupled = later_sources = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, records = _pair_analysis(pair)
                shapes.add(pair["input"].shape)
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
                periods.update(r[0] for r in info["runs"])
                object_counts.add(len(records))
                directions.add(info["turn"])
                backgrounds.add(info["background"])
                for o in records:
                    sizes.add(len(o["points"]))
                    handle_counts.add(len(o["handles"]))
                    locks.add(o["target"])
                    clipped += bool(o["clipped"])
                    delayed += o["delayed"]
                    multi_lock += len(o["locks"]) > 1
                    moved += o["delta"] != 0
                    unchanged += o["delta"] == 0
                    third_needed += o["last_needed"]
                    coupled += o["priority_sensitive"]
                    later_sources += o["later_source"]
    assert len(fingerprints) == 42 and len(structures) >= 200
    assert len(shapes) >= 55 and object_counts == {1, 2, 3, 4}
    assert directions == {0, 1, 2, 3} and len(backgrounds) >= 8
    assert {1, 2, 3} <= handle_counts and set(range(1, 8)) <= periods
    assert clipped >= 5 and delayed >= 84 and multi_lock >= 20 and moved >= 150
    assert third_needed >= 42 and coupled >= 84
    assert later_sources >= 42
    assert len(sizes) >= 15 and len(locks) >= 7
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4200 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0)
    for turn in range(4):
        q = {s: [{role: _orient(grid, turn, True) for role, grid in pair.items()}
                 for pair in p[s]] for s in p}
        validate_generated_puzzle(q)
    permutation = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], np.int8)
    validate_generated_puzzle({s: [{r: permutation[a] for r, a in pair.items()} for pair in p[s]] for s in p})
    bad = {s: [{r: a.copy() for r, a in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "normalized_inputs": len(structures),
            "periods": sorted(periods), "object_counts": sorted(object_counts),
            "handle_counts": sorted(handle_counts), "object_sizes": len(sizes),
            "clipped_objects": clipped, "delayed_locks": delayed,
            "multiple_candidate_locks": multi_lock, "moving_objects": moved,
            "already_docked_objects": unchanged,
            "effective_third_handles": third_needed, "priority_sensitive_objects": coupled,
            "sources_on_later_locks": later_sources,
            "scope": "generator-owned checks; no independent or blind solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
