"""Chamber-fuel crawlers: region counters coupled to translation.

Only Python/NumPy are needed. No files, parent generators, caches, training
examples or random state are consulted by solve(). No execution traces are saved.
"""
from __future__ import annotations

import numpy as np


_TURNS = {"U": 0, "D": 2, "L": 3, "R": 1}
_DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _need(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _need(np.all((a >= 0) & (a <= 9)), "color bounds")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    """Four-connected components, in the order of their first raster cell."""
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack, cells = [(row, col)], []
        seen[row, col] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _rooms(walls, direction):
    canonical = np.rot90(walls, _TURNS[direction])
    result = []
    for cells in _components(~canonical):
        ys, xs = zip(*cells)
        _need(min(ys) > 0 and min(xs) > 0 and max(ys) < canonical.shape[0] - 1
              and max(xs) < canonical.shape[1] - 1, "open outer frame")
        _need(len(cells) == (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1),
              "chambers must be rectangular")
        mask = np.zeros(canonical.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        result.append(np.rot90(mask, -_TURNS[direction]))
    _need(bool(result), "no enclosed chambers")
    # Every wall pixel belongs to the border of an elementary chamber.
    borders = np.zeros(walls.shape, dtype=bool)
    for room in result:
        a = np.argwhere(room)
        t, l = map(int, a.min(axis=0))
        b, r = map(int, a.max(axis=0))
        borders[t - 1, l - 1:r + 2] = True
        borders[b + 1, l - 1:r + 2] = True
        borders[t - 1:b + 2, l - 1] = True
        borders[t - 1:b + 2, r + 1] = True
    _need(np.array_equal(borders, walls), "stray wall or incomplete partition")
    return result


def _bar(box, direction, distance, shape):
    t, l, b, r = box
    if direction == "U":
        cells = [(t - 1 - distance, x) for x in range(l, r + 1)]
    elif direction == "D":
        cells = [(b + 1 + distance, x) for x in range(l, r + 1)]
    elif direction == "L":
        cells = [(y, l - 1 - distance) for y in range(t, b + 1)]
    else:
        cells = [(y, r + 1 + distance) for y in range(t, b + 1)]
    if any(not (0 <= y < shape[0] and 0 <= x < shape[1]) for y, x in cells):
        return None
    return cells


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _is_bar(grid, cells, color):
    if cells is None or any(int(grid[p]) != color for p in cells):
        return False
    y0, x0 = cells[0]
    y1, x1 = cells[-1]
    endpoints = ((y0, x0 - 1), (y1, x1 + 1)) if y0 == y1 else ((y0 - 1, x0), (y1 + 1, x1))
    return all(not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1])
               or grid[y, x] != color for y, x in endpoints)


def _read_with_marker(grid, marker):
    """Explain every non-background pixel under one candidate bar/state color."""
    colors = set(map(int, np.unique(grid))) - {0, 9, marker}
    _need(bool(colors), "no wall colors")
    cover = np.zeros(grid.shape, dtype=np.int8)
    swept = np.zeros(grid.shape, dtype=np.int8)
    vehicles = []
    for color in sorted(colors):
        points = np.argwhere(grid == color)
        t, l = map(int, points.min(axis=0))
        b, r = map(int, points.max(axis=0))
        _need(b - t >= 2 and r - l >= 2, "wall box too small")
        block = grid[t:b + 1, l:r + 1]
        visible = block == color
        _need(visible[0, 0] and visible[0, -1] and visible[-1, 0] and visible[-1, -1],
              "corners must be intact")
        walls = visible.copy()
        defects = block == 9
        for yy, xx in np.argwhere(defects):
            y, x = int(yy), int(xx)
            horizontal = 0 < x < block.shape[1] - 1 and visible[y, x - 1] and visible[y, x + 1]
            vertical = 0 < y < block.shape[0] - 1 and visible[y - 1, x] and visible[y + 1, x]
            _need(horizontal or vertical, "unsupported wall defect")
            walls[y, x] = True
        _need(walls[0].all() and walls[-1].all() and walls[:, 0].all() and walls[:, -1].all(),
              "outer border is incomplete")
        _need(not np.any(np.isin(block, list(colors - {color}))), "overlapping bodies")
        directions = [d for d in _TURNS if _is_bar(grid, _bar((t, l, b, r), d, 0, grid.shape), marker)]
        _need(len(directions) == 1, "ambiguous nose")
        direction = directions[0]
        nose = _bar((t, l, b, r), direction, 0, grid.shape)
        target = None
        for distance in range(1, 31):
            cells = _bar((t, l, b, r), direction, distance, grid.shape)
            if cells is None:
                break
            if _is_bar(grid, cells, marker):
                target = cells
                break
            _need(all(grid[p] == 0 for p in cells), "obstructed corridor or malformed dock")
        _need(target is not None, "missing aligned dock")
        rooms = _rooms(walls, direction)
        states = []
        for room in rooms:
            values = np.unique(block[room])
            _need(len(values) == 1 and int(values[0]) in (0, marker), "mixed or invalid chamber state")
            states.append(int(values[0]) == marker)
        spent = sum(states)
        _need(states == [True] * spent + [False] * (len(states) - spent), "chamber counter is not a prefix")
        cover[t:b + 1, l:r + 1] += 1
        for p in nose + target:
            cover[p] += 1
        ys = [t, b] + [p[0] for p in target]
        xs = [l, r] + [p[1] for p in target]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        swept[y0:y1 + 1, x0:x1 + 1] += 1
        vehicles.append(dict(color=color, marker=marker, box=(t, l, b, r), walls=walls,
                             rooms=rooms, spent=spent, distance=distance, direction=direction,
                             nose=nose, target=target, defects=int(defects.sum())))
    _need(np.max(cover) <= 1 and np.max(swept) <= 1, "shared structures or swept corridors")
    _need(np.all((grid == 0) | (cover == 1)), "unexplained foreground")
    return vehicles


def _decode(grid):
    explanations = []
    for marker in sorted(set(map(int, np.unique(grid))) - {0, 9}):
        try:
            explanations.append(_read_with_marker(grid, marker))
        except AssertionError:
            pass
    _need(len(explanations) == 1, "input needs one complete geometric explanation")
    return explanations[0]


def _render_solution(grid, vehicles, *, area_fuel=False, ignore_spent=False, ignore_fuel=False):
    output = grid.copy()
    for v in vehicles:
        t, l, b, r = v["box"]
        output[t:b + 1, l:r + 1] = 0
        _paint(output, v["nose"], 0)
    for v in vehicles:
        rooms = v["rooms"]
        start = 0 if ignore_spent else v["spent"]
        fuel = sum(int(a.sum()) for a in rooms[start:]) if area_fuel else len(rooms) - start
        movement = v["distance"] if ignore_fuel else min(fuel, v["distance"])
        spent = min(len(rooms), start + movement)
        body = np.zeros(v["walls"].shape, dtype=np.int8)
        body[v["walls"]] = v["color"]
        for room in rooms[:spent]:
            body[room] = v["marker"]
        dy, dx = _DELTAS[v["direction"]]
        t, l, b, r = v["box"]
        t, l = t + dy * movement, l + dx * movement
        output[t:t + body.shape[0], l:l + body.shape[1]] = body
        nose = _bar((t, l, t + body.shape[0] - 1, l + body.shape[1] - 1),
                    v["direction"], 0, output.shape)
        _paint(output, nose, v["marker"])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover chambers, consume one per step, translate; input only."""
    grid = _grid(input_grid)
    return _render_solution(grid, _decode(grid))


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 2) * ((r - l) // 2)


def _partition(rng, h, w, count, style):
    """Construct actual elementary rectangles, never a product-of-lines answer."""
    boxes = [(0, 0, h - 1, w - 1)]
    if _capacity(boxes[0]) < count:
        return None
    if style in ("strip", "grid"):
        factors = [(nr, count // nr) for nr in range(1, count + 1)
                   if count % nr == 0 and 2 * nr < h and 2 * (count // nr) < w
                   and (style == "grid" or nr == 1 or count // nr == 1)]
        if factors:
            nr, nc = factors[int(rng.integers(len(factors)))]
            def offsets(parts, span):
                gaps = np.full(parts, 2)
                for _ in range(span - 2 * parts):
                    gaps[int(rng.integers(parts))] += 1
                return [0] + list(map(int, np.cumsum(gaps)))
            ys, xs = offsets(nr, h - 1), offsets(nc, w - 1)
            boxes = [(ys[i], xs[j], ys[i + 1], xs[j + 1]) for i in range(nr) for j in range(nc)]
        else:
            style = "mixed"
    if style in ("partial", "mixed"):
        first_axis = None
        for step in range(count - 1):
            options = []
            for index, (t, l, b, r) in enumerate(boxes):
                for axis, low, high in ((0, t, b), (1, l, r)):
                    if style == "partial" and step == 1 and axis == first_axis:
                        continue
                    for cut in range(low + 2, high - 1):
                        halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                        capacity = sum(_capacity(a) for j, a in enumerate(boxes) if j != index)
                        if capacity + sum(_capacity(a) for a in halves) >= count:
                            options.append((index, axis, halves))
            if not options:
                return None
            index, axis, halves = options[int(rng.integers(len(options)))]
            if step == 0:
                first_axis = axis
            boxes[index:index + 1] = halves
    walls = np.zeros((h, w), dtype=bool)
    for t, l, b, r in boxes:
        walls[t, l:r + 1] = walls[b, l:r + 1] = True
        walls[t:b + 1, l] = walls[t:b + 1, r] = True
    ys = {a for t, _, b, _ in boxes for a in (t, b)}
    xs = {a for _, l, _, r in boxes for a in (l, r)}
    partial = count != (len(ys) - 1) * (len(xs) - 1)
    if style == "partial" and count >= 3 and not partial:
        return None
    # Construction order is independently specified by the boxes' interiors.
    boxes.sort(key=lambda a: (a[0] + 1, a[1] + 1))
    return walls, boxes


def _body(walls, boxes, spent, color, marker):
    body = np.zeros(walls.shape, dtype=np.int8)
    body[walls] = color
    for t, l, b, r in boxes[:spent]:
        body[t + 1:b, l + 1:r] = marker
    return body


def _damage(rng, body, walls, color, required):
    observed = body.copy()
    candidates = []
    h, w = walls.shape
    for y, x in np.argwhere(walls):
        y, x = int(y), int(x)
        if (0 < x < w - 1 and walls[y, x - 1] and walls[y, x + 1]) or (0 < y < h - 1 and walls[y - 1, x] and walls[y + 1, x]):
            candidates.append((y, x))
    rng.shuffle(candidates)
    amount = int(rng.integers(1, 4)) if required or rng.random() < .65 else 0
    for y, x in candidates:
        if amount == 0:
            break
        # All defects retain immediate, undamaged opposite supports.
        nearby = observed[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        if np.any(nearby == 9):
            continue
        observed[y, x] = 9
        amount -= 1
    return observed


def _object(rng, color, marker, count, direction, lesson=None, small=False, active=False):
    style = "partial" if lesson == "destination" else str(rng.choice(("strip", "grid", "partial", "mixed", "mixed")))
    for _ in range(160):
        maximum = 9 if small else 16
        h, w = int(rng.integers(3, maximum + 1)), int(rng.integers(3, maximum + 1))
        partition = _partition(rng, h, w, count, style)
        if partition is None:
            continue
        walls, boxes = partition
        if lesson == "destination":
            spent = int(rng.integers(1, count - 1))
            remaining = count - spent
            distance = int(rng.integers(1, min(remaining - 1, 5) + 1))
        elif lesson == "fuel":
            spent = int(rng.integers(1, count - 1))
            remaining = count - spent
            distance = remaining + int(rng.integers(1, 4))
            area = sum((b - t - 1) * (r - l - 1) for t, l, b, r in boxes[spent:])
            if area <= remaining:
                continue
        else:
            # The enlarged test body must actually exercise its chamber counter.
            # Optional companions still include fresh/exhausted stationary cases.
            spent = 0 if active and rng.random() < .35 else int(rng.integers(0, count - 1 if active else count + 1))
            remaining = count - spent
            regime = str(rng.choice(("destination", "fuel", "exact") if active else ("destination", "fuel", "exact", "free")))
            if regime == "destination" and remaining >= 2:
                distance = int(rng.integers(1, remaining))
            elif regime == "fuel":
                distance = remaining + int(rng.integers(1, 4))
            elif regime == "exact" and remaining:
                distance = remaining
            else:
                distance = int(rng.integers(1, 8 if not small else 5))
        if h + distance + 1 > 30:
            continue
        movement = min(remaining, distance)
        initial = _body(walls, boxes, spent, color, marker)
        damaged = _damage(rng, initial, walls, color, required=lesson is not None)
        final = _body(walls, boxes, spent + movement, color, marker)
        inp = np.zeros((h + distance + 1, w), dtype=np.int8)
        out = np.zeros_like(inp)
        inp[0] = out[0] = marker
        inp[distance] = marker
        inp[distance + 1:] = damaged
        out[distance - movement] = marker
        out[distance + 1 - movement:distance + 1 - movement + h] = final
        return np.rot90(inp, -_TURNS[direction]).copy(), np.rot90(out, -_TURNS[direction]).copy()
    raise AssertionError("bounded partition construction failed")


def _positions(occupied, shape):
    h, w = shape
    H, W = occupied.shape
    if h > H or w > W:
        return np.empty((0, 2), dtype=int)
    padded = np.pad(occupied, 1)
    blocked = np.zeros_like(occupied)
    for dy in range(3):
        for dx in range(3):
            blocked |= padded[dy:dy + H, dx:dx + W]
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _pair(rng, marker, ceiling, directions, *, lesson=None, direction=None, test=False):
    count = ceiling if lesson == "destination" else int(rng.integers(4, ceiling + 1)) if lesson == "fuel" else int(rng.integers(1, ceiling + 1))
    if test:
        count = ceiling + int(rng.integers(1, 4))
    palette = [int(c) for c in rng.permutation(np.arange(1, 9)) if c != marker]
    direction = direction or str(rng.choice(directions))
    main = _object(rng, palette[0], marker, count, direction, lesson, active=test)
    h, w = main[0].shape
    H, W = int(rng.integers(max(h, 14), 31)), int(rng.integers(max(w, 14), 31))
    inp, out = np.zeros((H, W), dtype=np.int8), np.zeros((H, W), dtype=np.int8)
    occupied = np.zeros((H, W), dtype=bool)
    wanted = int(rng.choice((1, 1, 2, 2, 3, 4)))
    for i in range(wanted):
        accepted = None
        for attempt in range(20 if i else 1):
            patch_in, patch_out = main if i == 0 else _object(
                rng, palette[i], marker, int(rng.integers(1, min(ceiling, 5) + 1)),
                str(rng.choice(directions)), small=True)
            ph, pw = patch_in.shape
            positions = _positions(occupied, (ph, pw))
            if len(positions):
                # Mixture of edge contact and unconstrained scatter.
                if rng.random() < .4:
                    edges = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0)
                                      | (positions[:, 0] + ph == H) | (positions[:, 1] + pw == W)]
                    if len(edges):
                        positions = edges
                y, x = map(int, positions[int(rng.integers(len(positions)))])
                accepted = (y, x, patch_in, patch_out)
                break
        if accepted is None:
            break  # Optional extra bodies never replace the required witness.
        y, x, patch_in, patch_out = accepted
        ph, pw = patch_in.shape
        inp[y:y + ph, x:x + pw] = patch_in
        out[y:y + ph, x:x + pw] = patch_out
        occupied[y:y + ph, x:x + pw] = True
    # Keep variable blank margins without clipping any swept region.
    if rng.random() < .8:
        points = np.argwhere(occupied)
        t, l = map(int, points.min(axis=0))
        b, r = map(int, points.max(axis=0))
        t, l = max(0, t - int(rng.integers(0, 4))), max(0, l - int(rng.integers(0, 4)))
        b, r = min(H, b + 1 + int(rng.integers(0, 4))), min(W, r + 1 + int(rng.integers(0, 4)))
        inp, out = inp[t:b, l:r].copy(), out[t:b, l:r].copy()
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("pair counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test and train > test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(1, 9))
    ceiling = int(rng.integers(5, 8))
    directions = tuple(map(str, rng.choice(list(_TURNS), size=2, replace=False)))
    lessons = ["destination", "fuel"] + [None] * (nt - 2)
    rng.shuffle(lessons)
    puzzle = {"train": [], "test": []}
    seen = set()
    lesson_directions = dict(zip(("destination", "fuel"), directions))
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            lesson = lessons[i] if split == "train" else None
            for _ in range(16):
                pair = _pair(rng, marker, ceiling, directions, lesson=lesson,
                             direction=lesson_directions.get(lesson), test=split == "test")
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("duplicate inputs after bounded attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid, vehicles):
    result = []
    for v in vehicles:
        rooms = v["rooms"]
        remaining = len(rooms) - v["spent"]
        ys, xs = set(), set()
        for room in rooms:
            points = np.argwhere(np.rot90(room, _TURNS[v["direction"]]))
            t, l = map(int, points.min(axis=0))
            b, r = map(int, points.max(axis=0))
            ys.update((t - 1, b + 1))
            xs.update((l - 1, r + 1))
        result.append(dict(count=len(rooms), spent=v["spent"], remaining=remaining,
                           movement=min(remaining, v["distance"]), distance=v["distance"],
                           direction=v["direction"], shape=v["walls"].shape,
                           partial=len(rooms) != (len(ys) - 1) * (len(xs) - 1),
                           areas=tuple(int(a.sum()) for a in rooms), defects=v["defects"],
                           marker=v["marker"], color=v["color"]))
    return result


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, markers = set(), set()
    features = {"train": [], "test": []}
    train_directions = set()
    evidence = dict(partial=False, destination=False, fuel=False, area=False, state=False, repair=False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "output shape changes")
            signature = _key(inp)
            _need(signature not in seen, "duplicate input")
            seen.add(signature)
            before = inp.copy()
            vs = _decode(inp)
            _need(np.array_equal(out, solve(inp)), "incorrect transformation")
            _need(np.array_equal(before, inp), "solver mutated input")
            _need(not np.any(out == 9), "unrepaired wall defects")
            records = _features(inp, vs)
            features[split].extend(records)
            markers.update(v["marker"] for v in vs)
            for v, record in zip(vs, records):
                dy, dx = _DELTAS[v["direction"]]
                m = record["movement"]
                t, l, b, r = v["box"]
                body = out[t + dy * m:b + 1 + dy * m, l + dx * m:r + 1 + dx * m]
                _need(np.all(body[v["walls"]] == v["color"]), "wall topology changed")
                states = [bool(np.all(body[room] == v["marker"])) for room in v["rooms"]]
                _need(states == [True] * (v["spent"] + m) + [False] * (len(states) - v["spent"] - m), "wrong chamber expenditure")
                _need(all(out[p] == v["marker"] for p in v["target"]), "dock moved")
            if split == "train":
                train_directions.update(v["direction"] for v in vs if len(v["rooms"]) > v["spent"])
                evidence["partial"] |= any(a["partial"] for a in records)
                evidence["destination"] |= any(0 < a["distance"] < a["remaining"] and a["spent"] > 0 for a in records)
                evidence["fuel"] |= any(0 < a["remaining"] < a["distance"] for a in records)
                evidence["repair"] |= any(a["defects"] for a in records)
                evidence["area"] |= not np.array_equal(out, _render_solution(inp, vs, area_fuel=True))
                evidence["state"] |= not np.array_equal(out, _render_solution(inp, vs, ignore_spent=True))
    _need(len(markers) == 1, "marker convention changed within puzzle")
    _need(all(evidence.values()), "training lacks concept/ablation witnesses")
    _need(len(train_directions) >= 2, "training lacks rotated counters")
    train_max = max(a["count"] for a in features["train"])
    for pair in puzzle["test"]:
        records = _features(pair["input"], _decode(_grid(pair["input"])))
        _need(max(a["count"] for a in records) > train_max, "missing chamber-count shift")
        _need(any(a["count"] > train_max and a["remaining"] >= 2 and a["movement"] > 0
                  for a in records), "enlarged test body must exercise the counter")
        _need(all(a["direction"] in train_directions for a in records), "untaught test orientation")
    return None


def self_test():
    """42 seeds, streaming metrics, count boundaries, metamorphic checks; no I/O."""
    shapes, geometries, palettes, directions, stops, counts = set(), set(), set(), set(), set(), set()
    partial = damaged = fresh = exhausted = multi = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                vs = _decode(grid)
                shapes.add(grid.shape)
                palettes.add(tuple(sorted(set(int(c) for c in np.unique(grid)))))
                multi += len(vs) > 1
                for a in _features(grid, vs):
                    geometries.add((a["count"], a["shape"], a["areas"], a["spent"], a["direction"]))
                    counts.add(a["count"])
                    directions.add(a["direction"])
                    regime = "zero" if a["remaining"] == 0 else "fuel" if a["remaining"] < a["distance"] else "exact" if a["remaining"] == a["distance"] else "destination"
                    stops.add((a["direction"], regime))
                    partial += a["partial"]
                    damaged += a["defects"] > 0
                    fresh += a["spent"] == 0
                    exhausted += a["remaining"] == 0
        if seed < 3:
            again = generate_puzzle(seed)
            _need(all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], again[s]) for k in a), "nondeterminism")
            pair = puzzle["train"][0]
            for turns in (1, 2, 3):
                _need(np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns)), "rotation failure")
            permutation = np.array([0, 8, 7, 6, 5, 4, 3, 2, 1, 9], dtype=np.int8)
            _need(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "color equivariance failure")
    _need(len(shapes) >= 70 and len(geometries) >= 180 and len(palettes) >= 20, "variation collapsed")
    _need(directions == set(_TURNS), "missing direction")
    _need(len(stops) == 16, "direction/stop correlation")
    _need({1, 2, 3, 5, 8}.issubset(counts), "chamber count range")
    _need(min(partial, damaged, fresh, exhausted, multi) > 10, "missing family variation")
    # Every legal explicit combination; reject invalid numeric types too.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            _need(len(p["train"]) == nt and len(p["test"]) == ne, "count ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "shapes": len(shapes), "geometries": len(geometries),
            "palettes": len(palettes), "partial_vehicles": partial, "damaged_vehicles": damaged,
            "fresh_vehicles": fresh, "exhausted_vehicles": exhausted, "multi_vehicle_pairs": multi}
