"""Tuned diagonal receivers: the receiver is both a ruler and a gate.

Colors 8 and 9 mean single and coincident impacts. All other roles are read
from the input's background, singleton pairs, and monochrome arrow geometry.
No parent generator, example bank, or generation state is used by solve.
"""
from collections import Counter
from itertools import combinations

import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))
SINGLE, COLLISION = 8, 9


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple(sorted((r + a, c + b) for a, b in offsets))


def _segment(a, b):
    a, b = sorted((a, b))
    distance = b[0] - a[0]
    assert distance >= 4 and abs(b[1] - a[1]) == distance
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k, a[1] + slope * k)
                 for k in range(distance + 1))


def _line(point, slope, shape):
    r, c = point
    return {(row, c + slope * (row - r)) for row in range(shape[0])
            if 0 <= c + slope * (row - r) < shape[1]}


def _near(cells):
    return {(r + a, c + b) for r, c in cells
            for a in (-1, 0, 1) for b in (-1, 0, 1)}


def _inside(points, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in points)


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _parse(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    assert counts.max() > grid.size / 2 and background < 8
    assert not np.any((grid == SINGLE) | (grid == COLLISION))
    foreground = set(map(tuple, np.argwhere(grid != background).tolist()))
    remaining = foreground.copy()
    dots, arrows = {}, []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            p = todo.pop()
            neighbors = _near((p,)) & remaining
            remaining -= neighbors
            component |= neighbors
            todo.extend(sorted(neighbors))
        object_colors = {int(grid[p]) for p in component}
        assert len(object_colors) == 1, "Input objects must have a blank separating gap"
        color = object_colors.pop()
        if len(component) == 1:
            dots.setdefault(color, []).append(start)
        else:
            assert len(component) == 3, "Expected a singleton or three-cell arrow"
            matches = [(tip, direction) for tip in sorted(component)
                       for direction in DIRECTIONS
                       if set(_arrow(tip, direction)) == component]
            assert len(matches) == 1, "Arrow direction must be unique"
            tip, direction = matches[0]
            arrows.append((color, tip, direction, tuple(sorted(component))))
    assert dots and arrows
    assert set(dots) == {a[0] for a in arrows}, "Every channel needs a ruler and arrows"
    receivers, span_union, endpoints = {}, set(), set()
    for color, points in sorted(dots.items()):
        assert len(points) == 2, "Exactly two endpoint dots per channel"
        span = _segment(*points)
        assert not span_union.intersection(span), "Receiver spans must be disjoint"
        span_union.update(span)
        endpoints.update((span[0], span[-1]))
        slope = 1 if span[-1][1] > span[0][1] else -1
        receivers[color] = (span, len(span) - 1, slope)
    assert span_union & foreground == endpoints, "Latent spans cannot contain input arrows"
    return grid, background, receivers, arrows, foreground


def _analyze(input_grid):
    grid, bg, receivers, arrows, foreground = _parse(input_grid)
    shots = []
    for color, tip, direction, cells in arrows:
        distance = receivers[color][1]
        point = tuple(tip[a] + distance * direction[a] for a in (0, 1))
        assert _inside((point,), grid.shape), "Every projection must stay on the canvas"
        assert point not in foreground, "Projected hits must be on originally empty cells"
        shots.append((color, point))
    counts = Counter(point for _, point in shots)
    active = {(color, point) for color, point in shots
              if point in receivers[color][0][1:-1]}
    beams = [(point, _line(point, -receivers[color][2], grid.shape))
             for color, point in sorted(active)]
    origins = {point for _, point in active}
    spans = set().union(*(set(v[0]) for v in receivers.values()))
    for point, cells in beams:
        assert not cells & foreground, "An emitted line must not cover input markers"
        assert cells & spans == {point}, "Emitted lines may meet receivers only at their origin"
    return {"grid": grid, "background": bg, "receivers": receivers,
            "arrows": arrows, "foreground": foreground, "shots": shots,
            "counts": counts, "active": active, "origins": origins, "beams": beams}


def _paint(analysis, *, gate=True, distance_override=None,
           count_override=False, infinite_gate=False, ignore_channel=False):
    """Normal paint plus explicit alternative-rule probes used by validation."""
    grid, receivers = analysis["grid"], analysis["receivers"]
    shots = analysis["shots"]
    if distance_override is not None:
        shots = [(color, tuple(tip[a] + distance_override * direction[a] for a in (0, 1)))
                 for color, tip, direction, _ in analysis["arrows"]]
    counts = Counter(point for _, point in shots)
    deposits = {p: SINGLE if n == 1 or count_override else COLLISION
                for p, n in counts.items() if _inside((p,), grid.shape)}
    if gate:
        for color, point in sorted(set(shots)):
            targets = receivers.values() if ignore_channel else (receivers[color],)
            for span, _, slope in targets:
                on_span = point in span[1:-1]
                if infinite_gate:
                    on_span = point[1] - span[0][1] == slope * (point[0] - span[0][0])
                if on_span:
                    value = SINGLE if counts[point] == 1 or count_override else COLLISION
                    for cell in _line(point, -slope, grid.shape):
                        deposits[cell] = max(deposits.get(cell, SINGLE), value)
    output = grid.copy()
    for color, (span, _, _) in receivers.items():
        for p in span:
            output[p] = color
    # Only original arrow-tip coincidences determine collision intensity.
    # Crossings of drawn lines do not count as additional shots.
    for p, value in deposits.items():
        output[p] = value
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive rulers, projections, collisions, and gates from this input only."""
    return _paint(_analyze(input_grid))


def _overlap_types(analysis):
    """Visibly consequential overlay cases, reconstructed from input events."""
    counts, beams = analysis["counts"], analysis["beams"]
    intensity = lambda p: SINGLE if counts[p] == 1 else COLLISION
    types = set()
    for i, (p, cells) in enumerate(beams):
        for q, other in beams[i + 1:]:
            if cells & other:
                types.add(tuple(sorted((intensity(p), intensity(q)))))
        for q in cells & (set(counts) - analysis["origins"]):
            types.add(tuple(sorted((intensity(p), intensity(q)))))
    # Two collision-colored deposits are 9 under either overlay or an extra
    # intersection-collision interpretation, so introduce no new consequence.
    return types - {(COLLISION, COLLISION)}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _sample_receivers(rng, shape, distances):
    receivers, endpoints, span_union = [], set(), set()
    parallel = int(rng.choice((-1, 1))) if rng.random() < .35 else None
    for distance in distances:
        for _ in range(32):
            slope = parallel or int(rng.choice((-1, 1)))
            r = int(rng.integers(shape[0] - distance))
            c = int(rng.integers(shape[1] - distance))
            if slope == -1:
                c += distance
            span = tuple((r + k, c + slope * k) for k in range(distance + 1))
            ends = {span[0], span[-1]}
            if span_union.intersection(span) or _near(ends) & endpoints:
                continue
            receivers.append(span)
            endpoints |= ends
            span_union.update(span)
            break
        else:
            return None
    return receivers


def _event_plan(rng, shape, receivers, witness):
    spans = set().union(*(set(s) for s in receivers))
    endpoints = {p for s in receivers for p in (s[0], s[-1])}
    events = []

    def add(channel, point, number):
        if point in endpoints or any(ch == channel and p == point for ch, p, _ in events):
            return False
        events.append((channel, point, number))
        return True

    if witness == "gate":
        span = receivers[0]
        add(0, _pick(rng, span[1:-1]), 1)
        slope = 1 if span[-1][1] > span[0][1] else -1
        beyond = [p for p in _line(span[0], slope, shape)
                  if p not in spans and min(abs(p[0] - span[0][0]),
                                            abs(p[0] - span[-1][0])) <= 5]
        if not beyond:
            return None
        add(0, _pick(rng, sorted(beyond)), int(rng.choice((2, 2, 3))))
        if len(receivers) > 1:
            add(1, _pick(rng, receivers[1][1:-1]), 1)
            unused = [p for p in receivers[1][1:-1]
                      if p not in {q for _, q, _ in events}]
            if not unused:
                return None
            # This hit lies on a receiver of the wrong color, so stays a point.
            add(0, _pick(rng, unused), 1)
    elif witness == "collision":
        indices = rng.choice(len(receivers[0]) - 2, size=2, replace=False) + 1
        add(0, receivers[0][int(indices[0])], int(rng.choice((2, 2, 3))))
        add(0, receivers[0][int(indices[1])], 1)

    for channel, span in enumerate(receivers):
        if witness == "gate" and channel < 2:
            continue
        if witness == "collision" and channel == 0:
            continue
        active_number = int(rng.choice((0, 1, 1, 2, 3)))
        active_number = min(active_number, len(span) - 2)
        for index in rng.permutation(len(span) - 2)[:active_number]:
            add(channel, span[int(index) + 1], int(rng.choice((1, 1, 1, 2, 3))))
        misses = int(rng.choice((0, 0, 1, 1, 2)))
        if not active_number:
            misses = max(1, misses)
        pool = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                if (r, c) not in span and (r, c) not in endpoints]
        for index in rng.permutation(len(pool))[:misses]:
            add(channel, pool[int(index)], int(rng.choice((1, 1, 2))))
    if sum(n for _, _, n in events) > 14:
        return None
    return events


def _place_arrows(rng, shape, receivers, events):
    spans = set().union(*(set(s) for s in receivers))
    endpoints = {p for s in receivers for p in (s[0], s[-1])}
    destinations = {p for _, p, _ in events}
    beams = []
    for channel, point, _ in events:
        span = receivers[channel]
        if point in span[1:-1]:
            slope = 1 if span[-1][1] > span[0][1] else -1
            line = _line(point, -slope, shape)
            if line & spans != {point} or line & endpoints:
                return None
            beams.append(line)
    beam_union = set().union(*beams) if beams else set()
    blocked = spans | destinations | beam_union | _near(endpoints)
    slots = []
    for channel, point, number in events:
        distance = len(receivers[channel]) - 1
        candidates = []
        for direction in DIRECTIONS:
            tip = tuple(point[a] - distance * direction[a] for a in (0, 1))
            cells = _arrow(tip, direction)
            if _inside(cells, shape) and not set(cells) & blocked:
                candidates.append((channel, tip, direction, cells))
        if len(candidates) < number:
            return None
        for _ in range(number):
            slots.append(candidates)
    # Constrained placement first, with a bounded backtracking search. Arrow
    # candidates are derived from events, not selected from saved whole puzzles.
    slots.sort(key=len)
    orders = [[options[int(i)] for i in rng.permutation(len(options))] for options in slots]
    nodes = 0

    def search(index, selected, near):
        nonlocal nodes
        if index == len(orders):
            return selected
        for arrow in orders[index]:
            nodes += 1
            if nodes > 500:
                return None
            if set(arrow[3]) & near:
                continue
            result = search(index + 1, selected + [arrow], near | _near(arrow[3]))
            if result is not None:
                return result
        return None

    return search(0, [], set())


def _render(shape, background, colors, receivers, arrows):
    grid = np.full(shape, background, dtype=np.int8)
    for color, span in zip(colors, receivers):
        grid[span[0]], grid[span[-1]] = color, color
    for channel, _, _, cells in arrows:
        for p in cells:
            grid[p] = colors[channel]
    return {"input": grid, "output": solve(grid)}


def _fallback(rng, background, colors, distances, witness, used):
    """A finite emergency geometry, obeying the same ruler/receiver rule.

    Capacity comes from reflections, rotations, distances, and blank cropping.
    It is only used after free geometric placement exhausts its retry budget.
    """
    query = distances[0] >= 8
    if query:
        distances, starts = distances[:1], ((12, 12),)
        p, q = (14, 14), (15, 15)
    else:
        distances, starts = distances[:2], ((8, 8), (20, 20))
        p, q = (10, 10), (11, 11)
    receivers = [tuple((r + k, c + k) for k in range(d + 1))
                 for (r, c), d in zip(starts, distances)]
    d = distances[0]
    if witness == "gate":
        specifications = [(0, p, (-1, -1)),
                          (0, (4, 4), (0, -1)), (0, (4, 4), (-1, 0))]
        if len(receivers) > 1:
            specifications += [(1, (22, 22), (0, 1)), (0, (21, 21), (1, 0))]
    else:
        specifications = [(0, p, (0, 1)), (0, p, (1, 0)), (0, q, (-1, -1))]
        if len(receivers) > 1:
            specifications.append((1, (22, 22), (0, 1)))
    arrows = []
    for channel, point, direction in specifications:
        tip = tuple(point[a] - distances[channel] * direction[a] for a in (0, 1))
        arrows.append((channel, tip, direction, _arrow(tip, direction)))
    pair = _render((30, 30), background, colors, receivers, arrows)
    for reflected in rng.permutation(2):
        for rotation in rng.permutation(4):
            transformed = {k: np.rot90(np.fliplr(g) if reflected else g,
                                     int(rotation)).copy() for k, g in pair.items()}
            # Extra blank bottom/right margins may be cropped when every shot
            # and input object remains inside. Recompute boundary extensions.
            for shape in ((30, 30), (29, 30), (30, 29), (29, 29)):
                candidate = transformed["input"][:shape[0], :shape[1]].copy()
                if _grid_key(candidate) in used:
                    continue
                try:
                    output = solve(candidate)
                except AssertionError:
                    continue
                return {"input": candidate, "output": output}
    raise AssertionError("Emergency geometry exhausted its finite uniqueness capacity")


def _make_pair(rng, background, colors, distances, witness, used, allowed_overlaps=None):
    for _ in range(220):
        floor = max(12, 2 * max(distances) + 3,
                    22 if len(distances) == 2 else 27 if len(distances) >= 3 else 0)
        shape = tuple(int(rng.integers(floor, 31)) for _ in range(2))
        if rng.random() < .25:
            shape = (max(shape), max(shape))
        receivers = _sample_receivers(rng, shape, distances)
        if receivers is None:
            continue
        events = _event_plan(rng, shape, receivers, witness)
        if events is None:
            continue
        arrows = _place_arrows(rng, shape, receivers, events)
        if arrows is None:
            continue
        if witness and len({int(a[2][0] * a[2][1] == 0) for a in arrows}) < 2:
            continue
        pair = _render(shape, background, colors, receivers, arrows)
        if allowed_overlaps is not None and not _overlap_types(_analyze(pair["input"])) <= allowed_overlaps:
            continue
        if _grid_key(pair["input"]) not in used:
            return pair
    return _fallback(rng, background, colors, distances, witness, used)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation(8)]
    background = palette[0]
    channel_colors = palette[1:4]
    training_distances = [int(v) for v in rng.permutation((4, 5, 6, 7))]
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", train_count), ("test", test_count)):
        allowed_overlaps = (set().union(*(_overlap_types(_analyze(p["input"]))
                                         for p in puzzle["train"]))
                            if split == "test" else None)
        for index in range(count):
            witness = ("gate", "collision")[index] if split == "train" and index < 2 else None
            channels = int(rng.choice((2, 2, 2, 3))) if witness == "gate" else int(rng.choice((1, 1, 2, 2, 3)))
            if witness == "collision":
                channels = min(channels, 2)
            pool = (4, 5, 6, 7) if split == "train" else (8, 9, 10, 11)
            primary = training_distances[index % 4] if split == "train" else int(rng.choice(pool))
            others = [int(v) for v in rng.permutation([d for d in pool if d != primary])]
            distances = [primary] + others[:channels - 1]
            colors = [int(v) for v in rng.permutation(channel_colors)[:channels]]
            pair = _make_pair(rng, background, colors, distances, witness, used, allowed_overlaps)
            puzzle[split].append(pair)
            used.add(_grid_key(pair["input"]))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a = _analyze(pair["input"])
    receivers = a["receivers"]
    beam_sets = [cells for _, cells in a["beams"]]
    beyond = False
    wrong_channel = False
    for color, p in a["shots"]:
        span, _, slope = receivers[color]
        beyond |= (p not in span and p[1] - span[0][1] == slope * (p[0] - span[0][0]))
        wrong_channel |= any(p in value[0][1:-1] for other, value in receivers.items()
                             if other != color) and (color, p) not in a["active"]
    return {"shape": a["grid"].shape, "background": a["background"],
            "channels": len(receivers), "distances": tuple(sorted(v[1] for v in receivers.values())),
            "slopes": tuple(sorted(v[2] for v in receivers.values())),
            "arrows": len(a["arrows"]), "directions": tuple(sorted({o[2] for o in a["arrows"]})),
            "active": len(a["active"]), "active_single": any(a["counts"][p] == 1 for _, p in a["active"]),
            "active_collision": any(a["counts"][p] >= 2 for _, p in a["active"]),
            "inactive_collision": any(n >= 2 and p not in a["origins"] for p, n in a["counts"].items()),
            "beyond": bool(beyond), "wrong_channel": bool(wrong_channel),
            "crossing_beams": any(x & y for x, y in combinations(beam_sets, 2)),
            "overlap_types": tuple(sorted(_overlap_types(a))),
            "max_convergence": max(a["counts"].values())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, backgrounds, records = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            assert inp.shape == out.shape and out.ndim == 2
            assert np.issubdtype(out.dtype, np.integer) and np.all((0 <= out) & (out <= 9))
            key = _grid_key(inp)
            assert key not in used, "Inputs must be distinct within a puzzle"
            used.add(key)
            a = _analyze(inp)
            backgrounds.add(a["background"])
            assert np.array_equal(out, solve(inp.copy())), "Incorrect output"
            assert not np.array_equal(inp, out)
            assert all(inp[p] == out[p] for p in a["foreground"]), "Input objects must be preserved"
            painted = set().union(*(set(v[0]) for v in a["receivers"].values())) | set(a["counts"])
            painted |= set().union(*(cells for _, cells in a["beams"])) if a["beams"] else set()
            untouched = np.ones(inp.shape, dtype=bool)
            for p in painted:
                untouched[p] = False
            assert np.array_equal(inp[untouched], out[untouched])
            records[split].append((_features(pair), a, out))
    assert len(backgrounds) == 1, "One shared background convention per puzzle"
    training = records["train"]
    assert any(f["active_single"] for f, _, _ in training)
    assert any(f["active_collision"] and f["active_single"] for f, _, _ in training)
    assert any(f["inactive_collision"] and f["beyond"] for f, _, _ in training)
    assert any(f["wrong_channel"] for f, _, _ in training)
    assert any(f["channels"] >= 2 and len(set(f["distances"])) >= 2 for f, _, _ in training)
    assert any(not np.array_equal(_paint(a, gate=False), out) for _, a, out in training), "B gate is consequential"
    assert any(not np.array_equal(_paint(a, count_override=True), out) for _, a, out in training), "A collisions are consequential"
    assert any(not np.array_equal(_paint(a, infinite_gate=True), out) for _, a, out in training), "Finite membership is taught"
    assert any(not np.array_equal(_paint(a, ignore_channel=True), out) for _, a, out in training), "Color association is taught"
    # No single fixed distance in the entire valid range fits the demonstrations.
    assert not any(all(np.array_equal(_paint(a, distance_override=d), out)
                       for _, a, out in training) for d in range(1, 30))
    train_ranges = {d for f, _, _ in training for d in f["distances"]}
    test_ranges = {d for f, _, _ in records["test"] for d in f["distances"]}
    assert train_ranges.isdisjoint(test_ranges), "Query ruler lengths must be new"
    taught_overlaps = set().union(*(_overlap_types(a) for _, a, _ in training))
    assert all(_overlap_types(a) <= taught_overlaps for _, a, _ in records["test"]), \
        "Teach any consequential overlap convention before querying it"


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(np.asarray(p["input"])), _grid_key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    records = [_features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert len({f["shape"] for f in records}) >= 35
    assert {1, 2, 3} <= {f["channels"] for f in records}
    assert len({f["arrows"] for f in records}) >= 7
    assert {d for f in records for d in f["distances"]} == set(range(4, 12))
    assert {d for f in records for d in f["directions"]} == set(DIRECTIONS)
    assert {s for f in records for s in f["slopes"]} == {-1, 1}
    assert len({f["background"] for f in records}) >= 6
    assert len({f["directions"] for f in records}) >= 30
    assert any(f["active"] == 0 for f in records)
    assert any(f["max_convergence"] == 3 for f in records)
    assert any(f["crossing_beams"] for f in records)
    # A fully specified hand fixture checks colors, finite membership, and the
    # fact that multiplicity is per destination, not pooled over a receiver.
    spans = [tuple((k, k) for k in range(8, 13)),
             tuple((k, k) for k in range(20, 28))]
    specifications = [(0, (10, 6), (0, 1)), (0, (6, 10), (1, 0)),
                      (0, (15, 15), (-1, -1)), (1, (22, 15), (0, 1))]
    arrows = [(ch, tip, di, _arrow(tip, di)) for ch, tip, di in specifications]
    fixture = _render((30, 30), 0, [2, 3], spans, arrows)
    out = fixture["output"]
    assert out[10, 10] == COLLISION and out[11, 11] == SINGLE
    expected = fixture["input"].copy()
    for k in range(8, 13):
        expected[k, k] = 2
    for k in range(20, 28):
        expected[k, k] = 3
    rows, cols = np.indices((30, 30))
    expected[(rows + cols == 22) | (rows + cols == 44)] = SINGLE
    expected[rows + cols == 20] = COLLISION
    assert np.array_equal(out, expected)
    a = _analyze(fixture["input"])
    assert sorted(a["counts"].values()) == [1, 1, 2]
    assert len(a["active"]) == 3
    for point, line in a["beams"]:
        expected = SINGLE if a["counts"][point] == 1 else COLLISION
        assert all(out[cell] == expected for cell in line)
    for d in range(4, 8):
        for mode in ("gate", "collision"):
            other = 4 if d != 4 else 7
            fallback = _fallback(np.random.default_rng(d), 0, [2, 3], [d, other], mode, set())
            features = _features(fallback)
            assert features["channels"] == 2
            if mode == "gate":
                assert features["beyond"] and features["wrong_channel"] and features["inactive_collision"]
            else:
                assert features["active_single"] and features["active_collision"]
    for d in range(8, 12):
        _fallback(np.random.default_rng(d), 0, [2], [d], None, set())
    # Rotation/reflection and channel recoloring leave the inferred mechanics
    # unchanged; the two reserved impact colors retain their fixed meanings.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
        assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
        mapping = np.array([7, 6, 5, 4, 3, 2, 1, 0, 8, 9], dtype=np.int8)
        assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(101, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
