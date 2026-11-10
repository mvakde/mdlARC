"""Active harbors: diagonal interval fields switch geometric docking sites on.

Only standard-library modules and NumPy are used. The input parser, interval
predicate, and exact-cover solver have no generation state. Labels are drawn
from construction-time target placements, then checked by that input-only solver.
"""
from itertools import combinations

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(cells, neighbors=N4):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _buffer(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _inside(cells, shape):
    t, l, b, r = _box(cells)
    return 0 <= t <= b < shape[0] and 0 <= l <= r < shape[1]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _runs(cells, slope):
    lines = {}
    for r, c in cells:
        lines.setdefault(c - slope * r, []).append(r)
    if len(lines) < 2:
        return None
    qs = sorted(lines)
    if any(b - a < 3 for a, b in zip(qs, qs[1:])):
        return None
    runs = []
    for q in qs:
        rows = sorted(lines[q])
        start = previous = rows[0]
        for r in rows[1:] + [None]:
            if r is not None and r == previous + 1:
                previous = r
                continue
            if previous == start:
                return None
            runs.append((q, 2 * start + slope * q,
                         2 * previous + slope * q))
            if r is not None:
                start = previous = r
    return runs


def _activation(shape, runs, slope, alternative=""):
    """Continuous intervals, including longitudinal levels between pixels."""
    rows, cols = np.indices(shape)
    q, t = cols - slope * rows, rows + slope * cols
    mask = np.zeros(shape, dtype=bool)
    if alternative == "consecutive":
        # Deliberately weaker: permanently neighboring rails only.
        ordered = sorted(runs)
        for a, b in zip(ordered, ordered[1:]):
            mask |= ((t >= max(a[1], b[1])) & (t <= min(a[2], b[2]))
                     & (q > a[0]) & (q < b[0]))
        return mask
    for level in range(int(t.min()), int(t.max()) + 1):
        active = [x[0] for x in runs if x[1] <= level <= x[2]
                  and (alternative != "centers" or (level - x[0]) % 2 == 0)]
        if len(active) >= 2:
            mask |= (t == level) & (q > min(active)) & (q < max(active))
    return mask


def _analyze(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must occupy a strict majority")
    by_color = {int(c): set(map(tuple, np.argwhere(grid == c).tolist()))
                for c in values if c != background}
    candidates = []
    for color, cells in by_color.items():
        for slope in (-1, 1):
            runs = _runs(cells, slope)
            if runs is not None:
                candidates.append((color, slope, runs))
    if len(candidates) != 1:
        raise ValueError("rail color and slope must be uniquely recognizable")
    rail_color, slope, runs = candidates[0]
    marker_colors = [c for c, cells in by_color.items() if c != rail_color
                     and all(len(x) == 1 for x in _components(cells, N8))]
    if len(marker_colors) != 1:
        raise ValueError("one sparse connector hue is required")
    marker = marker_colors[0]
    payload_cells = set().union(*(cells for c, cells in by_color.items()
                                 if c not in (rail_color, marker)))
    bodies = _components(payload_cells)
    if not bodies or any(len(b) < 2 for b in bodies):
        raise ValueError("payloads must be nontrivial orthogonally connected bodies")
    pieces = []
    owners = {}
    for i, body in enumerate(bodies):
        colors = {int(grid[q]) for q in body}
        if len(colors) != 1:
            raise ValueError("each body has exactly one payload hue")
        for q in body:
            owners[q] = i
        pieces.append(dict(body=body, pins=set(), color=colors.pop()))
    sites = set()
    for r, c in by_color[marker]:
        adjacent = {owners[(r + dr, c + dc)] for dr, dc in N4
                    if (r + dr, c + dc) in owners}
        if len(adjacent) > 1:
            raise ValueError("a pin touches more than one input piece")
        if adjacent:
            pieces[adjacent.pop()]["pins"].add((r, c))
        else:
            sites.add((r, c))
    if any(len(p["pins"]) not in (2, 3) for p in pieces):
        raise ValueError("every piece needs a two- or three-pin key")
    active = _activation(grid.shape, runs, slope)
    live = {q for q in sites if active[q]}
    return dict(background=background, rail_color=rail_color, marker=marker,
                rails=by_color[rail_color], slope=slope, runs=runs,
                pieces=pieces, sites=sites, live=live, active=active,
                shape=grid.shape)


def _placements(info, piece, sites):
    first = min(piece["pins"])
    forbidden = info["rails"] | info["sites"]
    result = []
    for r, c in sorted(sites):
        delta = r - first[0], c - first[1]
        pins = _shift(piece["pins"], delta)
        if not pins <= sites:
            continue
        body = _shift(piece["body"], delta)
        if not _inside(body | pins, info["shape"]) or body & forbidden:
            continue
        result.append((delta, body, pins))
    return result


def _draw(info, positions):
    output = np.full(info["shape"], info["background"], dtype=np.int8)
    for q in info["rails"]:
        output[q] = info["rail_color"]
    for q in info["sites"]:
        output[q] = info["marker"]
    for i, delta in positions.items():
        for q in _shift(info["pieces"][i]["body"], delta):
            output[q] = info["pieces"][i]["color"]
        for q in _shift(info["pieces"][i]["pins"], delta):
            output[q] = info["marker"]
    return output


def _reconstruct(info):
    pieces, live = info["pieces"], info["live"]
    if sum(len(p["pins"]) for p in pieces) != len(live):
        raise ValueError("live sites must exactly cover the pieces' pins")
    choices = [_placements(info, p, live) for p in pieces]
    if not all(choices):
        raise ValueError("a piece has no legal live docking position")
    answers, positions, states = {}, {}, 0

    def visit(remaining, used, occupied):
        nonlocal states
        states += 1
        if states > 20000:
            raise ValueError("complete uniqueness search exceeded its finite budget")
        if not remaining:
            if used == live:
                out = _draw(info, positions)
                answers.setdefault(out.tobytes(), (out, dict(positions)))
            return
        filtered = {i: [p for p in choices[i]
                        if not p[2] & used and not p[1] & occupied]
                    for i in remaining}
        i = min(remaining, key=lambda j: (len(filtered[j]), j))
        for delta, body, pins in filtered[i]:
            positions[i] = delta
            visit(remaining - {i}, used | pins, occupied | body)
            del positions[i]
            if len(answers) > 1:
                return

    visit(set(range(len(pieces))), set(), set())
    if len(answers) != 1:
        raise ValueError("docking must have exactly one distinct output grid")
    out, positions = next(iter(answers.values()))
    return out, positions, states


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _reconstruct(_analyze(_grid(input_grid)))[0]


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _body(rng, pin_count):
    if rng.random() < .17:
        h, w = 1, int(rng.integers(5, 8))
    else:
        h, w = int(rng.integers(2, 6)), int(rng.integers(3, 7))
    if rng.random() < .5:
        h, w = w, h
    rectangle = {(r, c) for r in range(h) for c in range(w)}
    boundary = ([(-1, c) for c in range(w)] + [(h, c) for c in range(w)]
                + [(r, -1) for r in range(h)] + [(r, w) for r in range(h)])
    options = [x for x in combinations(boundary, pin_count)
               if all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
                      for a, b in combinations(x, 2))]
    pins = set(options[int(rng.integers(len(options)))])
    required = {(min(h - 1, max(0, r)), min(w - 1, max(0, c))) for r, c in pins}
    style = int(rng.integers(5))
    if style == 0 or min(h, w) == 1:
        body = set(rectangle)
    elif style in (1, 2):
        missing_side = int(rng.integers(4)) if style == 2 else -1
        body = {(r, c) for r, c in rectangle
                if any((r == 0, c == w - 1, r == h - 1, c == 0)[i]
                       for i in range(4) if i != missing_side)}
    else:
        body = {min(required)}
        target = int(rng.integers(max(2, h * w // 3), max(3, 4 * h * w // 5 + 1)))
        while len(body) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N4}
                              & rectangle - body)
            if not frontier:
                break
            body.add(frontier[int(rng.integers(len(frontier)))])
    for target in sorted(required):
        r, c = min(body, key=lambda q: (abs(q[0] - target[0]) + abs(q[1] - target[1]), q))
        for axis in rng.permutation(2):
            if axis == 0:
                while r != target[0]:
                    r += 1 if target[0] > r else -1
                    body.add((r, c))
            else:
                while c != target[1]:
                    c += 1 if target[1] > c else -1
                    body.add((r, c))
    return dict(body=body, pins=pins)


def _rail_scene(rng, count, shape_min, skip):
    span = int(rng.integers(16, 25))
    gaps = 3 + rng.multinomial(span - 3 * (count - 1), [1 / (count - 1)] * (count - 1))
    qs = [0] + [int(x) for x in np.cumsum(gaps)]
    runs = []
    for i, q in enumerate(qs):
        if i in (0, count - 1):
            lo = int(rng.integers(0, 5))
            length = int(rng.integers(10, 14))
        else:
            lo = int(rng.integers(1, 13))
            length = int(rng.integers(2, 10))
        lo += (q - lo) % 2
        if skip and i == count // 2:
            lo = 10 + (q % 2)
            length = int(rng.integers(2, 4))
        runs.append((q, lo, lo + 2 * (length - 1)))
    cells = {(int((t - q) // 2), int((t + q) // 2))
             for q, lo, hi in runs for t in range(lo, hi + 1, 2)}
    t, l, b, r = _box(cells)
    bh, bw = b - t + 1, r - l + 1
    if max(bh, bw) > 30:
        return None
    h = int(rng.integers(max(shape_min, bh), 31))
    w = int(rng.integers(max(shape_min, bw), 31))
    offset = int(rng.integers(h - bh + 1)) - t, int(rng.integers(w - bw + 1)) - l
    cells = _shift(cells, offset)
    translated_runs = _runs(cells, 1)
    if translated_runs is None or _runs(cells, -1) is not None:
        return None
    return (h, w), cells, translated_runs


def _construct(rng, roles, count, pin_counts, rail_count, skip, overhang, minimum):
    scene = _rail_scene(rng, rail_count, minimum, skip)
    if scene is None:
        return None
    shape, rails, runs = scene
    active = _activation(shape, runs, 1)
    centers = _activation(shape, runs, 1, "centers")
    consecutive = _activation(shape, runs, 1, "consecutive")
    active_cells = sorted(map(tuple, np.argwhere(active).tolist()))
    targets, live, occupied, keys = [], set(), set(), set()
    for i in range(count):
        found = False
        for attempt in range(30):
            piece = _body(rng, pin_counts[i])
            key = _normalized(piece["pins"])
            if key in keys:
                continue
            first = min(piece["pins"])
            for index in rng.permutation(len(active_cells)):
                p = active_cells[int(index)]
                delta = p[0] - first[0], p[1] - first[1]
                pins, body = _shift(piece["pins"], delta), _shift(piece["body"], delta)
                if not _inside(pins | body, shape) or not all(active[q] for q in pins):
                    continue
                if body & (rails | live | occupied) or pins & (rails | occupied | _buffer(live)):
                    continue
                if i == 0 and (all(centers[q] for q in pins)
                               or (skip and all(consecutive[q] for q in pins))):
                    continue
                if i == 0 and overhang and all(active[q] for q in body):
                    continue
                targets.append(dict(body=body, pins=pins))
                live |= pins
                occupied |= body
                keys.add(key)
                found = True
                break
            if found:
                break
        if not found:
            return None
    # Copy a real key to a wholly dormant position. Its body would fit there,
    # making it a genuine geometric alternative when activation is omitted.
    decoy = None
    piece = targets[0]
    t, l, b, r = _box(piece["body"] | piece["pins"])
    offsets = [(a - t, c - l) for a in range(shape[0] - (b - t))
               for c in range(shape[1] - (r - l))]
    for index in rng.permutation(len(offsets)):
        delta = offsets[int(index)]
        pins, body = _shift(piece["pins"], delta), _shift(piece["body"], delta)
        if any(active[q] for q in pins):
            continue
        if pins & (rails | occupied | _buffer(live)) or body & (rails | live | occupied):
            continue
        decoy = dict(body=body, pins=pins, delta=delta)
        break
    if decoy is None:
        return None
    sites = live | decoy["pins"]
    for _ in range(int(rng.integers(0, 5))):
        for attempt in range(80):
            q = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            if not active[q] and q not in rails | occupied | _buffer(sites):
                sites.add(q)
                break
    scaffold = rails | sites
    background, rail_color, marker, colors = roles
    # Body hues carry no docking identity; repeated hues remain separate objects.
    palette_size = int(rng.integers(1, min(count, len(colors)) + 1))
    palette = [int(x) for x in rng.choice(colors, palette_size, replace=False)]
    body_colors = [palette[i % palette_size] for i in range(count)]
    rng.shuffle(body_colors)
    for p, color in zip(targets, body_colors):
        p["color"] = color
    for restart in range(20):
        blocked = _buffer(scaffold)
        scattered = {}
        success = True
        for i in range(count):
            p = targets[i]
            t, l, b, r = _box(p["body"] | p["pins"])
            offsets = [(a - t, c - l) for a in range(shape[0] - (b - t))
                       for c in range(shape[1] - (r - l))]
            rng.shuffle(offsets)
            if i == 0:
                # The dormant copy is closer than the live match, so nearest
                # docking without A is contradicted by a displayed example.
                offsets = [d for d in offsets if
                           abs(d[0] - decoy["delta"][0]) + abs(d[1] - decoy["delta"][1])
                           < abs(d[0]) + abs(d[1])]
                offsets.sort(key=lambda d: abs(d[0] - decoy["delta"][0])
                             + abs(d[1] - decoy["delta"][1]))
            for delta in offsets:
                if delta == (0, 0):
                    continue
                cells = _shift(p["body"] | p["pins"], delta)
                if cells & blocked:
                    continue
                scattered[i] = delta
                blocked |= _buffer(cells)
                break
            else:
                success = False
                break
        if success:
            break
    else:
        return None
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for q in rails:
        inp[q] = out[q] = rail_color
    for q in sites:
        inp[q] = out[q] = marker
    for i, p in enumerate(targets):
        for q in p["body"]:
            out[q] = p["color"]
        for q in _shift(p["body"], scattered[i]):
            inp[q] = p["color"]
        for q in _shift(p["pins"], scattered[i]):
            inp[q] = marker
    # Symmetries act on the whole scene; each piece still only translates.
    k = int(rng.integers(4))
    inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    try:
        predicted = solve(inp)
    except ValueError:
        return None
    if not np.array_equal(predicted, out):
        return None
    return {"input": inp, "output": out}


def _features(inp, out):
    info = _analyze(inp)
    predicted, positions, states = _reconstruct(info)
    assert np.array_equal(predicted, out), "incorrect docking output"
    bodies, matched = [], set()
    for i, piece in enumerate(info["pieces"]):
        delta = positions[i]
        assert delta != (0, 0), "generated loose pieces must move"
        body, pins = _shift(piece["body"], delta), _shift(piece["pins"], delta)
        assert pins <= info["live"] and not pins & matched
        assert not body & (info["rails"] | info["sites"] | set().union(*bodies))
        assert _inside(body | pins, inp.shape)
        assert all(out[q] == piece["color"] for q in body)
        bodies.append(body)
        matched |= pins
    assert matched == info["live"]
    assert all(out[q] == inp[q] for q in info["rails"] | info["sites"])
    assert np.count_nonzero(inp == info["marker"]) - np.count_nonzero(out == info["marker"]) == len(matched)
    for color in {p["color"] for p in info["pieces"]}:
        assert np.count_nonzero(inp == color) == np.count_nonzero(out == color)
    centers = _activation(inp.shape, info["runs"], info["slope"], "centers")
    consecutive = _activation(inp.shape, info["runs"], info["slope"], "consecutive")
    nearer = False
    for i, p in enumerate(info["pieces"]):
        distance = sum(abs(x) for x in positions[i])
        other_bodies = set().union(*(body for j, body in enumerate(bodies) if j != i))
        nearer |= any(sum(abs(x) for x in d) < distance and pins - info["live"]
                      and not body & other_bodies
                      for d, body, pins in _placements(info, p, info["sites"]))
    masks = [_normalized(p["pins"]) for p in info["pieces"]]
    # Removing finite endpoints turns the field into an infinite strip. A
    # stationary site in that strip but outside the actual field teaches why
    # endpoints, rather than just the two outer rail lines, matter.
    qlo, qhi = min(x[0] for x in info["runs"]), max(x[0] for x in info["runs"])
    endpoint = any(qlo < c - info["slope"] * r < qhi
                   for r, c in info["sites"] - matched)
    local_counts = [len(_placements(info, p, info["live"])) for p in info["pieces"]]
    return dict(shape=inp.shape, roles=(info["background"], info["rail_color"], info["marker"]),
                pieces=len(bodies), rails=len(info["runs"]), slope=info["slope"],
                pin_counts=tuple(sorted(len(m) for m in masks)), keys=tuple(masks),
                body_shapes=tuple(_normalized(p["body"]) for p in info["pieces"]),
                body_boxes=tuple((b[2]-b[0]+1, b[3]-b[1]+1) for b in map(_box, bodies)),
                continuous=any(not centers[q] for q in matched),
                skip=any(not consecutive[q] for q in matched),
                overhang=any(not info["active"][q] for body in bodies for q in body),
                nearer_dormant=bool(nearer),
                equal_area_keys=any(len(a) == len(b) and a != b
                                    for a, b in combinations(masks, 2)),
                reused_colors=len({p["color"] for p in info["pieces"]}) < len(bodies),
                endpoint=bool(endpoint),
                locally_ambiguous=any(n > 1 for n in local_counts),
                dormant=len(info["sites"] - matched), states=states,
                edge=any(r in (0, inp.shape[0]-1) or c in (0, inp.shape[1]-1)
                         for r, c in info["rails"]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    roles = (palette[0], palette[1], palette[2], palette[3:])
    puzzle, seen = {"train": [], "test": []}, set()
    taught_local_ambiguity = False
    for split, number in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(number):
            count = (2 if index == 0 else 3) if split == "train" and index < 2 else int(rng.integers(2, 5))
            if split == "test":
                count = int(rng.integers(3, 5))
                pin_counts = [2, 3] + [int(rng.integers(2, 4)) for _ in range(count - 2)]
                rng.shuffle(pin_counts)
                rail_count = int(rng.integers(3, 6))
            else:
                pin_counts = [2 if index % 2 == 0 else 3] * count
                rail_count = 2 if index == 0 else int(rng.integers(3, 6))
            skip = split == "train" and index == 1
            minimum = 22 if count == 2 else (25 if split == "test" else 24)
            for attempt in range(120):
                pair = _construct(rng, roles, count, pin_counts, rail_count,
                                  skip, split == "train" and index == 0,
                                  minimum if attempt < 90 else 30)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                feature = _features(pair["input"], pair["output"])
                if not feature["nearer_dormant"] or not feature["equal_area_keys"]:
                    continue
                if split == "train" and index == 0 and not feature["endpoint"]:
                    continue
                # Do not introduce a need for global conflict resolution only
                # in a query whose demonstrations all had unique local fits.
                if split == "test" and feature["locally_ambiguous"] and not taught_local_ambiguity:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    taught_local_ambiguity |= feature["locally_ambiguous"]
                break
            else:
                raise RuntimeError("bounded harbor construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    records, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            assert not np.array_equal(inp, out)
            record = _features(inp, out)
            assert record["dormant"] >= 2
            records[split].append(record)
    assert len({r["roles"] for rs in records.values() for r in rs}) == 1
    # Check visible witnesses, not merely the requested construction category.
    for name in ("continuous", "skip", "overhang", "nearer_dormant", "equal_area_keys",
                 "endpoint", "locally_ambiguous"):
        if any(r[name] for r in records["test"]):
            assert any(r[name] for r in records["train"]), "missing training witness: " + name
    assert any(r["continuous"] for r in records["train"])
    assert any(r["skip"] for r in records["train"])
    assert any(r["overhang"] for r in records["train"])
    assert any(r["nearer_dormant"] for r in records["train"])
    assert any(r["equal_area_keys"] for r in records["train"])
    assert any(r["endpoint"] for r in records["train"])
    taught = {n for r in records["train"] for n in r["pin_counts"]}
    assert {n for r in records["test"] for n in r["pin_counts"]} <= taught


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    # Analytic continuous-interval fixture, separate from random construction.
    rails = {(0, 2), (1, 1), (2, 0), (2, 4), (3, 3), (4, 2)}
    runs = _runs(rails, -1)
    mask = _activation((5, 5), runs, -1)
    expected = {(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)}
    assert set(map(tuple, np.argwhere(mask).tolist())) == expected
    assert not _activation((5, 5), runs, -1, "centers")[1, 2]
    extended = runs + [(4, -2, 0)]
    assert _activation((5, 5), extended, -1)[1, 3]
    assert not _activation((5, 5), extended, -1, "consecutive")[1, 3]
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            records.extend(_features(p["input"], p["output"]) for p in puzzle[split])
    assert len(fingerprints) == 42
    assert {r["pieces"] for r in records} == {2, 3, 4}
    assert {r["rails"] for r in records} >= {2, 3, 4, 5}
    assert {r["slope"] for r in records} == {-1, 1}
    assert len({r["shape"] for r in records}) >= 20
    assert len({key for r in records for key in r["keys"]}) >= 40
    assert len({body for r in records for body in r["body_shapes"]}) >= 80
    assert len({r["roles"] for r in records}) >= 30
    assert any(r["reused_colors"] for r in records) and any(not r["reused_colors"] for r in records)
    assert any(r["edge"] for r in records) and any(not r["edge"] for r in records)
    assert any(r["locally_ambiguous"] for r in records)
    assert _fingerprint(generate_puzzle(17)) == _fingerprint(generate_puzzle(17))
    # Input-only inference must be equivariant to whole-scene symmetries and
    # arbitrary bijective palette changes; there is no literal anchor hue.
    sample = generate_puzzle(0)["test"][0]
    recolor = np.array([7, 2, 8, 0, 5, 1, 9, 4, 6, 3], dtype=np.int8)
    for turns in range(4):
        inp, out = np.rot90(sample["input"], turns), np.rot90(sample["output"], turns)
        assert np.array_equal(solve(recolor[inp]), recolor[out])
        assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(101, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Geometrically incorrect labels are rejected even with a legal schema.
    puzzle = generate_puzzle(0, 2, 1)
    info = _analyze(puzzle["test"][0]["input"])
    q = min(info["live"])
    puzzle["test"][0]["output"][q] = info["background"]
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased live connector accepted")
    return {"seeds": 42, "pairs": len(records),
            "shapes": len({r["shape"] for r in records}),
            "pin_keys": len({key for r in records for key in r["keys"]}),
            "body_masks": len({body for r in records for body in r["body_shapes"]}),
            "feature_counts": {name: sum(bool(r[name]) for r in records) for name in
                               ("continuous", "skip", "overhang", "nearer_dormant",
                                "equal_area_keys", "endpoint", "locally_ambiguous",
                                "reused_colors", "edge")}}
