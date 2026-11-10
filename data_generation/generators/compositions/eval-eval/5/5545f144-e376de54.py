"""Elastic waypoint journeys: ordinal calibration in a moving local frame.

Only NumPy and the standard library are used. The public solver reads the grid;
generation constructs labels from sampled directions and distances, not solve().
"""
from __future__ import annotations

import copy
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
COUNTS = (3, 5, 7, 9)


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _arrow(span, nose, solid, rear):
    # Right-facing, symmetric head with a unique projecting tip. Paired rear
    # extensions cannot introduce a second, backward-pointing tip.
    cells = {(0, -i) for i in range(nose)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - nose - depth) for r in rows)
    for extra in range(1, rear + 1):
        cells.update((r, 1 - nose - span - extra) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(dict.fromkeys(_arrow(s, n, solid, rear)
                           for s in (1, 2) for n in (1, 2)
                           for solid in (False, True) for rear in (0, 1)))


def _footprint(mask, tip, direction):
    return {(tip[0] + dr, tip[1] + dc)
            for dr, dc in (_rotate(p, direction - 1) for p in mask)}


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _identify_arrow(cells):
    # Geometry, not MASKS, determines tip and facing. This also admits readable
    # family members with arrow geometries the sampler does not enumerate.
    matches = []
    for tip in sorted(cells):
        for direction in range(4):
            offsets = {_rotate((r - tip[0], c - tip[1]), 1 - direction)
                       for r, c in cells}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if max(abs(r) for r, c in offsets) == 0:
                continue
            if {(-r, c) for r, c in offsets} != offsets:
                continue
            matches.append((tuple(sorted(offsets)), tip, direction))
    if len(matches) != 1:
        raise ValueError("arrow must have one geometrically identifiable tip")
    return matches[0]


def _ray(origin, point):
    dr, dc = point[0] - origin[0], point[1] - origin[1]
    if dr == 0 and dc != 0:
        return (1 if dc > 0 else 3), abs(dc)
    if dc == 0 and dr != 0:
        return (2 if dr > 0 else 0), abs(dr)
    return None


def _order_waypoints(tip, waypoints):
    points, directions, lengths = [tip], [], []
    remaining = set(waypoints)
    while remaining:
        rays = {}
        for q in remaining:
            ray = _ray(points[-1], q)
            if ray is not None:
                d, distance = ray
                rays.setdefault(d, []).append((distance, q))
        if len(rays) != 1:
            raise ValueError("exactly one occupied orthogonal ray is required")
        d, candidates = next(iter(rays.items()))
        distance, q = min(candidates)
        points.append(q)
        directions.append(d)
        lengths.append(distance)
        remaining.remove(q)
    return points, directions, lengths


def _replay(start, directions, distance):
    points = [start]
    for d in directions:
        dr, dc = DIRS[d]
        r, c = points[-1]
        points.append((r + distance * dr, c + distance * dc))
    return points


def _bounds(cells):
    lo = (min(r for r, c in cells), min(c for r, c in cells))
    hi = (max(r for r, c in cells), max(c for r, c in cells))
    return lo, hi


def _within(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _check_grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any(g < 0) or np.any(g > 9)):
        raise ValueError("expected a 1..30 integer grid of ARC colors")
    return g


def _decode(input_grid):
    grid = _check_grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or max(counts) <= grid.size / 2:
        raise ValueError("need a majority background and two geometric ink roles")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for ink in colors:
        ink = int(ink)
        if ink == bg:
            continue
        parts = _components(map(tuple, np.argwhere(grid == ink)))
        if len(parts) == 1 and len(parts[0]) >= 3:
            candidates.append((ink, parts[0]))
    if len(candidates) != 1:
        raise ValueError("one connected arrow color is required")
    ink, cells = candidates[0]
    marker = next(int(c) for c in colors if c not in (bg, ink))
    parts = _components(map(tuple, np.argwhere(grid == marker)))
    if len(parts) not in COUNTS or any(len(p) != 1 for p in parts):
        raise ValueError("need 3, 5, 7, or 9 isolated waypoint dots")
    mask, tip, initial = _identify_arrow(cells)
    waypoints = {next(iter(p)) for p in parts}
    if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1
           for q in waypoints for r, c in cells):
        raise ValueError("initial arrow and waypoints must be visibly separate")
    points, directions, lengths = _order_waypoints(tip, waypoints)
    if directions[0] != initial or min(lengths) < 3:
        raise ValueError("first leg follows the tip; schematic distances are at least 3")
    reference = lengths[len(lengths) // 2]
    if any(-c + 1 >= reference or abs(r) + 1 >= reference for r, c in mask):
        raise ValueError("arrow depth and half-width must leave a gap before other station stamps")
    corrected = _replay(tip, directions, reference)
    if len(set(corrected)) != len(corrected):
        raise ValueError("calibrated stations must not revisit a station")
    for journey in (points, corrected):
        for i, p in enumerate(journey):
            d = initial if i == 0 else directions[i - 1]
            if not _within(_footprint(mask, p, d), grid.shape):
                raise ValueError("complete schematic and replayed arrows must stay inside the canvas")
    return dict(shape=grid.shape, bg=bg, ink=ink, marker=marker, mask=mask,
                tip=tip, initial=initial, points=points, directions=directions,
                lengths=lengths, reference=reference, corrected=corrected)


def _draw_trace(shape, palette, mask, points, directions):
    bg, ink, marker = palette
    out = np.full(shape, bg, dtype=np.int8)
    # Union of axial trace cells. Crossings of the same trace are idempotent.
    for a, b, d in zip(points, points[1:], directions):
        dr, dc = DIRS[d]
        distance = abs(b[0] - a[0]) + abs(b[1] - a[1])
        for step in range(distance + 1):
            out[a[0] + step * dr, a[1] + step * dc] = marker
    # Retain visible station identity, including the original tip as the start.
    for r, c in points[:-1]:
        out[r, c] = ink
    for r, c in _footprint(mask, points[-1], directions[-1]):
        out[r, c] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _decode(input_grid)
    return _draw_trace(a["shape"], (a["bg"], a["ink"], a["marker"]),
                       a["mask"], a["corrected"], a["directions"])


def _draw_input(shape, palette, mask, points, initial):
    bg, ink, marker = palette
    inp = np.full(shape, bg, dtype=np.int8)
    for r, c in points[1:]:
        inp[r, c] = marker
    for r, c in _footprint(mask, points[0], initial):
        inp[r, c] = ink
    return inp


def _sample_lengths(rng, n, reference, witness, single_edit=False):
    others = [i for i in range(n) if i != n // 2]
    lengths = [reference] * n
    if single_edit:
        i = int(rng.choice(others))
        short = reference >= 6 and bool(rng.integers(2))
        lengths[i] = int(rng.integers(3, reference - 2) if short
                         else rng.integers(reference + 3, 13))
    elif witness:
        majority_short = bool(rng.integers(2))
        minority = int(rng.choice(others))
        for i in others:
            short = majority_short != (i == minority)
            lengths[i] = int(rng.integers(3, reference) if short
                             else rng.integers(reference + 1, 13))
    else:
        mode = int(rng.integers(5))  # mixed, extend, trim, one extend, one trim
        rng.shuffle(others)
        for j, i in enumerate(others):
            if mode >= 3 and j > 0 or mode < 3 and j > 1 and rng.random() < .22:
                continue
            short = mode in (1, 3) or (mode == 0 and (j == 0 or j > 1 and rng.random() < .5))
            if short and reference > 3:
                lengths[i] = int(rng.integers(3, reference))
            else:
                lengths[i] = int(rng.integers(reference + 1, 13))
    return lengths


def _grow_itinerary(rng, lengths, reference, mask, witness, long_test):
    """Bounded randomized backtracking, not a route bank or fixed spiral.

    Every added dot is checked against ALL earlier tips. It can be on an earlier
    tip's occupied ray only beyond that tip's prescribed nearest successor.
    """
    n, budget = len(lengths), [6000]
    straight_weight = float(rng.uniform(.7, 2.5))

    def visit(points, calibrated, directions, occupied, initial_cells):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if len(directions) == n:
            turns = [(b - a) % 4 for a, b in zip(directions, directions[1:]) if a != b]
            if not turns:
                return None
            if witness and (directions[0] == directions[-1] or not {1, 3} <= set(turns)
                            or not any(a == b for a, b in zip(directions, directions[1:]))):
                return None
            if long_test and len(turns) < 2:
                return None
            return points, calibrated, directions, occupied
        order = list(range(4))
        weights = np.array([straight_weight if directions and d == directions[-1] else 1.
                            for d in order])
        order = [int(d) for d in rng.choice(order, 4, replace=False, p=weights / weights.sum())]
        step = len(directions)
        for d in order:
            dr, dc = DIRS[d]
            q = (points[-1][0] + lengths[step] * dr,
                 points[-1][1] + lengths[step] * dc)
            z = (calibrated[-1][0] + reference * dr,
                 calibrated[-1][1] + reference * dc)
            if q in points or z in calibrated:
                continue
            if any(max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 1 for p in points[1:]):
                continue
            body = initial_cells if initial_cells is not None else _footprint(mask, (0, 0), d)
            if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1 for r, c in body):
                continue
            valid = True
            for p, successor in zip(points[:-1], points[1:]):
                ray = _ray(p, q)
                if ray is not None:
                    known = _ray(p, successor)
                    if ray[0] != known[0] or ray[1] <= known[1]:
                        valid = False
                        break
            if not valid:
                continue
            used = occupied | body | _footprint(mask, q, d) | _footprint(mask, z, d)
            lo, hi = _bounds(used)
            if any(hi[j] - lo[j] >= 30 for j in (0, 1)):
                continue
            answer = visit(points + [q], calibrated + [z], directions + [d], used, body)
            if answer is not None:
                return answer
        return None

    return visit([(0, 0)], [(0, 0)], [], set(), None)


def _make_pair(rng, n, reference, palette, witness=False, avoid_mask=None, long_test=False,
               single_edit=False):
    eligible = [m for m in MASKS if m != avoid_mask
                and max(-c for r, c in m) + 1 < reference
                and max(abs(r) for r, c in m) + 1 < reference]
    for _ in range(160):
        mask = eligible[int(rng.integers(len(eligible)))]
        lengths = _sample_lengths(rng, n, reference, witness, single_edit)
        if witness:
            # Distinguish ordinal middle from halfway through total travel.
            halfway = sum(lengths) / 2
            index = next(i for i, total in enumerate(np.cumsum(lengths)) if total >= halfway)
            if index == n // 2 or lengths[index] == reference:
                continue
        result = _grow_itinerary(rng, lengths, reference, mask, witness, long_test)
        if result is None:
            continue
        original, corrected, directions, occupied = result
        lo, hi = _bounds(occupied)
        box = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1)
        shape = tuple(int(rng.integers(b, 31)) for b in box)
        offset = tuple(int(rng.integers(shape[j] - box[j] + 1)) - lo[j] for j in (0, 1))
        original = [(r + offset[0], c + offset[1]) for r, c in original]
        corrected = [(r + offset[0], c + offset[1]) for r, c in corrected]
        inp = _draw_input(shape, palette, mask, original, directions[0])
        out = _draw_trace(shape, palette, mask, corrected, directions)
        return {"input": inp, "output": out}, mask
    raise RuntimeError("bounded route construction exhausted")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _fingerprint(grid):
    g = np.asarray(grid)
    return g.shape, tuple(int(c) for c in g.flat)


def _puzzle_key(puzzle):
    return tuple(tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    shared_palette = bool(rng.random() < .2)
    palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    first_reference = int(rng.integers(4, 8))
    second_reference = int(rng.choice([r for r in range(3, 9) if r != first_reference]))
    first_mask = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = split == "train" and i == 0
            n = 5 if witness else 3 if split == "train" and i == 1 else int(rng.choice((3, 5) if split == "train" else (7, 9)))
            reference = first_reference if witness else second_reference if split == "train" and i == 1 else int(rng.integers(3, 9))
            if not shared_palette:
                palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
            for _ in range(20):
                pair, mask = _make_pair(rng, n, reference, palette, witness,
                                        first_mask if split == "train" and i == 1 else None,
                                        split == "test", split == "train" and i == 1)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            if witness:
                first_mask = mask
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _decode(inp)
            expected = _draw_trace(a["shape"], (a["bg"], a["ink"], a["marker"]),
                                   a["mask"], a["corrected"], a["directions"])
            assert np.array_equal(out, expected), "incorrect calibrated trace or rigid terminal arrow"
            assert len(a["points"]) == len(a["corrected"])
            assert all(_ray(p, q) == (d, a["reference"])
                       for p, q, d in zip(a["corrected"], a["corrected"][1:], a["directions"]))
            terminal = _footprint(a["mask"], a["corrected"][-1], a["directions"][-1])
            actual_ink = {tuple(p) for p in np.argwhere(out == a["ink"])}
            assert actual_ink == terminal | set(a["corrected"][:-1]), "mask cells or station stamps lost"
            assert len(actual_ink) == len(a["mask"]) + len(a["directions"])
            output_components = _components(actual_ink)
            output_arrows = [c for c in output_components if len(c) > 1]
            assert output_arrows == [terminal], "station stamp touches or changes the visible arrow"
            assert sum(len(c) == 1 for c in output_components) == len(a["directions"])
            for p in a["corrected"][:-1]:
                assert out[p] == a["ink"]
            for p in _footprint(a["mask"], a["corrected"][-1], a["directions"][-1]):
                assert out[p] == a["ink"]
            # Omitting B means tracing the uncalibrated route and moving to its
            # old endpoint. Every generated pair distinguishes that ablation.
            uncalibrated = _draw_trace(a["shape"], (a["bg"], a["ink"], a["marker"]),
                                       a["mask"], a["points"], a["directions"])
            assert not np.array_equal(out, uncalibrated), "calibration has no observable effect"
            analyses[split].append(a)
    train, test = analyses["train"], analyses["test"]
    assert len({a["reference"] for a in train}) >= 2
    assert len({a["mask"] for a in train}) >= 2
    assert {len(a["directions"]) for a in train} == {3, 5}
    assert all(len(a["directions"]) in (7, 9) for a in test)
    witness = train[0]
    ref, ls = witness["reference"], witness["lengths"]
    assert min(ls) < ref < max(ls) and sorted(ls)[len(ls) // 2] != ref
    assert witness["initial"] != witness["directions"][-1], "training must reveal rotation"
    halfway = sum(ls) / 2
    halfway_index = next(i for i, total in enumerate(np.cumsum(ls)) if total >= halfway)
    assert halfway_index != len(ls) // 2 and ls[halfway_index] != ref
    second = train[1]
    changes = [abs(l - second["reference"]) for l in second["lengths"] if l != second["reference"]]
    assert len(changes) == 1 and changes[0] >= 3, "second example distinguishes averaging"
    # Keeping the initial facing at the calibrated endpoint erases or invents
    # actual terminal cells. A is observable separately from B's resize.
    assert _footprint(witness["mask"], witness["corrected"][-1], witness["initial"]) != _footprint(
        witness["mask"], witness["corrected"][-1], witness["directions"][-1])


def measure_pair(pair):
    """Compact structural measurements recovered exclusively from the input."""
    a = _decode(pair["input"])
    ds, lengths = a["directions"], a["lengths"]
    turns = [(b - d) % 4 for d, b in zip(ds, ds[1:]) if d != b]
    return dict(shape=list(a["shape"]), count=len(ds), reference=a["reference"],
                lengths=lengths, directions=ds, mask=a["mask"],
                palette=[a["bg"], a["ink"], a["marker"]],
                mixed_turns=1 in turns and 3 in turns,
                straight_steps=sum(d == b for d, b in zip(ds, ds[1:])),
                extension=sum(l < a["reference"] for l in lengths),
                truncation=sum(l > a["reference"] for l in lengths),
                nonmedian=sorted(lengths)[len(lengths) // 2] != a["reference"],
                endpoint_moved=a["points"][-1] != a["corrected"][-1])


def self_test():
    rng = np.random.default_rng(92831)
    metrics, keys = {"train": [], "test": []}, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        keys.add(_puzzle_key(puzzle))
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ("train", "test"):
            for i, pair in enumerate(puzzle[split]):
                validate_input = pair["input"].copy()
                before = validate_input.copy()
                assert np.array_equal(solve(validate_input), pair["output"])
                assert np.array_equal(validate_input, before), "solve mutated input"
                assert np.array_equal(solve(validate_input.tolist()), pair["output"])
                metrics[split].append(measure_pair(pair))
                if i == 0:
                    perm = rng.permutation(10)
                    assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
                    for k in range(4):
                        for mirror in (False, True):
                            inp, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                            if mirror:
                                inp, out = np.fliplr(inp), np.fliplr(out)
                            assert np.array_equal(solve(inp), out)
    assert len(keys) == 42
    for split, rows in metrics.items():
        assert len({tuple(r["shape"]) for r in rows}) >= 20
        assert len({tuple(r["directions"]) for r in rows}) >= 20
        assert len({tuple(r["lengths"]) for r in rows}) >= 25
        assert len({r["reference"] for r in rows}) >= 4
        assert len({r["mask"] for r in rows}) >= 8
        assert {r["directions"][0] for r in rows} == set(range(4))
        assert {r["directions"][-1] for r in rows} == set(range(4))
        assert any(r["mixed_turns"] and r["straight_steps"] for r in rows)
        assert any(r["extension"] and r["truncation"] for r in rows)
        assert any(r["endpoint_moved"] for r in rows)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42 + 10 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    try:
        _order_waypoints((0, 0), {(0, 4), (4, 0), (4, 4)})
    except ValueError:
        pass
    else:
        raise AssertionError("junction accepted")
    p = generate_puzzle(123, 2, 1)
    broken = copy.deepcopy(p)
    a = _decode(broken["test"][0]["input"])
    broken["test"][0]["output"][a["corrected"][0]] = a["bg"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged station accepted")
    # An unseen readable arrow geometry must be carried as actual cells.
    unseen = tuple(p for p in _arrow(2, 1, True, 0) if p != (0, -2))
    assert unseen not in MASKS
    raw = [(7, 7), (7, 11), (13, 11), (13, 18)]
    ds = [1, 2, 1]
    fixed = _replay(raw[0], ds, 6)
    inp = _draw_input((24, 26), (0, 3, 8), unseen, raw, 1)
    assert np.array_equal(solve(inp), _draw_trace(inp.shape, (0, 3, 8), unseen, fixed, ds))
    # Guide ink must not camouflage a substituted or filled arrow mask.
    hollow = _arrow(1, 1, False, 0)
    solid = _arrow(1, 1, True, 0)
    inp_h = _draw_input(inp.shape, (0, 3, 8), hollow, raw, 1)
    inp_s = _draw_input(inp.shape, (0, 3, 8), solid, raw, 1)
    assert not np.array_equal(solve(inp_h), solve(inp_s))
    assert solve(inp_h)[fixed[-1][0], fixed[-1][1] - 1] == 8
    assert solve(inp_s)[fixed[-1][0], fixed[-1][1] - 1] == 3
    equal_area = _arrow(1, 2, False, 0)
    assert len(equal_area) == len(solid) and equal_area != solid
    inp_e = _draw_input(inp.shape, (0, 3, 8), equal_area, raw, 1)
    assert not np.array_equal(solve(inp_e), solve(inp_s))
    # The rule extends to reference distances outside the sampler's 3..8 range.
    large_raw = [(5, 5), (5, 8), (14, 8), (14, 20)]
    large_inp = _draw_input((20, 26), (0, 3, 8), solid, large_raw, 1)
    assert _decode(large_inp)["reference"] == 9
    assert np.array_equal(solve(large_inp), _draw_trace(large_inp.shape, (0, 3, 8), solid,
                                                     _replay(large_raw[0], ds, 9), ds))
    # Perturb the visible middle distance and translate its suffix, preserving
    # directions; calibration must change without any hidden generation state.
    changed = raw[:2] + [(r + 1, c) for r, c in raw[2:]]
    inp2 = _draw_input(inp.shape, (0, 3, 8), unseen, changed, 1)
    assert _decode(inp2)["reference"] == 7
    assert not np.array_equal(solve(inp2), solve(inp))
    return {"seeds": 42, "distinct_puzzles": len(keys),
            "shapes": {s: len({tuple(m["shape"]) for m in rows}) for s, rows in metrics.items()},
            "direction_sequences": {s: len({tuple(m["directions"]) for m in rows}) for s, rows in metrics.items()},
            "length_vectors": {s: len({tuple(m["lengths"]) for m in rows}) for s, rows in metrics.items()},
            "mask_geometries": {s: len({m["mask"] for m in rows}) for s, rows in metrics.items()}}
