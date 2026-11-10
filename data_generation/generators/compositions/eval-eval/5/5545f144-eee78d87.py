"""Roving four-tile stamp: route-oriented periodic printing and overlap color.

Only standard Python and NumPy are used.  solve reads geometry and roles from
its argument; generation parameters and seeds never enter the decoder.
"""

from __future__ import annotations

import copy
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _arrow(span, extension, solid, tails):
    # Canonical arrow faces right; (0,0) is its unique projecting tip.
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if solid else (-depth, depth)
        cells.update((r, 1 - extension - depth) for r in rows)
    for tail in range(1, tails + 1):
        cells.update((r, 1 - extension - span - tail) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(dict.fromkeys(_arrow(s, e, solid, t)
                           for s in (1, 2) for e in (1, 2)
                           for solid in (False, True) for t in (0, 1, 2)))


def _period(mask):
    return max(r for r, c in mask) - min(r for r, c in mask) + 1, 1 - min(c for r, c in mask)


def _single_cells(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, direction - 1) for p in mask)}


def _stamp_offsets(mask, direction):
    height, width = _period(mask)
    # Four complete bounding patches, including their unpainted carrier gaps.
    return tuple(_rotate((r + i * height, c - j * width), direction - 1)
                 for i in range(2) for j in range(2) for r, c in mask)


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _components(cells):
    pending, result = set(cells), []
    while pending:
        component = {pending.pop()}
        stack = list(component)
        while stack:
            r, c = stack.pop()
            found = {(r + dr, c + dc) for dr in (-1, 0, 1)
                     for dc in (-1, 0, 1)} & pending
            pending -= found
            component |= found
            stack.extend(sorted(found))
        result.append(component)
    return result


def _identify_arrow(cells):
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
        raise ValueError("arrow must have one symmetry axis with one projecting tip")
    return matches[0]


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if (dr == 0) == (dc == 0):
        raise ValueError("route points must be distinct and orthogonally aligned")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _route(tip, waypoints):
    remaining, points, directions = set(waypoints), [tip], []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        if len(rays) != 1:
            raise ValueError("exactly one occupied ray is required at every stop")
        direction, candidates = next(iter(rays.items()))
        tip = min(candidates, key=lambda q: abs(q[0] - tip[0]) + abs(q[1] - tip[1]))
        remaining.remove(tip)
        points.append(tip)
        directions.append(direction)
    return points, directions


def _array(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer ARC colors")
    return a.astype(np.int8, copy=True)


def _decode(grid):
    grid = _array(grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or counts.max() <= grid.size // 2:
        raise ValueError("three roles and a strict-majority background are required")
    bg = int(colors[np.argmax(counts)])
    roles = {}
    for color in colors:
        if color == bg:
            continue
        cells = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        comps = _components(cells)
        roles[int(color)] = (cells, comps)
    arrow_colors = [c for c, (cells, comps) in roles.items()
                    if len(comps) == 1 and len(cells) > 1]
    if len(arrow_colors) != 1:
        raise ValueError("one color must form a single pointed object")
    ink = arrow_colors[0]
    waypoint = next(c for c in roles if c != ink)
    if not all(len(c) == 1 for c in roles[waypoint][1]):
        raise ValueError("waypoints must be isolated singleton cells")
    mask, tip, initial = _identify_arrow(roles[ink][0])
    points, directions = _route(tip, roles[waypoint][0])
    if len(directions) < 2 or directions[0] != initial:
        raise ValueError("at least two waypoints; first leg follows the arrow")
    facings = [initial] + directions
    for p, d in zip(points, facings):
        if not _inside(_single_cells(mask, p, d), grid.shape):
            raise ValueError("the moving arrow itself must stay inside the board")
    return dict(shape=grid.shape, bg=bg, ink=ink, waypoint=waypoint,
                mask=mask, points=points, directions=directions, facings=facings)


def _coverage(info, copies=True, rotate=True, terminal_only=False, sweep=False,
              skip_initial=False):
    """Input-derived occupancy counts; variants support explicit shortcut audits."""
    counts = np.zeros(info["shape"], dtype=np.int16)
    stops = list(zip(info["points"], info["facings"]))
    if terminal_only:
        stops = stops[-1:]
    if skip_initial:
        stops = stops[1:]
    if sweep:
        stops = [stops[0]]
        for a, b, d in zip(info["points"], info["points"][1:], info["directions"]):
            dr, dc = DIRS[d]
            distance = abs(b[0] - a[0]) + abs(b[1] - a[1])
            stops.extend(((a[0] + k * dr, a[1] + k * dc), d)
                         for k in range(1, distance + 1))
    for p, d in stops:
        if not rotate:
            d = info["facings"][0]
        offsets = (_stamp_offsets(info["mask"], d) if copies else
                   tuple(_rotate(q, d - 1) for q in info["mask"]))
        for dr, dc in offsets:
            r, c = p[0] + dr, p[1] + dc
            if 0 <= r < counts.shape[0] and 0 <= c < counts.shape[1]:
                counts[r, c] += 1
    return counts


def _color_counts(counts, info):
    out = np.full(info["shape"], info["bg"], dtype=np.int8)
    out[counts == 1] = info["ink"]
    out[counts >= 2] = info["waypoint"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Print at the initial stop and every arrival, with occupied-cell saturation."""
    info = _decode(input_grid)
    return _color_counts(_coverage(info), info)


def _construct_output(shape, bg, ink, waypoint, mask, points, facings):
    # Construction by four physically translated copies. No input decoder and
    # no generator seed are used by solve; this path checks the renderer too.
    layers = []
    h, w = _period(mask)
    for p, d in zip(points, facings):
        layer = np.zeros(shape, dtype=np.int16)
        for i in range(2):
            for j in range(2):
                shift = _rotate((i * h, -j * w), d - 1)
                origin = p[0] + shift[0], p[1] + shift[1]
                for r, c in _single_cells(mask, origin, d):
                    if 0 <= r < shape[0] and 0 <= c < shape[1]:
                        layer[r, c] += 1
        layers.append(layer)
    total = np.sum(layers, axis=0)
    out = np.full(shape, bg, dtype=np.int8)
    out[total == 1] = ink
    out[total >= 2] = waypoint
    return out


def _grow(rng, shape, mask, length, straight=False, full_initial=False):
    # Incremental route geometry, not a bank of full paths or parent calls.
    for _ in range(50):
        initial = int(rng.integers(4))
        starts = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                  if _inside(_single_cells(mask, (r, c), initial), shape)
                  and (not full_initial or _inside(
                      {(r + dr, c + dc) for dr, dc in _stamp_offsets(mask, initial)}, shape))]
        if not starts:
            continue
        points = [starts[int(rng.integers(len(starts)))]]
        ds = []
        initial_cells = _single_cells(mask, points[0], initial)
        while len(ds) < length:
            options = {}
            choices = [initial] if not ds or straight else range(4)
            for d in choices:
                dr, dc = DIRS[d]
                r, c = points[-1]
                for distance in range(2, max(shape)):
                    q = (r + dr * distance, c + dc * distance)
                    if not _inside(_single_cells(mask, q, d), shape):
                        continue
                    if q in initial_cells or q in points:
                        continue
                    if any(max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 1 for p in points):
                        continue
                    valid = True
                    for p, nxt in zip(points[:-1], points[1:]):
                        if p[0] == q[0] or p[1] == q[1]:
                            if _direction(p, q) != _direction(p, nxt):
                                valid = False
                                break
                            if abs(q[0] - p[0]) + abs(q[1] - p[1]) <= abs(nxt[0] - p[0]) + abs(nxt[1] - p[1]):
                                valid = False
                                break
                    if valid:
                        options.setdefault(d, []).append(q)
            if not options:
                break
            dirs = sorted(options)
            weights = np.array([0.65 if ds and d == ds[-1] else 1.0 for d in dirs])
            d = int(rng.choice(dirs, p=weights / weights.sum()))
            points.append(options[d][int(rng.integers(len(options[d])))])
            ds.append(d)
        if len(ds) == length and (straight or ds[0] != ds[-1]):
            return points, [initial] + ds
    return None


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counterexamples(info, answer):
    result = {name: not np.array_equal(_color_counts(_coverage(info, **kwargs), info), answer)
            for name, kwargs in (("single_copy", {"copies": False}),
                                 ("no_rotation", {"rotate": False}),
                                 ("terminal_only", {"terminal_only": True}),
                                 ("sweep", {"sweep": True}),
                                 ("omit_initial", {"skip_initial": True}))}
    outgoing = dict(info, facings=info["directions"] + [info["directions"][-1]])
    result["outgoing"] = not np.array_equal(_color_counts(_coverage(outgoing), info), answer)
    result["exactly_two"] = bool(np.any(_coverage(info) >= 3))
    return result


def _make_pair(rng, mask, length, palette, straight=False, full_initial=False,
               need_triple=False):
    for _ in range(12):
        shape = tuple(map(int, rng.integers(14, 31, size=2)))
        route = _grow(rng, shape, mask, length, straight, full_initial)
        if route is None:
            continue
        points, facings = route
        bg, ink, waypoint = palette
        inp = np.full(shape, bg, dtype=np.int8)
        for r, c in _single_cells(mask, points[0], facings[0]):
            inp[r, c] = ink
        for r, c in points[1:]:
            inp[r, c] = waypoint
        out = _construct_output(shape, bg, ink, waypoint, mask, points, facings)
        info = _decode(inp)
        counts = _coverage(info)
        witnesses = _counterexamples(info, out)
        if not (np.any(counts == 1) and np.any(counts >= 2) and np.any(counts == 0)):
            continue
        if not all(witnesses[k] for k in ("single_copy", "terminal_only", "sweep")):
            continue
        if not straight and not witnesses["no_rotation"]:
            continue
        if need_triple and not (witnesses["exactly_two"] and witnesses["outgoing"]):
            continue
        assert np.array_equal(_color_counts(counts, info), out)
        return {"input": inp, "output": out}
    # Guaranteed roomy monotone route, with no junctions. It is a fallback,
    # not the normal route distribution, and still satisfies all witnesses.
    shape = (30, 30)
    h, w = _period(mask)
    points = [(h // 2 + 2, 2 * w - 1)]
    ds = []
    for i in range(length):
        d = 1 if straight or i % 2 == 0 else 2
        dr, dc = DIRS[d]
        points.append((points[-1][0] + 2 * dr, points[-1][1] + 2 * dc))
        ds.append(d)
    facings = [1] + ds
    bg, ink, waypoint = palette
    inp = np.full(shape, bg, dtype=np.int8)
    for r, c in _single_cells(mask, points[0], 1):
        inp[r, c] = ink
    for r, c in points[1:]:
        inp[r, c] = waypoint
    out = _construct_output(shape, bg, ink, waypoint, mask, points, facings)
    return {"input": inp, "output": out}


def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                    (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    shared = bool(rng.random() < 0.25)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    train_masks = []
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            pool = [m for m in MASKS if m not in train_masks] if split == "test" or i == 1 else list(MASKS)
            if split == "train" and i == 1:
                pool = [m for m in pool if _period(m) != _period(train_masks[0])]
            mask = pool[int(rng.integers(len(pool)))]
            if split == "train":
                train_masks.append(mask)
            length = (2 if i == 0 else 4 if i == 1 else int(rng.integers(2, 5))) if split == "train" else int(rng.integers(5, 9))
            if not shared:
                palette = tuple(map(int, rng.choice(10, 3, replace=False)))
            for _ in range(20):
                pair = _make_pair(rng, mask, length, palette,
                                  straight=split == "train" and i == 0,
                                  full_initial=split == "train" and i == 0,
                                  need_triple=split == "train" and i == 1)
                if _fingerprint(pair["input"]) not in seen:
                    break
            else:
                # Finite role-permutation fallback guarantees a new input;
                # at most 13 previous inputs versus 720 distinct assignments.
                old = _decode(pair["input"])
                done = False
                for bg in range(10):
                    for ink in range(10):
                        for wp in range(10):
                            if len({bg, ink, wp}) != 3:
                                continue
                            mapping = np.arange(10, dtype=np.int8)
                            mapping[[old["bg"], old["ink"], old["waypoint"]]] = [bg, ink, wp]
                            candidate = {k: mapping[v] for k, v in pair.items()}
                            if _fingerprint(candidate["input"]) not in seen:
                                pair, done = candidate, True
                                break
                        if done:
                            break
                    if done:
                        break
                assert done
            seen.add(_fingerprint(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(inp)
            coverage = _coverage(info)
            assert np.array_equal(out, _color_counts(coverage, info)), "incorrect print field"
            assert np.array_equal(out == info["bg"], coverage == 0), "carrier gaps changed"
            assert np.array_equal(out == info["ink"], coverage == 1)
            assert np.array_equal(out == info["waypoint"], coverage >= 2), "wrong overlap cells"
            assert all(np.any(coverage == k) for k in (0, 1)) and np.any(coverage >= 2)
            assert not np.array_equal(inp, out)
            witnesses = _counterexamples(info, out)
            assert witnesses["single_copy"] and witnesses["terminal_only"] and witnesses["sweep"]
            info["witnesses"] = witnesses
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    assert len({i["mask"] for i in train}) >= 2
    assert len({_period(i["mask"]) for i in train}) >= 2
    assert len({len(i["directions"]) for i in train}) >= 2
    assert any(i["witnesses"]["no_rotation"] for i in train)
    assert train[0]["witnesses"]["omit_initial"]
    assert train[1]["witnesses"]["outgoing"] and train[1]["witnesses"]["exactly_two"]
    assert len(train[0]["directions"]) == 2
    assert len(set(train[0]["directions"])) == 1
    assert _inside({(train[0]["points"][0][0] + r, train[0]["points"][0][1] + c)
                    for r, c in _stamp_offsets(train[0]["mask"], train[0]["facings"][0])}, train[0]["shape"])
    assert max(len(i["directions"]) for i in train) < min(len(i["directions"]) for i in test)
    assert not {i["mask"] for i in test} & {i["mask"] for i in train}


def _puzzle_key(puzzle):
    return tuple(tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def _route_signature(info):
    origin = info["points"][0]
    variants = []
    for mirror in (1, -1):
        for d in range(4):
            variants.append(tuple(_rotate((r - origin[0], mirror * (c - origin[1])), d)
                                  for r, c in info["points"]))
    return min(variants)


def self_test():
    """Bounded 42-seed structural, semantic, mutation and diversity checks."""
    metrics = {"puzzles": set(), "masks": set(), "routes": set(), "shapes": set(),
               "periods": set(), "palettes": set(), "initial": set(), "final": set(),
               "lengths": set(), "overlap_counts": set()}
    mixed = clipped = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        metrics["puzzles"].add(_puzzle_key(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            info = _decode(pair["input"])
            assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
            counts = _coverage(info)
            metrics["masks"].add(info["mask"])
            metrics["routes"].add(_route_signature(info))
            metrics["shapes"].add(info["shape"])
            metrics["periods"].add(_period(info["mask"]))
            metrics["palettes"].add((info["bg"], info["ink"], info["waypoint"]))
            metrics["initial"].add(info["facings"][0])
            metrics["final"].add(info["facings"][-1])
            metrics["lengths"].add(len(info["directions"]))
            metrics["overlap_counts"].add(int(np.count_nonzero(counts >= 2)))
            turns = {(b - a) % 4 for a, b in zip(info["directions"], info["directions"][1:])}
            mixed += {1, 3} <= turns
            clipped += any(not _inside({(p[0] + r, p[1] + c) for r, c in _stamp_offsets(info["mask"], d)}, info["shape"])
                           for p, d in zip(info["points"], info["facings"]))
    assert len(metrics["puzzles"]) == 42
    assert len(metrics["masks"]) >= 18 and len(metrics["routes"]) >= 150
    assert len(metrics["shapes"]) >= 80 and len(metrics["periods"]) >= 8
    assert len(metrics["palettes"]) >= 80 and len(metrics["overlap_counts"]) >= 40
    assert metrics["initial"] == metrics["final"] == set(range(4))
    assert metrics["lengths"] == {2, 3, 4, 5, 6, 7, 8}
    assert mixed >= 25 and clipped >= 80
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(4200 + 10 * a + b, a, b))
    assert _puzzle_key(generate_puzzle(77, 7, 3)) == _puzzle_key(generate_puzzle(77, 7, 3))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Regression for exact tile placement, gaps, saturation, and rotations.
    mask = _arrow(1, 2, False, 0)
    assert set(_stamp_offsets(mask, 1)) == {(r + 3 * i, c - 3 * j)
                                          for i in range(2) for j in range(2) for r, c in mask}
    assert len(set(_stamp_offsets(mask, 1))) == 4 * len(mask)
    for d in range(4):
        assert set(_stamp_offsets(mask, d)) == {_rotate(p, d - 1) for p in _stamp_offsets(mask, 1)}
    # Reject a junction instead of quietly applying forward priority.
    try:
        _route((5, 5), {(5, 8), (8, 5)})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous ray junction accepted")
    for mutation in ("carrier", "single_copy", "no_rotation", "terminal_only", "sweep", "no_overlap",
                     "exactly_two", "outgoing", "omit_initial"):
        bad = copy.deepcopy(generate_puzzle(123, 2, 1))
        pair = bad["train"][0 if mutation == "omit_initial" else 1]
        info = _decode(pair["input"])
        if mutation == "carrier":
            q = tuple(np.argwhere(pair["output"] == info["bg"])[0])
            pair["output"][q] = info["ink"]
        elif mutation == "no_overlap":
            pair["output"][pair["output"] == info["waypoint"]] = info["ink"]
        elif mutation == "exactly_two":
            pair["output"][_coverage(info) >= 3] = info["ink"]
        elif mutation == "outgoing":
            outgoing = dict(info, facings=info["directions"] + [info["directions"][-1]])
            pair["output"] = _color_counts(_coverage(outgoing), info)
        else:
            kwargs = {"single_copy": {"copies": False}, "no_rotation": {"rotate": False},
                      "terminal_only": {"terminal_only": True}, "sweep": {"sweep": True},
                      "omit_initial": {"skip_initial": True}}[mutation]
            pair["output"] = _color_counts(_coverage(info, **kwargs), info)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{mutation} corruption accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus legal-count and mutation checks)")
