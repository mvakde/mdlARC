"""Chroma docks: cardinal attraction determines local palette magnification.

Only NumPy and the standard library are used. solve never reads generation state.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _anchor_components(grid, color):
    anchors = []
    for cells in _components(grid == color):
        r, c = min(cells)
        if cells != {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}:
            return None
        anchors.append((r, c))
    return anchors or None


def _infer(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must be a strict majority")
    options = [(int(color), _anchor_components(grid, int(color)))
               for color in values if color != background]
    options = [(color, anchors) for color, anchors in options if anchors]
    if len(options) != 1:
        raise ValueError("anchor color is not uniquely identifiable")
    anchor_color, anchors = options[0]
    cargo_mask = (grid != background) & (grid != anchor_color)
    if (np.any(cargo_mask[:-1] & cargo_mask[1:])
            or np.any(cargo_mask[:, :-1] & cargo_mask[:, 1:])):
        raise ValueError("input cargo must be cardinally separated")
    cargo = [tuple(map(int, p)) for p in np.argwhere(cargo_mask)]
    if not cargo:
        raise ValueError("missing cargo")
    return background, anchor_color, anchors, cargo


def _candidates(grid, background, anchors, source):
    """Travel is to the exterior cell of the ORIGINAL 2x2 anchor."""
    r, c = source
    result = []
    for index, (ar, ac) in enumerate(anchors):
        possibilities = []
        if ar <= r < ar + 2:
            if c < ac:
                possibilities.append(("left", r - ar, (r, ac - 1)))
            elif c > ac + 1:
                possibilities.append(("right", r - ar, (r, ac + 2)))
        if ac <= c < ac + 2:
            if r < ar:
                possibilities.append(("above", c - ac, (ar - 1, c)))
            elif r > ar + 1:
                possibilities.append(("below", c - ac, (ar + 2, c)))
        for side, lane, destination in possibilities:
            dr = int(np.sign(destination[0] - r))
            dc = int(np.sign(destination[1] - c))
            distance = abs(destination[0] - r) + abs(destination[1] - c)
            path = tuple((r + dr * i, c + dc * i) for i in range(1, distance + 1))
            if all(grid[p] == background for p in path):
                result.append({"anchor": index, "side": side, "lane": lane,
                               "destination": destination, "distance": distance,
                               "path": path})
    return result


def _winner(candidates):
    if not candidates:
        return None
    minimum = min(item["distance"] for item in candidates)
    winners = [item for item in candidates if item["distance"] == minimum]
    return winners[0] if len(winners) == 1 else None


def _analyze(grid):
    background, anchor_color, anchors, cargo = _infer(grid)
    groups = [[] for _ in anchors]
    inert = []
    for source in cargo:
        candidates = _candidates(grid, background, anchors, source)
        chosen = _winner(candidates)
        item = {"source": source, "color": int(grid[source]),
                "candidates": candidates}
        if chosen is None:
            item["tied"] = bool(candidates)
            inert.append(item)
        else:
            item.update(chosen)
            groups[chosen["anchor"]].append(item)
    scales = [len({item["color"] for item in group}) for group in groups]
    if not all(1 <= k <= 3 for k in scales):
        raise ValueError("each anchor must receive a palette of 1..3 colors")
    for group in groups:
        ports = [(item["side"], item["lane"]) for item in group]
        if len(set(ports)) != len(ports):
            raise ValueError("two cargo cells want the same port")
    return {"background": background, "anchor_color": anchor_color,
            "anchors": anchors, "cargo": cargo, "groups": groups,
            "scales": scales, "inert": inert}


def _port_top_left(anchor, k, side, lane):
    r, c = anchor
    R, C = r - k + 1, c - k + 1
    return {"left": (R + lane * k, C - k),
            "right": (R + lane * k, C + 2 * k),
            "above": (R - k, C + lane * k),
            "below": (R + 2 * k, C + lane * k)}[side]


def _render(grid, facts, scales=None):
    scales = facts["scales"] if scales is None else scales
    background = facts["background"]
    output = grid.copy()
    for r, c in facts["anchors"]:
        output[r:r + 2, c:c + 2] = background
    for group in facts["groups"]:
        for item in group:
            output[item["source"]] = background
    occupied = output != background  # Inert cells must not be painted over.

    def paint(r, c, side, color):
        if not (0 <= r and 0 <= c and r + side <= grid.shape[0]
                and c + side <= grid.shape[1]):
            raise ValueError("growth would leave the canvas")
        if np.any(occupied[r:r + side, c:c + side]):
            raise ValueError("growth would overlap another output object")
        output[r:r + side, c:c + side] = color
        occupied[r:r + side, c:c + side] = True

    for anchor, group, k in zip(facts["anchors"], facts["groups"], scales):
        r, c = anchor
        paint(r - k + 1, c - k + 1, 2 * k, facts["anchor_color"])
        for item in group:
            r0, c0 = _port_top_left(anchor, k, item["side"], item["lane"])
            paint(r0, c0, k, item["color"])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive ownership, palettes, magnification and output from this input only."""
    grid = _grid(input_grid)
    return _render(grid, _analyze(grid))


def _footprint(anchor, k):
    r, c = anchor
    return r - 2 * k + 1, c - 2 * k + 1, r + 2 * k + 1, c + 2 * k + 1


def _reserve(shape, anchors, scales):
    mask = np.zeros(shape, dtype=bool)
    for anchor, k in zip(anchors, scales):
        r0, c0, r1, c1 = _footprint(anchor, k)
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            return None
        if mask[r0:r1, c0:c1].any():
            return None
        mask[r0:r1, c0:c1] = True
    return mask


def _layout(rng, mode, count, scales):
    if mode == "teaching_one":
        shape = tuple(map(int, rng.integers(16, 26, size=2)))
    elif mode == "teaching_two" or count == 3:
        shape = tuple(map(int, rng.integers(24, 31, size=2)))
    else:
        shape = tuple(map(int, rng.integers(17, 31, size=2)))
    if count == 3 and sum(k * k for k in scales) >= 22:
        # Three large reserved stencils need packing slack; no solver restriction.
        shape = tuple(map(int, rng.integers(28, 31, size=2)))
    if mode == "teaching_two":
        # Equal diagonal offsets give close H/V winners and an exact tie.
        gap = int(rng.integers(8, min(shape) - 7))
        r = int(rng.integers(1, shape[0] - 6 - gap))
        c = int(rng.integers(1, shape[1] - 6 - gap))
        anchors = [(r, c), (r + gap, c + gap)]
    else:
        anchors = []
        # Place the larger footprints first but retain the scale/anchor pairing.
        placed = {}
        for index in sorted(range(count), key=lambda i: -scales[i]):
            k = scales[index]
            for _ in range(100):
                coordinates = []
                for extent in shape:
                    low, high = 2 * k - 1, extent - 2 * k - 1
                    if count == 3 and rng.random() < 0.25:
                        coordinates.append(low if bool(rng.integers(2)) else high)
                    else:
                        coordinates.append(int(rng.integers(low, high + 1)))
                anchor = tuple(coordinates)
                trial = placed | {index: anchor}
                if _reserve(shape, list(trial.values()), [scales[i] for i in trial]) is not None:
                    placed = trial
                    break
            else:
                return None
        anchors = [placed[i] for i in range(count)]
    reserved = _reserve(shape, anchors, scales)
    if reserved is None:
        return None
    base = np.zeros(shape, dtype=np.int8)
    for r, c in anchors:
        base[r:r + 2, c:c + 2] = 1
    return base, anchors, reserved


def _make_geometry(rng, mode, scales, counts, require_cross=False):
    count = len(scales)
    for _ in range(240):
        layout = _layout(rng, mode, count, scales)
        if layout is None:
            continue
        base, anchors, reserved = layout
        sources = set()
        protected_paths = set()
        chosen = [[] for _ in anchors]
        used_ports = set()
        inert = []

        def can_add(source, candidates):
            if base[source] or source in sources or source in protected_paths:
                return False
            if any(abs(source[0] - r) + abs(source[1] - c) == 1 for r, c in sources):
                return False
            return not any(p in sources for item in candidates for p in item["path"])

        def add(source, candidates, winner):
            sources.add(source)
            protected_paths.update(p for item in candidates for p in item["path"])
            if winner is None:
                inert.append(source)
            else:
                item = dict(winner, source=source, candidates=candidates)
                chosen[winner["anchor"]].append(item)
                used_ports.add((winner["anchor"], winner["side"], winner["lane"]))

        if mode == "teaching_two":
            (ar, ac), (br, bc) = anchors
            required = [((ar, bc), 0), ((ar + 1, bc + 1), 1), ((br, ac + 1), None)]
            valid = True
            for source, expected in required:
                candidates = _candidates(base, 0, anchors, source)
                winner = _winner(candidates)
                if (not can_add(source, candidates)
                        or (winner is None) != (expected is None)
                        or (winner is not None and winner["anchor"] != expected)):
                    valid = False
                    break
                add(source, candidates, winner)
            if not valid:
                continue

        positions = set()
        for r, c in anchors:
            positions.update((rr, cc) for rr in (r, r + 1) for cc in range(base.shape[1]))
            positions.update((rr, cc) for cc in (c, c + 1) for rr in range(base.shape[0]))
        pool = []
        for source in sorted(positions):
            if base[source]:
                continue
            candidates = _candidates(base, 0, anchors, source)
            winner = _winner(candidates)
            if winner is not None:
                pool.append((source, candidates, winner))

        def pick(predicate):
            for index in rng.permutation(len(pool)):
                source, candidates, winner = pool[int(index)]
                a = winner["anchor"]
                port = (a, winner["side"], winner["lane"])
                if (len(chosen[a]) < counts[a] and port not in used_ports
                        and predicate(winner, candidates) and can_add(source, candidates)):
                    add(source, candidates, winner)
                    return True
            return False

        if require_cross and not pick(lambda w, cs: len({x["side"] in ("left", "right")
                                                       for x in cs}) == 2 and w["distance"] > 0):
            continue
        if mode == "teaching_one":
            if not all(pick(lambda w, cs, side=side: w["side"] == side)
                       for side in ("above", "below", "left", "right")):
                continue
        success = True
        for a in rng.permutation(count):
            a = int(a)
            while len(chosen[a]) < counts[a]:
                if not pick(lambda w, cs, a=a: w["anchor"] == a):
                    success = False
                    break
            if not success:
                break
        if not success:
            continue
        extra_inert = (1 if mode == "teaching_one" else int(rng.integers(0, 4)))
        for flat in rng.permutation(base.size):
            if extra_inert <= 0:
                break
            source = divmod(int(flat), base.shape[1])
            if reserved[source] or base[source]:
                continue
            candidates = _candidates(base, 0, anchors, source)
            if not candidates and can_add(source, candidates):
                add(source, candidates, None)
                extra_inert -= 1
        if extra_inert:
            continue
        return base, anchors, chosen, inert
    raise RuntimeError("bounded search could not build a compatible chroma-dock scene")


def _make_pair(rng, mode, roles, marker_palette, require_cross=False):
    if mode == "teaching_one":
        scales, counts = [2], [4]
    elif mode == "teaching_two":
        scales, counts = [1, 3], [4, 4]
    else:
        if mode == "test_three":
            n = 3
        else:
            n = int(rng.integers(1, 3) if mode == "extra" else rng.integers(2, 4))
        scales = list(map(int, rng.integers(1, 4, size=n)))
        counts = [int(rng.integers(max(2, k), 7)) for k in scales]
    base, anchors, groups, inert = _make_geometry(rng, mode, scales, counts, require_cross)
    background, anchor_color = roles
    grid = np.full(base.shape, background, dtype=np.int8)
    grid[base == 1] = anchor_color
    palettes = []
    for i, k in enumerate(scales):
        if mode == "teaching_two" and i == 1:
            shared = palettes[0][0]
            palette = [shared] + list(map(int, rng.choice(
                [c for c in marker_palette if c != shared], size=2, replace=False)))
        else:
            palette = list(map(int, rng.choice(marker_palette, size=k, replace=False)))
        palettes.append(palette)
        if mode == "teaching_two" and i == 1:
            # The required vertical crossing winner was added first. Its unique
            # color changes BOTH local factors if it is given to the other dock.
            tail = [palette[0], palette[2], palette[0]]
            rng.shuffle(tail)
            values = [palette[1]] + tail
        else:
            values = palette + [int(rng.choice(palette)) for _ in range(len(groups[i]) - k)]
            rng.shuffle(values)
        for item, color in zip(groups[i], values):
            grid[item["source"]] = color
    used_colors = set(c for palette in palettes for c in palette)
    outside = [c for c in marker_palette if c not in used_colors]
    for index, source in enumerate(inert):
        choices = outside if outside and index == 0 else marker_palette
        grid[source] = int(rng.choice(choices))
    if bool(rng.integers(2)):
        grid = grid.T
    if bool(rng.integers(2)):
        grid = np.flip(grid, axis=0)
    if bool(rng.integers(2)):
        grid = np.flip(grid, axis=1)
    grid = grid.copy()
    return {"input": grid, "output": solve(grid)}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    roles, marker_palette = colors[:2], colors[2:]
    modes = ["teaching_one", "teaching_two"] + ["extra"] * (num_train - 2)
    puzzle, seen = {"train": [], "test": []}, set()
    for section, plan in (("train", modes),
                          ("test", ["test_three"] + ["test_other"] * (num_test - 1))):
        for mode in plan:
            for _ in range(20):
                pair = _make_pair(rng, mode, roles, marker_palette,
                                  require_cross=mode == "test_three")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual_differs(grid, facts, kind):
    trial = copy.deepcopy(facts)
    if kind == "no_A":
        scales = [1] * len(facts["anchors"])
    elif kind == "global":
        n = len({int(grid[p]) for p in facts["cargo"]})
        scales = [n] * len(facts["anchors"])
    elif kind == "no_nearest":
        trial["groups"] = [[] for _ in facts["anchors"]]
        trial["inert"] = []
        for source in facts["cargo"]:
            candidates = _candidates(grid, facts["background"], facts["anchors"], source)
            if not candidates:
                trial["inert"].append({"source": source})
            for candidate in candidates:
                trial["groups"][candidate["anchor"]].append(
                    dict(candidate, source=source, color=int(grid[source])))
        scales = [len({item["color"] for item in group}) for group in trial["groups"]]
    else:
        raise ValueError(kind)
    try:
        return not np.array_equal(_render(grid, trial, scales), _render(grid, facts))
    except ValueError:
        return True  # The alternative demands an impossible overlapping/clipped output.


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    facts = _analyze(grid)
    _require(1 <= len(facts["anchors"]) <= 3, "invalid anchor count")
    _require(grid.shape == output.shape, "canvas dimensions changed")
    _require(np.array_equal(output, _render(grid, facts)), "wrong output")
    expected_counts = {facts["anchor_color"]: sum(4 * k * k for k in facts["scales"])}
    footprint = np.zeros(grid.shape, dtype=bool)

    def check_square(r, c, side, color):
        _require(0 <= r and 0 <= c and r + side <= grid.shape[0]
                 and c + side <= grid.shape[1], "square out of bounds")
        _require(not footprint[r:r + side, c:c + side].any(), "overlapping output squares")
        footprint[r:r + side, c:c + side] = True
        _require(np.all(output[r:r + side, c:c + side] == color), "nonuniform square")

    for anchor, group, k in zip(facts["anchors"], facts["groups"], facts["scales"]):
        ar, ac = anchor
        R, C = ar - k + 1, ac - k + 1
        _require(1 <= len(group) <= 8, "invalid number of local cargo cells")
        check_square(R, C, 2 * k, facts["anchor_color"])
        _require(2 * R + 2 * k - 1 == 2 * ar + 1, "core vertical center moved")
        _require(2 * C + 2 * k - 1 == 2 * ac + 1, "core horizontal center moved")
        for item in group:
            source, destination = item["source"], item["destination"]
            distances = [x["distance"] for x in item["candidates"]]
            _require(distances.count(min(distances)) == 1, "assigned a tied marker")
            _require(item["distance"] == abs(source[0] - destination[0])
                     + abs(source[1] - destination[1]), "wrong axial distance")
            # Independent coordinate check: expand the original snapped port cell.
            r = R + (destination[0] - ar) * k
            c = C + (destination[1] - ac) * k
            check_square(r, c, k, item["color"])
            expected_counts[item["color"]] = expected_counts.get(item["color"], 0) + k * k
    for item in facts["inert"]:
        source, color = item["source"], item["color"]
        _require(not footprint[source] and output[source] == color, "inert cargo was changed")
        footprint[source] = True
        expected_counts[color] = expected_counts.get(color, 0) + 1
    for color in range(10):
        if color != facts["background"]:
            _require(np.count_nonzero(output == color) == expected_counts.get(color, 0),
                     "wrong per-color area")
    _require(np.all(output[~footprint] == facts["background"]), "stray output pixels")
    return grid, facts


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    records = [_check_pair(pair) for pair in train + test]
    keys = [_key(grid) for grid, _ in records]
    _require(len(keys) == len(set(keys)), "duplicate inputs")
    _require(len({(f["background"], f["anchor_color"]) for _, f in records}) == 1,
             "structural color conventions changed")
    teaching = records[:len(train)]
    _require({1, 2, 3} <= {k for _, f in teaching for k in f["scales"]},
             "training must show three local factors")
    _require(any(len(f["anchors"]) == 1 and f["scales"] == [2]
                 and len(f["groups"][0]) == 4 for _, f in teaching), "missing single-dock contrast")
    _require(any(sorted(f["scales"]) == [1, 3]
                 and all(len(g) == 4 for g in f["groups"]) for _, f in teaching),
             "missing equal-count, unequal-scale contrast")
    _require(all(1 <= len(f["anchors"]) <= 2 for _, f in teaching), "training count shift violated")
    moves = [item for _, f in teaching for group in f["groups"] for item in group]
    _require({item["side"] for item in moves} == {"above", "below", "left", "right"},
             "training must show every side")
    for horizontal in (True, False):
        _require(any((item["side"] in ("left", "right")) == horizontal
                     and {c["side"] in ("left", "right") for c in item["candidates"]} == {True, False}
                     for item in moves), "missing cross-axis winner")
    _require(any(item["tied"] for _, f in teaching for item in f["inert"]), "missing tie demonstration")
    palette_sensitive_crossing = False
    for _, f in teaching:
        for a, group in enumerate(f["groups"]):
            for item in group:
                if sum(other["color"] == item["color"] for other in group) != 1:
                    continue
                for candidate in item["candidates"]:
                    b = candidate["anchor"]
                    if b != a and item["color"] not in {x["color"] for x in f["groups"][b]}:
                        palette_sensitive_crossing = True
    _require(palette_sensitive_crossing, "training needs a competition that controls both palettes")
    _require(any(len({item["color"] for group in f["groups"] for item in group})
                 < len({int(grid[p]) for p in f["cargo"]}) for grid, f in teaching),
             "missing inert-only color")
    for alternative in ("no_A", "no_nearest", "global"):
        _require(any(_counterfactual_differs(grid, f, alternative) for grid, f in teaching),
                 "training does not refute " + alternative)
    _require(len(records[len(train)][1]["anchors"]) == 3, "first test must use three anchors")
    _require(all(2 <= len(f["anchors"]) <= 3 for _, f in records[len(train):]),
             "invalid test anchor count")


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test():
    """Bounded family checks and descriptive variation across 42 fixed seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity collapsed")
    backgrounds, shapes, structures, local_patterns, cargo_counts = set(), set(), set(), set(), set()
    anchor_layouts, factors, approaches = set(), set(), set()
    zero_travel = long_travel = diagonal = same_axis = cross_tests = 0
    for puzzle in puzzles:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid, facts = _check_pair(pair)
                backgrounds.add(facts["background"])
                shapes.add(grid.shape)
                mapping = {}
                structures.add((grid.shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in grid.flat)))
                anchor_layouts.add((grid.shape, tuple(facts["anchors"])))
                factors.update(facts["scales"])
                cargo_counts.add(len(facts["cargo"]))
                local_patterns.update((k, len(group), tuple(sorted((x["side"], x["lane"]) for x in group)))
                                      for k, group in zip(facts["scales"], facts["groups"]))
                sources = facts["cargo"]
                diagonal += any(abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1
                                for i, a in enumerate(sources) for b in sources[i + 1:])
                for group in facts["groups"]:
                    for item in group:
                        approaches.add(item["side"])
                        zero_travel += item["distance"] == 0
                        long_travel += item["distance"] >= 8
                        same_axis += len(item["candidates"]) > 1 and len({
                            c["side"] in ("left", "right") for c in item["candidates"]}) == 1
                        cross_tests += section == "test" and len({
                            c["side"] in ("left", "right") for c in item["candidates"]}) == 2
    _require(len(backgrounds) >= 7 and factors == {1, 2, 3}, "insufficient role/factor diversity")
    _require(len(shapes) >= 40 and len(structures) >= 180 and len(anchor_layouts) >= 180,
             "insufficient structural diversity")
    _require(len(local_patterns) >= 100 and len(cargo_counts) >= 12, "insufficient local variation")
    _require(diagonal >= 42 and long_travel >= 42 and zero_travel >= 10 and cross_tests >= 42,
             "missing geometric variation")
    _require(approaches == {"above", "below", "left", "right"}, "missing side")
    repeated = generate_puzzle(0)
    _require(_signature(puzzles[0]) == _signature(repeated), "nondeterministic seed")
    # Symmetry must commute with the input-only rule, including tie decisions.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for turns in range(4):
            for reflect in (False, True):
                a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                if reflect:
                    a, b = np.fliplr(a), np.fliplr(b)
                _require(np.array_equal(solve(a), b), "symmetry changed the rule")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(712 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "wrong explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = copy.deepcopy(puzzles[0])
    grid = damaged["train"][0]["output"]
    grid[0, 0] = (int(grid[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": 42, "backgrounds": len(backgrounds),
            "input_shapes": len(shapes), "normalized_inputs": len(structures),
            "anchor_layouts": len(anchor_layouts), "local_port_patterns": len(local_patterns),
            "cargo_counts": sorted(cargo_counts), "zero_travel_cargo": zero_travel,
            "long_travel_cargo": long_travel, "diagonal_contact_scenes": diagonal,
            "same_axis_competitors": same_axis, "cross_axis_test_cargo": cross_tests}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
