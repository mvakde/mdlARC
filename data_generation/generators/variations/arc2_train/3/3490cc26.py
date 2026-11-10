"""Generate exact-alignment, nearest-clear-waypoint ARC route puzzles."""

import hashlib

import numpy as np


# Digests of the five solved source pairs, including dimensions and both grids.
_SOURCE_PAIR_DIGESTS = frozenset({
    "9d5c3f0bb874dd0d020ba700e9942029e79b5cdd12be800dab4868370bb05f7b",
    "65eb150db45fa43a39d991a7328d0bd0e97841093cc23fc2d8a152e87acd8083",
    "0ab8249de80d47ecc1c87870f92ab4fd3ce4521ee6a44b34e350672a9f1aa884",
    "7d30ed537604ba317876ab671102a3e66e870cfa84899330ec68d8e1238c10f1",
    "74361b4916bb84858c0f5011b0002334ec31d15a49d6f94d58a15c6350556432",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _separate(a: tuple[int, int], b: tuple[int, int], size: int) -> bool:
    return (a[0] + size < b[0] or b[0] + size < a[0]
            or a[1] + size < b[1] or b[1] + size < a[1])


def _relation(a: tuple[int, int], b: tuple[int, int], size: int):
    """Exact span alignment, orientation, positive gap, and direction."""
    ar, ac = a
    br, bc = b
    if ar == br and abs(ac - bc) > size:
        return ("h", abs(ac - bc) - size, 1 if bc > ac else -1)
    if ac == bc and abs(ar - br) > size:
        return ("v", abs(ar - br) - size, 1 if br > ar else -1)
    return None


def _segment(a: tuple[int, int], b: tuple[int, int], size: int, axis: str):
    if axis == "h":
        return tuple((r, c) for r in range(a[0], a[0] + size)
                     for c in range(min(a[1], b[1]) + size, max(a[1], b[1])))
    return tuple((r, c) for r in range(min(a[0], b[0]) + size, max(a[0], b[0]))
                 for c in range(a[1], a[1] + size))


def _trace(positions: list[tuple[int, int]], size: int) -> dict:
    """Walk to the nearest unvisited waypoint with a clear exact corridor."""
    occupied = {(r, c) for pr, pc in positions
                for r in range(pr, pr + size) for c in range(pc, pc + size)}
    separated = all(_separate(a, b, size) for i, a in enumerate(positions)
                    for b in positions[i + 1:])
    remaining = set(range(1, len(positions)))
    current = 0
    records = []
    tie = False
    blocked_terminal = False
    blocked_continuation = False
    terminal_blocked = ()
    while True:
        eligible = []
        blocked = []
        for index in sorted(remaining):
            relation = _relation(positions[current], positions[index], size)
            if relation is None:
                continue
            cells = _segment(positions[current], positions[index], size, relation[0])
            if any(cell in occupied for cell in cells):
                blocked.append(index)
            else:
                eligible.append((relation[1], index, relation, cells))
        if not eligible:
            blocked_terminal = bool(blocked)
            terminal_blocked = tuple(blocked)
            break
        if blocked:
            blocked_continuation = True
        gap = min(item[0] for item in eligible)
        closest = [item for item in eligible if item[0] == gap]
        if len(closest) != 1:
            tie = True
            break
        _, selected, relation, cells = closest[0]
        records.append({"source": current, "target": selected,
                        "relation": relation, "cells": cells,
                        "candidates": tuple(item[1] for item in eligible),
                        "blocked": tuple(blocked)})
        remaining.remove(selected)
        current = selected
    axes = [item["relation"][0] for item in records]
    directions = {(item["relation"][0], item["relation"][2]) for item in records}
    groups = _alignment_groups(positions, size)
    features = {
        "links": len(records), "distractors": len(remaining),
        "horizontal": "h" in axes, "vertical": "v" in axes,
        "turns": sum(a != b for a, b in zip(axes, axes[1:])),
        "reversal": any((axis, 1) in directions and (axis, -1) in directions
                        for axis in ("h", "v")),
        "straight_multilink": len(records) >= 2 and len(set(axes)) == 1,
        "competition": any(len(item["candidates"]) > 1 for item in records),
        "unused_competitor": any(index in remaining for item in records
                                 for index in item["candidates"]
                                 if index != item["target"]),
        "blocked_terminal": blocked_terminal,
        "blocked_continuation": blocked_continuation,
        "disconnected": groups > 1,
        "groups": groups,
        "largest_gap": max((item["relation"][1] for item in records), default=0),
    }
    return {"records": records, "remaining": frozenset(remaining),
            "tie": tie, "separated": separated, "features": features,
            "terminal_blocked": terminal_blocked}


def _alignment_components(positions: list[tuple[int, int]], size: int):
    unseen = set(range(len(positions)))
    groups = []
    while unseen:
        first = unseen.pop()
        group = {first}
        stack = [first]
        while stack:
            current = stack.pop()
            neighbors = [i for i in unseen
                         if _relation(positions[current], positions[i], size)]
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
            group.update(neighbors)
        groups.append(group)
    return groups


def _alignment_groups(positions: list[tuple[int, int]], size: int) -> int:
    return len(_alignment_components(positions, size))


def _shape(rng: np.random.Generator, links: int, size: int, attempt: int):
    if attempt >= 30:
        return (int(rng.integers(24, 31)), int(rng.integers(24, 31)))
    mode = int(rng.integers(0, 4))
    if mode == 0:
        low, high = (max(7, size * 2 + 3), 15) if links <= 2 else (15, 23)
        return (int(rng.integers(low, high)), int(rng.integers(low, high)))
    if mode == 1:
        short_low = max(7, size * 2 + 3)
        short_high = 13 if links <= 3 else 17
        dims = [int(rng.integers(short_low, short_high)),
                int(rng.integers(18, 31))]
        if int(rng.integers(0, 2)):
            dims.reverse()
        return tuple(dims)
    if mode == 2:
        return (int(rng.integers(12, 31)), int(rng.integers(12, 31)))
    return (int(rng.integers(21, 31)), int(rng.integers(21, 31)))


def _start(rng: np.random.Generator, shape: tuple[int, int], size: int):
    h, w = shape
    if rng.random() < 0.28:
        edge = int(rng.integers(0, 4))
        if edge == 0:
            return (0, int(rng.integers(0, w - size + 1)))
        if edge == 1:
            return (h - size, int(rng.integers(0, w - size + 1)))
        if edge == 2:
            return (int(rng.integers(0, h - size + 1)), 0)
        return (int(rng.integers(0, h - size + 1)), w - size)
    return (int(rng.integers(0, h - size + 1)),
            int(rng.integers(0, w - size + 1)))


def _chain(rng: np.random.Generator, shape: tuple[int, int], size: int,
           links: int, need_turn: bool):
    h, w = shape
    budget = [1800]

    def extend(positions):
        if len(positions) == links + 1:
            trace = _trace(positions, size)
            if not need_turn or (trace["features"]["horizontal"]
                                 and trace["features"]["vertical"]):
                return positions
            return None
        if budget[0] <= 0:
            return None
        current = positions[-1]
        options = []
        for axis in ("h", "v"):
            for sign in (-1, 1):
                limit = w if axis == "h" else h
                for distance in range(size + 1, limit):
                    candidate = (current[0] + sign * distance, current[1]) if axis == "v" else (current[0], current[1] + sign * distance)
                    if not (0 <= candidate[0] <= h - size
                            and 0 <= candidate[1] <= w - size):
                        continue
                    if all(_separate(candidate, other, size) for other in positions):
                        options.append(candidate)
        if not options:
            return None
        for choice in rng.permutation(len(options))[:90]:
            budget[0] -= 1
            candidate = options[int(choice)]
            trial = positions + [candidate]
            trace = _trace(trial, size)
            if trace["tie"] or not trace["separated"]:
                continue
            if [item["target"] for item in trace["records"]] != list(range(1, len(trial))):
                continue
            result = extend(trial)
            if result is not None:
                return result
        return None

    for _ in range(6):
        result = extend([_start(rng, shape, size)])
        if result is not None:
            return result
    return None


def _distractor_candidates(rng: np.random.Generator, positions, shape, size,
                           kind, active_count):
    h, w = shape
    candidates = []
    if kind == "isolated":
        for _ in range(120):
            candidates.append((int(rng.integers(0, h - size + 1)),
                               int(rng.integers(0, w - size + 1))))
    else:
        for _ in range(25):
            pool = positions[active_count:] if kind == "satellite" else positions
            if not pool:
                break
            source = pool[int(rng.integers(0, len(pool)))]
            axis = "h" if int(rng.integers(0, 2)) else "v"
            sign = -1 if int(rng.integers(0, 2)) else 1
            limit = w if axis == "h" else h
            distance = int(rng.integers(size + 1, limit))
            candidates.append((source[0], source[1] + sign * distance)
                              if axis == "h" else
                              (source[0] + sign * distance, source[1]))
    return candidates


def _add_distractors(rng: np.random.Generator, positions, shape, size, count,
                     opposite_first):
    h, w = shape
    active = list(range(1, len(positions)))
    active_count = len(positions)
    positions = positions.copy()
    detached_group = count >= 2 and rng.random() < 0.35 and not opposite_first
    for distractor_index in range(count):
        if opposite_first and distractor_index == 0:
            kinds = ["opposite"]
        elif detached_group and distractor_index == 0:
            kinds = ["isolated"]
        elif detached_group and distractor_index == 1:
            kinds = ["satellite"]
        else:
            kinds = ["isolated", "blocked", "far", "satellite"]
            kinds = [kinds[int(i)] for i in rng.permutation(4)] + ["any"]
        added = False
        for kind in kinds:
            if kind == "opposite":
                relation = _relation(positions[0], positions[1], size)
                axis, gap, sign = relation
                candidates = []
                for extra in rng.permutation(range(1, max(h, w))):
                    distance = size + gap + int(extra)
                    candidates.append((positions[0][0], positions[0][1] - sign * distance)
                                      if axis == "h" else
                                      (positions[0][0] - sign * distance, positions[0][1]))
            else:
                candidates = _distractor_candidates(
                    rng, positions, shape, size,
                    kind if kind != "any" else "isolated", active_count)
            for candidate in candidates:
                if not (0 <= candidate[0] <= h - size and 0 <= candidate[1] <= w - size):
                    continue
                if not all(_separate(candidate, other, size) for other in positions):
                    continue
                trial = positions + [candidate]
                trace = _trace(trial, size)
                if trace["tie"] or [r["target"] for r in trace["records"]] != active:
                    continue
                feature = trace["features"]
                new_index = len(trial) - 1
                if kind == "isolated" and any(
                    _relation(candidate, other, size) for other in positions):
                    continue
                if kind == "satellite" and not (
                    any(_relation(candidate, other, size)
                        for other in positions[active_count:])
                    and not any(_relation(candidate, other, size)
                                for other in positions[:active_count])):
                    continue
                if kind == "blocked" and not (
                    new_index in trace["terminal_blocked"]
                    or any(new_index in record["blocked"] for record in trace["records"])):
                    continue
                if kind == "far" and not any(
                    new_index in record["candidates"] for record in trace["records"]):
                    continue
                if kind == "opposite" and not (
                    feature["competition"] and
                    (new_index in trace["terminal_blocked"] or
                     any(new_index in record["blocked"] for record in trace["records"]))):
                    continue
                positions.append(candidate)
                added = True
                break
            if added:
                break
        if not added:
            return None
    return positions


def _make_pair(rng: np.random.Generator, palette: tuple[int, int, int, int],
               size: int, links: int, distractors: int, need_turn: bool,
               opposite_first: bool):
    for attempt in range(48):
        shape = _shape(rng, links, size, attempt)
        positions = _chain(rng, shape, size, links, need_turn)
        if positions is None:
            continue
        positions = _add_distractors(rng, positions, shape, size, distractors,
                                    opposite_first)
        if positions is None:
            continue
        trace = _trace(positions, size)
        if trace["tie"] or not trace["separated"] or len(trace["records"]) != links:
            continue
        if opposite_first and not (
            trace["features"]["competition"]
            and trace["features"]["unused_competitor"]
            and (trace["features"]["blocked_terminal"]
                 or trace["features"]["blocked_continuation"])
        ):
            continue
        background, start_color, waypoint_color, connector = palette
        inp = np.full(shape, background, dtype=np.int8)
        for index, (r, c) in enumerate(positions):
            inp[r:r + size, c:c + size] = start_color if index == 0 else waypoint_color
        out = inp.copy()
        for record in trace["records"]:
            for r, c in record["cells"]:
                out[r, c] = connector
        return {"input": inp, "output": out}
    raise ValueError("could not construct a clear unique route")


def _components(grid: np.ndarray, color: int):
    h, w = grid.shape
    unseen = {(r, c) for r in range(h) for c in range(w) if grid[r, c] == color}
    groups = []
    while unseen:
        first = unseen.pop()
        group = {first}
        stack = [first]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    group.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups


def _squares(grid: np.ndarray, color: int, size: int):
    result = []
    for group in _components(grid, color):
        top = min(r for r, _ in group)
        left = min(c for _, c in group)
        expected = {(r, c) for r in range(top, top + size)
                    for c in range(left, left + size)}
        _require(group == expected, "objects must be solid congruent squares")
        result.append((top, left))
    return sorted(result)


def _analyze(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input and output dimensions differ")
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 3, "input must have background, start and waypoints")
    counts_by_color = {int(c): int(n) for c, n in zip(colors, counts)}
    background = max(counts_by_color, key=counts_by_color.get)
    _require(list(counts_by_color.values()).count(counts_by_color[background]) == 1,
             "background must be identifiable")
    others = [int(c) for c in colors if int(c) != background]
    start = min(others, key=lambda c: counts_by_color[c])
    waypoint = next(c for c in others if c != start)
    area = counts_by_color[start]
    size = int(np.sqrt(area))
    _require(size in (2, 3) and size * size == area, "start is not a supported square")
    _require(counts_by_color[waypoint] >= 2 * area
             and counts_by_color[waypoint] % area == 0,
             "at least two same-size waypoints are required")
    start_positions = _squares(inp, start, size)
    waypoint_positions = _squares(inp, waypoint, size)
    _require(len(start_positions) == 1, "start must be unique")
    _require(len(waypoint_positions) * area == counts_by_color[waypoint],
             "waypoint components are malformed")
    positions = start_positions + waypoint_positions
    trace = _trace(positions, size)
    _require(trace["separated"] and not trace["tie"],
             "squares must be spaced and nearest clear choice unique")
    _require(trace["features"]["links"] >= 1, "at least one link is required")
    output_colors = set(int(c) for c in np.unique(out))
    connector_candidates = output_colors - set(int(c) for c in colors)
    _require(len(connector_candidates) == 1, "one output-only connector is required")
    connector = next(iter(connector_candidates))
    _require(output_colors == set(int(c) for c in colors) | {connector},
             "output lost an input color")
    expected = inp.copy()
    for record in trace["records"]:
        for r, c in record["cells"]:
            _require(inp[r, c] == background, "connector crosses an input object")
            expected[r, c] = connector
    _require(np.array_equal(expected, out), "output differs from the greedy clear route")
    return {"palette": (background, start, waypoint, connector),
            "size": size, "shape": inp.shape, "positions": positions,
            "trace": trace, "features": trace["features"]}


def _signature(grid: np.ndarray):
    return (grid.shape, np.asarray(grid, dtype=np.uint8).tobytes())


def _pair_signature(pair: dict):
    return (_signature(pair["input"]), _signature(pair["output"]))


def _source_digest(pair: dict):
    inp, out = pair["input"], pair["output"]
    payload = (str(inp.shape).encode()
               + np.asarray(inp, dtype=np.uint8).tobytes()
               + np.asarray(out, dtype=np.uint8).tobytes())
    return hashlib.sha256(payload).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    requested_train = _count("num_train", num_train, 2, 10)
    requested_test = _count("num_test", num_test, 1, 4)
    rng = np.random.default_rng(seed)
    if requested_train is None and requested_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
    elif requested_train is None:
        test_count = requested_test
        train_count = int(rng.integers(test_count + 1, 11))
    elif requested_test is None:
        train_count = requested_train
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = requested_train, requested_test
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")

    connector = int(rng.integers(0, 10))
    sizes = [2, 3] if int(rng.integers(0, 2)) else [3, 2]
    short_index = int(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    last_roles = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            size = sizes[index] if split == "train" and index < 2 else int(rng.integers(2, 4))
            if split == "train" and index < 2:
                if index == short_index:
                    links = int(rng.integers(1, 3))
                    distractors = int(rng.integers(1, 4))
                    need_turn = False
                    opposite_first = True
                else:
                    links = int(rng.integers(3, 7))
                    distractors = int(rng.integers(0, 4))
                    need_turn = True
                    opposite_first = False
            else:
                links = int(rng.integers(1, 8))
                distractors = int(rng.integers(0, 6 if split == "test" else 5))
                need_turn = False
                opposite_first = False
            if links == 1:
                distractors = max(1, distractors)
            for retry in range(10):
                roles = tuple(int(c) for c in rng.choice(
                    [c for c in range(10) if c != connector], size=3, replace=False))
                if split == "train" and index == 1 and last_roles is not None:
                    if any(roles[i] == last_roles[i] for i in range(3)):
                        continue
                palette = (*roles, connector)
                pair_rng = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
                try:
                    pair = _make_pair(pair_rng, palette, size, links, distractors,
                                      need_turn, opposite_first)
                except ValueError:
                    continue
                signature = _signature(pair["input"])
                if signature in used_inputs or _source_digest(pair) in _SOURCE_PAIR_DIGESTS:
                    continue
                used_inputs.add(signature)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    last_roles = roles
                break
            else:
                raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns,
             "illegal train/test counts")
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "each grid must be a 2D NumPy array")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30,
                         "grid dimensions must be 1 to 30")
                _require(np.issubdtype(grid.dtype, np.integer),
                         "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors must be 0 to 9")
            signature = _pair_signature(pair)
            _require(signature[0] not in inputs and signature not in pairs,
                     "duplicate input or pair")
            _require(_source_digest(pair) not in _SOURCE_PAIR_DIGESTS,
                     "source pair copied exactly")
            inputs.add(signature[0])
            pairs.add(signature)
            analyses[split].append(_analyze(pair))
    all_analyses = analyses["train"] + analyses["test"]
    connectors = {a["palette"][3] for a in all_analyses}
    _require(len(connectors) == 1, "connector color must be learned puzzle-wide")
    train = analyses["train"]
    _require({a["size"] for a in train} == {2, 3},
             "training must demonstrate both supported square sizes")
    _require(any(a["features"]["horizontal"] for a in train)
             and any(a["features"]["vertical"] for a in train),
             "training must demonstrate both route orientations")
    _require(any(a["features"]["links"] >= 3 and a["features"]["turns"] >= 1
                 for a in train), "training must demonstrate a turning greedy walk")
    _require(any(a["features"]["distractors"] >= 1 for a in train),
             "training must demonstrate unused waypoints")
    _require(any(a["features"]["competition"] and
                 a["features"]["unused_competitor"] for a in train),
             "training must show a uniquely nearest clear waypoint")
    _require(any(a["features"]["blocked_terminal"] or
                 a["features"]["blocked_continuation"] for a in train),
             "training must show a waypoint behind a visited block")
    for role in range(3):
        _require(len({a["palette"][role] for a in train}) >= 2,
                 "background, start and waypoint roles must vary between training pairs")


def _puzzle_signature(puzzle: dict):
    return tuple(tuple(_pair_signature(pair) for pair in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    source_first = _trace([(6, 8), (6, 1), (6, 13)], 2)
    _require([r["target"] for r in source_first["records"]] == [2]
             and source_first["features"]["blocked_terminal"],
             "the original opposite-side case must stop behind the visited start")
    automatic = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in automatic}) >= 4,
             "automatic train counts do not vary")
    _require(len({len(p["test"]) for p in automatic}) >= 3,
             "automatic test counts do not vary")
    _require(len({_puzzle_signature(p) for p in automatic}) >= 40,
             "different seeds repeat puzzles")
    shapes, layouts, palettes, link_counts = set(), set(), set(), set()
    feature_counts = {key: 0 for key in ("blocked_terminal", "blocked_continuation",
                                         "disconnected", "unused_competitor",
                                         "straight_multilink", "reversal")}
    review_counts = {key: 0 for key in ("compact", "thin", "edge", "long_gap",
                                         "no_unused", "short", "long", "opposite",
                                         "detached_group", "source_shape_size",
                                         "size3_edge_disconnected")}
    source_shapes = {(16, 18), (26, 28), (26, 25), (26, 18), (22, 23)}
    pairs = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        puzzle_analyses = [_analyze(pair) for split in ("train", "test")
                           for pair in puzzle[split]]
        _require(len({a["features"]["links"] for a in puzzle_analyses}) >= 2,
                 "route lengths do not vary within a puzzle")
        _require(len({a["features"]["distractors"] for a in puzzle_analyses}) >= 2,
                 "unused-waypoint counts do not vary within a puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                pairs += 1
                shapes.add(a["shape"])
                palettes.add(a["palette"])
                link_counts.add(a["features"]["links"])
                layouts.add((a["shape"], a["size"], tuple(a["positions"])))
                for key in feature_counts:
                    feature_counts[key] += bool(a["features"][key])
                h, w = a["shape"]
                size = a["size"]
                positions = a["positions"]
                features = a["features"]
                edge = any(r == 0 or c == 0 or r + size == h or c + size == w
                           for r, c in positions)
                opposite = any(
                    (positions[0][0] == positions[i][0] == positions[j][0]
                     and (positions[i][1] - positions[0][1])
                     * (positions[j][1] - positions[0][1]) < 0)
                    or (positions[0][1] == positions[i][1] == positions[j][1]
                        and (positions[i][0] - positions[0][0])
                        * (positions[j][0] - positions[0][0]) < 0)
                    for i in range(1, len(positions))
                    for j in range(i + 1, len(positions)))
                detached = any(0 not in group and len(group) >= 2
                               for group in _alignment_components(positions, size))
                for key, condition in (
                    ("compact", max(h, w) <= 15),
                    ("thin", min(h, w) <= 12 and max(h, w) >= 18),
                    ("edge", edge),
                    ("long_gap", features["largest_gap"] >= 7),
                    ("no_unused", features["distractors"] == 0),
                    ("short", features["links"] <= 2),
                    ("long", features["links"] >= 5),
                    ("opposite", opposite),
                    ("detached_group", detached),
                    ("source_shape_size", a["shape"] in source_shapes),
                    ("size3_edge_disconnected", size == 3 and edge
                     and features["disconnected"]),
                ):
                    review_counts[key] += bool(condition)
    _require(len(shapes) >= 40 and len(layouts) >= int(0.9 * pairs),
             "insufficient spatial diversity")
    _require(len(palettes) >= 50 and len(link_counts) >= 5,
             "insufficient palette or route-length diversity")
    _require(feature_counts["disconnected"] >= 5
             and feature_counts["straight_multilink"] >= 5,
             "missing disconnected or straight route examples")
    for key, minimum in (("blocked_terminal", 5),
                         ("blocked_continuation", 20),
                         ("unused_competitor", 20),
                         ("reversal", 20)):
        _require(feature_counts[key] >= minimum,
                 f"insufficient {key} examples")
    for key, minimum in (("compact", 5), ("thin", 10), ("edge", 20),
                         ("long_gap", 20), ("no_unused", 10),
                         ("short", 20), ("long", 20), ("opposite", 20),
                         ("detached_group", 5), ("source_shape_size", 1),
                         ("size3_edge_disconnected", 5)):
        _require(review_counts[key] >= minimum,
                 f"insufficient {key} review examples")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(2000 + 10 * train_count + test_count,
                                 train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts not honored")
    for index, (train_count, test_count) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "partial train count lost")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "partial test count lost")
    one = generate_puzzle(777, 5, 2)
    two = generate_puzzle(777, 5, 2)
    _require(_puzzle_signature(one) == _puzzle_signature(two),
             "generation is not deterministic")
    for arguments in ((1, None), (11, None), (None, 0), (None, 5),
                      (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
