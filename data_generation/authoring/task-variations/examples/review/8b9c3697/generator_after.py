"""Rectangles move to matching central spokes; only trail color is puzzle-wide.

Spoke lengths exclude the base. Object roles are inferred from visible shapes,
not from fixed colors. Construction and validation use separate geometry paths.
"""

import hashlib
from itertools import combinations

import numpy as np


_DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_SOURCE_DIGESTS = {
    "71a4aa19bf69e671b3244365f1f3bf02817a193e1d1b82f7bf571d10f3939e86",
    "2faa8e0b5ece606c25ca6d4819ca7e914aeb20d898f7e0a8571093253d954a63",
    "e121af1cadbc7492b4200b694a0e0f6a025b4e9743041ce378406b446a8d5123",
    "be89b3e823aa9ffc73ebe3ec11594075c5767d89bd215b110bac9b1f853755fd",
}


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS.values():
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    component.add(point)
                    stack.append(point)
        result.append(component)
    return result


def _rectangle(cells):
    rows, cols = zip(*cells)
    return len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _rect(r, c, h, w):
    return {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}


def _paint(grid, cells, color):
    for point in cells:
        grid[point] = color


def _expand(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _parse_fork(cells):
    """Recover a full-width base and three positive uniform spokes by rotation.

    The middle run is the stop, including when it is longer than both guards.
    No enumerated geometry template or comparison of lengths assigns this role.
    """
    parses = []
    for direction, (dr, dc) in _DIRS.items():
        outward, perp = (-dr, -dc), (dc, -dr)
        kmin = min(r * outward[0] + c * outward[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths) for k in range(1, length + 1)}
        if local != expected:
            continue
        runs = []
        q = 0
        while q < width:
            if not lengths[q]:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        center0, center1, center_length = runs[1]
        if center0 + center1 != width:
            continue
        parses.append({"direction": direction, "out": outward, "perp": perp,
                       "kmin": kmin, "qmin": qmin, "width": width,
                       "runs": tuple(runs), "lengths": tuple(run[2] for run in runs),
                       "q0": center0, "span": center1 - center0,
                       "stop": center_length + 1, "cells": cells})
    assert len(parses) == 1, "fork must have one base and three positive spokes"
    return parses[0]


def _cell(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    return (k * fork["out"][0] + q * fork["perp"][0],
            k * fork["out"][1] + q * fork["perp"][1])


def _candidate(fork, cells, grid, background):
    local = {(r * fork["out"][0] + c * fork["out"][1] - fork["kmin"],
              r * fork["perp"][0] + c * fork["perp"][1] - fork["qmin"])
             for r, c in cells}
    qs = {q for _, q in local}
    if qs != set(range(fork["q0"], fork["q0"] + fork["span"])):
        return None
    k0, k1 = min(k for k, _ in local), max(k for k, _ in local)
    if k0 <= fork["stop"]:
        return None
    route = {_cell(fork, k, q) for k in range(fork["stop"], k1 + 1) for q in qs}
    if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
           or ((r, c) not in cells and int(grid[r, c]) != background) for r, c in route):
        return None
    final = {_cell(fork, k, q) for k in range(fork["stop"], fork["stop"] + k1 - k0 + 1)
             for q in qs}
    return {"route": route, "final": final, "distance": k0 - fork["stop"],
            "length": k1 - k0 + 1}


def _analyze(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert sum(counts == counts.max()) == 1
    forks, tokens = [], []
    for color in values:
        if int(color) == background:
            continue
        for cells in _components(grid == color):
            if _rectangle(cells):
                tokens.append({"cells": cells, "color": int(color)})
            else:
                fork = _parse_fork(cells)
                fork["color"] = int(color)
                forks.append(fork)
    assert forks and tokens
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = []
        for ti, token in enumerate(tokens):
            candidate = _candidate(fork, token["cells"], grid, background)
            if candidate is not None:
                candidates.append((ti, candidate))
        assert len(candidates) <= 1, "multiple aligned tokens for one fork"
        if candidates:
            ti, candidate = candidates[0]
            assert ti not in used, "one token assigned to multiple forks"
            assert swept.isdisjoint(candidate["route"]), "interacting swept routes"
            swept.update(candidate["route"])
            used.add(ti)
            matches.append(dict(candidate, fi=fi, ti=ti))
    return {"background": background, "forks": forks, "tokens": tokens,
            "matches": matches, "unmatched": len(tokens) - len(used)}


def _solve(grid, trail):
    analysis = _analyze(grid)
    assert trail not in grid
    output = grid.copy()
    for token in analysis["tokens"]:
        _paint(output, token["cells"], analysis["background"])
    for match in analysis["matches"]:
        _paint(output, match["route"] - match["final"], trail)
        _paint(output, match["final"], analysis["tokens"][match["ti"]]["color"])
    return output, analysis


def _learn_trail(train_pairs):
    trails = []
    for pair in train_pairs:
        introduced = set(map(int, np.unique(pair["output"]))) - set(map(int, np.unique(pair["input"])))
        assert len(introduced) == 1
        trails.append(introduced.pop())
    assert len(set(trails)) == 1
    return trails[0]


def _predict(train_pairs, test_input):
    return _solve(test_input, _learn_trail(train_pairs))[0]


def _boxes(rng, h, w, count, fallback):
    """Compose random spatial partitions without visible dividers.

    The fallback uses disjoint row AND column bands: foreign tokens cannot
    align with another fork. Seven-cell squares accommodate every minimal spec.
    """
    if fallback:
        size = (min(h, w) - count + 1) // count
        return [(i * (size + 1), i * (size + 1), size, size) for i in range(count)]
    axis = int(rng.integers(2))
    cut = int(rng.integers(7, (h if axis == 0 else w) - 7))
    if count == 2:
        boxes = ([(0, 0, cut, w), (cut + 1, 0, h - cut - 1, w)] if axis == 0
                 else [(0, 0, h, cut), (0, cut + 1, h, w - cut - 1)])
    else:
        cross = int(rng.integers(7, (w if axis == 0 else h) - 7))
        if axis == 0:
            boxes = [(0, 0, cut, w), (cut + 1, 0, h - cut - 1, cross),
                     (cut + 1, cross + 1, h - cut - 1, w - cross - 1)]
        else:
            boxes = [(0, 0, h, cut), (0, cut + 1, cross, w - cut - 1),
                     (cross + 1, cut + 1, h - cross - 1, w - cut - 1)]
        if rng.integers(2):
            boxes = [(h - r - bh, w - c - bw, bh, bw) for r, c, bh, bw in boxes]
    return [boxes[int(i)] for i in rng.permutation(count)]


def _specs(rng, regime, index):
    if regime == "source_train":
        if index == 0:
            return [("L", 1, 1, True, "standard"), ("U", 2, 2, True, "unequal")]
        if index == 1:
            return [("D", 3, int(rng.integers(1, 3)), True, "long_center"),
                    (str(rng.choice(["L", "U", "D"])), int(rng.integers(1, 4)), 1, False, "random")]
        active = [True, bool(rng.integers(2))]
        rng.shuffle(active)
        return [(str(rng.choice(["L", "U", "D"])), int(rng.integers(1, 4)),
                 int(rng.integers(1, 3)), active[i], "random") for i in range(2)]
    return [("R", 3, int(rng.integers(1, 3)), True, "random"),
            ("U", int(rng.integers(1, 4)), int(rng.integers(1, 3)), True, "random"),
            ("D", int(rng.integers(1, 4)), int(rng.integers(1, 3)), True, "random")]


def _module(rng, box, spec):
    r0, c0, bh, bw = box
    direction, span, token_length, active, mode = spec
    dr, dc = _DIRS[direction]
    outward, perp = (-dr, -dc), (dc, -dr)
    cross, depth = (bw, bh) if dr else (bh, bw)
    choices = [(side, gap, tail) for side in (1, 2) for gap in range(1, 5) for tail in (0, 1)
               if span + 2 * (side + gap + tail) <= cross]
    side, clearance, tail = choices[int(rng.integers(len(choices)))]
    width = span + 2 * (side + clearance + tail)
    q0 = tail + side + clearance
    travel_gap = int(rng.integers(1, min(6, depth - token_length - 4) + 1)) if active else 1
    cap = min(8, depth - token_length - travel_gap - 1) if active else min(8, depth - 1)
    assert cap >= 3
    if mode == "random":
        mode = str(rng.choice(["standard", "unequal", "long_center", "independent"]))
    if mode == "standard":
        center = int(rng.integers(1, cap - token_length + 1))
        left = right = int(rng.integers(center + token_length, cap + 1))
    elif mode == "long_center":
        center = int(rng.integers(2, cap + 1))
        left, right = map(int, rng.integers(1, center, size=2))
    else:
        left, center, right = map(int, rng.integers(1, cap + 1, size=3))
        if mode == "unequal" and left == right:
            right = right % cap + 1
    structure = {(0, q) for q in range(width)}
    for qa, qb, length in ((tail, tail + side, left), (q0, q0 + span, center),
                            (width - tail - side, width - tail, right)):
        structure |= {(k, q) for k in range(1, length + 1) for q in range(qa, qb)}
    original, final, route = set(), set(), set()
    if active:
        start = max(left, center, right) + 1 + travel_gap
        original = _rect(start, q0, token_length, span)
        final = _rect(center + 1, q0, token_length, span)
        route = _rect(center + 1, q0, start + token_length - center - 1, span)
    all_cells = structure | route

    def raw(cells):
        return {(k * outward[0] + q * perp[0], k * outward[1] + q * perp[1]) for k, q in cells}

    raw_all = raw(all_cells)
    min_r, min_c = min(r for r, _ in raw_all), min(c for _, c in raw_all)
    height, wide = max(r for r, _ in raw_all) - min_r + 1, max(c for _, c in raw_all) - min_c + 1
    rr, cc = r0 + int(rng.integers(bh - height + 1)), c0 + int(rng.integers(bw - wide + 1))

    def placed(cells):
        return {(r - min_r + rr, c - min_c + cc) for r, c in raw(cells)}

    return {"structure": placed(structure), "original": placed(original),
            "final": placed(final), "route": placed(route), "active": active}


def _choose_palette(rng, colors, count, excluded):
    options = [p for p in combinations(sorted(colors), count) if p not in excluded]
    return options[int(rng.integers(len(options)))]


def _generate_pair(rng, regime, index, background, trail, previous, used_palettes):
    h, w = (19, 22) if regime == "source_train" else (int(rng.integers(23, 31)), int(rng.integers(24, 31)))
    specs = _specs(rng, regime, index)
    extra = int(rng.integers(2, 4) if regime == "source_train" else rng.integers(5, 9))
    fork_count = 2 if len(specs) == 2 else int(rng.integers(2, 4))
    colors = set(range(10)) - {background, trail}
    fork_palette = _choose_palette(rng, colors, fork_count, used_palettes | {previous[0]})
    token_count = int(rng.integers(2, min(4, sum(s[3] for s in specs) + extra) + 1))
    token_palette = _choose_palette(rng, colors - set(fork_palette), token_count, {previous[1]})
    fork_colors = list(rng.permutation(fork_palette))
    fork_colors += [int(rng.choice(fork_palette)) for _ in range(len(specs) - fork_count)]
    token_colors = list(rng.permutation(token_palette))
    token_colors += [int(rng.choice(token_palette)) for _ in range(sum(s[3] for s in specs) + extra - token_count)]
    # Only accidental cross-fork alignment is rejected. The final construction
    # has disjoint coordinate bands and cannot have that ambiguity.
    for attempt in range(17):
        boxes = _boxes(rng, h, w, len(specs), fallback=attempt == 16)
        modules = [_module(rng, box, spec) for box, spec in zip(boxes, specs)]
        grid = np.full((h, w), background, dtype=np.int8)
        output = grid.copy()
        ti, blocked = 0, set()
        for module, color in zip(modules, fork_colors):
            _paint(grid, module["structure"], color)
            _paint(output, module["structure"], color)
            blocked |= _expand(module["structure"] | module["route"])
            if module["active"]:
                _paint(grid, module["original"], token_colors[ti])
                _paint(output, module["route"] - module["final"], trail)
                _paint(output, module["final"], token_colors[ti])
                ti += 1
        try:
            predicted, analysis = _solve(grid, trail)
            if not np.array_equal(predicted, output):
                continue
        except AssertionError:
            continue
        # Reject exact matches and protect swept paths while placing distractors.
        # A singleton fallback avoids abandoning an otherwise valid crowded grid.
        for di in range(extra):
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            positions = []
            for small in (False, True):
                if small:
                    dh = dw = 1
                for r in range(h - dh + 1):
                    for c in range(w - dw + 1):
                        cells = _rect(r, c, dh, dw)
                        if not cells.isdisjoint(blocked):
                            continue
                        if any(_candidate(fork, cells, grid, background) is not None for fork in analysis["forks"]):
                            continue
                        positions.append(cells)
                if positions:
                    break
            assert positions, "separated modules leave space for distractors"
            cells = positions[int(rng.integers(len(positions)))]
            _paint(grid, cells, token_colors[ti + di])
            blocked |= _expand(cells)
        return {"input": grid, "output": output}, (fork_palette, token_palette)
    raise AssertionError("disjoint-coordinate fallback must be constructible")


def _valid_count(value, low, high):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and low <= int(value) <= high)


def _resolve_counts(seed_value, num_train, num_test):
    if num_train is not None and not _valid_count(num_train, 2, 10):
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not _valid_count(num_test, 1, 4):
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        minimum = num_test + 1
        num_train = minimum + (seed_value // 4) % (11 - minimum)
    elif num_train is None:
        minimum = int(num_test) + 1
        num_train = minimum + seed_value % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed_value % min(4, int(num_train) - 1)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    trail = int(rng.integers(10))
    backgrounds = list(map(int, rng.permutation([c for c in range(10) if c != trail])))
    puzzle, previous = {"train": [], "test": []}, (None, None)
    used_palettes = {color: set() for color in backgrounds}
    for split, count, regime in (("train", num_train, "source_train"), ("test", num_test, "target_test")):
        for index in range(count):
            ordinal = len(puzzle["train"]) + len(puzzle["test"])
            background = backgrounds[ordinal % 9]
            pair_rng = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
            pair, previous = _generate_pair(pair_rng, regime, index, background, trail, previous, used_palettes[background])
            # Repeated backgrounds use a new observable fork palette. This
            # guarantees distinct inputs without a retry or artificial marker.
            used_palettes[background].add(previous[0])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _length_mode(lengths):
    left, center, right = lengths
    if center > max(left, right):
        return "long_center"
    if left == right and center < left:
        return "standard"
    if left != right:
        return "unequal"
    return "equal"


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    for pairs in puzzle.values():
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
    trail = _learn_trail(puzzle["train"])
    inputs, pairs_seen, backgrounds, fork_palettes, token_palettes, widths = set(), set(), set(), set(), set(), set()
    train_directions, train_spans, train_lengths, train_active, train_modes = set(), set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid, output = pair["input"], pair["output"]
            key, out_key = _grid_key(grid), _grid_key(output)
            assert key not in inputs and (key, out_key) not in pairs_seen
            inputs.add(key)
            pairs_seen.add((key, out_key))
            digest = hashlib.sha256(bytes(grid.shape) + key[1] + out_key[1]).hexdigest()
            assert digest not in _SOURCE_DIGESTS
            expected, analysis = _solve(grid, trail)
            assert np.array_equal(expected, output) and not np.array_equal(grid, output)
            assert set(map(int, np.unique(output))) - set(map(int, np.unique(grid))) == {trail}
            forks, tokens, matches = analysis["forks"], analysis["tokens"], analysis["matches"]
            fcolors, tcolors = {f["color"] for f in forks}, {t["color"] for t in tokens}
            assert len(fcolors) >= 2 and len(tcolors) >= 2 and fcolors.isdisjoint(tcolors)
            backgrounds.add(analysis["background"])
            fork_palettes.add(tuple(sorted(fcolors)))
            token_palettes.add(tuple(sorted(tcolors)))
            structures, occupied = set(), set()
            for obj in forks + tokens:
                assert occupied.isdisjoint(_expand(obj["cells"])), "touching input objects"
                occupied |= obj["cells"]
            for fork in forks:
                assert min(fork["lengths"]) >= 1
                assert all(int(output[p]) == fork["color"] for p in fork["cells"])
                assert structures.isdisjoint(fork["cells"])
                structures |= fork["cells"]
                widths.add(fork["width"])
            assert matches and analysis["unmatched"] >= 2
            if split == "train":
                assert grid.shape == (19, 22) and len(forks) == 2
                assert {f["direction"] for f in forks} <= {"L", "U", "D"}
                assert 2 <= analysis["unmatched"] <= 3
                train_active.add(len(matches))
                for match in matches:
                    fork = forks[match["fi"]]
                    train_directions.add(fork["direction"])
                    train_spans.add(fork["span"])
                    train_lengths.add(match["length"])
                train_modes |= {_length_mode(f["lengths"]) for f in forks}
            else:
                assert 23 <= grid.shape[0] <= 30 and 24 <= grid.shape[1] <= 30
                assert len(forks) == len(matches) == 3 and 5 <= analysis["unmatched"] <= 8
                assert {f["direction"] for f in forks} == {"R", "U", "D"}
                assert next(f for f in forks if f["direction"] == "R")["span"] == 3
                # Predict without passing the test output or generation state.
                assert np.array_equal(_predict(puzzle["train"], grid.copy()), output)
    assert len(backgrounds) >= 3 and len(fork_palettes) >= 2 and len(token_palettes) >= 2
    assert len(widths) >= 2
    assert train_directions == {"L", "U", "D"} and train_spans == {1, 2, 3}
    assert train_lengths == {1, 2} and train_active == {1, 2}
    assert {"standard", "long_center"} <= train_modes


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, trails, shapes, layouts, geometries = set(), set(), set(), set(), set(), set()
    widths, lengths, combinations_seen, modes, puzzles = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_key(puzzle))
        trails.add(_learn_trail(puzzle["train"]))
        for pairs in puzzle.values():
            for pair in pairs:
                grid = pair["input"]
                analysis = _analyze(grid)
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple(sorted(tuple(sorted(f["cells"])) for f in analysis["forks"]))))
                for fork in analysis["forks"]:
                    widths.add(fork["width"])
                    lengths.update(fork["lengths"])
                    mode = _length_mode(fork["lengths"])
                    modes.add(mode)
                    geometries.add((fork["width"], fork["runs"]))
                    combinations_seen.add((fork["width"], mode))
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(puzzles) == 42
    assert len(trails) >= 5 and len(shapes) >= 15 and len(layouts) >= 250
    assert len(widths) >= 8 and len(geometries) >= 100 and len(combinations_seen) >= 25
    assert set(range(1, 9)) <= lengths and {"standard", "unequal", "long_center"} <= modes
    assert _puzzle_key(generate_puzzle(123456, 6, 3)) == _puzzle_key(generate_puzzle(123456, 6, 3))
    assert _puzzle_key(generate_puzzle(123456, 5, 2)) != _puzzle_key(generate_puzzle(123457, 5, 2))
    for index, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(9000 + index, nt, ne)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (4, 4), (2, 4)):
        try:
            generate_puzzle(7, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts did not raise ValueError")
    # Reject semantic corruption even when the ARC array schema is valid.
    puzzle = generate_puzzle(17, 2, 1)
    pair = puzzle["test"][0]
    match = _analyze(pair["input"])["matches"][0]
    pair["output"][min(match["final"])] = _learn_trail(puzzle["train"])
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted token accepted")
    malformed = _rect(0, 0, 1, 7) | _rect(1, 1, 3, 1) | _rect(1, 5, 3, 1)
    try:
        _parse_fork(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("zero-length center accepted")
