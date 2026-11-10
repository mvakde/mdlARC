"""Tuned avalanche: Manhattan phase determines which pegs stop edge bars.

Only Python's standard library and NumPy are used. No parent code is imported.
"""

from __future__ import annotations

import numpy as np


_RING = ((-1, 0), (-1, 1), (0, 1), (1, 1),
         (1, 0), (1, -1), (0, -1), (-1, -1))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _check(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _check(bool(np.all((0 <= a) & (a <= 9))), "illegal color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _parse(value):
    """Recover all conventions from this input, including its release edge."""
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _check(int(counts.max()) * 2 > a.size, "background is not dominant")
    _check(min(a.shape) >= 5, "no room for an interior clock")
    filled = np.lib.stride_tricks.sliding_window_view(a != bg, (3, 3)).all(axis=(-2, -1))
    clocks = []
    for corner in np.argwhere(filled):
        r, c = map(lambda x: int(x) + 1, corner)
        if not (2 <= r < a.shape[0] - 2 and 2 <= c < a.shape[1] - 2):
            continue
        mover = int(a[r, c])
        cycle = tuple(int(a[r + dr, c + dc]) for dr, dc in _RING)
        if len(set(cycle)) == 8 and mover not in cycle and bg not in cycle:
            clocks.append((r, c, mover, cycle))
    _check(len(clocks) == 1, "input must have exactly one color clock")
    r, c, mover, cycle = clocks[0]
    fixture = np.zeros(a.shape, dtype=bool)
    fixture[r - 1:r + 2, c - 1:c + 2] = True
    material = (a == mover) & ~fixture
    directions = []
    for rotation in range(4):
        m = np.rot90(material, -rotation)
        lengths = m.sum(axis=1).astype(int)
        if np.count_nonzero(lengths) < 2:
            continue
        valid = True
        for row in np.flatnonzero(lengths):
            length = int(lengths[row])
            if length >= m.shape[1] or not np.all(m[row, m.shape[1] - length:]):
                valid = False
                break
        if valid:
            directions.append((rotation, lengths))
    _check(len(directions) == 1, "source edge must be unique")
    rotation, lengths = directions[0]
    pegs = (a != bg) & (a != mover) & ~fixture
    return dict(a=a, bg=bg, mover=mover, cycle=cycle, center=(r, c),
                fixture=fixture, material=material, pegs=pegs,
                rotation=rotation, lengths=lengths)


def _hard(data, metric="manhattan"):
    a = data["a"]
    r, c = data["center"]
    rr, cc = np.indices(a.shape)
    dr, dc = np.abs(rr - r), np.abs(cc - c)
    if metric == "all":
        return data["fixture"] | data["pegs"]
    if metric == "constant":
        resonant = a == data["cycle"][0]
    else:
        distance = (dr if metric == "row" else dc if metric == "column"
                    else np.maximum(dr, dc) if metric == "chebyshev" else dr + dc)
        expected = np.asarray(data["cycle"], dtype=np.int8)[distance % 8]
        resonant = a == expected
        if metric == "nowrap":
            resonant &= distance < 8
    return data["fixture"] | (data["pegs"] & resonant)


def _closed_result(data, hard=None, rigid=False):
    """Closed-form reference used by construction and validation, not solve."""
    if hard is None:
        hard = _hard(data)
    k = data["rotation"]
    x = np.rot90(data["a"], -k)
    walls = np.rot90(hard, -k)
    records = []
    for row in np.flatnonzero(data["lengths"]):
        row = int(row)
        length = int(data["lengths"][row])
        start = x.shape[1] - length
        blockers = np.flatnonzero(walls[row, :start])
        stop = int(blockers[-1]) + 1 if len(blockers) else 0
        records.append(dict(row=row, length=length, start=start, stop=stop,
                            travel=start - stop, blockers=tuple(map(int, blockers))))
    if rigid:
        travel = min(record["travel"] for record in records)
        records = [dict(record, stop=record["start"] - travel, travel=travel)
                   for record in records]
    out = x.copy()
    for record in records:
        row, t, length = record["row"], record["stop"], record["length"]
        out[row, t:] = data["bg"]
        out[row, t:t + length] = data["mover"]
    return np.rot90(out, k).copy(), records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Find the clock and simulate each run's leading end one step at a time."""
    data = _parse(input_grid)
    k = data["rotation"]
    x = np.rot90(data["a"], -k)
    walls = np.rot90(_hard(data), -k)
    out = x.copy()
    for row, length in enumerate(data["lengths"]):
        length = int(length)
        if not length:
            continue
        leading = x.shape[1] - length
        while leading > 0 and not walls[row, leading - 1]:
            leading -= 1
        out[row, leading:] = data["bg"]
        out[row, leading:leading + length] = data["mover"]
    return np.rot90(out, k).copy()


def _paint_clock(grid, center, mover, cycle):
    r, c = center
    grid[r, c] = mover
    for color, (dr, dc) in zip(cycle, _RING):
        grid[r + dr, c + dc] = color


def _rotate_center(center, shape, k):
    r, c = center
    h, w = shape
    for _ in range(k):
        r, c, h, w = w - 1 - c, r, w, h
    return r, c


def _rows(rng, h, count):
    if rng.random() < 0.4:
        start = int(rng.integers(0, h - count + 1))
        return set(range(start, start + count))
    return set(map(int, rng.choice(h, count, replace=False)))


def _scene(rng, roles, cycle, shape, rotation, max_length,
           teaching=False, all_lengths=False):
    bg, mover = roles
    h, w = shape
    sr = int(rng.integers(2, h - 2))
    sc = int(rng.integers(2, w - max_length - 2))
    fixture = np.zeros(shape, dtype=bool)
    fixture[sr - 1:sr + 2, sc - 1:sc + 2] = True
    outside = [r for r in range(h) if abs(r - sr) > 1]
    count = int(rng.integers(max(2, max_length if all_lengths else 2),
                             min(h - 2, 6 if teaching else 11) + 1))
    active = _rows(rng, h, count)
    modes = {}
    wrap_column = None
    if teaching:
        candidates = [(r, c) for r in outside for c in range(2, w - max_length - 2)
                      if abs(r - sr) + abs(c - sc) == 8]
        _check(bool(candidates), "no wrap witness site")
        wr, wrap_column = candidates[int(rng.integers(len(candidates)))]
        other = list(map(int, rng.choice([r for r in outside if r != wr], 3, replace=False)))
        modes = {wr: "wrap", other[0]: "nearest", other[1]: "boundary",
                 other[2]: "stationary", sr: "fixture"}
        active.update(modes)
    active = sorted(active)
    lengths = {r: int(rng.integers(1, max_length + 1)) for r in active}
    if all_lengths:
        for length, row in enumerate(rng.permutation(active)[:max_length], 1):
            lengths[int(row)] = length
    pegs = {}

    def peg(row, col, solid, explicit_color=None):
        _check(not fixture[row, col], "peg overlaps fixture")
        distance = abs(row - sr) + abs(col - sc)
        phase = distance % 8
        color = cycle[phase] if solid else cycle[(phase + int(rng.integers(1, 8))) % 8]
        pegs[row, col] = int(color if explicit_color is None else explicit_color)

    density = float(rng.uniform(0.065, 0.235))
    for row in range(h):
        start = w - lengths[row] if row in lengths else w
        mode = modes.get(row, "random")
        b = None
        if row in lengths:
            minimum = sc + 1 if abs(row - sr) <= 1 else -1
            if mode == "wrap":
                b = wrap_column
            elif mode == "nearest":
                b = int(rng.integers(3, start - 2))
            elif mode == "boundary":
                b = -1
            elif mode == "stationary":
                b = start - 1
            elif mode == "fixture":
                b = sc + 1
            else:
                u = float(rng.random())
                if u < 0.22:
                    b = minimum
                elif u < 0.40:
                    b = start - 1
                else:
                    b = int(rng.integers(max(0, minimum), start))
        cluster_start = int(rng.integers(w))
        cluster_size = int(rng.integers(1, 5))
        for col in range(start):
            if fixture[row, col]:
                continue
            probability = density + (0.25 if cluster_start <= col < cluster_start + cluster_size else 0)
            if rng.random() < probability:
                solid = bool(rng.random() < 0.5) if b is None or col < b else col == b
                peg(row, col, solid)
        if b is not None and b >= 0 and not fixture[row, b]:
            peg(row, b, True)
        if mode == "nearest":
            peg(row, b - 2, True)
            peg(row, b - 1, False)  # shielded soft peg survives
            peg(row, int(rng.integers(b + 1, start)), False)
        elif mode == "boundary":
            peg(row, int(rng.integers(start)), False)
        elif mode == "stationary":
            peg(row, int(rng.integers(b)), False)
        elif mode == "wrap":
            choices = [c for c in range(b + 1, start)
                       if (abs(row - sr) + abs(c - sc)) % 8 != 0]
            _check(bool(choices), "no same-color soft decoy site")
            peg(row, int(rng.choice(choices)), False, cycle[0])
        elif mode == "fixture" and b + 1 < start:
            peg(row, b + 1, False)
    inactive = [r for r in range(h) if r not in lengths]
    if inactive:
        row = int(rng.choice(inactive))
        cols = [c for c in range(w) if not fixture[row, c]]
        peg(row, int(rng.choice(cols)), False)
    grid = np.full(shape, bg, dtype=np.int8)
    for (r, c), color in pegs.items():
        grid[r, c] = color
    _paint_clock(grid, (sr, sc), mover, cycle)
    for row, length in lengths.items():
        _check(not np.any(fixture[row, w - length:]), "source overlaps clock")
        grid[row, w - length:] = mover
    grid = np.rot90(grid, rotation).copy()
    # The clock is a compass-indexed code, so re-encode its north phase.
    center = _rotate_center((sr, sc), shape, rotation)
    _paint_clock(grid, center, mover, cycle)
    data = _parse(grid)
    output, _ = _closed_result(data)
    _check(not np.array_equal(grid, output), "no visible transformation")
    if teaching:
        _check_teaching(data, output)
    return {"input": grid, "output": output}


def _check_teaching(data, output):
    """Counterfactual checks, rather than trusting construction mode labels."""
    for metric in ("all", "constant", "row", "column", "chebyshev", "nowrap"):
        alternative, _ = _closed_result(data, _hard(data, metric))
        _check(not np.array_equal(alternative, output), "shortcut survives: " + metric)
    rigid, _ = _closed_result(data, rigid=True)
    _check(not np.array_equal(rigid, output), "rigid-motion shortcut survives")
    soft = data["pegs"] & ~_hard(data)
    globally_erased = output.copy()
    globally_erased[soft & (output != data["mover"])] = data["bg"]
    _check(not np.array_equal(globally_erased, output), "global deletion shortcut survives")


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " outside legal range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    roles, base_cycle = palette[:2], palette[2:]
    max_length = int(rng.integers(2, 5))
    ndirections = 1 if rng.random() < 0.55 else 2
    directions = list(map(int, rng.choice(4, ndirections, replace=False)))
    phases = list(map(int, rng.choice(8, 2, replace=False)))
    train_phases = phases + [int(rng.choice(phases)) for _ in range(num_train - 2)]
    train_directions = directions + [int(rng.choice(directions))
                                     for _ in range(num_train - ndirections)]
    shared_shape = bool(rng.random() < 0.35)
    base_shape = (int(rng.integers(12, 27)), int(rng.integers(16, 31)))
    phase_transfer = bool(rng.random() < 0.55)
    teaching_slot = int(rng.integers(2))
    length_slot = 1 - teaching_slot
    puzzle = {"train": [], "test": []}
    seen, train_shapes = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = split == "train" and index == teaching_slot
            all_lengths = split == "train" and index == length_slot
            if split == "train":
                phase, rotation = train_phases[index], train_directions[index]
            else:
                phase = int(rng.choice([p for p in range(8) if p not in phases]
                                       if phase_transfer else phases))
                rotation = int(rng.choice(directions))
            cycle = tuple(base_cycle[(phase + d) % 8] for d in range(8))
            for _ in range(200):
                if shared_shape and not (split == "test" and not phase_transfer):
                    shape = base_shape
                else:
                    shape = (int(rng.integers(12 if teaching else 9, 27)),
                             int(rng.integers(16 if teaching else 12, 31)))
                actual_shape = shape if rotation % 2 == 0 else shape[::-1]
                if split == "test" and not phase_transfer and actual_shape in train_shapes:
                    continue
                try:
                    pair = _scene(rng, roles, cycle, shape, rotation, max_length,
                                  teaching=teaching, all_lengths=all_lengths)
                except AssertionError:
                    continue
                key = _key(pair["input"])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_shapes.add(pair["input"].shape)
                break
            else:
                raise RuntimeError("could not construct a distinct, unambiguous scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "bad splits")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "bad counts")
    seen, conventions = set(), set()
    train_directions, train_phases, train_shapes, train_lengths = set(), set(), set(), set()
    evidence = {k: False for k in ("stationary", "boundary", "nearest", "fixture", "wrap",
                                    "soft_swept", "soft_preserved", "shear", "teaching")}
    test_data = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            data = _parse(pair["input"])
            a, b = data["a"], _grid(pair["output"])
            _check(a.shape == b.shape and not np.array_equal(a, b), "bad output dimensions or identity")
            key = _key(a)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            cycle = data["cycle"]
            successor = tuple(sorted((cycle[i], cycle[(i + 1) % 8]) for i in range(8)))
            conventions.add((data["bg"], data["mover"], successor))
            hard = _hard(data)
            expected, records = _closed_result(data, hard)
            _check(np.array_equal(expected, b), "wrong collision or sweep result")
            _check(np.array_equal(solve(a), b), "stepwise solver disagrees")
            _check(np.array_equal(a[hard], b[hard]), "hard cells moved or vanished")
            _check(np.count_nonzero(a == data["mover"]) == np.count_nonzero(b == data["mover"]),
                   "bar material was not conserved")
            sweep = np.zeros(a.shape, dtype=bool)
            canonical_sweep = np.rot90(sweep, -data["rotation"])
            for record in records:
                canonical_sweep[record["row"], record["stop"]:] = True
            _check(np.array_equal(a[~sweep], b[~sweep]), "unswept cells changed")
            soft = data["pegs"] & ~hard
            _check(np.all(b[soft & sweep] != a[soft & sweep]), "swept soft peg survived")
            # No phase-colored cells are ever created, and their exact removal mask is known.
            for color in cycle:
                _check(np.count_nonzero(b == color) == np.count_nonzero(a == color)
                       - np.count_nonzero((a == color) & soft & sweep), "incorrect peg deletion")
            if split == "test":
                test_data.append(data)
                continue
            train_directions.add(data["rotation"])
            train_phases.add(cycle[0])
            train_shapes.add(a.shape)
            train_lengths.update(record["length"] for record in records)
            evidence["soft_swept"] |= bool(np.any(soft & sweep))
            evidence["soft_preserved"] |= bool(np.any(soft & ~sweep))
            evidence["shear"] |= len({record["travel"] for record in records}) > 1
            canonical_fixture = np.rot90(data["fixture"], -data["rotation"])
            rr, cc = np.indices(a.shape)
            r, c = data["center"]
            distance8 = np.rot90((np.abs(rr - r) + np.abs(cc - c)) == 8,
                                -data["rotation"])
            for record in records:
                row, t = record["row"], record["stop"]
                evidence["stationary"] |= record["travel"] == 0
                evidence["boundary"] |= t == 0
                evidence["nearest"] |= len(record["blockers"]) >= 2 and not np.any(canonical_fixture[row])
                evidence["fixture"] |= t > 0 and bool(canonical_fixture[row, t - 1])
                evidence["wrap"] |= t > 0 and bool(distance8[row, t - 1]) and not bool(canonical_fixture[row, t - 1])
            try:
                _check_teaching(data, b)
            except AssertionError:
                pass
            else:
                evidence["teaching"] = True
    _check(len(conventions) == 1, "palette or successor changed within puzzle")
    _check(len(train_phases) >= 2, "training lacks phase transfer evidence")
    _check(all(evidence.values()), "missing training witness: " + str(evidence))
    _check(train_lengths == set(range(1, max(train_lengths) + 1)), "missing training run length")
    for data in test_data:
        _check(data["rotation"] in train_directions, "untrained release direction")
        _check(set(map(int, data["lengths"][data["lengths"] > 0])) <= train_lengths,
               "untrained run length")
        _check(data["cycle"][0] not in train_phases or data["a"].shape not in train_shapes,
               "test lacks promised phase or geometry shift")


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(np.asarray(pair["input"], dtype=np.int8)),
                                _key(np.asarray(pair["output"], dtype=np.int8)))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _hand_case_test():
    """A small specification example with manually chosen stops and deletion."""
    a = np.zeros((12, 17), dtype=np.int8)
    cycle = tuple(range(1, 9))
    _paint_clock(a, (3, 5), 9, cycle)
    # Row 7: hard color 1 at distance 8; same-colored soft peg at distance 11.
    for r, c, color in ((7, 9, 1), (7, 12, 1), (7, 3, 7), (7, 6, 2),
                        (8, 10, 1), (2, 10, 1), (5, 12, 2), (5, 11, 2),
                        (10, 13, 8), (10, 8, 7)):
        a[r, c] = color
    for row, length in ((7, 3), (8, 2), (2, 1), (5, 4)):
        a[row, 17 - length:] = 9
    expected = a.copy()
    # Nearest hard peg at column 9, boundary, clock edge at 6, stationary at 13.
    for row, stop, length in ((7, 10, 3), (8, 0, 2), (2, 7, 1)):
        expected[row, stop:] = 0
        expected[row, stop:stop + length] = 9
    _check(np.array_equal(solve(a), expected), "hand-computed specification example failed")
    # The rule is equivariant after the compass code is re-encoded.
    recolor = np.array([4, 8, 1, 7, 0, 9, 3, 2, 5, 6], dtype=np.int8)
    for k in range(4):
        x, y = np.rot90(a, k).copy(), np.rot90(expected, k).copy()
        center = _rotate_center((3, 5), a.shape, k)
        _paint_clock(x, center, 9, cycle)
        _paint_clock(y, center, 9, cycle)
        _check(np.array_equal(solve(x), y), "rotated hand case failed")
        _check(np.array_equal(solve(recolor[x]), recolor[y]), "recolored hand case failed")
    # Solve does not inherit the sampler's four-cell width cap.
    wide = np.zeros((12, 25), dtype=np.int8)
    _paint_clock(wide, (3, 5), 9, cycle)
    wide[7, 17:] = 9
    wide[8, 19:] = 9
    wide_expected = wide.copy()
    wide_expected[7:9] = 0
    wide_expected[7, :8] = 9
    wide_expected[8, :6] = 9
    _check(np.array_equal(solve(wide), wide_expected), "general run lengths failed")


def self_test():
    """Bounded semantic, diversity, API, and mutation checks over 42 seeds."""
    _hand_case_test()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(_puzzle_key(generate_puzzle(123, 7, 3)) == _puzzle_key(generate_puzzle(123, 7, 3)),
           "generation is nondeterministic")
    _check(len({_puzzle_key(p) for p in puzzles}) == 42, "whole puzzles collapsed")
    shapes, structures, directions, lengths, backgrounds, bars, cycles = [set() for _ in range(7)]
    stops = {k: 0 for k in ("stationary", "boundary", "fixture", "soft_swept", "soft_shielded")}
    phase_shift = geometry_shift = separated = within_shapes = small_test = 0
    centers, active_counts, densities = set(), set(), []
    for puzzle in puzzles:
        train = [_parse(p["input"]) for p in puzzle["train"]]
        phases = {d["cycle"][0] for d in train}
        train_shapes = {d["a"].shape for d in train}
        within_shapes += len({p["input"].shape for s in puzzle.values() for p in s}) > 1
        small_test += min(p["input"].size for p in puzzle["test"]) < max(p["input"].size for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _parse(pair["input"])
                a = data["a"]
                hard = _hard(data)
                _, records = _closed_result(data)
                if split == "test":
                    phase_shift += data["cycle"][0] not in phases
                    geometry_shift += a.shape not in train_shapes
                backgrounds.add(data["bg"])
                bars.add(data["mover"])
                cycles.add(tuple(sorted(zip(data["cycle"], data["cycle"][1:] + data["cycle"][:1]))))
                shapes.add(a.shape)
                directions.add(data["rotation"])
                lengths.update(r["length"] for r in records)
                centers.add((a.shape, data["center"]))
                active_counts.add(len(records))
                densities.append(float(np.mean(data["pegs"])))
                occupied = np.flatnonzero(data["lengths"])
                separated += bool(np.any(np.diff(occupied) > 1))
                normalized = np.zeros(a.shape, dtype=np.int8)
                normalized[data["pegs"] & ~hard] = 1
                normalized[data["pegs"] & hard] = 2
                normalized[data["fixture"]] = 3
                normalized[data["material"]] = 4
                structures.add(_key(normalized))
                fixture = np.rot90(data["fixture"], -data["rotation"])
                soft = np.rot90(data["pegs"] & ~hard, -data["rotation"])
                for record in records:
                    row, t = record["row"], record["stop"]
                    stops["stationary"] += record["travel"] == 0
                    stops["boundary"] += t == 0
                    stops["fixture"] += t > 0 and bool(fixture[row, t - 1])
                    stops["soft_swept"] += bool(np.any(soft[row, t:]))
                    stops["soft_shielded"] += bool(np.any(soft[row, :t]))
    _check(len(shapes) >= 45 and len(structures) >= 180, "structural variation collapsed")
    _check(directions == {0, 1, 2, 3} and lengths == {1, 2, 3, 4}, "direction or length missing")
    _check(len(backgrounds) >= 8 and len(bars) >= 8 and len(cycles) >= 40, "palette roles collapsed")
    _check(phase_shift >= 10 and geometry_shift >= 10, "transfer coverage collapsed")
    _check(separated >= 70 and within_shapes >= 20 and small_test >= 10, "layout variation collapsed")
    _check(len(centers) >= 180 and len(active_counts) >= 8 and max(densities) - min(densities) > 0.12,
           "clock, counts, or density collapsed")
    _check(all(v >= 25 for v in stops.values()), "collision regimes missing")
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(9000 + 10 * a + b, a, b)
        _check(len(p["train"]) == a and len(p["test"]) == b, "count request ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(9, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = puzzles[0]
    mutated = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]]
               for s in ("train", "test")}
    d = _parse(mutated["train"][0]["input"])
    r, c = d["center"]
    mutated["train"][0]["output"][r, c] = d["bg"]
    try:
        validate_generated_puzzle(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted clock destruction")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in ("train", "test")),
            "shapes": len(shapes), "semantic_structures": len(structures),
            "phase_transfers": phase_shift, "geometry_transfers": geometry_shift,
            "collision_witnesses": stops}


if __name__ == "__main__":
    print(self_test())
