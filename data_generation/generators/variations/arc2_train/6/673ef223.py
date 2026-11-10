"""Matched boundary bars with stopped source and full receiver rays."""

import hashlib
from itertools import combinations

import numpy as np


# SHA-256 of "height,width:" followed by uint8 input bytes from task.json.
_SOURCE_INPUTS = frozenset({
    "9ab71c19c30e33f0ca56d8b5e7569d0ca9acdac4dfb70ce23a29a0175adaba06",
    "46a6cfeca74528dd4ae81165eb385216451af53a6fef3bae4306208a6c548d9a",
    "cb77b202e37feb99cce1ad59232bdbf86524c9eb6183f8db35e5696d2079ce83",
    "9a3a730b5e2277b3e4257543bed1595859d2deebeaf52b3b7a1fc80257110a13",
})


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        train = max(2, test + 1) + (int(seed) // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = test + 1 + int(seed) % (10 - test)
    elif test is None:
        test = 1 + int(seed) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _input_digest(grid):
    rows, columns = grid.shape
    prefix = f"{rows},{columns}:".encode()
    return hashlib.sha256(prefix + np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _choose_slots(rng, length, count, asymmetric=False, adjacent=False):
    options = []
    for slots in combinations(range(length), count):
        if asymmetric and set(slots) == {length - 1 - slot for slot in slots}:
            continue
        if adjacent and not any(b == a + 1 for a, b in zip(slots, slots[1:])):
            continue
        options.append(slots)
    if not options:
        raise ValueError("no active-slot pattern satisfies the request")
    return options[int(rng.integers(len(options)))]


def _longest_run(slots):
    longest = run = 1
    for a, b in zip(slots, slots[1:]):
        run = run + 1 if b == a + 1 else 1
        longest = max(longest, run)
    return longest


def _choose_distances(rng, slots, depth):
    distances = []
    for index, slot in enumerate(slots):
        allowed = list(range(2, depth - 1))
        if index and slot == slots[index - 1] + 1:
            allowed = [d for d in allowed if d != distances[-1]]
        _require(bool(allowed), "no isolated marker position")
        mode = int(rng.integers(4))
        if mode == 0:
            preferred = [d for d in allowed if d <= 4]
        elif mode == 1:
            preferred = [d for d in allowed if d >= depth - 4]
        else:
            preferred = allowed
        choices = preferred or allowed
        distances.append(choices[int(rng.integers(len(choices)))])
    return tuple(distances)


def _vary_colors(rng, terminal, base, mode):
    if mode == "stable":
        return base
    colors = list(base)
    indices = (int(rng.integers(3)),) if mode == "one" else (0, 1, 2)
    for index in indices:
        forbidden = {terminal, *colors}
        choices = [value for value in range(10) if value not in forbidden]
        colors[index] = choices[int(rng.integers(len(choices)))]
    return tuple(colors)


def _make_pair(rng, terminal, colors, orientation, source_side,
               length, count, asymmetric=False, adjacent=False, size_mode="free"):
    slots = _choose_slots(rng, length, count, asymmetric, adjacent)
    along_min = 2 * length + 2
    if size_mode == "compact":
        along = int(rng.integers(along_min, 24))
    elif size_mode == "large":
        along = int(rng.integers(max(25, along_min), 31))
    else:
        along = int(rng.integers(along_min, 31))
    depth_min = max(6, 2 * _longest_run(slots) + 3)
    if size_mode == "compact":
        depth = int(rng.integers(depth_min, 21))
    elif size_mode == "large":
        depth = int(rng.integers(max(25, depth_min), 31))
    else:
        depth = int(rng.integers(depth_min, 31))
    gap = int(rng.integers(2, min(12, along - 2 * length) + 1))
    first_start = int(rng.integers(0, along - 2 * length - gap + 1))
    second_start = first_start + length + gap
    if bool(rng.integers(2)):
        source_start, receiver_start = first_start, second_start
    else:
        source_start, receiver_start = second_start, first_start
    distances = _choose_distances(rng, slots, depth)
    background, scaffold, signal = colors
    inp = np.full((along, depth), background, dtype=np.int8)
    source_edge = 0 if source_side == 0 else depth - 1
    receiver_edge = depth - 1 - source_edge
    inp[source_start:source_start + length, source_edge] = scaffold
    inp[receiver_start:receiver_start + length, receiver_edge] = scaffold
    out = inp.copy()
    for slot, distance in zip(slots, distances):
        row = source_start + slot
        column = distance if source_side == 0 else depth - 1 - distance
        inp[row, column] = signal
        if source_side == 0:
            out[row, 1:column] = signal
            out[receiver_start + slot, :depth - 1] = signal
        else:
            out[row, column + 1:depth - 1] = signal
            out[receiver_start + slot, 1:] = signal
        out[row, column] = terminal
    if orientation == "horizontal":
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _consecutive(values):
    return len(values) > 0 and np.array_equal(values, np.arange(values[0], values[0] + len(values)))


def _analyze_input(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "input must be a 2D array")
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 3 and np.count_nonzero(counts == counts.max()) == 1,
             "input needs three identifiable role colors")
    background = int(values[np.argmax(counts)])
    candidates = []
    for scaffold in map(int, values):
        if scaffold == background:
            continue
        for orientation in ("vertical", "horizontal"):
            canonical = grid if orientation == "vertical" else grid.T
            along, depth = canonical.shape
            positions = np.argwhere(canonical == scaffold)
            if not np.all((positions[:, 1] == 0) | (positions[:, 1] == depth - 1)):
                continue
            starts = []
            for edge in (0, depth - 1):
                rows = np.flatnonzero(canonical[:, edge] == scaffold)
                if not (4 <= len(rows) <= 9 and _consecutive(rows)):
                    break
                starts.append((int(rows[0]), len(rows)))
            if len(starts) != 2 or starts[0][1] != starts[1][1]:
                continue
            length = starts[0][1]
            if len(positions) != 2 * length:
                continue
            if abs(starts[0][0] - starts[1][0]) < length + 2:
                continue
            candidates.append((scaffold, orientation, starts[0][0], starts[1][0], length))
    _require(len(candidates) == 1, "input needs exactly two matched opposite bars")
    scaffold, orientation, start0, start1, length = candidates[0]
    canonical = grid if orientation == "vertical" else grid.T
    along, depth = canonical.shape
    signal = next(int(v) for v in values if int(v) not in (background, scaffold))
    markers = np.argwhere(canonical == signal)
    _require(1 <= len(markers) <= 5, "input needs one to five markers")
    _require(len(set(map(int, markers[:, 0]))) == len(markers), "markers need distinct slots")
    in0 = all(start0 <= int(row) < start0 + length for row in markers[:, 0])
    in1 = all(start1 <= int(row) < start1 + length for row in markers[:, 0])
    _require(in0 != in1, "markers must select exactly one bar")
    side = 0 if in0 else 1
    source_start = start0 if in0 else start1
    receiver_start = start1 if in0 else start0
    markers = sorted((int(row), int(col)) for row, col in markers)
    slots = tuple(row - source_start for row, _ in markers)
    distances = tuple(col if side == 0 else depth - 1 - col for _, col in markers)
    _require(all(2 <= distance <= depth - 2 for distance in distances),
             "markers need a visible source gap and interior endpoint")
    _require(all(b > a for a, b in zip(slots, slots[1:])), "marker slots repeat")
    _require(all(b != a + 1 or d2 != d1
                 for a, b, d1, d2 in zip(slots, slots[1:], distances, distances[1:])),
             "neighboring input markers must remain isolated")
    rebuilt = np.full_like(canonical, background)
    rebuilt[start0:start0 + length, 0] = scaffold
    rebuilt[start1:start1 + length, depth - 1] = scaffold
    for row, col in markers:
        rebuilt[row, col] = signal
    _require(np.array_equal(canonical, rebuilt), "input has extra cells")
    return {"orientation": orientation, "side": side, "source_start": source_start,
            "receiver_start": receiver_start, "length": length, "slots": slots,
            "distances": distances, "background": background, "scaffold": scaffold,
            "signal": signal, "markers": markers, "depth": depth, "along": along}


def _predict(grid, terminal):
    info = _analyze_input(grid)
    canonical = grid.copy() if info["orientation"] == "vertical" else grid.T.copy()
    signal, side, depth = info["signal"], info["side"], info["depth"]
    for slot, (_, col) in zip(info["slots"], info["markers"]):
        row = info["source_start"] + slot
        receiver_row = info["receiver_start"] + slot
        if side == 0:
            canonical[row, 1:col] = signal
            canonical[receiver_row, :depth - 1] = signal
        else:
            canonical[row, col + 1:depth - 1] = signal
            canonical[receiver_row, 1:] = signal
        canonical[row, col] = terminal
    return canonical if info["orientation"] == "vertical" else canonical.T.copy()


def _analyze_pair(inp, out):
    info = _analyze_input(inp)
    _require(inp.shape == out.shape, "output shape differs")
    new = set(map(int, np.unique(out))) - set(map(int, np.unique(inp)))
    _require(len(new) == 1, "output needs one new terminal color")
    terminal = new.pop()
    _require(np.array_equal(out, _predict(inp, terminal)), "ray transformation differs")
    info["terminal"] = terminal
    return info


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    terminal = int(rng.integers(10))
    base = tuple(int(v) for v in rng.choice([v for v in range(10) if v != terminal], 3, replace=False))
    variable_palette = bool(rng.integers(5))
    larger_test_mode = not bool(rng.integers(4))
    single_orientation = not bool(rng.integers(4))
    fixed_orientation = ("vertical", "horizontal")[int(rng.integers(2))]
    first_side = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for group, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(80):
                if single_orientation:
                    orientation = fixed_orientation
                elif group == "train" and index < 2:
                    orientation = ("vertical", "horizontal")[index]
                else:
                    orientation = ("vertical", "horizontal")[int(rng.integers(2))]
                source_side = first_side ^ index if group == "train" and index < 2 else int(rng.integers(2))
                length = int(rng.integers(4, 10))
                if group == "train" and index == 0:
                    active_count = int(rng.integers(2, min(4, length - 1) + 1))
                elif group == "test" and bool(rng.integers(2)):
                    active_count = int(rng.integers(3, min(5, length) + 1))
                else:
                    active_count = int(rng.integers(1, min(5, length) + 1))
                if group == "train" and index == 0:
                    colors = base
                elif not variable_palette:
                    colors = base
                elif group == "train" and index == 1:
                    colors = _vary_colors(rng, terminal, base, "one" if bool(rng.integers(2)) else "all")
                else:
                    mode = ("stable", "one", "all")[int(rng.integers(3))]
                    colors = _vary_colors(rng, terminal, base, mode)
                pair = _make_pair(rng, terminal, colors, orientation, source_side,
                                  length, active_count,
                                  asymmetric=group == "train" and index == 0,
                                  adjacent=group == "train" and index == 0,
                                  size_mode=("compact" if larger_test_mode else "free")
                                  if group == "train" else
                                  ("large" if larger_test_mode and index == 0 else "free"))
                key = _grid_key(pair["input"])
                if key not in seen and _input_digest(pair["input"]) not in _SOURCE_INPUTS:
                    seen.add(key)
                    puzzle[group].append(pair)
                    break
            else:
                raise RuntimeError("unable to produce a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen_inputs, seen_pairs, infos = set(), set(), {"train": [], "test": []}
    for group in ("train", "test"):
        for pair in puzzle[group]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(all(1 <= side <= 30 for side in grid.shape), "grid size outside ARC limits")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must use integer colors")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal ARC color")
            ikey = _grid_key(inp)
            pkey = (ikey, _grid_key(out))
            _require(ikey not in seen_inputs and pkey not in seen_pairs, "duplicate pair")
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            _require(_input_digest(inp) not in _SOURCE_INPUTS, "copied original input")
            infos[group].append(_analyze_pair(inp, out))
    terminals = {info["terminal"] for group in infos.values() for info in group}
    _require(len(terminals) == 1, "terminal color must stay fixed within a puzzle")
    _require({info["orientation"] for info in infos["test"]}.issubset(
                 {info["orientation"] for info in infos["train"]}),
             "test orientation lacks a training demonstration")
    _require({info["side"] for info in infos["train"]} == {0, 1},
             "training must demonstrate both inward directions")
    _require(any(set(info["slots"]) != {info["length"] - 1 - slot for slot in info["slots"]}
                 for info in infos["train"]), "training must distinguish direct slots from reflection")
    _require(any(any(b == a + 1 for a, b in zip(info["slots"], info["slots"][1:]))
                 for info in infos["train"]), "training must show adjacent active slots")
    train_palettes = {(info["background"], info["scaffold"], info["signal"])
                      for info in infos["train"]}
    all_palettes = {(info["background"], info["scaffold"], info["signal"])
                    for group in infos.values() for info in group}
    _require(len(train_palettes) > 1 or len(all_palettes) == 1,
             "test-only color variation lacks a demonstration")


def _puzzle_key(puzzle):
    return tuple((group, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[group])) for group in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, puzzles = set(), set(), set()
    shapes, layouts, spec_combos, palettes = set(), set(), set(), set()
    adjacent_pairs = near_pairs = long_pairs = later_sources = right_tests = 0
    small_tests = large_tests = varied_palettes = unequal_distances = 0
    mixed_orientation_puzzles = single_orientation_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_key(puzzle))
        local_palettes = set()
        train_shapes = [p["input"].shape for p in puzzle["train"]]
        orientations = {_analyze_input(p["input"])["orientation"] for group in ("train", "test")
                        for p in puzzle[group]}
        mixed_orientation_puzzles += len(orientations) == 2
        single_orientation_puzzles += len(orientations) == 1
        for group in ("train", "test"):
            for pair in puzzle[group]:
                info = _analyze_pair(pair["input"], pair["output"])
                _require(not np.array_equal(pair["input"], pair["output"]), "empty transformation")
                _require(np.array_equal(_predict(pair["input"], info["terminal"]), pair["output"]),
                         "input-only prediction disagrees")
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, info["orientation"], info["side"],
                             info["source_start"], info["receiver_start"],
                             info["length"], info["slots"], info["distances"]))
                spec_combos.add((info["orientation"], info["side"], info["length"], len(info["slots"])))
                palette = (info["background"], info["scaffold"], info["signal"], info["terminal"])
                palettes.add(palette)
                local_palettes.add(palette)
                adjacent_pairs += any(b == a + 1 for a, b in zip(info["slots"], info["slots"][1:]))
                near_pairs += min(info["distances"]) == 2
                long_pairs += max(info["distances"]) >= 12
                later_sources += info["source_start"] > info["receiver_start"]
                unequal_distances += len(set(info["distances"])) > 1
                if group == "test":
                    right_tests += info["side"] == 1
                    small_tests += all(pair["input"].shape[i] <= max(s[i] for s in train_shapes)
                                       for i in (0, 1))
                    large_tests += any(pair["input"].shape[i] > max(s[i] for s in train_shapes)
                                       for i in (0, 1))
        varied_palettes += len(local_palettes) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack range")
    _require(len(puzzles) == 42 and len(shapes) >= 60 and len(layouts) >= 150,
             "structural diversity is too narrow")
    _require(len(spec_combos) >= 35 and len(palettes) >= 40, "role or geometry combinations too narrow")
    _require(all(x >= 10 for x in (adjacent_pairs, near_pairs, long_pairs, later_sources,
                                  right_tests, small_tests, large_tests, unequal_distances)),
             "a requested variation is underrepresented")
    _require(varied_palettes >= 20, "within-puzzle color variation is too rare")
    _require(mixed_orientation_puzzles >= 20 and single_orientation_puzzles >= 5,
             "orientation regimes lack puzzle-level diversity")
    a = generate_puzzle(918273, 7, 3)
    _require(_puzzle_key(a) == _puzzle_key(generate_puzzle(918273, 7, 3)), "nondeterministic")
    _require(_puzzle_key(a) != _puzzle_key(generate_puzzle(918274, 7, 3)), "seeds coincide")
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        _require(train is None or len(p["train"]) == train, "train count ignored")
        _require(test is None or len(p["test"]) == test, "test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(999, *args)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {args}")
