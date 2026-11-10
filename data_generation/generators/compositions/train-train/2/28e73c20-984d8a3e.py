"""Gated spiral conveyors: count-preserving band shifts in curved chambers.

The solver reads only the supplied grid. Generation uses a cell walker and
closed-form block exchange; solving uses boundary lists and one-cell exchanges.
"""

from itertools import permutations
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _walk_path(height, width):
    """A's ordered spiral, including its intentional touching inner ending."""
    row = col = 0
    path = [(row, col)]
    for _ in range(width - 1):
        col += 1
        path.append((row, col))
    directions = ((1, 0), (0, -1), (-1, 0), (0, 1))
    run = 0
    while True:
        length = (height if run % 2 == 0 else width) - 1 - 2 * (run // 2)
        if length <= 0:
            return path
        dr, dc = directions[run % 4]
        for _ in range(length):
            row, col = row + dr, col + dc
            path.append((row, col))
        run += 1


def _boundary_path(height, width):
    """Separate ordered construction: enumerate successively inset edges."""
    path = [(0, col) for col in range(width)]
    for inset in range(30):
        if height - 1 - 2 * inset <= 0:
            break
        if inset % 2 == 0:
            path.extend((row, width - 1 - inset)
                        for row in range(inset + 1, height - inset))
        else:
            path.extend((row, inset - 1)
                        for row in range(height - inset - 1, inset, -1))
        if width - 1 - 2 * inset <= 0:
            break
        if inset % 2 == 0:
            path.extend((height - 1 - inset, col)
                        for col in range(width - inset - 2, inset - 1, -1))
        else:
            path.extend((inset + 1, col)
                        for col in range(inset, width - inset - 1))
    return path


def _oriented_path(shape, orientation, builder):
    turns = orientation % 4
    height, width = shape[::-1] if turns % 2 else shape
    labels = np.zeros((height, width), dtype=np.int16)
    for index, cell in enumerate(builder(height, width), 1):
        labels[cell] = index
    if orientation >= 4:
        labels = np.fliplr(labels)
    labels = np.rot90(labels, turns)
    coordinates = np.argwhere(labels > 0)
    order = np.argsort(labels[labels > 0])
    return [tuple(int(x) for x in coordinates[index]) for index in order]


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return grid


def _profile(values, roles):
    first, band, opposite = roles
    index = 0
    while index < len(values) and values[index] == first:
        index += 1
    a = index
    while index < len(values) and values[index] == band:
        index += 1
    b = index - a
    if b == 0 or any(color != opposite for color in values[index:]):
        return None
    return a, b, len(values) - index


def _decode(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not all(np.all(edge == background) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("the spiral needs a complete background margin")
    occupied = np.argwhere(grid != background)
    if not len(occupied):
        raise ValueError("missing spiral")
    top, left = occupied.min(axis=0)
    bottom, right = occupied.max(axis=0) + 1
    cropped = grid[top:bottom, left:right]
    if min(cropped.shape) < 7:
        raise ValueError("spiral bounding rectangle must have sides at least 7")
    candidates = []
    for orientation in range(8):
        path = _oriented_path(cropped.shape, orientation, _boundary_path)
        mask = np.zeros(cropped.shape, dtype=bool)
        for cell in path:
            mask[cell] = True
        if np.array_equal(mask, cropped != background):
            candidates.append((orientation, path))
    if len(candidates) != 1:
        raise ValueError("spiral footprint does not determine one ordered path")
    orientation, path = candidates[0]
    path = [(row + int(top), col + int(left)) for row, col in path]
    sequence = [int(grid[cell]) for cell in path]
    gate = sequence[0]
    if sequence[-1] != gate:
        raise ValueError("both path ends must be gates of the same color")
    gates = [i for i, color in enumerate(sequence) if color == gate]
    intervals = [(a + 1, b) for a, b in zip(gates, gates[1:])]
    if not intervals or any(hi <= lo for lo, hi in intervals):
        raise ValueError("gates must enclose nonempty chambers")
    material_colors = set(sequence) - {gate}
    if len(material_colors) != 3:
        raise ValueError("expected three material colors in addition to gates")
    choices = []
    for roles in permutations(sorted(material_colors)):
        profiles = [_profile(sequence[lo:hi], roles) for lo, hi in intervals]
        if all(profile is not None for profile in profiles):
            choices.append((roles, profiles))
    if len(choices) != 1:
        raise ValueError("ordered material roles are ambiguous")
    roles, profiles = choices[0]
    return dict(grid=grid, background=background, gate=gate, path=path,
                sequence=sequence, intervals=intervals, roles=roles,
                profiles=profiles, orientation=orientation,
                box=(int(top), int(left), int(bottom), int(right)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Slide bands by adjacent arclength steps; never use generation metadata."""
    data = _decode(input_grid)
    output = data["grid"].copy()
    for (lo, hi), (a, b, _) in zip(data["intervals"], data["profiles"]):
        values = data["sequence"][lo:hi].copy()
        start, end = a, a + b
        pivot = len(values) // 2 + 1  # number of allowed cells, not a zero-based index
        while start > 0 and end > pivot:
            values[start - 1], values[end - 1] = values[end - 1], values[start - 1]
            start -= 1
            end -= 1
        for cell, color in zip(data["path"][lo:hi], values):
            output[cell] = color
    return output


def _event(profile, length):
    a, b, c = profile
    pivot = length // 2 + 1
    shift = min(a, max(0, a + b - pivot))
    events = set()
    if a + b < pivot:
        events.add("before")
    if a + b == pivot:
        events.add("exact")
    if 0 < shift < a:
        events.add("partial")
    if a > 0 and shift == a:
        events.add("full")
    if a > 0 and b > pivot and shift == a:
        events.add("residual")
    if a == 0:
        events.add("no_first")
    if c == 0:
        events.add("no_suffix")
    return events


def _exchange(a, b, c, roles):
    first, band, opposite = roles
    pivot = (a + b + c) // 2 + 1
    shift = min(a, max(0, a + b - pivot))
    return [first] * (a - shift) + [band] * b + [first] * shift + [opposite] * c


def _straight_chunks(path, lo, hi):
    start, previous = lo, None
    for index in range(lo + 1, hi):
        direction = (path[index][0] - path[index - 1][0],
                     path[index][1] - path[index - 1][1])
        if previous is not None and direction != previous:
            yield start, index
            start = index
        previous = direction
    yield start, hi


def _legwise_ablation(data):
    """Wrong alternative: give each straight fragment its own midpoint."""
    answer = data["grid"].copy()
    for lo, hi in data["intervals"]:
        for start, end in _straight_chunks(data["path"], lo, hi):
            values = data["sequence"][start:end]
            profile = _profile(values, data["roles"])
            if profile is not None:
                for cell, color in zip(data["path"][start:end],
                                       _exchange(*profile, data["roles"])):
                    answer[cell] = color
    return answer


def _global_midpoint_ablation(data):
    """Wrong alternative: all chambers use the whole cable's halfway point."""
    answer = data["grid"].copy()
    first, band, opposite = data["roles"]
    pivot = len(data["path"]) // 2 + 1
    for (lo, hi), (a, b, c) in zip(data["intervals"], data["profiles"]):
        shift = min(a, max(0, lo + a + b - pivot))
        values = ([first] * (a - shift) + [band] * b
                  + [first] * shift + [opposite] * c)
        for cell, color in zip(data["path"][lo:hi], values):
            answer[cell] = color
    return answer


def _normalized_cells(cells):
    if not cells:
        return ()
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _observations(data, output):
    lengths, events = [], set()
    bends = shape_changes = changed = 0
    even_pivot_witness = False
    for (lo, hi), profile in zip(data["intervals"], data["profiles"]):
        length = hi - lo
        lengths.append(length)
        events.update(_event(profile, length))
        a, b, c = profile
        shift = min(a, max(0, a + b - (length // 2 + 1)))
        if shift:
            changed += 1
            even_pivot_witness |= length % 2 == 0 and shift < a
            old_cells = data["path"][lo + a:lo + a + b]
            new_cells = data["path"][lo + a - shift:lo + a + b - shift]
            shape_changes += _normalized_cells(old_cells) != _normalized_cells(new_cells)
            bends += len(list(_straight_chunks(data["path"], lo + a - shift,
                                              lo + a + b))) > 1
    return dict(lengths=lengths, events=events, changed=changed, bends=bends,
                shape_changes=shape_changes,
                even_pivot_witness=even_pivot_witness,
                global_diff=not np.array_equal(_global_midpoint_ablation(data), output),
                legwise_diff=not np.array_equal(_legwise_ablation(data), output))


def _sample_profile(rng, length, event):
    p = length // 2 + 1
    if event in ("before", "exact"):
        endpoint = int(rng.integers(2, p)) if event == "before" else p
        a = int(rng.integers(1, endpoint))
        b = endpoint - a
    elif event == "partial":
        b = int(rng.integers(1, p))
        a = int(rng.integers(p - b + 1, length - b))
    elif event == "residual":
        b = int(rng.integers(p + 1, length - 1))
        a = int(rng.integers(1, length - b))
    elif event == "no_first":
        a, b = 0, int(rng.integers(1, length))
    elif event == "no_suffix":
        a = int(rng.integers(1, length))
        b = length - a
    else:
        a = int(rng.integers(1, length - 1))
        b = int(rng.integers(1, length - a))
    return a, b, length - a - b


def _partition(rng, total, count, minimum=9, maximum=50):
    parts = []
    for remaining_count in range(count - 1, -1, -1):
        low = max(minimum, total - remaining_count * maximum)
        high = min(maximum, total - remaining_count * minimum)
        value = int(rng.integers(low, high + 1))
        parts.append(value)
        total -= value
    rng.shuffle(parts)
    return parts


def _make_pair(rng, index, test, palette):
    for _ in range(160):
        # Independent sides and margins let deep, wide, tall and square tracks mix.
        low, high = (12, 28) if test else (9, 22)
        shape = (int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1)))
        orientation = int(rng.integers(8))
        path = _oriented_path(shape, orientation, _walk_path)
        n = len(path)
        if test:
            max_count = min(5, (n - 52) // 10)
            if max_count < 2:
                continue
            count = int(rng.integers(2, max_count + 1))
            total = n - count - 1
            long_length = int(rng.integers(60, total - 9 * (count - 1) + 1))
            lengths = _partition(rng, total - long_length, count - 1,
                                 maximum=total - long_length)
            long_index = int(rng.integers(count))
            lengths.insert(long_index, long_length)
        else:
            min_count = max(3, (n - 1 + 50) // 51)
            max_count = min(7, (n - 1) // 10)
            if min_count > max_count:
                continue
            count = int(rng.integers(min_count, max_count + 1))
            lengths = _partition(rng, n - count - 1, count)
            if len({length % 2 for length in lengths}) < 2:
                continue
        events = [str(rng.choice(["ordinary", "partial", "partial", "before",
                                  "exact", "residual", "no_first", "no_suffix"]))
                  for _ in lengths]
        if test:
            events[long_index] = str(rng.choice(["partial", "residual"]))
        else:
            slots = rng.permutation(count)
            if index == 0:
                # A partial shift in an even chamber distinguishes the two
                # possible middle cells, even with only two demonstrations.
                partial_slot = int(rng.choice([i for i, length in enumerate(lengths)
                                               if length % 2 == 0]))
                before_slot = int(rng.choice([i for i in range(count) if i != partial_slot]))
                events[partial_slot], events[before_slot] = "partial", "before"
            else:
                required = ("exact", "residual") if index == 1 else ("partial",)
                for slot, event in zip(slots, required):
                    events[int(slot)] = event
        profiles = [_sample_profile(rng, length, event)
                    for length, event in zip(lengths, events)]
        background, gate, first, band, opposite = palette
        roles = first, band, opposite
        before, after = [gate], [gate]
        for (a, b, c) in profiles:
            before.extend([first] * a + [band] * b + [opposite] * c + [gate])
            after.extend(_exchange(a, b, c, roles) + [gate])
        # Margins can be asymmetric, but every border stays background.
        vertical_space, horizontal_space = 30 - shape[0], 30 - shape[1]
        top = int(rng.integers(1, min(3, vertical_space - 1) + 1))
        bottom = int(rng.integers(1, min(3, vertical_space - top) + 1))
        left = int(rng.integers(1, min(3, horizontal_space - 1) + 1))
        right = int(rng.integers(1, min(3, horizontal_space - left) + 1))
        inp = np.full((shape[0] + top + bottom, shape[1] + left + right),
                      background, dtype=np.int8)
        out = inp.copy()
        for (row, col), old, new in zip(path, before, after):
            inp[row + top, col + left] = old
            out[row + top, col + left] = new
        data = _decode(inp)
        obs = _observations(data, out)
        if obs["shape_changes"] and obs["legwise_diff"] and obs["global_diff"]:
            return {"input": inp, "output": out}
    raise RuntimeError("could not sample a chamber with a visible curved shift")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first_palette = tuple(int(x) for x in rng.choice(10, 5, replace=False))
    stable_scaffold = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if section == "train" and index == 0:
                palette = first_palette
            elif section == "train" and index == 1:
                bg, gate, a, b, c = first_palette
                palette = (bg, gate, c, a, b)
            elif stable_scaffold:
                bg, gate = first_palette[:2]
                materials = rng.choice([c for c in range(10) if c not in (bg, gate)],
                                       3, replace=False)
                palette = (bg, gate, *(int(x) for x in materials))
            else:
                palette = tuple(int(x) for x in rng.choice(10, 5, replace=False))
            for _ in range(30):
                pair = _make_pair(rng, index, section == "test", palette)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("failed to construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle fields")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_events, train_lengths, test_lengths = set(), set(), [], []
    even_pivot_witness = False
    train_roles = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair fields")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data = _decode(inp)
            _require(np.array_equal(inp == data["background"], out == data["background"]),
                     "spiral footprint or background changed")
            _require(np.array_equal(inp == data["gate"], out == data["gate"]), "gate moved")
            for (lo, hi), (a, b, c) in zip(data["intervals"], data["profiles"]):
                values = [int(out[cell]) for cell in data["path"][lo:hi]]
                expected = _exchange(a, b, c, data["roles"])
                _require(values == expected, "wrong bounded chamber exchange")
                _require(np.array_equal(np.bincount(values, minlength=10),
                                        np.bincount(data["sequence"][lo:hi], minlength=10)),
                         "chamber color counts changed")
                if c:
                    _require(values[-c:] == [data["roles"][2]] * c, "opposite suffix changed")
            _require(np.array_equal(solve(inp), out), "input-only solver disagrees")
            _require(not np.array_equal(inp, out), "identity pair")
            obs = _observations(data, out)
            _require(obs["shape_changes"] > 0 and obs["legwise_diff"] and obs["global_diff"],
                     "pair does not distinguish curved arclength from rigid/legwise motion")
            if section == "train":
                train_events.update(obs["events"])
                even_pivot_witness |= obs["even_pivot_witness"]
                train_lengths.extend(obs["lengths"])
                train_roles.append(data["roles"])
            else:
                test_lengths.append(max(obs["lengths"]))
    _require({"before", "exact", "partial", "full", "residual"} <= train_events,
             "training lacks essential shift evidence")
    _require({length % 2 for length in train_lengths} == {0, 1}, "missing pivot parity")
    _require(even_pivot_witness, "even midpoint convention is not distinguished")
    _require(all(length > max(train_lengths) for length in test_lengths),
             "tests must contain longer chambers than demonstrations")
    _require(any(color in roles and roles.index(color) != train_roles[0].index(color)
                 for roles in train_roles[1:] for color in train_roles[0]),
             "training does not expose material role reassignment")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded semantic, diversity, determinism and API checks over 42 seeds."""
    for height in range(7, 29):
        for width in range(7, 29):
            path = _walk_path(height, width)
            _require(path == _boundary_path(height, width), "ordered path builders disagree")
            _require(len(path) == len(set(path)), "spiral revisits a cell")
            _require(all(0 <= r < height and 0 <= c < width for r, c in path), "path outside box")
            _require(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                         for a, b in zip(path, path[1:])), "nonadjacent path steps")
    # Valid family members outside the random sampler's narrower regimes:
    # one chamber, tiny chambers, extreme aspect ratios, all orientations,
    # and the minimum supported footprint. These have no training metadata.
    for shape in ((7, 7), (7, 28), (28, 7), (8, 8), (28, 28)):
        for orientation in range(8):
            path = _oriented_path(shape, orientation, _walk_path)
            n = len(path)
            for profiles in (((2, n - 5, 1),),
                             ((0, 1, 0), (1, 2, 0), (2, n - 11, 1))):
                old, new = [5], [5]
                for a, b, c in profiles:
                    old.extend([2] * a + [8] * b + [1] * c + [5])
                    new.extend(_exchange(a, b, c, (2, 8, 1)) + [5])
                _require(len(old) == n, "manual fixture length")
                inp = np.zeros((shape[0] + 2, shape[1] + 2), dtype=np.int8)
                out = inp.copy()
                for (r, c), before, after in zip(path, old, new):
                    inp[r + 1, c + 1], out[r + 1, c + 1] = before, after
                _require(np.array_equal(solve(inp), out), "general family solver fixture")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, footprints, partitions, roles, orientations, structures = set(), set(), set(), set(), set(), set()
    counts, lengths, shifts, parity_pairs, test_spatial_relations = set(), set(), set(), set(), set()
    rectangles = squares = omitted = shape_changes = bent_changes = 0
    for puzzle in puzzles:
        train_areas = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                data = _decode(pair["input"])
                obs = _observations(data, pair["output"])
                top, left, bottom, right = data["box"]
                h, w = bottom - top, right - left
                if section == "train":
                    train_areas.append(h * w)
                else:
                    test_spatial_relations.add("larger" if h * w > max(train_areas)
                                               else "smaller" if h * w < min(train_areas) else "overlap")
                shapes.add(pair["input"].shape)
                footprints.add((h, w, data["orientation"]))
                partitions.add(tuple(obs["lengths"]))
                counts.add(len(data["intervals"]))
                lengths.update(obs["lengths"])
                roles.add((data["background"], data["gate"], *data["roles"]))
                orientations.add(data["orientation"])
                parity_pairs.add((h % 2, w % 2))
                rectangles += h != w
                squares += h == w
                omitted += bool({"no_first", "no_suffix"} & obs["events"])
                shape_changes += obs["shape_changes"]
                bent_changes += obs["bends"]
                structures.add((h, w, data["orientation"], tuple(obs["lengths"]),
                                tuple(data["profiles"])))
                for (a, b, c) in data["profiles"]:
                    shifts.add(min(a, max(0, a + b - ((a + b + c) // 2 + 1))))
        _require(np.array_equal(solve(puzzle["test"][0]["input"].tolist()),
                                puzzle["test"][0]["output"]), "nested-list solving")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "duplicate seeded puzzles")
    _require(len(shapes) >= 70 and len(structures) >= 200 and len(partitions) >= 180,
             "insufficient geometry/partition/profile diversity")
    _require(orientations == set(range(8)) and parity_pairs == {(0, 0), (0, 1), (1, 0), (1, 1)},
             "missing orientation or geometry parity")
    _require(rectangles >= 100 and squares >= 3 and omitted >= 20, "narrow geometry/edge cases")
    _require(len(roles) >= 120 and len(counts) >= 5 and len(lengths) >= 60 and len(shifts) >= 20,
             "narrow palettes, chamber counts, lengths or shifts")
    _require({"larger", "smaller", "overlap"} <= test_spatial_relations, "spatial scale diversity")
    _require(_puzzle_key(generate_puzzle(794203441)) == _puzzle_key(generate_puzzle(794203441)),
             "seed not reproducible")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(1000 + train * 10 + test, train, test)
            _require((len(p["train"]), len(p["test"])) == (train, test), "explicit counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validation must notice corruption, including a changed fixed gate.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    data = _decode(bad["train"][0]["input"])
    bad["train"][0]["output"][data["path"][0]] = data["roles"][0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("gate corruption was not detected")
    return dict(seeds=42, pairs=210, shapes=len(shapes), footprints=len(footprints),
                partitions=len(partitions), structures=len(structures), palettes=len(roles),
                orientations=sorted(orientations), chamber_counts=sorted(counts),
                chamber_lengths=len(lengths), shift_distances=len(shifts),
                rectangles=rectangles, squares=squares, omitted_outer_run_pairs=omitted,
                shape_changing_bands=shape_changes, bent_changed_bands=bent_changes,
                test_spatial_relations=sorted(test_spatial_relations))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
