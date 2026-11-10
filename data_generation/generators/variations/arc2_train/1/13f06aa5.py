"""Synthetic ARC puzzles with directional, signal-bearing bars."""

import hashlib
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_WIDTHS = (3, 5, 7)
_SOURCE_INPUT_DIGESTS = {
    "44e6084d4e5fe49753abb0db88db83d9dfe1571f983ecb52a18bac75fa0311ed",
    "786b8e0b91388316ed4810c1877531e23abf57a68b27246b5c5468deee5176f2",
    "bf9e5b608e59efdcb48cdb18eea5977d60914dd855b6bc9ee96c250b908c6808",
    "0d5b4c99734ea2c008c9cc7a6c7062314155abcc0445297907eac35bdcb628b3",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _signature(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _source_digest(grid):
    h, w = grid.shape
    return hashlib.sha256(bytes((h, w)) + grid.astype(np.uint8).tobytes()).hexdigest()


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        # Independent draws avoid encoding a direction schedule in the counts.
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _footprint(center, direction, width):
    row, col = center
    dr, dc = _DIRECTIONS[direction]
    pr, pc = -dc, dr
    half = width // 2
    front = {(row + offset * pr, col + offset * pc) for offset in range(-half, half + 1)}
    rear = {(row - dr + offset * pr, col - dc + offset * pc)
            for offset in range(-half + 1, half)}
    return frozenset(front | rear)


def _ray(shape, center, direction):
    h, w = shape
    row, col = center
    dr, dc = _DIRECTIONS[direction]
    result = []
    while True:
        row += dr
        col += dc
        if not (0 <= row < h and 0 <= col < w):
            return tuple(result)
        result.append((row, col))


def _separated(first, second):
    return all(max(abs(a - c), abs(b - d)) > 1
               for a, b in first for c, d in second)


def _components(grid, background):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            r, c = frontier.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (r + dr, c + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
        components.append(frozenset(component))
    return components


def _parse_input(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    modal = np.flatnonzero(counts == counts.max())
    _require(len(modal) == 1, "background must be the unique modal input color")
    background = int(modal[0])
    objects = []
    for cells in _components(grid, background):
        matches = []
        for direction in range(4):
            for width in _WIDTHS:
                if len(cells) != 2 * width - 2:
                    continue
                for center in cells:
                    if _footprint(center, direction, width) != cells:
                        continue
                    signal = int(grid[center])
                    carriers = {int(grid[pos]) for pos in cells if pos != center}
                    if len(carriers) != 1:
                        continue
                    carrier = carriers.pop()
                    if signal in (background, carrier) or carrier == background:
                        continue
                    matches.append(dict(center=center, direction=direction,
                                        width=width, signal=signal, carrier=carrier,
                                        cells=cells))
        _require(len(matches) == 1, "component must have one directional interpretation")
        objects.append(matches[0])
    _require(1 <= len(objects) <= 4, "each pair needs one to four controllers")
    directions = [obj["direction"] for obj in objects]
    _require(len(set(directions)) == len(directions), "one controller per side")
    for i, first in enumerate(objects):
        for second in objects[i + 1:]:
            _require(_separated(first["cells"], second["cells"]), "objects must be separated")
            if (first["direction"] - second["direction"]) % 4 in (1, 3):
                _require(first["signal"] != second["signal"],
                         "adjacent sides require distinct signals")
    rays = []
    for obj in objects:
        ray = frozenset(_ray(grid.shape, obj["center"], obj["direction"]))
        _require(len(ray) >= 1, "signal must be interior")
        _require(all(int(grid[pos]) == background for pos in ray), "ray is obstructed")
        _require(all(ray.isdisjoint(other) for other in rays), "rays cross")
        rays.append(ray)
    return background, objects


def _junction_color(train):
    # The convention is exposed by an adjacent-side training demonstration.
    colors = []
    for pair in train:
        _, objects = _parse_input(pair["input"])
        sides = {obj["direction"] for obj in objects}
        h, w = pair["input"].shape
        for pos, incident in (((0, 0), (0, 3)), ((0, w - 1), (0, 1)),
                              ((h - 1, w - 1), (2, 1)), ((h - 1, 0), (2, 3))):
            if set(incident) <= sides:
                colors.append(int(pair["output"][pos]))
    _require(colors and len(set(colors)) == 1, "training must reveal one junction color")
    return colors[0]


def _solve(grid, junction):
    background, objects = _parse_input(grid)
    _require(junction != background and all(junction not in (o["carrier"], o["signal"])
                                           for o in objects), "junction color is reserved")
    output = grid.copy()
    sides = {}
    for obj in objects:
        signal = obj["signal"]
        direction = obj["direction"]
        sides[direction] = signal
        for distance, pos in enumerate(_ray(grid.shape, obj["center"], direction), 1):
            if distance % 2 == 0:
                output[pos] = signal
    if 0 in sides:
        output[0, :] = sides[0]
    if 1 in sides:
        output[:, -1] = sides[1]
    if 2 in sides:
        output[-1, :] = sides[2]
    if 3 in sides:
        output[:, 0] = sides[3]
    h, w = grid.shape
    for pos, incident in (((0, 0), (0, 3)), ((0, w - 1), (0, 1)),
                          ((h - 1, w - 1), (2, 1)), ((h - 1, 0), (2, 3))):
        if all(side in sides for side in incident):
            output[pos] = junction
    return output


def _placement_candidates(rng, shape, direction, width, required_distance=None):
    h, w = shape
    candidates = []
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            center = (row, col)
            cells = _footprint(center, direction, width)
            if not all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in cells):
                continue
            ray = frozenset(_ray(shape, center, direction))
            if required_distance == "odd_long" and (len(ray) < 5 or len(ray) % 2 == 0):
                continue
            if required_distance == "short" and len(ray) > 3:
                continue
            if cells.isdisjoint(ray):
                candidates.append((center, cells, ray))
    order = rng.permutation(len(candidates))
    return [candidates[int(i)] for i in order]


def _place(rng, shape, directions, widths, teaching):
    candidates = []
    for index, (direction, width) in enumerate(zip(directions, widths)):
        target = "odd_long" if teaching and index == 0 else None
        # Short rays are deliberately common, including distance one.
        if target is None and rng.random() < 0.32:
            target = "short"
        options = _placement_candidates(rng, shape, direction, width, target)
        if not options and target == "short":
            options = _placement_candidates(rng, shape, direction, width)
        if not options:
            raise RuntimeError("no placement for a controller")
        candidates.append(options)
    placed = []
    def search(index, occupied, rays):
        if index == len(candidates):
            return True
        for center, cells, ray in candidates[index][:150]:
            if (cells.isdisjoint(occupied) and _separated(cells, occupied)
                    and cells.isdisjoint(rays) and ray.isdisjoint(occupied)
                    and ray.isdisjoint(rays)):
                placed.append((center, cells, ray))
                if search(index + 1, occupied | cells, rays | ray):
                    return True
                placed.pop()
        return False
    if not search(0, frozenset(), frozenset()):
        raise RuntimeError("could not place separated, unobstructed controllers")
    return placed


def _dimension(rng, count, regime):
    minimum = {1: 8, 2: 10, 3: 12, 4: 14}[count]
    category = int(rng.choice(3, p=(.30, .40, .30) if regime == "source_train"
                              else (.22, .38, .40)))
    if category == 0:
        upper = min(15, minimum + 5)
        size = int(rng.integers(minimum, upper + 1))
    elif category == 1:
        size = int(rng.integers(max(13, minimum), 23))
    else:
        size = int(rng.integers(max(21, minimum), 31))
    if rng.random() < .065:
        size = 30
    return size


def _colors(rng, count, junction):
    allowed = [c for c in range(1, 10) if c != junction]
    background = int(rng.choice(allowed))
    choices = [c for c in allowed if c != background]
    carrier_count = 2 if count > 1 and rng.random() < .45 else 1
    carrier_palette = [int(c) for c in rng.choice(choices, carrier_count, replace=False)]
    carriers = [carrier_palette[int(rng.integers(len(carrier_palette)))]
                for _ in range(count)]
    if carrier_count == 2:
        carriers[0], carriers[-1] = carrier_palette
    signals_available = [c for c in choices if c not in carrier_palette]
    signals = [int(c) for c in rng.choice(signals_available, count, replace=False)]
    # Repeated colors on opposite sides are compatible with the source rule.
    if count >= 3 and rng.random() < .22:
        signals[2] = signals[0]
    return background, carriers, signals


def _make_pair(rng, regime, directions, widths, junction, teaching=False,
               preferred_shape=None):
    count = len(directions)
    for attempt in range(80):
        shape = (preferred_shape if preferred_shape is not None and attempt < 40
                 else (_dimension(rng, count, regime), _dimension(rng, count, regime)))
        if attempt >= 60:
            shape = (30, 30)
        try:
            placed = _place(rng, shape, directions, widths, teaching)
        except RuntimeError:
            continue
        background, carriers, signals = _colors(rng, count, junction)
        grid = np.full(shape, background, dtype=np.int8)
        for (center, cells, _), carrier, signal in zip(placed, carriers, signals):
            for pos in cells:
                grid[pos] = carrier
            grid[center] = signal
        # Parse the visible input to construct ground truth, independent of placement metadata.
        try:
            output = _solve(grid, junction)
        except AssertionError:
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded placement failed")


def _directions(rng, count, adjacent=None):
    if count == 2 and adjacent is not None:
        first = int(rng.integers(4))
        step = int(rng.choice((1, 3) if adjacent else (2,)))
        return [first, (first + step) % 4]
    return [int(x) for x in rng.choice(4, count, replace=False)]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    junction = 0 if rng.random() < .75 else 9
    difficulty_pattern = int(rng.choice(3, p=(.22, .22, .56)))
    puzzle = {"train": [], "test": []}
    inputs = set()
    seen_widths = set()

    def add(split, directions, widths, teaching=False, preferred_shape=None):
        for _ in range(12):
            pair = _make_pair(rng, "source_train" if split == "train" else "target_test",
                              directions, widths, junction, teaching, preferred_shape)
            signature = _signature(pair["input"])
            if signature not in inputs and _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                inputs.add(signature)
                puzzle[split].append(pair)
                return
        raise RuntimeError("could not construct a unique non-source pair")

    # The first training example reveals dotted phase, border override, and junction color.
    directions = _directions(rng, 2, adjacent=True)
    widths = [5, int(rng.choice(_WIDTHS))]
    add("train", directions, widths, teaching=True)
    seen_widths.update(widths)
    # A second example changes scale and often the number of active sides.
    if difficulty_pattern == 0:
        second_count = int(rng.integers(1, 3))
    elif difficulty_pattern == 1:
        second_count = 4
    else:
        second_count = int(rng.integers(1, 5))
    second_directions = _directions(rng, second_count,
                                    adjacent=bool(rng.integers(2)) if second_count == 2 else None)
    alternate = int(rng.choice([w for w in _WIDTHS if w not in seen_widths])) if len(seen_widths) < 3 else int(rng.choice(_WIDTHS))
    second_widths = [alternate] + [int(rng.choice(_WIDTHS)) for _ in range(second_count - 1)]
    add("train", second_directions, second_widths)
    seen_widths.update(second_widths)
    for _ in range(2, train_count):
        count = int(rng.integers(1, 3)) if difficulty_pattern == 0 else int(rng.integers(1, 5))
        directions = _directions(rng, count,
                                 adjacent=bool(rng.integers(2)) if count == 2 else None)
        widths = [int(rng.choice(_WIDTHS)) for _ in range(count)]
        add("train", directions, widths)
        seen_widths.update(widths)
    for _ in range(test_count):
        if difficulty_pattern == 0:
            count = int(rng.integers(3, 5))
        elif difficulty_pattern == 1:
            count = int(rng.integers(1, 3))
        else:
            count = int(rng.integers(1, 5))
        directions = _directions(rng, count,
                                 adjacent=bool(rng.integers(2)) if count == 2 else None)
        widths = [int(rng.choice(sorted(seen_widths))) for _ in range(count)]
        preferred_shape = None
        if rng.random() < .30:
            minimum = {1: 8, 2: 10, 3: 12, 4: 14}[count]
            compatible = [pair["input"].shape for pair in puzzle["train"]
                          if min(pair["input"].shape) >= minimum]
            if compatible:
                preferred_shape = compatible[int(rng.integers(len(compatible)))]
        add("test", directions, widths, preferred_shape=preferred_shape)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    input_signatures, pair_signatures = set(), set()
    junction = _junction_color(puzzle["train"])
    _require(junction in range(10), "invalid junction color")
    train_widths = set()
    all_analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integral")
                _require(all(1 <= d <= 30 for d in grid.shape), "illegal dimensions")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "shape changed or transformation degenerate")
            si, so = _signature(inp), _signature(out)
            _require(si not in input_signatures and (si, so) not in pair_signatures,
                     "duplicate pair or input")
            input_signatures.add(si)
            pair_signatures.add((si, so))
            _require(_source_digest(inp) not in _SOURCE_INPUT_DIGESTS, "source pair copied")
            background, objects = _parse_input(inp)
            _require(junction != background and all(junction not in
                     (o["carrier"], o["signal"]) for o in objects), "junction role changed")
            _require(np.array_equal(_solve(inp, junction), out), "output violates rule")
            all_analyses[split].append(objects)
            if split == "train":
                train_widths.update(o["width"] for o in objects)
    _require(any(any((a["direction"] - b["direction"]) % 4 in (1, 3)
                     for i, a in enumerate(objects) for b in objects[i + 1:])
                 for objects in all_analyses["train"]), "training does not show junction")
    _require(any(len(_ray(pair["input"].shape, obj["center"], obj["direction"])) >= 5
                 and len(_ray(pair["input"].shape, obj["center"], obj["direction"])) % 2 == 1
                 for pair, objects in zip(puzzle["train"], all_analyses["train"])
                 for obj in objects), "training does not reveal border override and dots")
    _require(len(train_widths) >= 2, "training must show scale variation")
    _require(all(o["width"] in train_widths for objects in all_analyses["test"]
                 for o in objects), "test uses unseen object width")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((17, None, None), (23, 6, 3), (100, 2, 1),
                              (101, 10, 1), (102, 10, 4), (201, None, 1),
                              (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require(_equal(puzzle, generate_puzzle(seed, train, test)), "not deterministic")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(300, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, counts, sizes, structures = set(), set(), set(), set(), set()
    widths, junctions, short, carrier_mix, backgrounds = set(), set(), 0, 0, set()
    opposite, four_side, mixed_width, distance_one, axis_30 = 0, 0, 0, 0, 0
    varied_background, varied_carrier, role_reassignment = 0, 0, 0
    same_shape, test_more, test_less, held_rotation, novel_signal = 0, 0, 0, 0, 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        junctions.add(_junction_color(puzzle["train"]))
        profile = []
        puzzle_backgrounds, puzzle_carriers, train_signals, test_signals = set(), set(), set(), set()
        train_shapes, test_shapes, train_directions = set(), set(), set()
        split_counts = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, objects = _parse_input(inp)
                puzzle_backgrounds.add(bg)
                puzzle_carriers.update(o["carrier"] for o in objects)
                (train_signals if split == "train" else test_signals).update(
                    o["signal"] for o in objects)
                (train_shapes if split == "train" else test_shapes).add(inp.shape)
                if split == "train":
                    train_directions.update(o["direction"] for o in objects)
                split_counts[split].append(len(objects))
                counts.add((split, len(objects)))
                sizes.add(inp.shape)
                backgrounds.add(bg)
                widths.update(o["width"] for o in objects)
                carrier_mix += len({o["carrier"] for o in objects}) > 1
                mixed_width += len({o["width"] for o in objects}) > 1
                opposite += len(objects) == 2 and (
                    objects[0]["direction"] - objects[1]["direction"]) % 4 == 2
                four_side += len(objects) == 4
                axis_30 += 30 in inp.shape
                short += sum(len(_ray(inp.shape, o["center"], o["direction"])) <= 3
                             for o in objects)
                distance_one += sum(len(_ray(inp.shape, o["center"], o["direction"])) == 1
                                    for o in objects)
                if split == "test":
                    held_rotation += any(o["direction"] not in train_directions for o in objects)
                profile.append((split, inp.shape, tuple(sorted(
                    (o["center"], o["direction"], o["width"]) for o in objects))))
        structures.add(tuple(profile))
        varied_background += len(puzzle_backgrounds) > 1
        varied_carrier += len(puzzle_carriers) > 1
        role_reassignment += bool(puzzle_backgrounds & puzzle_carriers)
        same_shape += bool(train_shapes & test_shapes)
        test_more += max(split_counts["test"]) > max(split_counts["train"])
        test_less += max(split_counts["test"]) < max(split_counts["train"])
        novel_signal += bool(test_signals - train_signals)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity missing")
    _require(len(sizes) >= 60 and len(structures) >= 40, "structural diversity missing")
    _require(widths == set(_WIDTHS) and len(backgrounds) >= 7, "palette or width diversity missing")
    _require(short >= 20 and carrier_mix >= 10, "short-ray or carrier diversity missing")
    _require(distance_one >= 10 and opposite >= 10 and four_side >= 10,
             "shortest rays or side combinations missing")
    _require(mixed_width >= 20 and axis_30 >= 10, "scale or dimension diversity missing")
    _require(varied_background >= 30 and varied_carrier >= 30 and role_reassignment >= 20,
             "within-puzzle color variation missing")
    _require(same_shape >= 5 and test_more >= 5 and test_less >= 5,
             "train/test size and difficulty patterns missing")
    _require(held_rotation >= 1 and novel_signal >= 10,
             "held orientation or locally visible new signal missing")
    _require(junctions == {0, 9}, "junction convention did not vary")
    _require(all((split, count) in counts for split in ("train", "test")
                 for count in range(1, 5)), "object counts did not span both splits")
