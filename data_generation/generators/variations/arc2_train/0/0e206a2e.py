"""Relocate connected templates by registering three colored landmarks."""

import hashlib
import itertools

import numpy as np


SOURCE_HASHES = {
    ((14, 15), "737c875f7e9fd11d4f64db8fa1b1b121bb6a7822078e872a4e581beb057cb98d"),
    ((14, 18), "967b739028109cdc0f8c2a196b89e8485edc50913490447c23ad042d6a2f263b"),
    ((16, 14), "7347b6640ab68b14201adc5b47e8e03051b1be0f5ab3bd900701ad0571b45427"),
    ((24, 19), "840f3323d61d6fa9cc306206f995bd5e29dfe5063fb4ac7b9685b8a6d2b00b60"),
}
SOURCE_DIMENSIONS = ((14, 15), (14, 18), (16, 14), (24, 19))
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def pose(cell, kind):
    r, c = cell
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (-r, c), (r, -c), (c, r), (-c, -r))[kind]


def components(grid, background):
    height, width = grid.shape
    seen, result = set(), []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) == background or (row, col) in seen:
                continue
            stack, part = [(row, col)], []
            seen.add((row, col))
            while stack:
                cell = stack.pop()
                part.append(cell)
                for dr, dc in NEIGHBORS:
                    other = (cell[0] + dr, cell[1] + dc)
                    if (0 <= other[0] < height and 0 <= other[1] < width
                            and other not in seen and int(grid[other]) != background):
                        seen.add(other)
                        stack.append(other)
            result.append(part)
    return result


def noncollinear(points):
    a, b, c = points
    return ((b[0] - a[0]) * (c[1] - a[1])
            != (c[0] - a[0]) * (b[1] - a[1]))


def solve_input(inp):
    """Recover the unique output from input colors and geometry only."""
    height, width = inp.shape
    frequencies = np.bincount(inp.astype(np.uint8).ravel(), minlength=10)
    background = int(np.argmax(frequencies))
    assert frequencies[background] > inp.size // 2
    parts = components(inp, background)
    objects = [part for part in parts if len(part) > 1]
    markers = [part[0] for part in parts if len(part) == 1]
    count = len(objects)
    assert 1 <= count <= 3 and len(markers) == 3 * count
    parsed, carrier, identities = [], None, None
    for part in objects:
        groups = {}
        for cell in part:
            groups.setdefault(int(inp[cell]), []).append(cell)
        repeated = [color for color, cells in groups.items() if len(cells) > 1]
        assert len(groups) == 4 and len(repeated) == 1
        local_carrier = repeated[0]
        assert len(groups[local_carrier]) >= 4
        landmarks = {color: cells[0] for color, cells in groups.items()
                     if color != local_carrier and len(cells) == 1}
        assert len(landmarks) == 3 and noncollinear(list(landmarks.values()))
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2
                   for a, b in itertools.combinations(landmarks.values(), 2))
        if carrier is None:
            carrier, identities = local_carrier, tuple(sorted(landmarks))
        else:
            assert carrier == local_carrier and identities == tuple(sorted(landmarks))
        parsed.append(([(cell, int(inp[cell])) for cell in part], landmarks))
    by_color = {color: [] for color in identities}
    for cell in markers:
        color = int(inp[cell])
        assert color in by_color
        by_color[color].append(cell)
    assert all(len(cells) == count for cells in by_color.values())
    all_markers = set(markers)
    source_cells = {cell for part in objects for cell in part}
    options = []
    for cells, landmarks in parsed:
        found = []
        for triple in itertools.product(*(by_color[color] for color in identities)):
            targets = dict(zip(identities, triple))
            used = frozenset(triple)
            for kind in range(8):
                first = identities[0]
                moved = pose(landmarks[first], kind)
                delta = (targets[first][0] - moved[0], targets[first][1] - moved[1])
                if any(tuple(pose(landmarks[color], kind)[i] + delta[i]
                             for i in (0, 1)) != targets[color]
                       for color in identities[1:]):
                    continue
                painted = []
                for cell, color in cells:
                    moved = pose(cell, kind)
                    dest = (moved[0] + delta[0], moved[1] + delta[1])
                    if not (0 <= dest[0] < height and 0 <= dest[1] < width):
                        break
                    painted.append((dest, color))
                else:
                    occupied = frozenset(cell for cell, _ in painted)
                    if occupied & source_cells or occupied & (all_markers - used):
                        continue
                    found.append((kind, used, painted, occupied))
        assert found
        options.append(found)
    solutions = []

    def search(index, used_markers, used_cells, chosen):
        if len(solutions) >= 2:
            return
        if index == count:
            if used_markers == all_markers:
                solutions.append(chosen.copy())
            return
        for option in options[index]:
            _, option_markers, _, occupied = option
            if option_markers & used_markers or occupied & used_cells:
                continue
            chosen.append(option)
            search(index + 1, used_markers | option_markers,
                   used_cells | occupied, chosen)
            chosen.pop()

    search(0, set(), set(), [])
    assert len(solutions) == 1
    chosen = solutions[0]
    out = np.full(inp.shape, background, dtype=np.int8)
    moves, spans, source_centers, target_centers = [], [], [], []
    for part, option in zip(objects, chosen):
        for cell, color in option[2]:
            out[cell] = color
        src = (sum(r for r, _ in part) / len(part),
               sum(c for _, c in part) / len(part))
        dst = (sum(cell[0] for cell, _ in option[2]) / len(part),
               sum(cell[1] for cell, _ in option[2]) / len(part))
        source_centers.append(src)
        target_centers.append(dst)
        moves.append((dst[0] - src[0], dst[1] - src[1]))
        spans.append((max(r for r, _ in part) - min(r for r, _ in part) + 1,
                      max(c for _, c in part) - min(c for _, c in part) + 1,
                      len(part)))
    info = {"n": count, "background": background, "carrier": carrier,
            "landmarks": frozenset(identities),
            "poses": tuple(option[0] for option in chosen),
            "moves": tuple(moves), "spans": tuple(spans),
            "source_centers": tuple(source_centers),
            "target_centers": tuple(target_centers)}
    return out, info


def inspect_pair(inp, out):
    expected, info = solve_input(inp)
    assert np.array_equal(expected, out) and not np.array_equal(inp, out)
    assert set(np.unique(inp)) == set(np.unique(out))
    return info


def motif(rng, landmarks, carrier):
    """Build sparse bars, compact branches, and irregular connected shapes."""
    for _ in range(100):
        mode = int(rng.integers(3))
        if mode == 0:
            length = int(rng.integers(6, 9))
            vertical = bool(rng.integers(2))
            cells = {(i, 0) if vertical else (0, i) for i in range(length)}
            target_size = length + int(rng.integers(1, 5))
        else:
            cells = {(0, 0)}
            target_size = int(rng.integers(7, 14 if mode == 1 else 17))
            limit = 2 if mode == 1 else 3
        while len(cells) < target_size:
            fringe = {(r + dr, c + dc) for r, c in cells for dr, dc in NEIGHBORS
                      if (r + dr, c + dc) not in cells}
            if mode == 0:
                fringe = {cell for cell in fringe
                          if -2 <= (cell[1] if vertical else cell[0]) <= 2
                          and -1 <= (cell[0] if vertical else cell[1]) <= length}
            else:
                fringe = {cell for cell in fringe
                          if abs(cell[0]) <= limit and abs(cell[1]) <= limit}
            if not fringe:
                break
            ordered = sorted(fringe)
            cells.add(ordered[int(rng.integers(len(ordered)))])
        if len(cells) < 7:
            continue
        min_row = min(r for r, _ in cells)
        min_col = min(c for _, c in cells)
        choices = sorted((r - min_row, c - min_col) for r, c in cells)
        for _ in range(40):
            indices = rng.choice(len(choices), size=3, replace=False)
            points = [choices[int(i)] for i in indices]
            if not noncollinear(points):
                continue
            if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) < 2
                   for a, b in itertools.combinations(points, 2)):
                continue
            shape = {cell: carrier for cell in choices}
            for color, point in zip(landmarks, points):
                shape[point] = color
            return shape
    raise RuntimeError("Could not construct landmark template")


def place(shape, kind, top, left):
    rotated = [(pose(cell, kind), color) for cell, color in shape.items()]
    min_row = min(cell[0] for cell, _ in rotated)
    min_col = min(cell[1] for cell, _ in rotated)
    return {(cell[0] - min_row + top, cell[1] - min_col + left): color
            for cell, color in rotated}


def separated(cells, occupied):
    return all((r + dr, c + dc) not in occupied
               for r, c in cells for dr, dc in ((0, 0),) + NEIGHBORS)


def palette(rng):
    background = 0 if rng.random() < 0.52 else int(rng.integers(1, 10))
    available = [color for color in range(10) if color != background]
    chosen = rng.choice(available, size=4, replace=False)
    return background, tuple(int(c) for c in chosen[:3]), int(chosen[3])


def dimensions(rng, regime, count):
    minimum = (13 if regime == "source_train" else 14) if count == 1 else (17 if count == 2 else 21)
    if rng.random() < 0.23:
        choices = [shape for shape in SOURCE_DIMENSIONS if min(shape) >= minimum]
        if choices:
            return choices[int(rng.integers(len(choices)))]
    upper = 25 if regime == "source_train" else 30
    return int(rng.integers(minimum, upper)), int(rng.integers(minimum, upper))


def random_placement(rng, shape, kind, height, width, near=None):
    oriented = place(shape, kind, 0, 0)
    shape_h = max(row for row, _ in oriented) + 1
    shape_w = max(col for _, col in oriented) + 1
    if shape_h > height or shape_w > width:
        return None
    if near is not None and rng.random() < 0.42:
        top = near[0] + int(rng.integers(-10, 11))
        left = near[1] + int(rng.integers(-10, 11))
        if 0 <= top <= height - shape_h and 0 <= left <= width - shape_w:
            return place(shape, kind, top, left)
    top = int(rng.integers(height - shape_h + 1))
    left = int(rng.integers(width - shape_w + 1))
    return place(shape, kind, top, left)


def make_pair(rng, regime, forced_pose=None):
    for _ in range(320):
        count = (int(rng.choice((1, 2), p=(0.53, 0.47)))
                 if regime == "source_train"
                 else int(rng.choice((1, 2, 3), p=(0.28, 0.52, 0.20))))
        height, width = dimensions(rng, regime, count)
        background, landmarks, carrier = palette(rng)
        shapes = [motif(rng, landmarks, carrier) for _ in range(count)]
        kinds = [int(rng.integers(8)) for _ in range(count)]
        if forced_pose is not None:
            kinds[int(rng.integers(count))] = forced_pose
        sources, source_cells = [], set()
        for shape in shapes:
            for _ in range(120):
                source = random_placement(rng, shape, 0, height, width)
                if source is not None and separated(source, source_cells):
                    sources.append(source)
                    source_cells.update(source)
                    break
            else:
                break
        if len(sources) != count:
            continue
        targets, target_cells, marker_cells = [], set(), set()
        for shape, kind, source in zip(shapes, kinds, sources):
            near = (min(r for r, _ in source), min(c for _, c in source))
            for _ in range(180):
                target = random_placement(rng, shape, kind, height, width, near)
                if target is None:
                    continue
                cells = set(target)
                marks = {cell for cell, color in target.items() if color in landmarks}
                if cells & source_cells or not separated(cells, target_cells):
                    continue
                if not separated(marks, source_cells | marker_cells):
                    continue
                if not separated(marker_cells, marks):
                    continue
                targets.append(target)
                target_cells.update(cells)
                marker_cells.update(marks)
                break
            else:
                break
        if len(targets) != count:
            continue
        inp = np.full((height, width), background, dtype=np.int8)
        out = np.full((height, width), background, dtype=np.int8)
        for source in sources:
            for cell, color in source.items():
                inp[cell] = color
        for target in targets:
            for cell, color in target.items():
                out[cell] = color
                if color in landmarks:
                    inp[cell] = color
        try:
            info = inspect_pair(inp, out)
            if sorted(info["poses"]) == sorted(kinds):
                return inp, out
        except AssertionError:
            pass
    raise RuntimeError("Could not construct an unambiguous pair")


def counts(seed, num_train, num_test, rng):
    def legal(value, low, high):
        return (isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_)) and low <= value <= high)
    if num_train is not None and not legal(num_train, 2, 10):
        raise ValueError("num_train must be an integer from 2 to 10")
    if num_test is not None and not legal(num_test, 1, 4):
        raise ValueError("num_test must be an integer from 1 to 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = counts(seed, num_train, num_test, rng)
    # Identity-based rigid registration is the only puzzle-wide convention.
    required = [int(rng.choice((1, 2, 3))),
                int(rng.choice((4, 5, 6, 7)))]
    rng.shuffle(required)
    puzzle, seen = {"train": [], "test": []}, set()
    for regime, count, key in (("source_train", train_count, "train"),
                               ("target_test", test_count, "test")):
        for index in range(count):
            forced = required[index] if key == "train" and index < 2 else None
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                inp, out = make_pair(pair_rng, regime, forced)
                signature = (inp.shape, inp.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[key].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("Could not make distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs, train_infos = set(), set(), []
    for key in ("train", "test"):
        for pair in puzzle[key]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_bytes = inp.astype(np.uint8, copy=False).tobytes()
            output_bytes = out.astype(np.uint8, copy=False).tobytes()
            input_key = (inp.shape, input_bytes)
            pair_key = (inp.shape, input_bytes, output_bytes)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            digest = hashlib.sha256(input_bytes + output_bytes).hexdigest()
            assert (inp.shape, digest) not in SOURCE_HASHES
            info = inspect_pair(inp, out)
            assert info["carrier"] not in info["landmarks"]
            assert info["background"] not in info["landmarks"]
            assert info["background"] != info["carrier"]
            if key == "train":
                train_infos.append(info)
                assert 1 <= info["n"] <= 2
                assert 13 <= inp.shape[0] <= 24 and 13 <= inp.shape[1] <= 24
            else:
                assert 1 <= info["n"] <= 3
                assert 14 <= inp.shape[0] <= 29 and 14 <= inp.shape[1] <= 29
    train_poses = {kind for info in train_infos for kind in info["poses"]}
    assert train_poses & {1, 2, 3} and train_poses & {4, 5, 6, 7}


def same_puzzle(first, second):
    return all(len(first[key]) == len(second[key])
               and all(np.array_equal(a[field], b[field])
                       for a, b in zip(first[key], second[key])
                       for field in ("input", "output"))
               for key in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert same_puzzle(generate_puzzle(13, 10, 4), generate_puzzle(13, 10, 4))
    assert not same_puzzle(puzzles[0], puzzles[1])
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    pairs = [pair for puzzle in puzzles for key in ("train", "test")
             for pair in puzzle[key]]
    infos = [inspect_pair(pair["input"], pair["output"]) for pair in pairs]
    assert set(itertools.chain.from_iterable(i["poses"] for i in infos)) == set(range(8))
    assert {i["n"] for i in infos} == {1, 2, 3}
    assert len({i["background"] for i in infos}) >= 5
    assert len({i["carrier"] for i in infos}) >= 6
    assert len({i["landmarks"] for i in infos}) >= 15
    assert any(any(size == 7 for _, _, size in i["spans"]) for i in infos)
    assert any(any(max(h, w) >= 6 for h, w, _ in i["spans"]) for i in infos)
    assert any(any(dr < -2 for dr, _ in i["moves"]) for i in infos)
    assert any(any(dr > 2 for dr, _ in i["moves"]) for i in infos)
    assert any(any(abs(dr) < 3 and abs(dc) > 2 for dr, dc in i["moves"]) for i in infos)
    assert len({pair["input"].shape for pair in pairs}) >= 30
    assert any(pair["input"].shape in SOURCE_DIMENSIONS for pair in pairs)
    train_shapes = {pair["input"].shape for puzzle in puzzles for pair in puzzle["train"]}
    test_shapes = {pair["input"].shape for puzzle in puzzles for pair in puzzle["test"]}
    assert len(train_shapes & test_shapes) >= 4
    assert sum(i["background"] != 0 for i in infos) >= 40
    assert sum(any(size == 7 for _, _, size in i["spans"]) for i in infos) >= 20
    assert sum(any(max(h, w) >= 6 for h, w, _ in i["spans"]) for i in infos) >= 50
    assert sum(any(max(abs(dr), abs(dc)) <= 7 for dr, dc in i["moves"])
               for i in infos) >= 40
    assert sum(any(max(abs(dr), abs(dc)) >= 12 for dr, dc in i["moves"])
               for i in infos) >= 20
    assert sum(any(abs(dr) <= 2 and abs(dc) > 2 for dr, dc in i["moves"])
               for i in infos) >= 20
    assert sum(i["n"] == 2 and
               (i["source_centers"][0][1] - i["source_centers"][1][1]) *
               (i["target_centers"][0][1] - i["target_centers"][1][1]) < 0
               for i in infos) >= 15
    assert sum(i["n"] == 2 and
               (i["source_centers"][0][1] - i["source_centers"][1][1]) *
               (i["target_centers"][0][1] - i["target_centers"][1][1]) > 0
               for i in infos) >= 15
    role_exchanges = 0
    repeated_carriers = 0
    varying_landmarks = 0
    for puzzle in puzzles:
        local = [inspect_pair(pair["input"], pair["output"])
                 for key in ("train", "test") for pair in puzzle[key]]
        role_exchanges += any(a["carrier"] in b["landmarks"]
                              for a in local for b in local if a is not b)
        repeated_carriers += len({a["carrier"] for a in local}) < len(local)
        varying_landmarks += len({a["landmarks"] for a in local}) > 1
    assert role_exchanges >= 15 and repeated_carriers >= 15
    assert varying_landmarks >= 30
    assert sum(i["background"] != 0 and
               any(size == 7 for _, _, size in i["spans"]) and
               any(dr < -2 for dr, _ in i["moves"])
               for i in infos) >= 2
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(123, train_count, test_count))
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts accepted")
    damaged = {"train": puzzles[0]["train"].copy(),
               "test": [dict(pair) for pair in puzzles[0]["test"]]}
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    old = int(damaged["test"][0]["output"][0, 0])
    damaged["test"][0]["output"][0, 0] = (old + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
