"""Nearest-color loom: intersect nearest-marker bands, then conditionally paint.

Only the standard library and NumPy are used. No parent code or data is loaded.
"""

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return grid.astype(np.int8, copy=True)


def _decode(value):
    """Read the background and rotate the visibly marked corner to top-left."""
    grid = _grid(value)
    if min(grid.shape) < 5:
        raise ValueError("a loom needs two rulers and an interior")
    background = int(grid[0, 0])
    if not all(int(v) == background for v in
               (grid[0, -1], grid[-1, 0], grid[-1, -1])):
        raise ValueError("all four corners must show the background")
    edges = (grid[0, 1:-1], grid[1:-1, -1],
             grid[-1, 1:-1], grid[1:-1, 0])
    active = frozenset(i for i, edge in enumerate(edges) if np.any(edge != background))
    rotations = {frozenset((0, 3)): 0, frozenset((0, 1)): 1,
                 frozenset((1, 2)): 2, frozenset((2, 3)): 3}
    if active not in rotations:
        raise ValueError("exactly two adjacent edges must contain control markers")
    turns = rotations[active]
    canonical = np.rot90(grid, turns).copy()
    for rail in (canonical[:, 0], canonical[0, :]):
        positions = np.flatnonzero(rail != background)
        if not 2 <= len(positions) <= 5:
            raise ValueError("each ruler must have 2..5 markers")
        if np.any(np.diff(positions) < 2):
            raise ValueError("ruler markers must have a background cell between them")
        if any(rail[a] == rail[b] for a, b in zip(positions, positions[1:])):
            raise ValueError("consecutive ruler labels must differ")
    return canonical, background, turns


def _nearest_owners(rail, background):
    positions = np.flatnonzero(rail != background)
    distances = np.abs(np.arange(len(rail))[:, None] - positions[None, :])
    winners = distances.argmin(axis=1)
    owners = rail[positions[winners]].astype(int)
    tied = (distances == distances.min(axis=1)[:, None]).sum(axis=1) != 1
    owners[tied] = -1
    return owners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the queried input alone, including its marked orientation."""
    scene, background, turns = _decode(input_grid)
    rows = _nearest_owners(scene[:, 0], background)
    columns = _nearest_owners(scene[0, :], background)
    result = scene.copy()
    result[1:, 0] = np.where(rows[1:] >= 0, rows[1:], background)
    result[0, 1:] = np.where(columns[1:] >= 0, columns[1:], background)
    agreed = (rows[1:, None] == columns[None, 1:]) & (rows[1:, None] >= 0)
    commands = np.broadcast_to(rows[1:, None], scene[1:, 1:].shape)
    changed = np.where(scene[1:, 1:] == commands, background, commands)
    result[1:, 1:] = np.where(agreed, changed, scene[1:, 1:])
    return np.rot90(result, -turns).copy()


def _interval_owners(rail, background):
    """Separate reference calculation: integer half-gap intervals, no argmin."""
    seeds = [(i, int(c)) for i, c in enumerate(rail) if c != background]
    owners = [-1] * len(rail)
    for index, (position, color) in enumerate(seeds):
        low = 0 if index == 0 else (seeds[index - 1][0] + position) // 2 + 1
        high = (len(rail) - 1 if index == len(seeds) - 1
                else (position + seeds[index + 1][0] - 1) // 2)
        for coordinate in range(low, high + 1):
            owners[coordinate] = color
    return owners


def _reference(scene, background, variant="correct"):
    rows = _interval_owners(scene[:, 0], background)
    columns = _interval_owners(scene[0, :], background)
    if variant == "equal_bands":
        for rail, owners in ((scene[:, 0], rows), (scene[0, :], columns)):
            labels = [int(c) for c in rail if c != background]
            for coordinate in range(1, len(rail)):
                owners[coordinate] = labels[min(len(labels) - 1,
                    (coordinate - 1) * len(labels) // (len(rail) - 1))]
    result = scene.copy()
    for r in range(scene.shape[0]):
        for c in range(scene.shape[1]):
            if r == c == 0:
                continue
            if r == 0:
                result[r, c] = background if columns[c] < 0 else columns[c]
            elif c == 0:
                result[r, c] = background if rows[r] < 0 else rows[r]
            else:
                color = rows[r]
                active = color >= 0 and color == columns[c]
                if variant == "seed_crossings":
                    active = (active and scene[r, 0] != background
                              and scene[0, c] != background)
                if variant == "row_priority":
                    active = color >= 0
                if active:
                    result[r, c] = (color if variant == "solid" or scene[r, c] != color
                                    else background)
    return result


def _information(scene, background):
    rows = np.array(_interval_owners(scene[:, 0], background))
    columns = np.array(_interval_owners(scene[0, :], background))
    core = scene[1:, 1:]
    color = np.broadcast_to(rows[1:, None], core.shape)
    active = (color >= 0) & (color == columns[None, 1:])
    rp = tuple(int(x) for x in np.flatnonzero(scene[:, 0] != background))
    cp = tuple(int(x) for x in np.flatnonzero(scene[0, :] != background))
    rl = tuple(int(scene[r, 0]) for r in rp)
    cl = tuple(int(scene[0, c]) for c in cp)
    return {
        "geometry": (scene.shape, rp, cp), "counts": (len(rp), len(cp)),
        "background": background, "row_labels": rl, "column_labels": cl,
        "matching": int(np.count_nonzero(active & (core == color))),
        "contrasting": int(np.count_nonzero(active & (core != color) & (core != background))),
        "extension": int(np.count_nonzero(active & (core == background))),
        "protected_payload": int(np.count_nonzero(~active & (core != background))),
        "row_ties": int(np.count_nonzero(rows[1:] < 0)),
        "column_ties": int(np.count_nonzero(columns[1:] < 0)),
        "repeated": len(set(rl)) < len(rl) or len(set(cl)) < len(cl),
        "unpaired": bool(set(rl) ^ set(cl)),
        "density": float(np.mean(scene[1:-1, 1:-1] != background)),
    }


def _positions(rng, length, count, first_parity=None, off_center=False):
    for _ in range(128):
        gaps = rng.integers(2, 8, size=count - 1).astype(int)
        if first_parity is not None:
            gaps[0] = int(rng.choice((4, 6, 8) if first_parity == 0 else (3, 5, 7)))
        spare = length - 3 - int(gaps.sum())
        if spare < 0:
            continue
        start = int(rng.integers(1, spare + 2))
        levels = tuple([start] + [start + int(v) for v in np.cumsum(gaps)])
        if off_center and abs((levels[0] + levels[1]) / 2 - (length - 1) / 2) < 2:
            continue
        return levels
    # A finite constructive fallback, also covering narrow legal sizes.
    gap = 4 if first_parity == 0 else 3 if first_parity == 1 else 2
    levels = tuple(1 + gap * i for i in range(count))
    if levels[-1] >= length - 1:
        levels = tuple(1 + 2 * i for i in range(count))
    if off_center and abs((levels[0] + levels[1]) / 2 - (length - 1) / 2) < 2:
        raise RuntimeError("insufficient length for off-center demonstration")
    return levels


def _labels(rng, count, shared, private, repeated=False):
    a, b = shared
    labels = [a, b, a] if repeated else [a, b]
    if not repeated and count >= 3:
        labels.append(private)
    while len(labels) < count:
        choices = [v for v in (a, b, private) if v != labels[-1]]
        labels.append(int(rng.choice(choices)))
    if rng.random() < 0.5:
        labels.reverse()
    return labels


def _box(rng, height, width):
    rh = int(rng.integers(2, height - 1))
    rw = int(rng.integers(2, width - 1))
    row = int(rng.integers(1, height - rh))
    col = int(rng.integers(1, width - rw))
    return row, col, rh, rw


def _paint_scene(scene, rng, palette, test):
    height, width = scene.shape
    for _ in range(int(rng.integers(3, 9 if test else 7))):
        row, col, rh, rw = _box(rng, height, width)
        color = int(rng.choice(palette))
        style = int(rng.integers(5))
        if style == 0:
            scene[row:row + rh, col:col + rw] = color
        elif style == 1:
            scene[row:row + rh, col] = color
            scene[row:row + rh, col + rw - 1] = color
            scene[row, col:col + rw] = color
            scene[row + rh - 1, col:col + rw] = color
        elif style == 2:
            scene[row:row + rh, col:col + max(1, rw // 3)] = color
            scene[row + rh - max(1, rh // 3):row + rh, col:col + rw] = color
        elif style == 3:
            scene[row:row + rh:2, col:col + rw] = color
        else:
            for dr in range(rh):
                end = max(1, (dr + 1) * rw // rh)
                scene[row + dr, col:col + end] = color


def _crossing_patch(scene, rng, r, c, color):
    """A small actual region crossing a commanded window, always interior."""
    rh, rw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    row = max(1, min(r - int(rng.integers(rh)), scene.shape[0] - rh - 1))
    col = max(1, min(c - int(rng.integers(rw)), scene.shape[1] - rw - 1))
    scene[row:row + rh, col:col + rw] = color


def _make_pair(rng, kind, max_markers, turns, old_shapes=()):
    if kind == "simple":
        nr = nc = 2
    elif kind == "teaching":
        nr = nc = max_markers
    else:
        nr, nc = (int(rng.integers(2, max_markers + 1)) for _ in range(2))
    minimum = max(12, 2 * max(nr, nc) + 5)
    height, width = (int(rng.integers(minimum, 31)) for _ in range(2))
    if rng.random() < 0.22:
        width = height
    if kind == "test" and old_shapes and rng.random() < 0.35:
        compatible = [shape for shape in old_shapes if min(shape) >= minimum]
        if compatible:
            height, width = compatible[int(rng.integers(len(compatible)))]
    rp = _positions(rng, height, nr, 0 if kind == "simple" else None,
                    off_center=kind == "simple")
    cp = _positions(rng, width, nc, 1 if kind == "simple" else
                    0 if kind == "teaching" else None)
    background, a, b, private_r, private_c, *extra = [int(v) for v in rng.permutation(10)]
    if kind == "simple":
        rl, cl = [a, b], [a, b]
        if rng.random() < 0.5:
            cl.reverse()
    elif kind == "teaching":
        rl = _labels(rng, nr, (a, b), private_r, repeated=True)
        cl = _labels(rng, nc, (b, a), private_c)
    else:
        rl = _labels(rng, nr, (a, b), private_r,
                     repeated=nr >= 3 and rng.random() < 0.5)
        cl = _labels(rng, nc, (a, b), private_c,
                     repeated=nc >= 3 and rng.random() < 0.5)
    scene = np.full((height, width), background, dtype=np.int8)
    if kind == "simple":
        scene[rp[0]:rp[-1] + 1, cp[0]:cp[-1] + 1] = a
    else:
        if kind == "teaching":
            # Keep the second indispensable demonstration readable even when
            # only two training pairs were requested: a hollow region plus
            # small crossing patches, rather than a dense composite scene.
            rh = int(rng.integers(3, min(9, height - 1)))
            rw = int(rng.integers(3, min(9, width - 1)))
            row = int(rng.integers(1, height - rh))
            col = int(rng.integers(1, width - rw))
            scene[row, col:col + rw] = private_r
            scene[row + rh - 1, col:col + rw] = private_r
            scene[row:row + rh, col] = private_r
            scene[row:row + rh, col + rw - 1] = private_r
            _crossing_patch(scene, rng, rp[rl.index(a)], cp[cl.index(private_c)], b)
        else:
            palette = (a, b, private_r, private_c, *extra[:int(rng.integers(0, 4))])
            _paint_scene(scene, rng, palette, kind == "test")
        # Every test has real matching and contrasting crossings, rather than
        # allowing the conditional part of the rule to be irrelevant.
        ar, ac = rp[rl.index(a)], cp[cl.index(a)]
        br, bc = rp[rl.index(b)], cp[cl.index(b)]
        _crossing_patch(scene, rng, br, bc, a)
        _crossing_patch(scene, rng, ar, ac, a)
    for row, color in zip(rp, rl):
        scene[row, 0] = color
    for col, color in zip(cp, cl):
        scene[0, col] = color
    inp = np.rot90(scene, -turns).copy()
    return {"input": inp, "output": solve(inp)}


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count = _integer(num_train, "num_train", 2, 10)
    test_count = _integer(num_test, "num_test", 1, 4)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    max_markers = int(rng.integers(3, 6))
    first_turn = int(rng.integers(4))
    teaching_turn = (first_turn + int(rng.choice((1, 3)))) % 4
    puzzle = {"train": [], "test": []}
    geometries, seen, train_shapes = set(), set(), []
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        kind = ("simple" if index == 0 else "teaching" if index == 1 else
                "test" if split == "test" else "complex")
        turns = first_turn if index == 0 else teaching_turn if index == 1 else int(rng.integers(4))
        for _ in range(160):
            pair = _make_pair(rng, kind, max_markers, turns, train_shapes)
            scene, background, _ = _decode(pair["input"])
            info = _information(scene, background)
            signature = _key(pair["input"])
            if (signature in seen or info["geometry"] in geometries
                    or not info["matching"] or not info["contrasting"]
                    or not info["extension"] or not info["protected_payload"]):
                continue
            break
        else:
            raise RuntimeError("unable to construct a distinct informative loom")
        puzzle[split].append(pair)
        seen.add(signature)
        geometries.add(info["geometry"])
        if split == "train":
            train_shapes.append(scene.shape)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, records = set(), {"train": [], "test": []}
    failures = {name: 0 for name in ("solid", "equal_bands", "seed_crossings", "row_priority")}
    orientations = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            signature = _key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            scene, background, turns = _decode(inp)
            prediction = np.rot90(_reference(scene, background), -turns)
            _require(np.array_equal(prediction, out), "output violates interval/conditional rule")
            _require(not np.array_equal(inp, out), "unchanged output")
            info = _information(scene, background)
            _require(len(set(info["row_labels"]) & set(info["column_labels"])) >= 2,
                     "need at least two shared command colors")
            for feature in ("matching", "contrasting", "extension", "protected_payload"):
                _require(info[feature] > 0, f"missing {feature} witness")
            records[split].append(info)
            if split == "train":
                orientations.add(turns)
                for name in failures:
                    failures[name] += not np.array_equal(_reference(scene, background, name),
                                                         np.rot90(out, turns))
    training = records["train"]
    _require(len(orientations) >= 2, "training must expose visible orientation")
    _require(any(i["row_ties"] for i in training) and any(i["column_ties"] for i in training),
             "training must expose neutral seams on both rulers")
    _require(any(i["repeated"] for i in training), "training omits repeated-label windows")
    _require(any(i["unpaired"] for i in training), "training omits unpaired labels")
    _require(all(failures.values()), "training permits a documented shortcut")
    train_geometry = {i["geometry"] for i in training}
    _require(len(train_geometry) >= 2, "training needs varied geometry")
    maxima = tuple(max(i["counts"][axis] for i in training) for axis in (0, 1))
    for info in records["test"]:
        _require(info["geometry"] not in train_geometry, "test repeats training geometry")
        _require(all(info["counts"][axis] <= maxima[axis] for axis in (0, 1)),
                 "test uses more markers than demonstrated")


def _puzzle_key(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded structural, semantic, metamorphic, count, and breadth checks."""
    shapes, geometry, backgrounds, counts, orientations, structures = (set() for _ in range(6))
    signatures, densities = set(), []
    observations = {"matching": 0, "contrasting": 0, "protected_payload": 0,
                    "row_ties": 0, "column_ties": 0, "repeated": 0, "unpaired": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene, bg, turns = _decode(pair["input"])
                info = _information(scene, bg)
                shapes.add(scene.shape)
                geometry.add(info["geometry"])
                backgrounds.add(bg)
                counts.update(info["counts"])
                orientations.add(turns)
                densities.append(info["density"])
                mapping = {}
                structures.add((scene.shape, tuple(mapping.setdefault(int(c), len(mapping))
                                                    for c in scene.flat)))
                for feature in observations:
                    observations[feature] += int(info[feature] > 0)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
        # Rotation and arbitrary color renaming preserve the entire operation.
        pair = puzzle["test"][0]
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], turn)),
                                   np.rot90(pair["output"], turn)), "rotation equivariance failed")
        permutation = np.random.default_rng(40000 + seed).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "color equivariance failed")
    _require(len(signatures) == 42, "seed diversity failed")
    _require(len(shapes) >= 50 and len(geometry) >= 180 and len(structures) >= 200,
             "structural diversity collapsed")
    _require(backgrounds == set(range(10)) and orientations == set(range(4)),
             "background/orientation diversity collapsed")
    _require(counts == {2, 3, 4, 5}, "marker-count variation collapsed")
    _require(min(densities) < 0.20 and max(densities) > 0.55, "payload density lacks range")
    _require(min(observations.values()) >= 42, "a core structural feature lacks coverage")
    # Exercise legal inputs beyond the sampled payload vocabulary and size
    # floor, including the 5x5 minimum, many ties, and arbitrary cell colors.
    fuzz_rng = np.random.default_rng(731)
    for index in range(42):
        nr, nc = (2, 2) if index == 0 else (int(fuzz_rng.integers(2, 6)) for _ in range(2))
        floor = 2 * max(nr, nc) + 1
        shape = (5, 5) if index == 0 else tuple(int(fuzz_rng.integers(floor, 31)) for _ in range(2))
        scene = np.zeros(shape, dtype=np.int8)
        scene[1:-1, 1:-1] = fuzz_rng.integers(0, 10, size=(shape[0] - 2, shape[1] - 2))
        for position, label in zip(_positions(fuzz_rng, shape[0], nr),
                                   _labels(fuzz_rng, nr, (2, 7), 3, nr >= 3)):
            scene[position, 0] = label
        for position, label in zip(_positions(fuzz_rng, shape[1], nc),
                                   _labels(fuzz_rng, nc, (2, 7), 4)):
            scene[0, position] = label
        expected = _reference(scene, 0)
        _require(np.array_equal(solve(scene), expected), "small/arbitrary-payload input failed")
        original = scene.copy()
        solve(scene)
        _require(np.array_equal(scene, original), "solve mutated its input")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(10000 + 10 * train + test, train, test)
            _require((len(p["train"]), len(p["test"])) == (train, test), "count ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {train}/{test}")
    _require(_puzzle_key(generate_puzzle(314159, 7, 3)) ==
             _puzzle_key(generate_puzzle(314159, 7, 3)), "nondeterministic generation")
    return {"seeds": 42, "pairs": 210, "additional_small_inputs": 42, "shapes": len(shapes),
            "geometries": len(geometry), "normalized_inputs": len(structures),
            "backgrounds": sorted(backgrounds), "orientations": sorted(orientations),
            "marker_counts": sorted(counts), "feature_pair_counts": observations,
            "density_range": [min(densities), max(densities)]}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
