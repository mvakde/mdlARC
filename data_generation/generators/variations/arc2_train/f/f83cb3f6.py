import numpy as np


def _grid_signature(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes()


def _array_from_rows(encoded: str) -> np.ndarray:
    rows = encoded.split("/")
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_INPUT_SIGNATURES = {
    _grid_signature(_array_from_rows(encoded))
    for encoded in (
        (
            "00000800000001/10000800000000/00000801000000/"
            "00000800000000/00000800000000/00000800100000/"
            "00000801000000/00000800001000/00100800100000/"
            "00000000000000/10000000000000/00000000001010"
        ),
        (
            "0050000000/0000000000/0500005000/0000000000/"
            "0000000005/0000050000/5000000000/0000000005/"
            "8888888888/0000000000/0050000000/0000000000/"
            "0005000000/0000005000/5000000500/0500000000/"
            "0000000000"
        ),
        (
            "0020020000/2000200000/0000000000/2000000002/"
            "0020000000/2000200000/0000000000/8880088880/"
            "0000000000/0000000002/2000000000/0000002020/"
            "0002000002/0000000000/0000000000"
        ),
        (
            "000000080000300/000000080000000/000000080030000/"
            "000000000000303/300030000000000/000003080000300/"
            "000000080000000/000300080000000/300003000003000/"
            "000000080300003"
        ),
    )
}


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
            ):
                raise ValueError(f"{name} must be an integer or None")
            if not low <= int(value) <= high:
                raise ValueError(
                    f"{name} must be between {low} and {high}"
                )

    train_count = None if num_train is None else int(num_train)
    test_count = None if num_test is None else int(num_test)

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif train_count is None:
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif test_count is None:
        maximum = min(4, train_count - 1)
        if maximum < 1:
            raise ValueError("no legal test count exists")
        test_count = int(rng.integers(1, maximum + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """The shared convention is semantic projection, not fixed literal colors."""
    colors = [int(color) for color in rng.permutation(10)]
    return {"base_background": colors[0], "base_separator": colors[1]}


def _run_count(positions: list[int]) -> int:
    return 1 + sum(b != a + 1 for a, b in zip(positions, positions[1:]))


def _sample_size(rng: np.random.Generator, low: int) -> int:
    band = int(rng.choice(3, p=[0.35, 0.40, 0.25]))
    limits = ((low, 11), (12, 20), (21, 30))[band]
    return int(rng.integers(limits[0], limits[1] + 1))


def _sample_occupied(length: int, kind: str, rng: np.random.Generator) -> list[int]:
    if kind == "full":
        return list(range(length))
    if kind == "partial":
        span = int(rng.integers(4, length))
        start = int(rng.integers(0, length - span + 1))
        return list(range(start, start + span))

    # Always retain the ends of the chosen span, so a missing position
    # strictly between structural cells is a visible hole.
    span = int(rng.integers(6, length + 1))
    start = int(rng.integers(0, length - span + 1))
    mask = np.ones(span, dtype=bool)
    style = int(rng.integers(0, 3))
    if style == 0:
        hole = int(rng.integers(1, span - 1))
        mask[hole] = False
    elif style == 1:
        hole_size = int(rng.integers(2, max(3, span - 3)))
        hole_size = min(hole_size, span - 4)
        hole = int(rng.integers(1, span - hole_size))
        mask[hole:hole + hole_size] = False
    else:
        holes = int(rng.integers(2, min(5, span - 3)))
        chosen = rng.choice(np.arange(1, span - 1), size=holes, replace=False)
        mask[chosen] = False
        if np.count_nonzero(mask) < 4:
            mask[int(chosen[0])] = True
    return [start + index for index in range(span) if mask[index]]


def _choose_colors(rng: np.random.Generator, parameters: dict,
                   index: int, train_payloads: list[int], test: bool) -> tuple[int, int, int]:
    base_background = parameters["base_background"]
    base_separator = parameters["base_separator"]
    background = base_background if index == 0 or rng.random() < 0.68 else int(rng.integers(0, 10))
    available = [c for c in range(10) if c != background]
    separator = (base_separator if base_separator != background and
                 (index == 0 or rng.random() < 0.68) else int(rng.choice(available)))
    available = [c for c in available if c != separator]
    if test and train_payloads and rng.random() < 0.52:
        reusable = [c for c in train_payloads if c in available]
        if reusable:
            return background, separator, int(rng.choice(reusable))
    if test:
        novel = [c for c in available if c not in train_payloads]
        if novel:
            available = novel
    elif index == 1:
        distinct = [c for c in available if c != train_payloads[0]]
        if distinct:
            available = distinct
    return background, separator, int(rng.choice(available))


def _make_pair(rng: np.random.Generator, orientation: str, kind: str,
               colors: tuple[int, int, int], profile: str) -> dict:
    axis_length = _sample_size(rng, 6)
    perpendicular = _sample_size(rng, 5)
    if orientation == "vertical":
        height, width = axis_length, perpendicular
    else:
        height, width = perpendicular, axis_length
    if rng.random() < 0.34:
        line = int(rng.choice([1, perpendicular - 2]))
    else:
        line = int(rng.integers(1, perpendicular - 1))
    occupied = _sample_occupied(axis_length, kind, rng)
    background, separator, payload = colors
    grid = np.full((height, width), background, dtype=np.int8)
    if orientation == "vertical":
        grid[occupied, line] = separator
    else:
        grid[line, occupied] = separator
    rooms = {-1: line, 1: perpendicular - line - 1}
    cells: set[tuple[int, int]] = set()

    def add(position: int, side: int, distance: int) -> None:
        coordinate = ((position, line + side * distance) if orientation == "vertical"
                      else (line + side * distance, position))
        cells.add(coordinate)

    # Density, direction, distance, and multiplicity are separately sampled.
    mode = str(rng.choice(["sparse", "adjacent", "medium", "dense"],
                          p=[0.22, 0.17, 0.38, 0.23]))
    if kind == "full" and mode == "adjacent":
        mode = "sparse"
    one_sided = profile != "rich" and rng.random() < 0.30
    sides = [-1, 1]
    if one_sided:
        sides = [int(rng.choice(sides))]
    rays = [(position, side) for position in occupied for side in sides]
    rng.shuffle(rays)
    if mode in ("sparse", "adjacent"):
        ray_count = int(rng.integers(1, min(4, len(rays)) + 1))
    else:
        density = float(rng.uniform(0.23, 0.52) if mode == "medium" else rng.uniform(0.53, 0.82))
        ray_count = max(1, min(len(rays), int(round(density * len(rays)))))
    cap = max(5, (height * width) // 5)
    for position, side in rays[:ray_count]:
        room = rooms[side]
        if mode == "adjacent":
            distances = [1]
        else:
            multiplicity = 1
            if mode == "dense" and room > 1 and rng.random() < 0.50:
                multiplicity = int(rng.integers(2, min(5, room + 1)))
            elif mode == "medium" and room > 1 and rng.random() < 0.24:
                multiplicity = 2
            multiplicity = min(room, multiplicity)
            distances = [int(v) for v in rng.choice(np.arange(1, room + 1), size=multiplicity, replace=False)]
        for distance in distances:
            if len(cells) < cap:
                add(position, side, distance)

    if profile == "rich":
        position = int(rng.choice(occupied))
        deep_side = -1 if rooms[-1] >= rooms[1] else 1
        add(position, deep_side, 1)
        add(position, deep_side, min(rooms[deep_side], 3))
        add(position, -deep_side, 1)
        if rooms[deep_side] >= 3 and rng.random() < 0.6:
            add(position, deep_side, 2)

    missing = [position for position in range(axis_length) if position not in occupied]
    internal = [position for position in missing if occupied[0] < position < occupied[-1]]
    ignored_count = int(rng.choice([0, 1, 2, 3, 4, 5, 6],
                                   p=[0.25, 0.18, 0.18, 0.14, 0.11, 0.08, 0.06])) if missing else 0
    if profile == "gap_demo":
        ignored_count = max(ignored_count, 2)
        assert internal
        focus = int(rng.choice(internal))
        add(focus, -1, 1)
        add(focus, 1, min(rooms[1], 2))
    elif internal and ignored_count >= 3 and rng.random() < 0.45:
        focus = int(rng.choice(internal))
        add(focus, -1, 1)
        add(focus, 1, 1)
        ignored_count -= 2
    for _ in range(ignored_count):
        if not missing or len(cells) >= cap:
            break
        pool = internal if internal and rng.random() < 0.72 else missing
        position = int(rng.choice(pool))
        side = int(rng.choice([-1, 1]))
        distance = 1 if internal and position in internal and rng.random() < 0.42 else int(rng.integers(1, rooms[side] + 1))
        add(position, side, distance)

    # Keep a sparse structural fragment uniquely identifiable by its
    # collinearity, even when all payload traffic is on one side.
    def payload_collinear() -> bool:
        if len(cells) < 4:
            return False
        return (len({r for r, _ in cells}) == 1 or len({c for _, c in cells}) == 1)
    if payload_collinear():
        if mode == "adjacent":
            cells = set(sorted(cells)[:3])
        else:
            position = int(occupied[-1] if occupied[-1] != occupied[0] else occupied[0])
            candidate = ((position, line + 1) if orientation == "vertical" else (line + 1, position))
            cells.add(candidate)
            if payload_collinear():
                cells = set(sorted(cells)[:3])

    for cell in cells:
        grid[cell] = payload
    output = _project(grid, background, separator, payload, orientation, line, occupied)
    if np.array_equal(grid, output):
        if missing:
            side = -1
            add(int(missing[0]), side, 1)
        else:
            side = -1 if rooms[-1] >= 2 else 1
            add(int(occupied[0]), side, 2)
        for cell in cells:
            grid[cell] = payload
        output = _project(grid, background, separator, payload, orientation, line, occupied)
    return {"input": grid, "output": output}


def _project(grid: np.ndarray, background: int, separator: int, payload: int,
             orientation: str, line: int, occupied: list[int]) -> np.ndarray:
    result = np.full(grid.shape, background, dtype=np.int8)
    for position in occupied:
        if orientation == "vertical":
            result[position, line] = separator
            if np.any(grid[position, :line] == payload):
                result[position, line - 1] = payload
            if np.any(grid[position, line + 1:] == payload):
                result[position, line + 1] = payload
        else:
            result[line, position] = separator
            if np.any(grid[:line, position] == payload):
                result[line - 1, position] = payload
            if np.any(grid[line + 1:, position] == payload):
                result[line + 1, position] = payload
    return result


def _inspect_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    background = int(values[np.argmax(counts)])
    assert int(np.max(counts)) > int(np.sort(counts)[-2])
    candidates = []
    for value in values:
        color = int(value)
        if color == background:
            continue
        locations = np.argwhere(grid == color)
        if len(locations) < 4:
            continue
        same_row = bool(np.all(locations[:, 0] == locations[0, 0]))
        same_column = bool(np.all(locations[:, 1] == locations[0, 1]))
        if same_row != same_column:
            candidates.append((color, locations, "vertical" if same_column else "horizontal"))
    assert len(candidates) == 1
    separator, locations, orientation = candidates[0]
    payload = next(int(v) for v in values if int(v) not in (background, separator))
    line = int(locations[0, 1] if orientation == "vertical" else locations[0, 0])
    occupied = sorted(int(v) for v in locations[:, 0 if orientation == "vertical" else 1])
    axis_length = grid.shape[0 if orientation == "vertical" else 1]
    perpendicular = grid.shape[1 if orientation == "vertical" else 0]
    assert 0 < line < perpendicular - 1
    internal = [i for i in range(occupied[0] + 1, occupied[-1]) if i not in occupied]
    kind = "gapped" if internal else ("full" if len(occupied) == axis_length else "partial")
    return {"background": background, "separator": separator, "payload": payload,
            "orientation": orientation, "line": line, "occupied": occupied,
            "kind": kind, "internal": internal, "axis_length": axis_length}


def _predict_from_input(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # The training outputs establish projection, while roles and geometry
    # are recovered from the test input itself. No generator state is read.
    assert len(train_pairs) >= 2
    for pair in train_pairs:
        info = _inspect_input(pair["input"])
        assert np.array_equal(pair["output"], _project(pair["input"], info["background"],
                              info["separator"], info["payload"], info["orientation"],
                              info["line"], info["occupied"]))
    info = _inspect_input(test_input)
    return _project(test_input, info["background"], info["separator"],
                    info["payload"], info["orientation"], info["line"], info["occupied"])


def _features(grid: np.ndarray, info: dict) -> dict:
    payload = info["payload"]
    line = info["line"]
    occupied = info["occupied"]
    negative = []
    positive = []
    for position in occupied:
        if info["orientation"] == "vertical":
            negative.append(int(np.count_nonzero(grid[position, :line] == payload)))
            positive.append(int(np.count_nonzero(grid[position, line + 1:] == payload)))
        else:
            negative.append(int(np.count_nonzero(grid[:line, position] == payload)))
            positive.append(int(np.count_nonzero(grid[line + 1:, position] == payload)))
    locations = np.argwhere(grid == payload)
    axis_index = 0 if info["orientation"] == "vertical" else 1
    perp_index = 1 - axis_index
    ignored = [tuple(int(v) for v in point) for point in locations
               if int(point[axis_index]) not in occupied]
    internal_ignored = [point for point in ignored if point[axis_index] in info["internal"]]
    distant = any(int(point[axis_index]) in occupied and abs(int(point[perp_index]) - line) > 1
                  for point in locations)
    return {"collapse": any(v > 1 for v in negative + positive),
            "both_sides": any(v > 0 for v in negative) and any(v > 0 for v in positive),
            "same_position_both_sides": any(a and b for a, b in zip(negative, positive)),
            "distant": distant, "ignored": len(ignored), "internal_ignored": len(internal_ignored),
            "immediate_hole": any(abs(point[perp_index] - line) == 1 for point in internal_ignored),
            "one_sided": not (any(v > 0 for v in negative) and any(v > 0 for v in positive)),
            "unhit": sum(a == 0 and b == 0 for a, b in zip(negative, positive)),
            "max_ray": max(negative + positive), "payload_count": len(locations),
            "adjacent_only": all(abs(int(point[perp_index]) - line) == 1 for point in locations)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    facts: dict[str, list[tuple[dict, dict]]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = _grid_signature(inp)
            assert signature not in seen and signature not in _SOURCE_INPUT_SIGNATURES
            seen.add(signature)
            info = _inspect_input(inp)
            assert np.array_equal(out, _project(inp, info["background"], info["separator"],
                                                info["payload"], info["orientation"],
                                                info["line"], info["occupied"]))
            assert set(int(v) for v in np.unique(out)) == set(int(v) for v in np.unique(inp))
            assert np.array_equal(inp == info["separator"], out == info["separator"])
            facts[split].append((info, _features(inp, info)))
    train = facts["train"]
    assert train[0][1]["collapse"] and train[0][1]["both_sides"] and train[0][1]["distant"]
    assert train[0][1]["same_position_both_sides"]
    assert train[1][0]["kind"] == "gapped" and train[1][1]["internal_ignored"] >= 1
    assert train[1][1]["immediate_hole"]
    for info, flags in facts["test"]:
        if flags["internal_ignored"]:
            assert any(f["internal_ignored"] for _, f in train)
        if flags["collapse"]:
            assert any(f["collapse"] for _, f in train)
        if flags["same_position_both_sides"]:
            assert any(f["same_position_both_sides"] for _, f in train)
    for pair in puzzle["test"]:
        assert np.array_equal(pair["output"], _predict_from_input(puzzle["train"], pair["input"].copy()))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    shared_orientation = str(rng.choice(["vertical", "horizontal"]))
    composition = str(rng.choice(["same", "mixed"], p=[0.29, 0.71]))
    other = "horizontal" if shared_orientation == "vertical" else "vertical"
    train_orientations = [shared_orientation, shared_orientation if composition == "same" else other]
    train_orientations += [shared_orientation if composition == "same" else str(rng.choice([shared_orientation, other]))
                           for _ in range(train_count - 2)]
    test_orientations = [shared_orientation if composition == "same" else str(rng.choice([shared_orientation, other]))
                         for _ in range(test_count)]
    train_kinds = [str(rng.choice(["full", "partial"])), "gapped"]
    train_kinds += [str(rng.choice(["full", "partial", "gapped"], p=[0.24, 0.31, 0.45]))
                    for _ in range(train_count - 2)]
    test_kinds = [str(rng.choice(["full", "partial", "gapped"], p=[0.18, 0.28, 0.54]))
                  for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_payloads: list[int] = []
    index = 0
    for split, orientations, kinds in (("train", train_orientations, train_kinds),
                                       ("test", test_orientations, test_kinds)):
        for local_index, (orientation, kind) in enumerate(zip(orientations, kinds)):
            colors = _choose_colors(rng, parameters, index, train_payloads, split == "test")
            profile = ("rich" if split == "train" and local_index == 0 else
                       "gap_demo" if split == "train" and local_index == 1 else "free")
            for _ in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, orientation, kind, colors, profile)
                signature = _grid_signature(pair["input"])
                if signature in seen or signature in _SOURCE_INPUT_SIGNATURES:
                    continue
                try:
                    info = _inspect_input(pair["input"])
                    if not np.array_equal(pair["output"], _project(pair["input"], info["background"],
                                                                     info["separator"], info["payload"],
                                                                     info["orientation"], info["line"],
                                                                     info["occupied"])):
                        continue
                except AssertionError:
                    continue
                puzzle[split].append(pair)
                seen.add(signature)
                break
            else:
                raise RuntimeError("could not construct an identifiable distinct pair")
            if split == "train":
                train_payloads.append(colors[2])
            index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return (all(len(first[split]) == len(second[split]) and
                all(np.array_equal(a[key], b[key]) for a, b in zip(first[split], second[split])
                    for key in ("input", "output")) for split in ("train", "test")))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    dimensions, layouts, orientations, kinds = set(), set(), set(), set()
    flags_seen = {name: 0 for name in ("collapse", "both_sides", "one_sided", "adjacent_only",
                                       "distant", "immediate_hole")}
    ignored_counts, offsets, occupied_counts, palettes = set(), set(), set(), set()
    same_orientation_puzzles = 0
    reused_test_payloads = 0
    changed_background_puzzles = 0
    changed_separator_puzzles = 0
    simple_pairs = 0
    review_counts = {name: 0 for name in (
        "near_edge", "ignored_zero", "ignored_multiple", "ignored_four_plus",
        "gapped_many_cells", "single_hole", "full_span_one_hole", "long_holes", "three_at_hole",
        "train_vertical_gapped", "test_horizontal", "test_continuous",
        "train_full_vertical", "within_ignored_varies", "within_density_varies",
        "novel_test_payload")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = []
        puzzle_features = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_input(pair["input"])
                flags = _features(pair["input"], info)
                assert np.array_equal(pair["output"], _predict_from_input(puzzle["train"], pair["input"]))
                dimensions.add(tuple(pair["input"].shape))
                layouts.add((tuple(pair["input"].shape), info["orientation"], info["line"],
                             tuple(info["occupied"])))
                orientations.add(info["orientation"])
                kinds.add(info["kind"])
                occupied_counts.add(len(info["occupied"]))
                offsets.add(info["line"])
                palettes.add((info["background"], info["separator"], info["payload"]))
                ignored_counts.add(flags["ignored"])
                for name in flags_seen:
                    flags_seen[name] += bool(flags[name])
                simple_pairs += not flags["collapse"] and not flags["both_sides"]
                infos.append(info)
                puzzle_features.append(flags)
                perpendicular = pair["input"].shape[1 if info["orientation"] == "vertical" else 0]
                review_counts["near_edge"] += min(info["line"], perpendicular - info["line"] - 1) == 1
                review_counts["ignored_zero"] += flags["ignored"] == 0
                review_counts["ignored_multiple"] += flags["ignored"] >= 2
                review_counts["ignored_four_plus"] += flags["ignored"] >= 4
                review_counts["gapped_many_cells"] += info["kind"] == "gapped" and len(info["occupied"]) >= 5
                review_counts["single_hole"] += info["kind"] == "gapped" and len(info["internal"]) == 1
                review_counts["full_span_one_hole"] += (info["kind"] == "gapped" and
                                                        len(info["internal"]) == 1 and
                                                        info["occupied"][0] == 0 and
                                                        info["occupied"][-1] == info["axis_length"] - 1)
                review_counts["long_holes"] += info["kind"] == "gapped" and len(info["internal"]) >= 3
                if info["kind"] == "gapped":
                    axis_index = 0 if info["orientation"] == "vertical" else 1
                    hole_hits = [int(point[axis_index]) for point in np.argwhere(pair["input"] == info["payload"])
                                 if int(point[axis_index]) in info["internal"]]
                    review_counts["three_at_hole"] += any(hole_hits.count(pos) >= 3 for pos in info["internal"])
                review_counts["train_vertical_gapped"] += split == "train" and info["orientation"] == "vertical" and info["kind"] == "gapped"
                review_counts["test_horizontal"] += split == "test" and info["orientation"] == "horizontal"
                review_counts["test_continuous"] += split == "test" and info["kind"] != "gapped"
                review_counts["train_full_vertical"] += split == "train" and info["orientation"] == "vertical" and info["kind"] == "full"
        same_orientation_puzzles += len({i["orientation"] for i in infos}) == 1
        review_counts["within_ignored_varies"] += len({f["ignored"] for f in puzzle_features}) > 1
        review_counts["within_density_varies"] += len({f["payload_count"] for f in puzzle_features}) > 1
        changed_background_puzzles += len({i["background"] for i in infos}) > 1
        changed_separator_puzzles += len({i["separator"] for i in infos}) > 1
        train_payloads = {i["payload"] for i in infos[:len(puzzle["train"])]}
        reused_test_payloads += sum(i["payload"] in train_payloads for i in infos[len(puzzle["train"]):])
        review_counts["novel_test_payload"] += sum(i["payload"] not in train_payloads
                                                    for i in infos[len(puzzle["train"]):])
    assert len(dimensions) >= 55 and len(layouts) >= 150
    assert len(occupied_counts) >= 12 and len(offsets) >= 10 and len(palettes) >= 30
    assert orientations == {"vertical", "horizontal"} and kinds == {"full", "partial", "gapped"}
    assert len(ignored_counts) >= 5 and simple_pairs >= 10
    assert all(v > 0 for v in flags_seen.values())
    assert same_orientation_puzzles >= 4 and reused_test_payloads >= 10
    assert changed_background_puzzles >= 10 and changed_separator_puzzles >= 10
    assert all(value > 0 for value in review_counts.values())
    assert review_counts["within_ignored_varies"] >= 35
    assert review_counts["within_density_varies"] >= 35
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(901 + train_count * 10 + test_count, train_count, test_count)
        validate_generated_puzzle(p)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(930, train_count, test_count))
    assert _puzzles_equal(generate_puzzle(950), generate_puzzle(950))
    assert _puzzles_equal(generate_puzzle(951, 7, 3), generate_puzzle(951, 7, 3))
    assert not _puzzles_equal(generate_puzzle(952), generate_puzzle(953))
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(960, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
