"""Compositional generators for the marker-centered ARC scaffold."""

import numpy as np


# Direction points from the marker toward its full-canvas foundation line.
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SOURCE_LAYOUTS = (
    ((10, 10), ((4, 4),)),
    ((16, 16), ((4, 11), (10, 3))),
    ((17, 19), ((3, 4), (8, 11), (13, 6))),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    rng: np.random.Generator, seed: int,
    num_train: int | None, num_test: int | None,
) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        minimum = max(2, test + 1)
        train = minimum + (int(seed) // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _geometry(rng: np.random.Generator) -> tuple[int, int, int, int]:
    # Source geometry remains common; the other choices are learned per puzzle.
    if rng.random() < 0.32:
        return 0, 2, 2, 2
    direction = int(rng.choice(4, p=[0.4, 0.2, 0.2, 0.2]))
    return direction, int(rng.integers(1, 4)), int(rng.integers(2, 4)), int(rng.integers(2, 4))


def _at(row: int, column: int, direction: int, t: int, s: int) -> tuple[int, int]:
    dr, dc = _DIRECTIONS[direction]
    return row + t * dr - s * dc, column + t * dc + s * dr


def _render(
    input_grid: np.ndarray,
    geometry: tuple[int, int, int, int],
    anchor_color: int,
    colors: tuple[int, int, int],
) -> np.ndarray:
    direction, halfwidth, top, bottom = geometry
    cap, support, foundation = colors
    anchors = [(int(r), int(c)) for r, c in np.argwhere(input_grid == anchor_color)]
    output = input_grid.copy()
    dr, dc = _DIRECTIONS[direction]

    # Draw each full-width line once. All local segments are placed afterward,
    # so two constructions on one line retain both foundation segments.
    for axis in {r if dr else c for r, c in anchors}:
        line = axis + bottom * (dr if dr else dc)
        if dr:
            output[line, :] = support
        else:
            output[:, line] = support

    for row, column in anchors:
        for s in range(-halfwidth, halfwidth + 1):
            rr, cc = _at(row, column, direction, -top, s)
            output[rr, cc] = cap
            rr, cc = _at(row, column, direction, bottom, s)
            output[rr, cc] = foundation
        rr, cc = _at(row, column, direction, -top + 1, 0)
        output[rr, cc] = cap
        for t in range(-top + 1, bottom):
            for s in (-halfwidth, halfwidth):
                rr, cc = _at(row, column, direction, t, s)
                output[rr, cc] = support
        output[row, column] = anchor_color
    return output


def _signature(grid: np.ndarray) -> tuple:
    grid = np.ascontiguousarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _source_copy(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    source_geometry = (0, 2, 2, 2)
    for shape, anchors in _SOURCE_LAYOUTS:
        if input_grid.shape != shape:
            continue
        original = np.zeros(shape, dtype=np.int8)
        for row, column in anchors:
            original[row, column] = 3
        if np.array_equal(input_grid, original) and np.array_equal(
            output_grid, _render(original, source_geometry, 3, (5, 2, 8))
        ):
            return True
    return False


def _partition(rng: np.random.Generator, total: int, bins: int) -> list[int]:
    if total == 0:
        return [0] * bins
    probabilities = rng.dirichlet(np.ones(bins))
    return [int(x) for x in rng.multinomial(total, probabilities)]


def _dimension(rng: np.random.Generator, minimum: int) -> int:
    if minimum == 30:
        return 30
    draw = rng.random()
    if draw < 0.17:
        return minimum
    if draw < 0.28:
        return 30
    return int(rng.integers(minimum, 31))


def _uv_to_rc(u: int, v: int, direction: int, t_length: int, s_length: int) -> tuple[int, int]:
    if direction == 0:
        return u, v
    if direction == 1:
        return t_length - 1 - u, s_length - 1 - v
    if direction == 2:
        return s_length - 1 - v, u
    return v, t_length - 1 - u


def _group_sizes(
    rng: np.random.Generator,
    markers: int,
    geometry: tuple[int, int, int, int],
    layout: str,
) -> list[int]:
    _, halfwidth, top, bottom = geometry
    band_height = top + bottom + 1
    width = 2 * halfwidth + 1
    separation = width + 1
    capacity = 1 + (30 - (width + 2)) // separation
    if layout == "single":
        return [1]
    if layout == "vertical":
        return [1] * markers
    if layout == "shared":
        return [markers]
    candidates = [
        bands for bands in range(1, min(markers, 30 // band_height) + 1)
        if bands * capacity >= markers
    ]
    bands = int(rng.choice(candidates))
    sizes = [1] * bands
    for _ in range(markers - bands):
        choices = [index for index, size in enumerate(sizes) if size < capacity]
        sizes[int(rng.choice(choices))] += 1
    rng.shuffle(sizes)
    return sizes


def _layout(
    rng: np.random.Generator,
    geometry: tuple[int, int, int, int],
    group_sizes: list[int],
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    direction, halfwidth, top, bottom = geometry
    band_height = top + bottom + 1
    width = 2 * halfwidth + 1
    separation = width + 1
    min_t = len(group_sizes) * band_height
    min_s = width + 2 + (max(group_sizes) - 1) * separation
    t_length = _dimension(rng, min_t)
    s_length = _dimension(rng, min_s)
    if rng.random() < 0.22:
        side = _dimension(rng, max(min_t, min_s))
        t_length = s_length = side

    gaps = _partition(rng, t_length - min_t, len(group_sizes) + 1)
    anchors_uv: list[tuple[int, int]] = []
    u = top + gaps[0]
    prior_centers: list[int] = []
    for index, size in enumerate(group_sizes):
        slack = s_length - width - (size - 1) * separation
        side_gaps = _partition(rng, slack, size + 1)
        centers: list[int] = []
        v = halfwidth + side_gaps[0]
        for j in range(size):
            if j:
                v += separation + side_gaps[j]
            centers.append(v)

        # Align groups occasionally without changing their nonoverlap.
        if prior_centers and rng.random() < 0.42:
            candidates = []
            for old in prior_centers:
                for j, center in enumerate(centers):
                    shifted = [x + old - center for x in centers]
                    if shifted[0] >= halfwidth and shifted[-1] <= s_length - halfwidth - 1:
                        candidates.append(shifted)
            if candidates:
                centers = list(candidates[int(rng.integers(len(candidates)))])
        anchors_uv.extend((u, center) for center in centers)
        prior_centers.extend(centers)
        if index + 1 < len(group_sizes):
            u += band_height + gaps[index + 1]

    anchors = tuple(sorted(
        _uv_to_rc(u, v, direction, t_length, s_length) for u, v in anchors_uv
    ))
    shape = (t_length, s_length) if direction < 2 else (s_length, t_length)
    return shape, anchors


def _input_grid(shape: tuple[int, int], anchors: tuple[tuple[int, int], ...],
                background: int, marker: int) -> np.ndarray:
    grid = np.full(shape, background, dtype=np.int8)
    for row, column in anchors:
        grid[row, column] = marker
    return grid


def _input_roles(
    rng: np.random.Generator,
    available: tuple[int, ...],
    stable_roles: tuple[int, int] | None,
    index: int,
) -> tuple[int, int]:
    if stable_roles is not None:
        return stable_roles
    # Advance the background through a shuffled seven-color cycle. This
    # guarantees actual within-puzzle input-palette variation.
    background = available[index % len(available)]
    marker = available[(index + 1 + int(rng.integers(0, len(available) - 1))) % len(available)]
    if marker == background:
        marker = available[(index + 1) % len(available)]
    return background, marker


def _infer_geometry(
    input_grid: np.ndarray, output_grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]]:
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) == 2, "input must contain a background and markers")
    _require(counts[0] != counts[1], "background must be identifiable")
    background = int(values[int(np.argmax(counts))])
    marker = int(values[int(np.argmin(counts))])
    anchors = np.argwhere(input_grid == marker)
    _require(len(anchors) == 1, "the first training pair must show one marker")
    row, column = (int(x) for x in anchors[0])
    extra = set(int(x) for x in np.unique(output_grid)) - {background, marker}
    _require(len(extra) == 3, "training must expose cap, support, and foundation")
    candidates = []
    for direction in range(4):
        for halfwidth in (1, 2, 3):
            for top in (2, 3):
                for bottom in (2, 3):
                    points = [
                        _at(row, column, direction, -top, 0),
                        _at(row, column, direction, 0, halfwidth),
                        _at(row, column, direction, bottom, 0),
                    ]
                    if not all(
                        0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]
                        for r, c in points
                    ):
                        continue
                    colors = tuple(int(output_grid[r, c]) for r, c in points)
                    if set(colors) != extra:
                        continue
                    geometry = (direction, halfwidth, top, bottom)
                    if np.array_equal(output_grid, _render(input_grid, geometry, marker, colors)):
                        candidates.append((geometry, colors))
    _require(len(candidates) == 1, "training does not uniquely reveal its construction")
    return candidates[0]


def _inspect_pair(
    pair: dict, geometry: tuple[int, int, int, int],
    colors: tuple[int, int, int],
) -> dict:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
        _require(np.issubdtype(grid.dtype, np.integer), "grid must have an integer dtype")
        _require(all(1 <= x <= 30 for x in grid.shape), "grid size out of bounds")
        _require(0 <= int(grid.min()) <= int(grid.max()) <= 9, "invalid ARC color")
    _require(inp.shape == out.shape, "input and output dimensions differ")
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 2 and counts[0] != counts[1], "input roles are ambiguous")
    background = int(values[int(np.argmax(counts))])
    marker = int(values[int(np.argmin(counts))])
    anchors = tuple((int(r), int(c)) for r, c in np.argwhere(inp == marker))
    _require(len(anchors) >= 1 and counts.max() > len(anchors), "invalid sparse markers")
    _require(len({background, marker, *colors}) == 5, "color roles must be distinct")

    direction, halfwidth, top, bottom = geometry
    dr, dc = _DIRECTIONS[direction]
    band_height = top + bottom + 1
    groups: dict[int, list[int]] = {}
    for row, column in anchors:
        for t in (-top, bottom):
            for s in (-halfwidth, halfwidth):
                rr, cc = _at(row, column, direction, t, s)
                _require(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1],
                         "construction is clipped")
        axis = row if dr else column
        across = column if dr else row
        groups.setdefault(axis, []).append(across)
    axes = sorted(groups)
    _require(all(b - a >= band_height for a, b in zip(axes, axes[1:])),
             "different construction bands overlap")
    for group in groups.values():
        group.sort()
        _require(all(b - a >= 2 * halfwidth + 2 for a, b in zip(group, group[1:])),
                 "local footprints on a shared line overlap")
    _require(np.array_equal(out, _render(inp, geometry, marker, colors)),
             "output breaks the learned stamping rule")
    _require(set(int(x) for x in np.unique(out)) == {background, marker, *colors},
             "output omits a semantic role")
    _require(not _source_copy(inp, out), "pair copies a source example")
    return {
        "shape": inp.shape, "anchors": anchors, "count": len(anchors),
        "groups": tuple(tuple(group) for group in groups.values()),
        "shared": any(len(group) > 1 for group in groups.values()),
        "input_roles": (background, marker),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    first = puzzle["train"][0]
    _require(isinstance(first, dict) and set(first) == {"input", "output"},
             "invalid first training pair")
    geometry, colors = _infer_geometry(first["input"], first["output"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair, geometry, colors)
            infos[split].append(info)
            input_key = _signature(pair["input"])
            pair_key = input_key, _signature(pair["output"])
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

    _require(infos["train"][0]["count"] == 1, "first training pair must isolate the motif")
    _require(infos["train"][1]["count"] == 2 and not infos["train"][1]["shared"],
             "second training pair must demonstrate independent foundations")
    if train_count >= 3:
        _require(infos["train"][2]["count"] == 2 and infos["train"][2]["shared"],
                 "training must demonstrate a shared foundation")
    else:
        _require(not any(info["shared"] for info in infos["test"]),
                 "shared-line tests need a training demonstration")
    _require(len({info["shape"] for info in infos["train"]}) >= 2,
             "training needs multiple canvas shapes")
    # Each marker's input identity is visible in its pair. The three new output
    # colors and the entire geometry are recovered from the first training pair.
    _require(all(set(colors).isdisjoint(info["input_roles"])
                 for split in infos.values() for info in split),
             "an input role collides with a learned output color")


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    geometry = _geometry(rng)
    palette = [int(x) for x in rng.choice(10, size=10, replace=False)]
    colors = tuple(palette[:3])
    available = tuple(palette[3:])
    stable_roles = tuple(available[:2]) if rng.random() < 0.30 else None

    train_specs = [(1, "single"), (2, "vertical")]
    if train_count >= 3:
        train_specs.append((2, "shared"))
    while len(train_specs) < train_count:
        train_specs.append((int(rng.choice((1, 2, 3, 4, 5, 6),
                                           p=(.16, .23, .23, .18, .12, .08))), "random"))
    test_specs = []
    for _ in range(test_count):
        maximum = 6 if train_count >= 3 else 3
        choices = np.arange(1, maximum + 1)
        weights = np.array((.13, .18, .28, .20, .13, .08)[:maximum])
        weights = weights / weights.sum()
        test_specs.append((int(rng.choice(choices, p=weights)), "random" if train_count >= 3 else "vertical"))

    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (split, specification) in enumerate(
        [("train", x) for x in train_specs] + [("test", x) for x in test_specs]
    ):
        markers, layout = specification
        # Pair streams are independent of the number of failed uniqueness
        # attempts in earlier pairs.
        pair_rng = np.random.default_rng([int(seed), index + 1, 314159])
        background, marker = _input_roles(pair_rng, available, stable_roles, index)
        for _ in range(120):
            groups = _group_sizes(pair_rng, markers, geometry, layout)
            shape, anchors = _layout(pair_rng, geometry, groups)
            inp = _input_grid(shape, anchors, background, marker)
            out = _render(inp, geometry, marker, colors)
            key = _signature(inp)
            if key in seen or _source_copy(inp, out):
                continue
            if split == "train" and len(puzzle["train"]) == 1:
                if shape == puzzle["train"][0]["input"].shape:
                    continue
            puzzle[split].append({"input": inp, "output": out})
            seen.add(key)
            break
        else:
            raise RuntimeError("could not construct a unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split]) and
        all(np.array_equal(a[key], b[key])
            for a, b in zip(first[split], second[split])
            for key in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    shared_input = _input_grid((10, 16), ((4, 4), (4, 11)), 0, 3)
    shared_output = _render(shared_input, (0, 2, 2, 2), 3, (5, 2, 8))
    _require(
        "".join(str(int(x)) for x in shared_output[6]) == "2288888228888822",
        "shared foundation erased a segment",
    )
    boundary_input = _input_grid((5, 8), ((2, 2),), 0, 3)
    boundary_output = _render(boundary_input, (0, 2, 2, 2), 3, (5, 2, 8))
    _require(
        np.all(boundary_output[0, :5] == 5)
        and np.all(boundary_output[4, :5] == 8)
        and np.all(boundary_output[4, 5:] == 2),
        "top/bottom boundary construction is clipped",
    )
    _require(_puzzles_equal(generate_puzzle(123456), generate_puzzle(123456)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(314159, 7, 3),
                            generate_puzzle(314159, 7, 3)),
             "explicit generation is not deterministic")
    for seed, train, test in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4),
                              (9101, None, 1), (9102, None, 4),
                              (9103, 2, None), (9104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(puzzle["train"]) == train, "requested train count changed")
        if test is not None:
            _require(len(puzzle["test"]) == test, "requested test count changed")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(9200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")

    train_counts, test_counts, shapes, palettes, layouts = set(), set(), set(), set(), set()
    geometries, marker_counts, input_role_changes = set(), set(), 0
    shared_train = shared_test = square_train = square_test = 0
    aligned = boundary = side_touch = touching = wide_gap = unequal_gap = 0
    rectangular_train = familiar_test = smaller_test = count_shift = 0
    varied_palette = stable_palette = combinations = 0
    puzzle_signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        geometry, colors = _infer_geometry(
            puzzle["train"][0]["input"], puzzle["train"][0]["output"])
        geometries.add(geometry)
        palettes.add(colors)
        puzzle_signatures.add(tuple(
            (split, _signature(pair["input"]), _signature(pair["output"]))
            for split in ("train", "test") for pair in puzzle[split]))
        roles = set()
        train_max_count = max(
            len(np.argwhere(pair["input"] != np.bincount(pair["input"].ravel()).argmax()))
            for pair in puzzle["train"]
        )
        train_max_side = max(max(pair["input"].shape) for pair in puzzle["train"])
        if any(
            len(np.argwhere(pair["input"] != np.bincount(pair["input"].ravel()).argmax()))
            > train_max_count for pair in puzzle["test"]
        ):
            count_shift += 1
        if any(max(pair["input"].shape) <= train_max_side for pair in puzzle["test"]):
            smaller_test += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair, geometry, colors)
                shapes.add(info["shape"])
                layouts.add((info["shape"], info["anchors"], geometry))
                marker_counts.add(info["count"])
                roles.add(info["input_roles"])
                if split == "train" and info["shape"][0] != info["shape"][1]:
                    rectangular_train += 1
                if split == "test" and info["count"] <= train_max_count:
                    familiar_test += 1
                if info["shape"][0] == info["shape"][1]:
                    if split == "train":
                        square_train += 1
                    else:
                        square_test += 1
                if info["shared"]:
                    if split == "train":
                        shared_train += 1
                    else:
                        shared_test += 1
                anchors = info["anchors"]
                if len({c if geometry[0] < 2 else r for r, c in anchors}) < len(anchors):
                    aligned += 1
                direction, halfwidth, top, bottom = geometry
                dr, dc = _DIRECTIONS[direction]
                if any(
                    any(_at(r, c, direction, t, s)[axis] in (0, info["shape"][axis] - 1)
                        for axis in (0, 1))
                    for r, c in anchors for t in (-top, bottom)
                    for s in (-halfwidth, halfwidth)
                ):
                    boundary += 1
                if any(
                    (c if dr else r) in (halfwidth, (info["shape"][1] if dr else info["shape"][0]) - halfwidth - 1)
                    for r, c in anchors
                ):
                    side_touch += 1
                axes = sorted({r if dr else c for r, c in anchors})
                gaps = [b - a - (top + bottom + 1) for a, b in zip(axes, axes[1:])]
                touching += 0 in gaps
                wide_gap += any(gap >= 3 for gap in gaps)
                unequal_gap += len(set(gaps)) > 1
                if info["count"] >= 4 and info["shared"] and info["shape"][0] != info["shape"][1]:
                    combinations += 1
        if len(roles) > 1:
            input_role_changes += 1
            varied_palette += 1
        else:
            stable_palette += 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity collapsed")
    _require(len(geometries) >= 10 and len({g[0] for g in geometries}) == 4,
             "geometry diversity collapsed")
    _require(len({g[1] for g in geometries}) == 3, "width diversity collapsed")
    _require(len(palettes) >= 25 and len(shapes) >= 35 and len(layouts) >= 100,
             "palette, shape, or layout diversity collapsed")
    _require(marker_counts == set(range(1, 7)), "marker-count coverage collapsed")
    _require(input_role_changes >= 15, "input colors did not vary within puzzles")
    _require(shared_train > 0 and shared_test > 0 and aligned > 0 and boundary > 0,
             "placement diversity collapsed")
    _require(square_train > 0 and square_test > 0, "square-canvas coverage collapsed")
    _require(rectangular_train > 0 and familiar_test > 0 and smaller_test > 0
             and count_shift > 0, "train/test regimes collapsed")
    _require(side_touch > 0 and touching > 0 and wide_gap > 0 and unequal_gap > 0,
             "margin and spacing diversity collapsed")
    _require(varied_palette > 0 and stable_palette > 0 and combinations > 0,
             "composable palette and layout diversity collapsed")
    _require(len(puzzle_signatures) == 42, "different seeds produced identical puzzles")
