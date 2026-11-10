"""Anchored stencil completion puzzles with learned marker operations."""

import numpy as np


_NEIGHBORS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if dr or dc
)
_TRANSFORMS = (
    "identity", "horizontal", "vertical", "half_turn",
    "quarter_right", "quarter_left", "main_diagonal", "anti_diagonal",
)
_SOURCE_INPUTS = (
    ('0000000000000','0000000000200','0038800000000','0080000000000','0000000000000','0000000000000','0000000000000','0000004200000','0300000440000','0000000000000','0000000000000','0000000000000','0000000000000'),
    ('0000000000000','0000004000000','0000043400000','0000044000000','0000000000000','0000000000000','0030000000000','0000000000000','0000000000000','0000000003000','0000000000000','0000000000000','0000000000000'),
    ('0000000000000','0000000444000','0002000024000','0000000004000','0000000000000','0000000000000','0000000000000','0000000000110','0000000001310','0000300000000','0000000000000','0000000020000','0000000000000'),
    ('0000000000000','0000000000000','0000000000000','0002100000000','0001100000000','0000010000000','0000000000000','0000000000000','0000000200000','0000000000000','0000000000000','0000000000000','0000000000000'),
    ('0000000000000','0010030002000','0120000000000','0110000000000','0000000000000','0000000088800','0000000003800','0000000008000','0000000000000','0002000000000','0000000300000','0000000000000','0000000000000'),
)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _clean_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _clean_count(num_train, "num_train", 2, 10)
    num_test = _clean_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _transform(offsets, kind):
    result = []
    for r, c in offsets:
        if kind == "identity":
            point = (r, c)
        elif kind == "horizontal":
            point = (r, -c)
        elif kind == "vertical":
            point = (-r, c)
        elif kind == "half_turn":
            point = (-r, -c)
        elif kind == "quarter_right":
            point = (c, -r)
        elif kind == "quarter_left":
            point = (-c, r)
        elif kind == "main_diagonal":
            point = (c, r)
        else:
            point = (-c, -r)
        result.append(point)
    return tuple(sorted(result))


def _sample_offsets(rng, count, span, teaching):
    for _ in range(180):
        cells = {(0, 0)}
        while len(cells) <= count:
            choices = set()
            for base in cells:
                for dr, dc in _NEIGHBORS:
                    candidate = (base[0] + dr, base[1] + dc)
                    if candidate in cells:
                        continue
                    proposed = cells | {candidate}
                    rows, cols = zip(*proposed)
                    if max(rows) - min(rows) < span and max(cols) - min(cols) < span:
                        choices.add(candidate)
            if not choices:
                break
            cells.add(tuple(sorted(choices))[int(rng.integers(len(choices)))])
        if len(cells) <= count:
            continue
        offsets = tuple(sorted(cells - {(0, 0)}))
        if teaching and len({_transform(offsets, op) for op in _TRANSFORMS}) != 8:
            continue
        return offsets
    raise RuntimeError("could not sample stencil")


def _dimension(rng):
    draw = float(rng.random())
    if draw < 0.07:
        return 13
    if draw < 0.13:
        return 30
    return int(rng.integers(10, 31))


def _place(rng, height, width, specs, edge_choice, interior):
    order = list(rng.permutation(len(specs)))
    if edge_choice is not None:
        order.remove(edge_choice[0])
        order.insert(0, edge_choice[0])
    anchors = [None] * len(specs)
    blocked = set()
    for index in order:
        offsets = specs[index]["footprint"]
        rows, cols = zip(*offsets)
        r0, r1 = -min(rows), height - 1 - max(rows)
        c0, c1 = -min(cols), width - 1 - max(cols)
        if interior:
            r0 += 1
            r1 -= 1
            c0 += 1
            c1 -= 1
        if r0 > r1 or c0 > c1:
            return None
        candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if edge_choice is not None and index == edge_choice[0]:
            side = edge_choice[1]
            candidates = [
                (r, c) for r, c in candidates
                if (side == 0 and r + min(rows) == 0)
                or (side == 1 and r + max(rows) == height - 1)
                or (side == 2 and c + min(cols) == 0)
                or (side == 3 and c + max(cols) == width - 1)
            ]
        if not candidates:
            return None
        chosen = None
        for candidate_index in rng.permutation(len(candidates)):
            r, c = candidates[int(candidate_index)]
            footprint = {(r + dr, c + dc) for dr, dc in offsets}
            if footprint.isdisjoint(blocked):
                chosen = (r, c, footprint)
                break
        if chosen is None:
            return None
        r, c, footprint = chosen
        anchors[index] = (r, c)
        for y, x in footprint:
            for dr, dc in _NEIGHBORS + ((0, 0),):
                blocked.add((y + dr, x + dc))
    return anchors


def _generate_pair(rng, regime, markers, operations, backgrounds, companion_pool,
                   active, background_index, teaching_marker=None, forbidden=None,
                   forbidden_total=None):
    for attempt in range(250):
        height, width = _dimension(rng), _dimension(rng)
        if len(active) <= 2 and float(rng.random()) < 0.09:
            height = width = 13
        if len(active) == 3:
            height = max(height, 17)
            width = max(width, 17)
        background = backgrounds[background_index % len(backgrounds)]
        companions = {marker: companion_pool[int(rng.integers(len(companion_pool)))]
                      for marker in active}
        offsets = {}
        for marker in active:
            teaching = marker == teaching_marker
            count = int(rng.integers(5, 8)) if teaching else int(rng.integers(2, 9))
            if min(height, width) <= 13:
                count = min(count, 5)
            span = int(rng.integers(3, 7))
            count = min(count, span * span - 1)
            offsets[marker] = _sample_offsets(rng, count, span, teaching)
        if forbidden is not None and any(offsets[m] in forbidden.get(m, set())
                                         for m in active):
            continue

        if regime == "source_train":
            counts = {marker: 1 for marker in active}
            extras = int(rng.integers(0, 4 - len(active)))
            for _ in range(extras):
                marker = active[int(rng.integers(len(active)))]
                counts[marker] += 1
        else:
            counts = {marker: int(rng.integers(1, 5)) for marker in active}
            while sum(counts.values()) > 7:
                reducible = [marker for marker in active if counts[marker] > 1]
                marker = reducible[int(rng.integers(len(reducible)))]
                counts[marker] -= 1
            if min(height, width) <= 13:
                counts = {marker: 1 for marker in active}
        if forbidden_total is not None and sum(counts.values()) == forbidden_total:
            continue

        specs = []
        for marker in active:
            specs.append({"marker": marker, "kind": "prototype",
                          "footprint": ((0, 0),) + offsets[marker]})
            transformed = _transform(offsets[marker], operations[marker])
            for _ in range(counts[marker]):
                specs.append({"marker": marker, "kind": "target",
                              "footprint": ((0, 0),) + transformed})
        edge_roll = float(rng.random())
        interior = edge_roll < 0.22
        edge_choice = None
        if edge_roll > 0.52:
            edge_choice = (int(rng.integers(len(specs))), int(rng.integers(4)))
        anchors = _place(rng, height, width, specs, edge_choice, interior)
        if anchors is None:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for spec, anchor in zip(specs, anchors):
            row, col = anchor
            marker = spec["marker"]
            input_grid[row, col] = marker
            output_grid[row, col] = marker
            for dr, dc in spec["footprint"]:
                if dr == dc == 0:
                    continue
                if spec["kind"] == "prototype":
                    input_grid[row + dr, col + dc] = companions[marker]
                output_grid[row + dr, col + dc] = companions[marker]
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not place a pair")


def _components(grid, background):
    seen = set()
    components = []
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) == background or (row, col) in seen:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            component = set()
            while stack:
                r, c = stack.pop()
                component.add((r, c))
                for dr, dc in _NEIGHBORS:
                    y, x = r + dr, c + dc
                    if (0 <= y < height and 0 <= x < width and
                            (y, x) not in seen and int(grid[y, x]) != background):
                        seen.add((y, x))
                        stack.append((y, x))
            components.append(component)
    return components


def _inspect_pair(pair):
    source, target = pair["input"], pair["output"]
    _require(source.shape == target.shape, "dimensions changed")
    background = int(np.argmax(np.bincount(source.ravel(), minlength=10)))
    _require(int(np.count_nonzero(source == background)) > source.size // 2,
             "background is not a distinguishable majority")
    _require(int(np.argmax(np.bincount(target.ravel(), minlength=10))) == background,
             "background changed")
    inputs = _components(source, background)
    outputs = _components(target, background)
    singleton = {int(source[next(iter(c))]): next(iter(c))
                 for c in inputs if len(c) == 1}
    # Multiple destinations of one color are recovered below, independently of this map.
    marker_set = set(singleton)
    anchors = {}
    prototypes = {}
    for component in inputs:
        if len(component) == 1:
            point = next(iter(component))
            anchors.setdefault(int(source[point]), []).append(point)
            continue
        marker_cells = [point for point in component if int(source[point]) in marker_set]
        _require(len(marker_cells) == 1, "prototype has no unique marker")
        anchor = marker_cells[0]
        marker = int(source[anchor])
        _require(marker not in prototypes, "duplicate prototype for marker")
        colors = {int(source[p]) for p in component - {anchor}}
        _require(len(colors) == 1, "prototype companion is not monochrome")
        color = next(iter(colors))
        _require(color not in marker_set and color != background,
                 "companion conflicts with marker or background")
        offsets = tuple(sorted((r - anchor[0], c - anchor[1])
                               for r, c in component - {anchor}))
        _require(len(offsets) >= 2, "stencil too small")
        prototypes[marker] = (anchor, color, offsets, component)
    _require(set(prototypes) == set(anchors) and bool(prototypes),
             "every marker needs one prototype and a destination")
    _require(np.all(target[source != background] == source[source != background]),
             "input content was changed")
    _require(len(inputs) == len(outputs), "component count changed")
    output_by_anchor = {}
    for component in outputs:
        owned = [point for point in component
                 if int(source[point]) in marker_set and source[point] != background]
        _require(len(owned) == 1, "output components merged or lost anchors")
        output_by_anchor[owned[0]] = component
    observations = {}
    for marker, (anchor, color, offsets, cells) in prototypes.items():
        _require(output_by_anchor.get(anchor) == cells, "prototype changed")
        finished = []
        for point in anchors[marker]:
            component = output_by_anchor.get(point)
            _require(component is not None, "destination missing")
            _require(all(int(target[p]) == color for p in component - {point}),
                     "destination uses wrong companion color")
            relative = tuple(sorted((r - point[0], c - point[1])
                                    for r, c in component - {point}))
            finished.append(relative)
        observations[marker] = (offsets, finished, color)
    _require(len(output_by_anchor) == sum(1 + len(v) for v in anchors.values()),
             "unexpected output component")
    return {"background": background, "observations": observations,
            "counts": {m: len(v) for m, v in anchors.items()},
            "edge_input": bool(np.any(source[0, :] != background) or
                               np.any(source[-1, :] != background) or
                               np.any(source[:, 0] != background) or
                               np.any(source[:, -1] != background)),
            "edge_output": bool(np.any(target[0, :] != background) or
                                np.any(target[-1, :] != background) or
                                np.any(target[:, 0] != background) or
                                np.any(target[:, -1] != background))}


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "illegal pair counts")
    seen = set()
    analyzed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= size <= 30 for size in grid.shape),
                         "grid dimension outside ARC bounds")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         bool(np.all((grid >= 0) & (grid <= 9))),
                         "grid has illegal dtype or color")
            source, target = pair["input"], pair["output"]
            _require(not np.array_equal(source, target), "degenerate output")
            key = _grid_key(source)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if source.shape == (13, 13):
                rows = tuple(''.join(str(int(x)) for x in row) for row in source)
                _require(rows not in _SOURCE_INPUTS, "source input copied")
            analysis = _inspect_pair(pair)
            total = sum(analysis["counts"].values())
            if split == "train":
                _require(1 <= len(analysis["counts"]) <= 3 and 1 <= total <= 3,
                         "training family or destination count invalid")
            else:
                _require(1 <= len(analysis["counts"]) <= 3 and 1 <= total <= 7,
                         "test family or destination count invalid")
            analyzed[split].append(analysis)

    candidates = {}
    train_markers = set()
    for analysis in analyzed["train"]:
        for marker, (offsets, finished, _) in analysis["observations"].items():
            train_markers.add(marker)
            options = {op for op in _TRANSFORMS
                       if all(_transform(offsets, op) == shape for shape in finished)}
            candidates[marker] = candidates.get(marker, set(_TRANSFORMS)) & options
    _require(all(len(ops) == 1 for ops in candidates.values()),
             "training does not distinguish anchored rotations and reflections")
    learned = {marker: next(iter(ops)) for marker, ops in candidates.items()}
    _require(set(learned.values()) == {"identity", "horizontal"},
             "training does not demonstrate both operations")
    train_stencils = {}
    for analysis in analyzed["train"]:
        for marker, (offsets, _, _) in analysis["observations"].items():
            train_stencils.setdefault(marker, set()).add(offsets)
    for analysis in analyzed["test"]:
        _require(set(analysis["observations"]).issubset(train_markers),
                 "test uses an unseen marker")
        for marker, (offsets, finished, _) in analysis["observations"].items():
            _require(offsets not in train_stencils[marker],
                     "test repeats a training stencil for its marker")
            _require(all(_transform(offsets, learned[marker]) == shape
                         for shape in finished), "test completion contradicts training")
    return None


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    nt, nv = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    family_count = 3 if nt >= 3 and float(rng.random()) < 0.27 else 2
    palette = [int(x) for x in rng.permutation(10)]
    markers = palette[:family_count]
    operations = {markers[0]: "identity", markers[1]: "horizontal"}
    if family_count == 3:
        operations[markers[2]] = ("identity", "horizontal")[int(rng.integers(2))]
    backgrounds = palette[family_count:family_count + 3]
    companion_pool = palette[family_count + 3:]
    pair_entropy = rng.integers(0, 2**32, size=(nt + nv, 4), dtype=np.uint32)
    puzzle = {"train": [], "test": []}

    training_modes = [(marker,) for marker in markers]
    while len(training_modes) < nt:
        count = int(rng.integers(1, min(family_count, 3) + 1))
        choice = tuple(markers[int(i)] for i in rng.choice(family_count, count, replace=False))
        training_modes.append(choice)
    rng.shuffle(training_modes)
    taught = set()
    for index, active in enumerate(training_modes):
        teaching_marker = next((m for m in active if m not in taught), None)
        if teaching_marker is not None:
            taught.add(teaching_marker)
        local = np.random.default_rng(np.random.SeedSequence([int(x) for x in pair_entropy[index]]))
        pair = _generate_pair(local, "source_train", markers, operations,
                              backgrounds, companion_pool, active, index,
                              teaching_marker)
        puzzle["train"].append(pair)
    for local_index in range(nv):
        index = nt + local_index
        local = np.random.default_rng(np.random.SeedSequence([int(x) for x in pair_entropy[index]]))
        choice = int(rng.integers(5))
        if choice == 0:
            active = (markers[0],)
        elif choice == 1:
            active = (markers[1],)
        elif choice == 2 and family_count == 3:
            active = tuple(markers)
        else:
            active = tuple(markers[:2])
        train_stencils = {}
        for train_pair in puzzle["train"]:
            for marker, (offsets, _, _) in _inspect_pair(train_pair)["observations"].items():
                train_stencils.setdefault(marker, set()).add(offsets)
        prior_counts = [sum(_inspect_pair(previous)["counts"].values())
                        for previous in puzzle["train"] + puzzle["test"]]
        forbidden_total = prior_counts[0] if (local_index == nv - 1 and
                                               len(set(prior_counts)) == 1) else None
        pair = _generate_pair(local, "target_test", markers, operations,
                              backgrounds, companion_pool, active, index,
                              forbidden=train_stencils,
                              forbidden_total=forbidden_total)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "duplicate puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    shapes, stencils, layouts, backgrounds, counts = set(), set(), set(), set(), set()
    edge_sides = set()
    train_edge_sides = set()
    test_edge_sides = set()
    diagonal = 0
    shared_companion = 0
    test_modes = set()
    operation_modes = set()
    edge_free_tests = 0
    smaller_tests = 0
    larger_tests = 0
    high_copy_tests = 0
    unequal_tests = 0
    mixed_size_pairs = 0
    symmetric_stencils = 0
    three_family_tests = 0
    combined_tests = 0
    exact_source_size = {"train": 0, "test": 0}
    varied_counts = 0
    varied_shapes = 0
    total = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_backgrounds = set()
        puzzle_counts = set()
        puzzle_shapes = set()
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        possible = {}
        for pair in puzzle["train"]:
            for marker, (offsets, finished, _) in _inspect_pair(pair)["observations"].items():
                options = {op for op in _TRANSFORMS
                           if all(_transform(offsets, op) == shape for shape in finished)}
                possible[marker] = possible.get(marker, set(_TRANSFORMS)) & options
        _require(all(len(options) == 1 for options in possible.values()),
                 "training failed to distinguish all eight anchored symmetries")
        learned = {marker: next(iter(options)) for marker, options in possible.items()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                grid = pair["input"]
                info = _inspect_pair(pair)
                shapes.add(grid.shape)
                backgrounds.add(info["background"])
                puzzle_backgrounds.add(info["background"])
                puzzle_counts.add(sum(info["counts"].values()))
                puzzle_shapes.add(grid.shape)
                if grid.shape == (13, 13):
                    exact_source_size[split] += 1
                counts.add((split, len(info["counts"]), sum(info["counts"].values())))
                layouts.add((grid.shape, tuple(map(tuple, np.argwhere(grid != info["background"])))))
                colors = []
                sizes_in_pair = []
                diagonal_only = False
                for marker, (offsets, _, color) in info["observations"].items():
                    stencils.add(offsets)
                    colors.append(color)
                    sizes_in_pair.append(len(offsets))
                    if any(abs(r) == 1 and abs(c) == 1 for r, c in offsets):
                        diagonal += 1
                    points = {(0, 0)} | set(offsets)
                    if any(
                        not any((r + dr, c + dc) in points
                                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                        and any((r + dr, c + dc) in points
                                for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
                        for r, c in offsets
                    ):
                        diagonal_only = True
                    if len({_transform(offsets, op) for op in _TRANSFORMS}) < 8:
                        symmetric_stencils += 1
                if len(sizes_in_pair) > 1 and max(sizes_in_pair) - min(sizes_in_pair) >= 3:
                    mixed_size_pairs += 1
                if len(colors) > len(set(colors)):
                    shared_companion += 1
                if split == "test":
                    test_modes.add(len(info["counts"]))
                    operation_modes.add(tuple(sorted({learned[m] for m in info["counts"]})))
                    if grid.size < min(train_areas):
                        smaller_tests += 1
                    if grid.size > max(train_areas):
                        larger_tests += 1
                    if sum(info["counts"].values()) >= 4:
                        high_copy_tests += 1
                    if len(set(info["counts"].values())) > 1:
                        unequal_tests += 1
                    if len(info["counts"]) == 3:
                        three_family_tests += 1
                out = pair["output"]
                bg = info["background"]
                touches_edge = False
                for side, edge in enumerate((out[0], out[-1], out[:, 0], out[:, -1])):
                    if np.any(edge != bg):
                        edge_sides.add(side)
                        (train_edge_sides if split == "train" else test_edge_sides).add(side)
                        touches_edge = True
                if split == "test" and not touches_edge:
                    edge_free_tests += 1
                if (split == "test" and len(info["counts"]) == 3 and
                        len(set(info["counts"].values())) > 1 and
                        diagonal_only and touches_edge):
                    combined_tests += 1
        _require(len(puzzle_backgrounds) >= 2, "pair backgrounds did not vary")
        varied_counts += len(puzzle_counts) > 1
        varied_shapes += len(puzzle_shapes) > 1
    _require(len(shapes) >= 35 and len(stencils) >= 90 and
             len(layouts) >= total * 0.9, "structural diversity is too narrow")
    _require(13 in {size for shape in shapes for size in shape} and
             30 in {size for shape in shapes for size in shape}, "dimension extremes missing")
    _require(len(backgrounds) >= 5 and diagonal >= 20 and shared_companion >= 1,
             "color or diagonal coverage missing")
    _require(len(edge_sides) == 4 and {1, 2}.issubset(test_modes),
             "edge or family-count coverage missing")
    _require(len(counts) >= 10, "destination counts lack variation")
    _require(len(train_edge_sides) == len(test_edge_sides) == 4 and edge_free_tests >= 10,
             "edge regimes lack variation")
    _require({("identity",), ("horizontal",), ("horizontal", "identity")} <= operation_modes,
             "test operation mixtures lack variation")
    _require(smaller_tests >= 5 and larger_tests >= 5 and high_copy_tests >= 10,
             "test size or copy-count regimes lack variation")
    _require(unequal_tests >= 10 and three_family_tests >= 1 and mixed_size_pairs >= 10,
             "family, destination, or stencil-size combinations lack variation")
    _require(symmetric_stencils >= 10 and varied_counts == 42 and varied_shapes >= 40,
             "symmetry or within-puzzle diversity is insufficient")
    _require(exact_source_size["train"] >= 5 and exact_source_size["test"] >= 2
             and combined_tests >= 1,
             "13-by-13 or combined-axis coverage missing")
    _require(_signature(generate_puzzle(7001, 6, 3)) ==
             _signature(generate_puzzle(7001, 6, 3)), "not deterministic")
    _require(_signature(generate_puzzle(7001, 6, 3)) !=
             _signature(generate_puzzle(7002, 6, 3)), "seeds did not differ")
    for seed, nt, nv in ((8201, 2, 1), (8202, 10, 1), (8203, 10, 4),
                         (8301, None, 1), (8302, None, 4),
                         (8303, 2, None), (8304, 10, None)):
        puzzle = generate_puzzle(seed, nt, nv)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "requested train count ignored")
        if nv is not None:
            _require(len(puzzle["test"]) == nv, "requested test count ignored")
    for nt in range(2, 11):
        for nv in range(1, min(4, nt - 1) + 1):
            _require(len(generate_puzzle(9000 + nt * 5 + nv, nt, nv)["test"]) == nv,
                     "legal explicit count failed")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(10001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
