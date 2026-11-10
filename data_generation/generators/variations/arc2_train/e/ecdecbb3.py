"""Compositional generator for band-bounded signal projection puzzles."""

import numpy as np


_SOURCE_SIGNATURES = {
    ((13, 18), 0, 8, 2, "vertical", (3, 14), ((4, 8),)),
    ((13, 13), 0, 8, 2, "horizontal", (6,), ((2, 2), (10, 8))),
    ((17, 12), 0, 8, 2, "horizontal", (7, 13), ((2, 8), (10, 4))),
    ((17, 19), 0, 8, 2, "vertical", (4, 12),
     ((1, 15), (2, 1), (8, 8), (14, 16))),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _normalize_count("num_train", num_train)
    num_test = _normalize_count("num_test", num_test)
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _bounding_guides(perpendicular: int, guides: tuple[int, ...]) -> tuple[int, ...]:
    if perpendicular < guides[0]:
        return (guides[0],)
    if perpendicular > guides[-1]:
        return (guides[-1],)
    for left, right in zip(guides[:-1], guides[1:]):
        if left < perpendicular < right:
            return left, right
    raise ValueError("signal lies on a guide")


def _geometry(orientation, guides, source):
    """Return the exact ring/ray writes made by one isolated source."""
    row, column = source
    perpendicular = column if orientation == "vertical" else row
    writes = {}
    centers = []
    for guide in _bounding_guides(perpendicular, guides):
        center = (row, guide) if orientation == "vertical" else (guide, column)
        centers.append(center)
        for ring_row in range(center[0] - 1, center[0] + 2):
            for ring_column in range(center[1] - 1, center[1] + 2):
                writes[ring_row, ring_column] = "guide"
        writes[center] = "signal"
        step = 1 if guide > perpendicular else -1
        for value in range(perpendicular, guide - step, step):
            cell = (row, value) if orientation == "vertical" else (value, column)
            writes[cell] = "signal"
    return writes, tuple(centers)


def _safe_layout(shape, orientation, guides, signals):
    """Check footprints, including corridors on distinct nearby guides."""
    height, width = shape
    along_length = height if orientation == "vertical" else width
    perp_length = width if orientation == "vertical" else height
    if any(not 1 <= guide <= perp_length - 2 for guide in guides):
        return False
    if any(right - left < 4 for left, right in zip(guides[:-1], guides[1:])):
        return False
    occupied = set()
    for index, source in enumerate(signals):
        row, column = source
        along = row if orientation == "vertical" else column
        perpendicular = column if orientation == "vertical" else row
        if not 1 <= along <= along_length - 2:
            return False
        if perpendicular in guides:
            return False
        if any(abs(perpendicular - guide) < 2
               for guide in _bounding_guides(perpendicular, guides)):
            return False
        if any(max(abs(row - other[0]), abs(column - other[1])) <= 1
               for other in signals[:index]):
            return False
        writes, _ = _geometry(orientation, guides, source)
        if any(not (0 <= r < height and 0 <= c < width)
               for r, c in writes):
            return False
        if occupied.intersection(writes):
            return False
        occupied.update(writes)
    return True


def _apply_rule(input_grid, orientation, guides, signals, guide_color, signal_color):
    output = np.array(input_grid, copy=True)
    for source in signals:
        writes, _ = _geometry(orientation, guides, source)
        for cell, role in writes.items():
            output[cell] = guide_color if role == "guide" else signal_color
    return output


def _reference_transform(input_grid, orientation, guides, signals,
                         guide_color, signal_color):
    """Separate coordinate-walk implementation for validation."""
    result = input_grid.copy()
    for row, column in signals:
        p = column if orientation == "vertical" else row
        j = int(np.searchsorted(np.asarray(guides), p))
        targets = ([guides[0]] if j == 0 else
                   [guides[-1]] if j == len(guides) else
                   [guides[j - 1], guides[j]])
        for guide in targets:
            cr, cc = (row, guide) if orientation == "vertical" else (guide, column)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    result[cr + dr, cc + dc] = guide_color
            result[cr, cc] = signal_color
            ray = (range(p, guide - 1) if p < guide else
                   range(guide + 2, p + 1))
            for q in ray:
                cell = (row, q) if orientation == "vertical" else (q, column)
                result[cell] = signal_color
    return result


def _composition(rng, total, parts):
    if parts == 1:
        return [total]
    cuts = np.sort(rng.choice(total + parts - 1, size=parts - 1, replace=False))
    bounds = [-1] + [int(x) for x in cuts] + [total + parts - 1]
    return [bounds[i + 1] - bounds[i] - 1 for i in range(parts)]


def _sample_guides(rng, length, count, required_bands):
    left_margin = 2 if 0 in required_bands else 1
    right_margin = 2 if count in required_bands else 1
    slack = length - 1 - right_margin - left_margin - 4 * (count - 1)
    if slack < 0:
        raise ValueError("perpendicular dimension too small")
    edge_sides = []
    if left_margin == 1:
        edge_sides.append("left")
    if right_margin == 1:
        edge_sides.append("right")
    if slack and edge_sides and rng.random() < 0.27:
        side = edge_sides[int(rng.integers(len(edge_sides)))]
        extras = ([0] + _composition(rng, slack, count) if side == "left"
                  else _composition(rng, slack, count) + [0])
    else:
        extras = _composition(rng, slack, count + 1)
    guides = [left_margin + extras[0]]
    for index in range(1, count):
        guides.append(guides[-1] + 4 + extras[index])
    return tuple(guides)


def _band_positions(length, guides, band):
    if band == 0:
        return list(range(0, guides[0] - 1))
    if band == len(guides):
        return list(range(guides[-1] + 2, length))
    return list(range(guides[band - 1] + 2, guides[band] - 1))


def _place_signals(rng, shape, orientation, guides, bands, require_long):
    height, width = shape
    along_length = height if orientation == "vertical" else width
    perp_length = width if orientation == "vertical" else height
    sources = []
    occupied = set()
    long_found = False
    for band in bands:
        positions = _band_positions(perp_length, guides, band)
        if not positions:
            return None
        priorities = rng.permutation(len(positions))
        distances = [min(abs(p - g) for g in _bounding_guides(p, guides))
                     for p in positions]
        bias = rng.random()
        if bias < 0.24:
            priorities = sorted(priorities, key=lambda i: distances[i])
        elif bias < 0.48:
            priorities = sorted(priorities, key=lambda i: -distances[i])
        alongs = list(map(int, rng.permutation(np.arange(1, along_length - 1))))
        if sources and rng.random() < 0.36:
            preferred = int(sources[int(rng.integers(len(sources)))][
                0 if orientation == "vertical" else 1])
            alongs.sort(key=lambda a: (abs(a - preferred),
                                       int(rng.integers(0, 2))))
        chosen = None
        for position_index in priorities:
            p = positions[int(position_index)]
            for along in alongs:
                source = (along, p) if orientation == "vertical" else (p, along)
                if any(max(abs(source[0] - other[0]),
                           abs(source[1] - other[1])) <= 1 for other in sources):
                    continue
                writes, _ = _geometry(orientation, guides, source)
                if occupied.intersection(writes):
                    continue
                if any(not (0 <= r < height and 0 <= c < width)
                       for r, c in writes):
                    continue
                chosen = source
                occupied.update(writes)
                long_found |= any(abs(p - g) >= 4
                                  for g in _bounding_guides(p, guides))
                break
            if chosen is not None:
                break
        if chosen is None:
            return None
        sources.append(chosen)
    if require_long and not long_found:
        return None
    return tuple(sorted(sources))


def _construct_pair(rng, orientation, guide_count, required_bands,
                    signal_count, require_long, colors):
    for _ in range(100):
        minimum_perp = 4 * (guide_count - 1) + 5
        if guide_count == 1 and set(required_bands) == {0, 1}:
            minimum_perp = max(minimum_perp, 10)
        perp_length = int(rng.integers(minimum_perp, 31))
        along_low = max(9, 2 * signal_count + 3,
                        17 if signal_count >= 5 else 9)
        along_length = int(rng.integers(along_low, 31))
        shape = ((along_length, perp_length) if orientation == "vertical"
                 else (perp_length, along_length))
        try:
            guides = _sample_guides(rng, perp_length, guide_count,
                                    required_bands)
        except ValueError:
            continue
        eligible = [b for b in range(guide_count + 1)
                    if _band_positions(perp_length, guides, b)]
        if any(b not in eligible for b in required_bands):
            continue
        bands = list(required_bands)
        while len(bands) < signal_count:
            if (rng.random() < 0.44 and guide_count > 1 and
                    any(0 < b < guide_count for b in eligible)):
                options = [b for b in eligible if 0 < b < guide_count]
            else:
                options = eligible
            bands.append(int(options[int(rng.integers(len(options)))]))
        rng.shuffle(bands)
        sources = _place_signals(rng, shape, orientation, guides,
                                 bands, require_long)
        if sources is None or not _safe_layout(shape, orientation, guides, sources):
            continue
        background, guide_color, signal_color = colors
        input_grid = np.full(shape, background, dtype=np.int8)
        if orientation == "vertical":
            input_grid[:, list(guides)] = guide_color
        else:
            input_grid[list(guides), :] = guide_color
        for source in sources:
            input_grid[source] = signal_color
        output_grid = _apply_rule(input_grid, orientation, guides, sources,
                                  guide_color, signal_color)
        return input_grid, output_grid
    raise RuntimeError("could not place a conflict-free pair")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} is not an ndarray")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(all(1 <= side <= 30 for side in grid.shape),
             f"{label} dimensions are out of range")
    _require(np.issubdtype(grid.dtype, np.integer),
             f"{label} must have integer dtype")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
             f"{label} has an invalid color")


def _pair_analysis(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape,
             "input/output dimensions differ")
    colors, counts = np.unique(input_grid, return_counts=True)
    _require(len(colors) == 3, "input must have three role colors")
    _require(np.count_nonzero(counts == max(counts)) == 1,
             "background is not uniquely identifiable")
    background = int(colors[int(np.argmax(counts))])
    remaining = [int(c) for c in colors if c != background]
    candidates = []
    for color in remaining:
        vertical = tuple(int(i) for i in np.flatnonzero(
            np.all(input_grid == color, axis=0)))
        horizontal = tuple(int(i) for i in np.flatnonzero(
            np.all(input_grid == color, axis=1)))
        if vertical and not horizontal:
            candidates.append((color, "vertical", vertical))
        if horizontal and not vertical:
            candidates.append((color, "horizontal", horizontal))
    _require(len(candidates) == 1, "guide role is not unique")
    guide_color, orientation, guides = candidates[0]
    signal_color = next(c for c in remaining if c != guide_color)
    _require(1 <= len(guides) <= 4, "invalid guide count")
    signals = tuple(sorted((int(r), int(c)) for r, c in
                           np.argwhere(input_grid == signal_color)))
    _require(1 <= len(signals) <= 6, "invalid signal count")
    reconstructed = np.full(input_grid.shape, background, dtype=input_grid.dtype)
    if orientation == "vertical":
        reconstructed[:, list(guides)] = guide_color
    else:
        reconstructed[list(guides), :] = guide_color
    for source in signals:
        reconstructed[source] = signal_color
    _require(np.array_equal(reconstructed, input_grid),
             "malformed guide or noisy input")
    _require(_safe_layout(input_grid.shape, orientation, guides, signals),
             "overlapping, out-of-bounds, or ambiguous footprints")
    expected = _reference_transform(input_grid, orientation, guides, signals,
                                    guide_color, signal_color)
    _require(np.array_equal(expected, output_grid),
             "output violates band-bounded projection")
    _require(not np.array_equal(input_grid, output_grid), "unchanged pair")
    bands = tuple(int(np.searchsorted(np.asarray(guides),
                 c if orientation == "vertical" else r)) for r, c in signals)
    distances = tuple(abs((c if orientation == "vertical" else r) - g)
                      for r, c in signals
                      for g in _bounding_guides(
                          c if orientation == "vertical" else r, guides))
    return dict(shape=input_grid.shape, background=background,
                guide=guide_color, signal=signal_color,
                orientation=orientation, guides=guides, signals=signals,
                bands=bands, distances=distances)


def _source_signature(analysis):
    return (analysis["shape"], analysis["background"], analysis["guide"],
            analysis["signal"], analysis["orientation"],
            analysis["guides"], analysis["signals"])


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and
             isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and
             train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{split}[{index}] needs input and output")
            _validate_grid(pair["input"], f"{split}[{index}].input")
            _validate_grid(pair["output"], f"{split}[{index}].output")
            key = _grid_key(pair["input"])
            pair_key = key, _grid_key(pair["output"])
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate pair or input")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analysis = _pair_analysis(pair["input"], pair["output"])
            _require(_source_signature(analysis) not in _SOURCE_SIGNATURES,
                     "source pair copied")
            analyses[split].append(analysis)
    train, test = analyses["train"], analyses["test"]
    _require(len({(a["background"], a["guide"], a["signal"])
                  for a in train}) >= 2,
             "training must demonstrate pair-local color roles")
    _require(any(len(a["guides"]) >= 2 and
                 any(0 < b < len(a["guides"]) for b in a["bands"])
                 for a in train), "missing interior demonstration")
    _require(any(0 in a["bands"] for a in train) and
             any(len(a["guides"]) in a["bands"] for a in train),
             "missing both exterior directions")
    _require(any(max(a["distances"]) >= 4 for a in train),
             "training does not show a nontrivial ray")
    train_orientations = {a["orientation"] for a in train}
    _require(all(a["orientation"] in train_orientations for a in test),
             "test orientation is not demonstrated")
    for a in test:
        n = len(a["guides"])
        _require(any(len(t["guides"]) >= n for t in train),
                 "test guide count is not demonstrated")
        if n >= 3:
            _require(any(len(t["guides"]) >= n and
                         set(range(1, len(t["guides"]))).issubset(t["bands"])
                         for t in train),
                     "multi-guide interior bands are not demonstrated")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The projection rule is the only puzzle-wide convention. Colors are
    # visible local roles, so pairs can recolor independently.
    max_guides = int(rng.choice([2, 3, 4], p=[0.40, 0.38, 0.22]))
    first_orientation = "vertical" if rng.random() < 0.5 else "horizontal"
    same_orientation = rng.random() < 0.28
    second_orientation = (first_orientation if same_orientation or
                          rng.random() < 0.40 else
                          "horizontal" if first_orientation == "vertical"
                          else "vertical")
    specifications = [
        (first_orientation, max_guides, list(range(1, max_guides)),
         int(rng.integers(max_guides - 1, min(5, max_guides + 2) + 1)), False),
        (second_orientation, 1, [0, 1], int(rng.integers(2, 5)), True),
    ]
    rng.shuffle(specifications)
    demonstrated_orientations = {s[0] for s in specifications}
    for _ in range(train_count - 2):
        orientation = (first_orientation if same_orientation else
                       "vertical" if rng.random() < 0.5 else "horizontal")
        demonstrated_orientations.add(orientation)
        guides = int(rng.integers(1, max_guides + 1))
        signals = int(rng.integers(1, 6))
        required = []
        if guides > 1 and signals > 1 and rng.random() < 0.45:
            required.append(int(rng.integers(1, guides)))
        specifications.append((orientation, guides, required, signals, False))
    tests = []
    for _ in range(test_count):
        orientation = sorted(demonstrated_orientations)[
            int(rng.integers(len(demonstrated_orientations)))]
        guides = int(rng.integers(1, max_guides + 1))
        signals = int(rng.integers(1, 7))
        required = []
        if guides > 1 and signals >= 2 and rng.random() < 0.50:
            required = [int(rng.integers(1, guides)),
                        int(rng.choice([0, guides]))]
        tests.append((orientation, guides, required, signals, False))
    palette = list(map(int, rng.permutation(10)))
    all_specs = [("train", s) for s in specifications] + [("test", s) for s in tests]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=len(all_specs), dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_train_colors = None
    for index, (split, spec) in enumerate(all_specs):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        accepted = False
        for _ in range(120):
            colors = tuple(int(c) for c in pair_rng.choice(palette, 3,
                                                           replace=False))
            if split == "train" and index == 1 and colors == first_train_colors:
                continue
            try:
                input_grid, output_grid = _construct_pair(pair_rng, *spec,
                                                           colors=colors)
            except RuntimeError:
                continue
            key = _grid_key(input_grid)
            if key in seen_inputs:
                continue
            analysis = _pair_analysis(input_grid, output_grid)
            if _source_signature(analysis) in _SOURCE_SIGNATURES:
                continue
            if index == 0:
                first_train_colors = colors
            puzzle[split].append({"input": input_grid, "output": output_grid})
            seen_inputs.add(key)
            accepted = True
            break
        if not accepted:
            raise RuntimeError("could not build a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(len(left[split]) == len(right[split]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(left[split], right[split])
        for k in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in automatic}) >= 4,
             "train-count diversity missing")
    _require(len({len(p["test"]) for p in automatic}) >= 3,
             "test-count diversity missing")
    shapes, structures, roles = set(), set(), set()
    guide_counts, signal_counts, orientations = set(), set(), set()
    train_three, repeated_interior, repeated_exterior = 0, 0, 0
    train_repeat_interior, train_repeat_exterior = 0, 0
    horizontal_tests, sparse_tests, close_rows = 0, 0, 0
    tight_gaps, edge_guides, near_sources, long_sources = 0, 0, 0, 0
    same_orientation_puzzles, pair_recolors = 0, 0
    tests_not_denser = 0
    all_pairs = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        orientation_set = { _pair_analysis(p["input"], p["output"])["orientation"]
                            for p in puzzle["train"] + puzzle["test"] }
        same_orientation_puzzles += len(orientation_set) == 1
        training_signal_max = max(len(_pair_analysis(p["input"], p["output"])["signals"])
                                  for p in puzzle["train"])
        tests_not_denser += any(len(_pair_analysis(p["input"], p["output"])["signals"])
                                <= training_signal_max for p in puzzle["test"])
        train_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _pair_analysis(pair["input"], pair["output"])
                all_pairs += 1
                shapes.add(a["shape"])
                role = a["background"], a["guide"], a["signal"]
                roles.add(role)
                if split == "train":
                    train_roles.add(role)
                    train_three += len(a["signals"]) >= 3
                else:
                    horizontal_tests += a["orientation"] == "horizontal"
                    sparse_tests += len(a["signals"]) <= 2
                guide_counts.add(len(a["guides"]))
                signal_counts.add(len(a["signals"]))
                orientations.add(a["orientation"])
                structures.add((a["shape"], a["orientation"], a["guides"],
                                a["signals"]))
                repeated_interior += any(a["bands"].count(b) >= 2
                                         for b in range(1, len(a["guides"])))
                repeated_exterior += (a["bands"].count(0) >= 2 or
                                      a["bands"].count(len(a["guides"])) >= 2)
                if split == "train":
                    train_repeat_interior += any(a["bands"].count(b) >= 2
                                                 for b in range(1, len(a["guides"])))
                    train_repeat_exterior += (a["bands"].count(0) >= 2 or
                                              a["bands"].count(len(a["guides"])) >= 2)
                tight_gaps += any(y - x == 4 and
                                  any((c if a["orientation"] == "vertical" else r)
                                      == x + 2 for r, c in a["signals"])
                                  for x, y in zip(a["guides"][:-1],
                                                  a["guides"][1:]))
                perp = a["shape"][1 if a["orientation"] == "vertical" else 0]
                edge_guides += any(g == 1 or g == perp - 2
                                   for r, c in a["signals"]
                                   for g in _bounding_guides(
                                       c if a["orientation"] == "vertical" else r,
                                       a["guides"]))
                near_sources += 2 in a["distances"]
                long_sources += any(d >= 5 for d in a["distances"])
                alongs = [r if a["orientation"] == "vertical" else c
                          for r, c in a["signals"]]
                close_rows += any(abs(x - y) <= 2 for i, x in enumerate(alongs)
                                  for y in alongs[i + 1:])
        pair_recolors += len(train_roles) >= 2
    _require(len(structures) >= int(0.85 * all_pairs) and len(shapes) >= 45,
             "layout/size diversity missing")
    _require(len(roles) >= 30 and pair_recolors == 42,
             "pair-local color variation missing")
    _require(guide_counts == {1, 2, 3, 4} and
             {1, 2, 3, 4, 5, 6}.issubset(signal_counts) and
             orientations == {"vertical", "horizontal"},
             "count/orientation support missing")
    _require(min(train_three, repeated_interior, repeated_exterior,
                 train_repeat_interior, train_repeat_exterior,
                 horizontal_tests, sparse_tests, close_rows, tight_gaps,
                 edge_guides, near_sources, long_sources,
                 same_orientation_puzzles, tests_not_denser) >= 1,
             "review feature coverage missing")
    source_input = np.zeros((17, 19), dtype=np.int8)
    source_input[:, [4, 12]] = 8
    for source in ((1, 15), (2, 1), (8, 8), (14, 16)):
        source_input[source] = 2
    source_output = _apply_rule(source_input, "vertical", (4, 12),
                                ((1, 15), (2, 1), (8, 8), (14, 16)), 8, 2)
    _pair_analysis(source_input, source_output)
    three_gap_input = np.zeros((12, 11), dtype=np.int8)
    three_gap_input[:, 5] = 8
    three_gap_signals = ((4, 2), (7, 2))
    for source in three_gap_signals:
        three_gap_input[source] = 2
    _pair_analysis(three_gap_input,
                   _apply_rule(three_gap_input, "vertical", (5,),
                               three_gap_signals, 8, 2))
    _require(not _safe_layout((12, 11), "vertical", (5,),
                              ((4, 2), (6, 2))),
             "overlapping same-guide rings were accepted")
    _require(_puzzles_equal(generate_puzzle(314159, 6, 3),
                            generate_puzzle(314159, 6, 3)),
             "generation is nondeterministic")
    for seed, ntrain, ntest in ((7001, 2, 1), (7002, 10, 1),
                                (7003, 10, 4), (7004, 7, 3),
                                (8001, None, 1), (8002, None, 4),
                                (8003, 2, None), (8004, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            _require(len(puzzle["train"]) == ntrain, "num_train ignored")
        if ntest is not None:
            _require(len(puzzle["test"]) == ntest, "num_test ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3},
                   {"num_train": 4, "num_test": 4},
                   {"num_train": 2.5}, {"num_test": True}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")
