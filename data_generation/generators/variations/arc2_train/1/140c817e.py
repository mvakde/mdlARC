"""Generate crosses from isolated points with learned center and accent colors."""

from functools import lru_cache

import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    (9, 9, 9, 1, ((3, 3), (6, 6))),
    (13, 13, 8, 1, ((1, 11), (3, 2), (6, 9), (9, 5))),
    (11, 11, 7, 1, ((1, 8), (4, 2), (9, 6))),
    (14, 14, 8, 1, ((2, 7), (5, 2), (11, 4), (13, 13))),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _coerce_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _coerce_count("num_train", num_train, 2, 10)
    test = _coerce_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return (grid.shape, np.ascontiguousarray(grid, dtype=np.int64).tobytes())


def _source_signature(grid, background, marker_color, markers):
    return (grid.shape[0], grid.shape[1], background, marker_color, tuple(sorted(markers)))


def _clear_geometry(markers, shape):
    height, width = shape
    if len(markers) != len(set(markers)) or not markers:
        return False
    if any(not (0 <= r < height and 0 <= c < width) for r, c in markers):
        return False
    # A one-cell row or column offset would put an accent on another cross.
    return all(
        abs(r1 - r2) != 1 and abs(c1 - c2) != 1
        for i, (r1, c1) in enumerate(markers)
        for r2, c2 in markers[i + 1 :]
    )


def _accent_cells(markers, shape):
    height, width = shape
    return {
        (r + dr, c + dc)
        for r, c in markers
        for dr in (-1, 1)
        for dc in (-1, 1)
        if 0 <= r + dr < height and 0 <= c + dc < width
    }


def _render_output(shape, markers, background, line, center, accent):
    result = np.full(shape, background, dtype=np.int8)
    for row, column in markers:
        result[row, :] = line
        result[:, column] = line
    for row, column in markers:
        result[row, column] = center
    for row, column in _accent_cells(markers, shape):
        result[row, column] = accent
    return result


def _sample_axis(rng, size, required, target):
    """Draw a nonadjacent set of coordinates containing the required ones."""
    required = frozenset(required)
    allowed = frozenset(range(1, size - 1)) | required
    if any(abs(a - b) == 1 for a in required for b in required if a != b):
        return None

    @lru_cache(None)
    def ways(index, left, previous):
        if left < 0:
            return 0
        if index == size:
            return int(left == 0)
        if index in required:
            return 0 if previous else ways(index + 1, left - 1, True)
        skip = ways(index + 1, left, False)
        take = 0 if previous or index not in allowed else ways(index + 1, left - 1, True)
        return skip + take

    for wanted in range(min(target, size), len(required) - 1, -1):
        if not ways(0, wanted, False):
            continue
        chosen = []
        index, left, previous = 0, wanted, False
        while index < size:
            if index in required:
                take = True
            else:
                skip_ways = ways(index + 1, left, False)
                take_ways = (
                    0 if previous or index not in allowed
                    else ways(index + 1, left - 1, True)
                )
                take = bool(take_ways and int(rng.integers(skip_ways + take_ways)) >= skip_ways)
            if take:
                chosen.append(index)
                left -= 1
            previous = take
            index += 1
        return tuple(chosen)
    return None


def _boundary_markers(rng, shape, profile):
    height, width = shape
    if profile == "interior":
        return ()
    if profile == "edge":
        side = int(rng.integers(4))
        return (((0, int(rng.integers(1, width - 1))) if side == 0 else
                 (height - 1, int(rng.integers(1, width - 1))) if side == 1 else
                 (int(rng.integers(1, height - 1)), 0) if side == 2 else
                 (int(rng.integers(1, height - 1)), width - 1)),)
    if profile == "corner":
        return ((int(rng.choice((0, height - 1))), int(rng.choice((0, width - 1)))),)
    if profile == "multi":
        columns = [(a, b) for a in range(1, width - 1) for b in range(1, width - 1)
                   if abs(a - b) != 1]
        a, b = columns[int(rng.integers(len(columns)))]
        return ((0, a), (height - 1, b))
    if profile == "opposite":
        return (((0, 0), (height - 1, width - 1)) if int(rng.integers(2)) == 0
                else ((0, width - 1), (height - 1, 0)))
    raise ValueError("unknown boundary profile")


def _has_shared_axis(markers):
    return (len({r for r, _ in markers}) < len(markers)
            or len({c for _, c in markers}) < len(markers))


def _sample_markers(rng, shape, count, profile, mode):
    height, width = shape
    fixed = _boundary_markers(rng, shape, profile)
    if count < len(fixed):
        return None
    rows = _sample_axis(rng, height, (r for r, _ in fixed), count)
    columns = _sample_axis(rng, width, (c for _, c in fixed), count)
    if rows is None or columns is None:
        return None
    fixed_set = set(fixed)
    candidates = [
        (r, c) for r in rows for c in columns
        if (r, c) not in fixed_set and 0 < r < height - 1 and 0 < c < width - 1
    ]
    markers = list(fixed)
    if mode == "distinct":
        free_rows = [r for r in rows if r not in {p[0] for p in fixed}]
        free_columns = [c for c in columns if c not in {p[1] for p in fixed}]
        need = count - len(fixed)
        if len(free_rows) < need or len(free_columns) < need:
            return None
        rng.shuffle(free_rows)
        rng.shuffle(free_columns)
        markers.extend(zip(free_rows[:need], free_columns[:need]))
    else:
        if mode == "shared" and not _has_shared_axis(markers):
            all_positions = list(fixed) + candidates
            choices = [
                (a, b) for i, a in enumerate(all_positions)
                for b in all_positions[i + 1 :]
                if (a[0] == b[0] or a[1] == b[1])
                and len(fixed_set | {a, b}) <= count
            ]
            if not choices:
                return None
            a, b = choices[int(rng.integers(len(choices)))]
            markers = list(fixed_set | {a, b})
        remaining = [p for p in candidates if p not in markers]
        need = count - len(markers)
        if len(remaining) < need:
            return None
        if need:
            indices = rng.choice(len(remaining), size=need, replace=False)
            markers.extend(remaining[int(i)] for i in indices)
    markers = tuple(sorted(markers))
    if len(markers) != count or not _clear_geometry(markers, shape):
        return None
    if mode == "shared" and not _has_shared_axis(markers):
        return None
    if mode == "distinct" and _has_shared_axis(markers):
        return None
    return markers


def _sample_shape(rng, seed, index, section):
    regime = seed % 3
    if regime == 0 and section == "train":
        low, high = 6, 16
    elif regime == 0 and section == "test":
        low, high = 16, 30
    elif regime == 2 and section == "test":
        low, high = 6, 20
    else:
        low, high = 6, 30
    if section == "test" and index == 0 and seed % 17 == 0:
        return 30, 30
    roll = rng.random()
    if roll < 0.07 and high == 30:
        return 30, 30
    height = int(rng.integers(low, high + 1))
    width = height if roll < 0.23 else int(rng.integers(low, high + 1))
    return height, width


def _profile(seed, index, section, rng):
    regime = seed % 3
    if section == "train":
        if index == 1 and seed % 4 != 0:
            return "interior"
        if regime == 0:
            return "interior"
        if index == 0:
            return ("edge", "corner", "multi", "opposite")[(seed // 3) % 4]
    elif regime == 2 and index == 0:
        return "interior"
    elif regime == 0 and index == 0:
        return ("edge", "corner", "multi", "opposite")[(seed // 3) % 4]
    return ("interior", "edge", "corner", "multi", "opposite")[int(rng.integers(5))]


def _marker_count(rng, shape, minimum, mode, seed, index, section):
    height, width = shape
    cap = min(15, max(1, height * width // 20))
    if mode == "distinct":
        cap = min(cap, (height - 2 + 1) // 2, (width - 2 + 1) // 2)
    cap = max(minimum, cap)
    if section == "test" and index == 0 and seed % 17 == 0:
        return 15
    if cap >= 10 and rng.random() < 0.25:
        return int(rng.integers(max(minimum, 8), cap + 1))
    return int(rng.integers(minimum, cap + 1))


def _build_pair(rng, shape, count, profile, mode, background, line, center, accent):
    for _ in range(40):
        markers = _sample_markers(rng, shape, count, profile, mode)
        if markers is None:
            continue
        input_grid = np.full(shape, background, dtype=np.int8)
        for row, column in markers:
            input_grid[row, column] = line
        if _source_signature(input_grid, background, line, markers) in _SOURCE_INPUT_SIGNATURES:
            continue
        output_grid = _render_output(shape, markers, background, line, center, accent)
        if len(np.unique(output_grid)) != 4:
            continue
        return {"input": input_grid, "output": output_grid}
    return None


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    center, accent = (int(v) for v in rng.choice(10, size=2, replace=False))
    palette = [v for v in range(10) if v not in (center, accent)]
    background_order = [int(v) for v in rng.permutation(palette)]
    line_order = [int(v) for v in rng.permutation(palette)]
    shared_puzzle = seed % 4 != 0
    pair_seeds = np.random.SeedSequence(seed).spawn(train_count + test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_backgrounds = set()
    for section, total, start in (("train", train_count, 0), ("test", test_count, train_count)):
        for index in range(total):
            pair_rng = np.random.default_rng(pair_seeds[start + index])
            profile = _profile(seed, index, section, pair_rng)
            if section == "test" and index == 0 and seed % 17 == 0:
                profile = "opposite"
            if section == "train" and index == 1 and shared_puzzle:
                profile, mode = "interior", "shared"
            elif section == "train" and index == 0:
                mode = "distinct"
            elif section == "test" and shared_puzzle and index == 0 and profile in ("interior", "edge"):
                mode = "shared"
            else:
                mode = "free" if shared_puzzle else "distinct"
            shape = _sample_shape(pair_rng, seed, index, section)
            minimum = max(len(_boundary_markers(pair_rng, shape, profile)), 2 if mode == "shared" else 1)
            if section == "train" and index == 0:
                minimum = max(minimum, 2)
            count = _marker_count(pair_rng, shape, minimum, mode, seed, index, section)
            background = background_order[(start + index) % len(background_order)]
            if section == "test" and seed % 2 and len(train_backgrounds) < len(palette):
                novel = [v for v in background_order if v not in train_backgrounds]
                background = novel[index % len(novel)]
            line = line_order[(3 * (start + index) + 1) % len(line_order)]
            if line == background:
                line = next(v for v in line_order if v != background)
            if section == "train":
                train_backgrounds.add(background)
            pair = None
            for _ in range(8):
                pair = _build_pair(pair_rng, shape, count, profile, mode,
                                   background, line, center, accent)
                if pair is not None and _grid_key(pair["input"]) not in seen:
                    break
                pair = None
            if pair is None:
                raise RuntimeError(f"could not construct pair for seed {seed}, {section} {index}")
            seen.add(_grid_key(pair["input"]))
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    input_grid, output = pair["input"], pair["output"]
    _require(input_grid.shape == output.shape, "output dimensions changed")
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) == 2, "input needs background and one marker color")
    bg_index = int(np.argmax(counts))
    _require(counts[bg_index] > counts[1 - bg_index], "background must be majority")
    background, line = int(values[bg_index]), int(values[1 - bg_index])
    markers = tuple(map(tuple, np.argwhere(input_grid == line).tolist()))
    _require(_clear_geometry(markers, input_grid.shape), "conflicting marker geometry")
    accents = _accent_cells(markers, input_grid.shape)
    _require(bool(accents), "no visible diagonal accent")
    rows, columns = {r for r, _ in markers}, {c for _, c in markers}
    _require(all(r not in rows and c not in columns for r, c in accents),
             "accent overlaps a cross")
    centers = {int(output[r, c]) for r, c in markers}
    accent_colors = {int(output[r, c]) for r, c in accents}
    _require(len(centers) == len(accent_colors) == 1, "inconsistent output roles")
    center, accent = centers.pop(), accent_colors.pop()
    _require(len({background, line, center, accent}) == 4, "semantic colors collide")
    expected = _render_output(input_grid.shape, markers, background, line, center, accent)
    _require(np.array_equal(output, expected), "cross construction is incorrect")
    _require(set(np.unique(output).tolist()) == {background, line, center, accent},
             "all four colors must be visible")
    _require(_source_signature(input_grid, background, line, markers)
             not in _SOURCE_INPUT_SIGNATURES, "source pair was copied")
    return {"shape": input_grid.shape, "background": background, "line": line,
            "center": center, "accent": accent, "markers": markers,
            "accents": accents, "shared": _has_shared_axis(markers)}


def _solve_using_training(train, test_input):
    examples = [_analyze_pair(pair) for pair in train]
    center = examples[0]["center"]
    accent = examples[0]["accent"]
    _require(all(a["center"] == center and a["accent"] == accent for a in examples),
             "training output palette is contradictory")
    values, counts = np.unique(test_input, return_counts=True)
    _require(len(values) == 2, "test input has no identifiable markers")
    bg_index = int(np.argmax(counts))
    background, line = int(values[bg_index]), int(values[1 - bg_index])
    markers = tuple(map(tuple, np.argwhere(test_input == line).tolist()))
    return _render_output(test_input.shape, markers, background, line, center, accent)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys must be train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "sections must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair schema is invalid")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape), "grid size out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid color out of range")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[section].append(_analyze_pair(pair))
    train = analyses["train"]
    all_pairs = train + analyses["test"]
    learned = (train[0]["center"], train[0]["accent"])
    _require(all((a["center"], a["accent"]) == learned for a in all_pairs),
             "output colors are not learned from training")
    _require(any(len(a["markers"]) >= 2 for a in train),
             "training does not demonstrate multiple crosses")
    _require(any(any(0 < r < a["shape"][0] - 1 and 0 < c < a["shape"][1] - 1
                     for r, c in a["markers"]) for a in train),
             "training lacks a complete diagonal pattern")
    if any(a["shared"] for a in analyses["test"]):
        _require(any(a["shared"] for a in train),
                 "test shares an axis without a training demonstration")
    for pair in puzzle["test"]:
        _require(np.array_equal(_solve_using_training(puzzle["train"], pair["input"]),
                                pair["output"]),
                 "test output cannot be reconstructed from training and input")


def _puzzle_fingerprint(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, marker_counts, layouts, palettes, fingerprints = set(), set(), set(), set(), set()
    profiles = {"train": set(), "test": set()}
    shared_train = shared_test = 0
    rectangles = square_30 = single_marker = fifteen_markers = 0
    shared_accents = mixed_boundary_rectangles = distinct_pairs = 0
    visible_accent_counts, corner_orientations = set(), set()
    interior_train_boundary_test = matched_boundary_splits = 0
    novel_test_backgrounds = novel_test_lines = interior_tests = 0
    equal_or_smaller_tests = larger_tests = 0
    all_backgrounds, all_lines, all_centers, all_accents = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        analyses = {s: [_analyze_pair(p) for p in puzzle[s]] for s in ("train", "test")}
        palettes.add((analyses["train"][0]["center"], analyses["train"][0]["accent"]))
        shared_train += any(a["shared"] for a in analyses["train"])
        shared_test += any(a["shared"] for a in analyses["test"])
        train_backgrounds = {a["background"] for a in analyses["train"]}
        train_lines = {a["line"] for a in analyses["train"]}
        _require(len(train_backgrounds) >= 2 and len(train_lines) >= 2,
                 "pair-local input colors did not vary in training")
        train_boundary = any(
            any(r in (0, a["shape"][0] - 1) or c in (0, a["shape"][1] - 1)
                for r, c in a["markers"]) for a in analyses["train"])
        test_boundary = any(
            any(r in (0, a["shape"][0] - 1) or c in (0, a["shape"][1] - 1)
                for r, c in a["markers"]) for a in analyses["test"])
        interior_train_boundary_test += not train_boundary and test_boundary
        matched_boundary_splits += train_boundary and test_boundary
        novel_test_backgrounds += any(
            a["background"] not in train_backgrounds for a in analyses["test"])
        novel_test_lines += any(
            a["line"] not in train_lines for a in analyses["test"])
        interior_tests += any(
            all(0 < r < a["shape"][0] - 1 and 0 < c < a["shape"][1] - 1
                for r, c in a["markers"]) for a in analyses["test"])
        largest_train_area = max(a["shape"][0] * a["shape"][1]
                                 for a in analyses["train"])
        equal_or_smaller_tests += any(
            a["shape"][0] * a["shape"][1] <= largest_train_area
            for a in analyses["test"])
        larger_tests += any(
            a["shape"][0] * a["shape"][1] > largest_train_area
            for a in analyses["test"])
        for section in ("train", "test"):
            for pair, a in zip(puzzle[section], analyses[section]):
                h, w = a["shape"]
                dimensions.add((h, w))
                marker_counts.add(len(a["markers"]))
                layouts.add((a["shape"], a["markers"]))
                rectangles += h != w
                square_30 += (h, w) == (30, 30)
                single_marker += len(a["markers"]) == 1
                fifteen_markers += len(a["markers"]) == 15
                distinct_pairs += not a["shared"]
                all_backgrounds.add(a["background"])
                all_lines.add(a["line"])
                all_centers.add(a["center"])
                all_accents.add(a["accent"])
                individual_accents = sum(
                    0 <= r + dr < h and 0 <= c + dc < w
                    for r, c in a["markers"]
                    for dr in (-1, 1) for dc in (-1, 1))
                shared_accents += individual_accents > len(a["accents"])
                for r, c in a["markers"]:
                    count = sum(0 <= r + dr < h and 0 <= c + dc < w
                                for dr in (-1, 1) for dc in (-1, 1))
                    visible_accent_counts.add(count)
                boundary = [p for p in a["markers"] if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
                corners = [p for p in boundary if p[0] in (0, h - 1) and p[1] in (0, w - 1)]
                corner_orientations.update(
                    (int(r == h - 1), int(c == w - 1)) for r, c in corners)
                mixed_boundary_rectangles += h != w and bool(boundary) and a["shared"]
                label = ("interior" if not boundary else "opposite" if len(corners) >= 2
                         else "multi" if len(boundary) >= 2 else "corner" if corners else "edge")
                profiles[section].add(label)
                if section == "test":
                    _require(np.array_equal(_solve_using_training(puzzle["train"], pair["input"]),
                                            pair["output"]), "learned prediction failed")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack variation")
    _require(len(fingerprints) >= 40 and len(layouts) >= 180, "layout diversity is insufficient")
    _require(len(dimensions) >= 80 and len(marker_counts) >= 10, "size or count diversity is insufficient")
    _require(len(palettes) >= 20, "learned palette diversity is insufficient")
    _require(shared_train >= 10 and shared_test >= 5, "shared-axis examples are too rare")
    _require(rectangles >= 150 and square_30 >= 3 and single_marker >= 10
             and fifteen_markers >= 3, "size, aspect, or density coverage is insufficient")
    _require(shared_accents >= 20 and mixed_boundary_rectangles >= 10
             and distinct_pairs >= 50, "compatible geometry combinations are insufficient")
    _require(visible_accent_counts == {1, 2, 4}
             and len(corner_orientations) == 4, "clipping coverage is insufficient")
    _require(interior_train_boundary_test >= 5 and matched_boundary_splits >= 5
             and interior_tests >= 5, "train/test boundary regimes lack variety")
    _require(novel_test_backgrounds >= 5 and novel_test_lines >= 5,
             "unseen pair-local test colors are too rare")
    _require(equal_or_smaller_tests >= 5 and larger_tests >= 5,
             "train/test size regimes lack variety")
    _require(all_backgrounds == all_lines == all_centers == all_accents == set(range(10)),
             "semantic color roles do not span the palette")
    _require(profiles["train"] == profiles["test"] ==
             {"interior", "edge", "corner", "multi", "opposite"},
             "boundary regimes lack coverage")
    for seed, nt, ns in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                         (1003, 7, 3), (2000, None, 1), (2001, None, 4),
                         (2002, 2, None), (2003, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        _require((nt is None or len(p["train"]) == nt) and
                 (ns is None or len(p["test"]) == ns), "requested counts were ignored")
    _require(_puzzle_fingerprint(generate_puzzle(3100)) ==
             _puzzle_fingerprint(generate_puzzle(3100)), "generation is not deterministic")
    _require(_puzzle_fingerprint(generate_puzzle(3101, 7, 3)) ==
             _puzzle_fingerprint(generate_puzzle(3101, 7, 3)), "explicit generation is not deterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}, {"num_train": 3.0},
                   {"num_test": True}):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts {kwargs}")
    try:
        generate_puzzle(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative seed")
