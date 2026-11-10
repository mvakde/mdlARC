"""Compositional slice-projection puzzles from ARC task 05a7bcf2."""

import hashlib

import numpy as np


SOURCE_PAIR_DIGESTS = frozenset({
    "9aea10d0e7cfecdcb3e6dda7f8169822d7790934652ecb7d2869826ec48c2762",
    "3fb1dacaececbe02863c11da8141a0b016088c577a37cf7ce2bee2f99112bfc5",
    "aa2ba14639d56982e30ba6188db6abd01e08f3c77a32cf4440bd8bddfa680879",
    "dff5632c57a0d67e903bcfa533dd4087c3dee7a4d2cace526b13da5c4361b05e",
})


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _component_count(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            count += 1
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                r, c = stack.pop()
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
    return count


def _analyze_input(grid):
    """Infer every geometric and input-color role directly from a grid."""
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "input must be a grid")
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) >= 4, "background, two lines, and sources are required")
    background = int(values[int(np.argmax(counts))])
    _require(int(counts.max()) > int(np.partition(counts, -2)[-2]),
             "background must be uniquely most frequent")
    candidates = []
    for orientation in ("vertical", "horizontal"):
        canonical = grid if orientation == "vertical" else grid.T
        cross_length, axis_length = canonical.shape
        full = {}
        for axis in range(axis_length):
            color = int(canonical[0, axis])
            if color != background and bool(np.all(canonical[:, axis] == color)):
                full.setdefault(color, []).append(axis)
        if len(full) != 2:
            continue
        for divider, profile in ((a, b) for a in full for b in full if a != b):
            if len(full[divider]) != 1:
                continue
            d = full[divider][0]
            source_colors = set(int(v) for v in values) - {background, divider, profile}
            if not source_colors or len(source_colors) > 6:
                continue
            source_mask = np.isin(canonical, list(source_colors))
            source_axes = np.argwhere(source_mask)[:, 1]
            if not len(source_axes):
                continue
            profile_axes = full[profile]
            if min(profile_axes) > d:
                direction = 1
                spine = max(profile_axes)
                if not bool(np.all(source_axes <= d - 2)):
                    continue
            elif max(profile_axes) < d:
                direction = -1
                spine = min(profile_axes)
                if not bool(np.all(source_axes >= d + 2)):
                    continue
            else:
                continue
            if int(np.count_nonzero(canonical == divider)) != cross_length:
                continue
            lengths = np.zeros(cross_length, dtype=np.int64)
            active = np.zeros(cross_length, dtype=bool)
            source_at = np.zeros(cross_length, dtype=np.int64)
            valid = True
            for cross in range(cross_length):
                positions = np.flatnonzero(canonical[cross] == profile)
                if not len(positions) or bool(np.any(np.diff(positions) != 1)):
                    valid = False
                    break
                if direction == 1:
                    if positions[-1] != spine or positions[0] <= d + 1:
                        valid = False
                        break
                elif positions[0] != spine or positions[-1] >= d - 1:
                    valid = False
                    break
                lengths[cross] = len(positions)
                cells = np.flatnonzero(source_mask[cross])
                if len(cells):
                    if bool(np.any(np.diff(cells) != 1)):
                        valid = False
                        break
                    colors_here = set(int(v) for v in canonical[cross, cells])
                    if len(colors_here) != 1:
                        valid = False
                        break
                    active[cross] = True
                    source_at[cross] = colors_here.pop()
            if valid and bool(np.any(active)) and bool(np.any(~active)):
                candidates.append({
                    "orientation": orientation, "background": background,
                    "divider": divider, "profile": profile,
                    "source_colors": source_colors, "source_at": source_at,
                    "divider_position": d, "spine_position": spine,
                    "direction": direction, "axis_length": axis_length,
                    "cross_length": cross_length, "profile_lengths": lengths,
                    "active": active, "source_mask": source_mask,
                })
    _require(len(candidates) == 1, "input geometry or semantic roles are ambiguous")
    return candidates[0]


def _construct_output(input_grid, info, residue):
    _require(0 <= residue <= 9 and residue not in set(int(v) for v in np.unique(input_grid)),
             "residue must be a new ARC color")
    output = input_grid.copy()
    source = input_grid if info["orientation"] == "vertical" else input_grid.T
    target = output if info["orientation"] == "vertical" else output.T
    d = info["divider_position"]
    axis_length = info["axis_length"]
    for cross in np.flatnonzero(info["active"]):
        cells = np.flatnonzero(info["source_mask"][cross])
        source_color = int(source[cross, cells[0]])
        target[cross, cells] = residue
        length = int(info["profile_lengths"][cross])
        if info["direction"] == 1:
            target[cross, cells[-1] + 1:d] = source_color
            target[cross, d:axis_length - length] = info["divider"]
            target[cross, axis_length - length:axis_length] = info["profile"]
        else:
            target[cross, d + 1:cells[0]] = source_color
            target[cross, 0:length] = info["profile"]
            target[cross, length:d + 1] = info["divider"]
    return output


def _cellwise_rule_holds(input_grid, output_grid, info, residue):
    """Check the transformation per cell without using the output builder."""
    before = input_grid if info["orientation"] == "vertical" else input_grid.T
    after = output_grid if info["orientation"] == "vertical" else output_grid.T
    d = info["divider_position"]
    for cross in range(info["cross_length"]):
        if not info["active"][cross]:
            if not np.array_equal(before[cross], after[cross]):
                return False
            continue
        source_cells = np.flatnonzero(info["source_mask"][cross])
        color = int(before[cross, source_cells[0]])
        depth = int(info["profile_lengths"][cross])
        for axis in range(info["axis_length"]):
            expected = int(before[cross, axis])
            if info["source_mask"][cross, axis]:
                expected = residue
            elif info["direction"] == 1:
                if source_cells[-1] < axis < d:
                    expected = color
                elif d <= axis < info["axis_length"] - depth:
                    expected = info["divider"]
                elif axis >= info["axis_length"] - depth:
                    expected = info["profile"]
            else:
                if d < axis < source_cells[0]:
                    expected = color
                elif axis < depth:
                    expected = info["profile"]
                elif depth <= axis <= d:
                    expected = info["divider"]
            if int(after[cross, axis]) != expected:
                return False
    return True


def _sample_bands(rng, cross_length, count, informative):
    widths = [3 if informative and i == 0 else 1 for i in range(count)]
    minimum_gaps = count - 1
    reserve_inactive = int(count == 1)
    available = cross_length - sum(widths) - minimum_gaps - reserve_inactive
    _require(available >= 0, "cross axis cannot fit source bands")
    budget = int(rng.integers(0, min(available, 15) + 1))
    for _ in range(budget):
        choices = [i for i, width in enumerate(widths) if width < 8]
        if not choices:
            break
        widths[int(rng.choice(choices))] += 1
    slack = cross_length - sum(widths) - minimum_gaps
    gaps = rng.multinomial(slack, np.full(count + 1, 1 / (count + 1))).astype(int)
    if count > 1:
        gaps[1:count] += 1
    if count == 1 and gaps[0] + gaps[1] == 0:
        gaps[1] = 1
    bands = []
    pos = int(gaps[0])
    for index, width in enumerate(widths):
        bands.append((pos, pos + width))
        pos += width + int(gaps[index + 1])
    _require(pos == cross_length, "band construction lost cells")
    return bands


def _profile_lengths(rng, cross_length, maximum, active, first_band, informative):
    mode = str(rng.choice(("jagged", "plateau", "ramp", "steps", "repeat", "uniform")))
    if mode == "uniform":
        lengths = np.full(cross_length, int(rng.integers(1, maximum + 1)), dtype=int)
    elif mode == "ramp":
        period = int(rng.integers(3, min(cross_length, 10) + 1))
        lengths = 1 + np.floor((np.arange(cross_length) % period) * maximum / period).astype(int)
    elif mode == "steps":
        lengths = np.repeat(np.arange(1, maximum + 1), 2)
        lengths = np.resize(lengths, cross_length).astype(int)
    elif mode == "repeat":
        motif = rng.integers(1, maximum + 1, size=int(rng.integers(2, 6)))
        lengths = np.resize(motif, cross_length).astype(int)
    elif mode == "plateau":
        lengths = np.zeros(cross_length, dtype=int)
        pos = 0
        while pos < cross_length:
            width = int(rng.integers(2, 7))
            lengths[pos:pos + width] = int(rng.integers(1, maximum + 1))
            pos += width
    else:
        lengths = rng.integers(1, maximum + 1, size=cross_length).astype(int)
    if informative:
        start, stop = first_band
        _require(stop - start >= 3 and maximum >= 3, "informative band needs three depths")
        lengths[start:start + 3] = [1, maximum, 2]
        inactive = np.flatnonzero(~active)
        lengths[int(rng.choice(inactive))] = maximum
    elif maximum >= 2 and mode != "uniform" and rng.random() < 0.3:
        lengths[active] = np.minimum(lengths[active], maximum - 1)
        lengths[int(rng.choice(np.flatnonzero(~active)))] = maximum
    return lengths


def _paint_object(rng, canonical, band, divider, color, boundary, informative):
    start, stop = band
    width = stop - start
    styles = ("flat", "rectangle", "step", "slant", "bent", "irregular", "thin_bar")
    style = "step" if informative else str(rng.choice(styles))
    if boundary:
        base_face = int(rng.integers(0, min(6, divider - 2) + 1))
    elif informative:
        base_face = int(rng.integers(3, divider - 1))
    else:
        base_face = int(rng.integers(1, divider - 1))
    base_thickness = 1 if style == "thin_bar" else int(rng.integers(1, min(6, base_face + 1) + 1))
    for offset, cross in enumerate(range(start, stop)):
        if informative:
            shift = 0 if offset % 2 == 0 else -1
        elif style == "slant":
            shift = min(offset, 2) - 1
        elif style == "step":
            shift = (offset // 2) % 3 - 1
        elif style == "bent":
            shift = min(offset, width - offset - 1, 2) - 1
        elif style == "irregular":
            shift = int(rng.integers(-1, 2))
        else:
            shift = 0
        face = max(0 if boundary else 1, min(divider - 2, base_face + shift))
        if boundary:
            far = 0
        else:
            thickness = min(base_thickness, face)
            if style in ("flat", "bent", "irregular") and not informative:
                thickness = max(1, min(face, thickness + int(rng.integers(-1, 2))))
            if style in ("slant", "step") and width > 1:
                thickness = max(2, min(face, thickness)) if face >= 2 else 1
            far = face - thickness + 1
        canonical[cross, far:face + 1] = color


def _pair_colors(rng, params):
    if params["palette_mode"] == "fixed":
        return params["fixed_colors"]
    colors = [int(x) for x in rng.permutation([c for c in range(10) if c != params["residue"]])]
    return colors[:3], colors[3:]


def _generate_pair(rng, params, direction_name, informative):
    for _ in range(200):
        full_size = bool(rng.random() < 0.06)
        compact = not informative and not full_size and rng.random() < 0.26
        axis_length = 30 if full_size else int(
            rng.integers(12 if compact else 18, 19 if compact else 31))
        cross_length = 30 if full_size else int(
            rng.integers(8 if compact else 11, 20 if compact else 31))
        if informative:
            axis_length = max(18, axis_length)
            cross_length = max(12, cross_length)
        divider = int(rng.integers(6 if informative else 4, axis_length - 4))
        spine = int(rng.integers(divider + 3, axis_length))
        maximum_fit = min(8, spine - divider - 1, axis_length - divider - 2)
        if maximum_fit < (3 if informative else 1):
            continue
        maximum = int(rng.integers(3 if informative else 1, maximum_fit + 1))
        count_cap = min(6, (cross_length - 1) // 2 if informative
                        else (cross_length + 1) // 2)
        count = int(rng.integers(2 if informative else 1, count_cap + 1))
        bands = _sample_bands(rng, cross_length, count, informative)
        active = np.zeros(cross_length, dtype=bool)
        for start, stop in bands:
            active[start:stop] = True
        if not bool(np.any(~active)):
            continue
        roles, source_choices = _pair_colors(rng, params)
        background, divider_color, profile_color = roles
        if params["multicolor"] and count >= 2:
            colors_used = int(rng.integers(2, min(3, count) + 1))
        else:
            colors_used = 1
        selected = [int(v) for v in source_choices[:colors_used]]
        object_colors = [selected[i % colors_used] for i in range(count)]
        rng.shuffle(object_colors)
        canonical = np.full((cross_length, axis_length), background, dtype=np.int8)
        canonical[:, divider] = divider_color
        for index, band in enumerate(bands):
            boundary = bool(rng.random() < 0.28)
            _paint_object(rng, canonical, band, divider, object_colors[index],
                          boundary, informative and index == 0)
        source_mask = np.isin(canonical, selected)
        if _component_count(source_mask) != count:
            continue
        lengths = _profile_lengths(rng, cross_length, maximum, active, bands[0], informative)
        for cross, length in enumerate(lengths):
            canonical[cross, spine - int(length) + 1:spine + 1] = profile_color
        negative = direction_name in ("left", "up")
        horizontal = direction_name in ("down", "up")
        if negative:
            canonical = canonical[:, ::-1].copy()
        input_grid = canonical.T.copy() if horizontal else canonical
        try:
            info = _analyze_input(input_grid)
            output_grid = _construct_output(input_grid, info, params["residue"])
        except AssertionError:
            continue
        return input_grid, output_grid
    raise ValueError("could not fit a valid source and profile layout")


def _pair_key(grid):
    return (grid.shape, grid.tobytes())


def _source_digest(input_grid, output_grid):
    return hashlib.sha256(bytes(input_grid.shape) + input_grid.astype(np.int8).tobytes()
                          + output_grid.astype(np.int8).tobytes()).hexdigest()


def _residue_from_training(train):
    colors = []
    for pair in train:
        input_colors = set(int(v) for v in np.unique(pair["input"]))
        output_colors = set(int(v) for v in np.unique(pair["output"]))
        novel = output_colors - input_colors
        _require(len(novel) == 1, "every training pair must expose one residue")
        colors.append(novel.pop())
    _require(len(set(colors)) == 1, "training residue convention is inconsistent")
    return colors[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    records = []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional NumPy arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(all(1 <= size <= 30 for size in grid.shape), "illegal grid size")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal ARC color")
            _require(input_grid.shape == output_grid.shape, "dimensions must be preserved")
            key = _pair_key(input_grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_source_digest(input_grid, output_grid) not in SOURCE_PAIR_DIGESTS,
                     "source pair was copied")
            info = _analyze_input(input_grid)
            input_colors = set(int(v) for v in np.unique(input_grid))
            output_colors = set(int(v) for v in np.unique(output_grid))
            novel = output_colors - input_colors
            _require(len(novel) == 1 and output_colors == input_colors | novel,
                     "one output-only residue is required")
            residue = novel.pop()
            _require(np.array_equal(output_grid, _construct_output(input_grid, info, residue)),
                     "output violates the slice projection")
            _require(_cellwise_rule_holds(input_grid, output_grid, info, residue),
                     "cellwise source footprint or extrusion check failed")
            _require(not np.array_equal(input_grid, output_grid), "pair must activate the rule")
            records.append((split, pair, info, residue))
    learned_residue = _residue_from_training(train)
    _require(all(residue == learned_residue for _, _, _, residue in records),
             "test residue differs from the training convention")
    informative = False
    inactive_match = False
    for split, pair, info, _ in records:
        if split != "train":
            continue
        lengths = info["profile_lengths"]
        active = info["active"]
        if set(int(v) for v in lengths[active]) & set(int(v) for v in lengths[~active]):
            inactive_match = True
        source_mask = info["source_mask"]
        canonical = source_mask if info["orientation"] == "vertical" else source_mask.T
        for start in np.flatnonzero(active):
            if start > 0 and active[start - 1]:
                continue
            stop = start
            while stop < len(active) and active[stop]:
                stop += 1
            if stop - start >= 3 and len(set(int(v) for v in lengths[start:stop])) >= 3:
                informative = True
        _require(bool(np.any(canonical)), "train source is missing")
    _require(informative, "training must show contrasting local profile lengths")
    _require(inactive_match, "training must show an inactive matching profile length")
    for split, pair, info, _ in records:
        if split == "test":
            prediction = _construct_output(pair["input"], _analyze_input(pair["input"]),
                                           learned_residue)
            _require(np.array_equal(prediction, pair["output"]),
                     "test is not predicted from training residue and test input")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    residue = int(rng.integers(0, 10))
    available_colors = [c for c in range(10) if c != residue]
    fixed = [int(v) for v in rng.permutation(available_colors)]
    params = {
        "residue": residue,
        "palette_mode": "fixed" if rng.random() < 0.35 else "varied",
        "fixed_colors": (fixed[:3], fixed[3:]),
        "multicolor": bool(rng.random() < 0.38),
    }
    directions = ("right", "left", "down", "up")
    schedule = str(rng.choice(("mixed", "mixed", "common", "heldout", "original")))
    if schedule == "common":
        direction = str(rng.choice(directions))
        train_directions = [direction] * train_count
        test_directions = [direction] * test_count
    elif schedule == "heldout":
        heldout = str(rng.choice(directions))
        other = [d for d in directions if d != heldout]
        train_directions = [str(rng.choice(other)) for _ in range(train_count)]
        test_directions = [heldout] * test_count
    elif schedule == "original":
        train_directions = ["right" if i % 2 == 0 else "down" for i in range(train_count)]
        test_directions = ["left"] * test_count
    else:
        train_directions = [str(rng.choice(directions)) for _ in range(train_count)]
        test_directions = [str(rng.choice(directions)) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, sequence in (("train", train_directions), ("test", test_directions)):
        for index, direction in enumerate(sequence):
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                input_grid, output_grid = _generate_pair(
                    pair_rng, params, direction, split == "train" and index == 0)
                key = _pair_key(input_grid)
                if key not in seen and _source_digest(input_grid, output_grid) not in SOURCE_PAIR_DIGESTS:
                    seen.add(key)
                    puzzle[split].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise ValueError("could not generate distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[split], b[split])) for split in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(12345)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(24680, 10, 4)
    _require(_puzzles_equal(explicit, generate_puzzle(24680, 10, 4)),
             "fixed seed must reproduce the whole puzzle")
    _require(not _puzzles_equal(explicit, generate_puzzle(24681, 10, 4)),
             "neighboring seeds should differ")
    for seed, train, test in ((300, 2, 1), (301, 10, 1), (302, 10, 4),
                              (303, None, 1), (304, None, 4),
                              (305, 2, None), (306, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {kwargs}")
    train_counts, test_counts, dimensions, directions = set(), set(), set(), set()
    role_signatures, layouts = set(), set()
    flat, varied_palettes, stable_palettes = 0, 0, 0
    multicolor, depth_inactive, large, compact = 0, 0, 0, 0
    wide, long_bar, rectangle, interior_step = 0, 0, 0, 0
    common_direction, heldout_direction, shallower_test = 0, 0, 0
    boundary_counts, object_counts = set(), {"train": set(), "test": set()}
    directions_by_split = {"train": set(), "test": set()}
    all_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_roles = set()
        puzzle_directions = {"train": set(), "test": set()}
        puzzle_depths = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                all_inputs.add(_pair_key(grid))
                dimensions.add(grid.shape)
                direction = ("right" if info["direction"] == 1 else "left") if info["orientation"] == "vertical" else ("down" if info["direction"] == 1 else "up")
                directions.add(direction)
                directions_by_split[split].add(direction)
                puzzle_directions[split].add(direction)
                roles = (info["background"], info["divider"], info["profile"],
                         tuple(sorted(info["source_colors"])))
                pair_roles.add(roles)
                role_signatures.add(roles)
                layouts.add((grid.shape, direction, info["divider_position"],
                             info["spine_position"], _component_count(info["source_mask"]),
                             tuple(int(v) for v in info["profile_lengths"])))
                flat += int(len(set(int(v) for v in info["profile_lengths"])) == 1)
                multicolor += int(len(info["source_colors"]) > 1)
                depth_inactive += int(int(info["profile_lengths"].max()) >
                                      int(info["profile_lengths"][info["active"]].max()))
                large += int(grid.shape == (30, 30))
                compact += int(min(grid.shape) <= 17)
                puzzle_depths[split].append(int(info["profile_lengths"].max()))
                normalized = grid if info["orientation"] == "vertical" else grid.T
                if info["direction"] == -1:
                    normalized = normalized[:, ::-1]
                mask = np.isin(normalized, list(info["source_colors"]))
                active = info["active"]
                boundaries = 0
                count = 0
                cross = 0
                while cross < len(active):
                    if not active[cross]:
                        cross += 1
                        continue
                    start = cross
                    while cross < len(active) and active[cross]:
                        cross += 1
                    count += 1
                    far_edges, facing_edges, thicknesses = [], [], []
                    for row in range(start, cross):
                        cells = np.flatnonzero(mask[row])
                        far_edges.append(int(cells[0]))
                        facing_edges.append(int(cells[-1]))
                        thicknesses.append(len(cells))
                    boundaries += int(min(far_edges) == 0)
                    wide += int(cross - start >= 5)
                    long_bar += int(cross - start >= 5 and max(thicknesses) == 1)
                    rectangle += int(cross - start >= 3 and min(thicknesses) >= 3
                                     and len(set(far_edges)) == len(set(facing_edges)) == 1)
                    interior_step += int(min(far_edges) > 0 and len(set(facing_edges)) > 1)
                object_counts[split].add(count)
                boundary_counts.add(min(boundaries, 2))
                _require(count == _component_count(info["source_mask"]),
                         "source bands must remain separate connected objects")
        varied_palettes += int(len(pair_roles) > 1)
        stable_palettes += int(len(pair_roles) == 1)
        common_direction += int(len(puzzle_directions["train"] | puzzle_directions["test"]) == 1)
        heldout_direction += int(bool(puzzle_directions["test"] - puzzle_directions["train"]))
        shallower_test += int(min(puzzle_depths["test"]) < max(puzzle_depths["train"]))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack range")
    _require(len(dimensions) >= 45 and len(layouts) >= 150 and len(all_inputs) >= 150,
             "geometry lacks diversity")
    _require(len(role_signatures) >= 40 and varied_palettes >= 10,
             "palette variation is insufficient")
    _require(directions == {"right", "left", "down", "up"} and
             all(len(s) == 4 for s in directions_by_split.values()),
             "both splits must exercise all four directions")
    _require(flat > 0 and multicolor > 0 and depth_inactive > 0 and compact > 0 and large > 0,
             "profile, source-color, or size modes are missing")
    _require(stable_palettes > 0 and common_direction > 0 and heldout_direction > 0,
             "stable palettes or direction schedules are missing")
    _require(shallower_test > 0 and boundary_counts == {0, 1, 2},
             "test depth and boundary-contact schedules lack diversity")
    _require(object_counts["train"] == set(range(1, 7)) and
             object_counts["test"] == set(range(1, 7)),
             "both splits must contain one through six objects")
    _require(wide > 0 and long_bar > 0 and rectangle > 0 and interior_step > 0,
             "larger and variable-facing source shapes are missing")
