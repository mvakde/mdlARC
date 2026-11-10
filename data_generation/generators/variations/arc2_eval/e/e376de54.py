import numpy as np


_CANONICAL_UNITS = {
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "diag_down_right": (1, 1),
    "diag_down_left": (1, -1),
}


def _normalize_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_pair_counts(seed_word, num_train, num_test):
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed_word % 4
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed_word // 4) % number_of_choices
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + seed_word % number_of_choices
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists for num_train")
        test_count = 1 + seed_word % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return int(train_count), int(test_count)


def sample_puzzle_parameters(rng):
    """Sampling regimes are not hidden conventions: geometry supplies the rule."""
    level = int(rng.integers(3))
    regime = str(rng.choice(("common", "mixed", "transfer"), p=(.25, .55, .20)))
    directions = [int(x) for x in rng.permutation(8)]
    pool = tuple(int(x) for x in rng.choice(10, size=int(rng.integers(3, 10)), replace=False))
    return {"level": level, "regime": regime, "directions": directions, "pool": pool}


def _directed_axis(index):
    orientation = tuple(_CANONICAL_UNITS)[index // 2]
    dr, dc = _CANONICAL_UNITS[orientation]
    sign = 1 if index % 2 == 0 else -1
    return (dr * sign, dc * sign), (dc, -dr)


def _sample_gaps(rng, count, diagonal, equal, witness):
    # Each diagonal baseline step advances both row and column. Reserve room
    # for a visible reference and a longer stroke even for nine-stroke arrays.
    limit = 24 if diagonal else 28
    if witness:
        gaps = [2] * (count - 1)
        side = int(rng.integers(2))
        index = 0 if side == 0 else count - 2
        gaps[index] += int(rng.choice([extra for extra in (5, 7, 9) if extra <= limit - sum(gaps)]))
        return gaps
    if equal:
        gap = int(rng.integers(2, min(7, limit // (count - 1)) + 1))
        return [gap] * (count - 1)
    gaps = [2] * (count - 1)
    budget = int(rng.integers(1, limit - sum(gaps) + 1))
    for _ in range(budget):
        eligible = [i for i, gap in enumerate(gaps) if gap < 10]
        if not eligible:
            break
        gaps[int(rng.choice(eligible))] += 1
    if len(set(gaps)) == 1:
        gaps[0] += 1
        gaps[-1] -= 1
    return gaps


def _sample_lengths(rng, count, reference, capacity, mode, witness):
    lengths = [reference] * count
    others = [int(i) for i in rng.permutation(count) if i != count // 2]
    low = 1 if rng.random() < .30 else 2
    low = min(low, reference - 1)

    def short():
        return int(rng.integers(low, reference))

    def long():
        return int(rng.integers(reference + 1, capacity + 1))

    if witness:
        # With at least five strokes, a majority on one side of the reference
        # distinguishes positional selection from the statistical median.
        majority_short = bool(rng.integers(2))
        for i in others:
            lengths[i] = short() if majority_short else long()
        lengths[others[0]] = long() if majority_short else short()
    elif mode in ("one_extend", "one_trim"):
        lengths[others[0]] = short() if mode == "one_extend" else long()
    else:
        # An already-correct noncentral member is a real possibility.
        for i in others:
            if rng.random() < .22:
                continue
            lengths[i] = short() if mode == "extend" or (mode == "mixed" and rng.random() < .5) else long()
        if mode == "mixed":
            lengths[others[0]], lengths[others[1]] = short(), long()
        elif all(length == reference for length in lengths):
            lengths[others[0]] = short() if mode == "extend" else long()
    return lengths


def _choose_colors(rng, pool, count, palette_mode, required=()):
    maximum = min(len(pool), count)
    if palette_mode == "mono":
        size = 1
    elif palette_mode == "multi":
        size = int(rng.integers(2, maximum + 1))
    else:
        size = 1 if rng.random() < .25 else int(rng.integers(2, maximum + 1)) if maximum > 1 else 1
    size = max(size, len(required))
    available = [c for c in pool if c not in required]
    palette = list(required) + [int(x) for x in rng.choice(available, size=size - len(required), replace=False)]
    weights = rng.dirichlet(np.full(size, float(rng.uniform(.25, 2))))
    colors = palette + [int(x) for x in rng.choice(palette, count - size, p=weights)]
    # Repetitions can be clustered or scattered, with no frequency balancing.
    if rng.random() < .22:
        colors.sort()
    else:
        rng.shuffle(colors)
    return colors


def _draw_pair(shape, background, anchors, direction, lengths, colors, reference):
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for (r, c), length, color in zip(anchors, lengths, colors):
        for grid, span in ((inp, length), (out, reference)):
            for step in range(span):
                row, col = r + step * direction[0], c + step * direction[1]
                assert 0 <= row < shape[0] and 0 <= col < shape[1]
                assert grid[row, col] == background
                grid[row, col] = color
    return {"input": inp, "output": out}


def _generate_pair(rng, regime, parameters, descriptor, used_shapes):
    assert regime in ("source_train", "target_test")
    level = parameters["level"]
    count = descriptor["count"]
    direction, perpendicular = _directed_axis(descriptor["direction"])
    diagonal = all(direction)
    gaps = _sample_gaps(rng, count, diagonal, descriptor["equal"], descriptor["witness"])
    if descriptor.get("source_scale"):
        if descriptor["equal"]:
            gaps = [2 if diagonal else int(rng.integers(2, 4))] * (count - 1)
        else:
            gaps = [2] * (count - 1)
            for i in rng.choice(count - 1, size=int(rng.integers(1, 4)), replace=False):
                gaps[int(i)] += 1
    span = sum(gaps)
    capacity = min((12, 22, 28)[level], 29 - span if diagonal else 28)
    if descriptor.get("source_scale"):
        capacity = min(capacity, 16 - span if diagonal else 16)
    choices = [i for i in range(2, min((9, 14, 22)[level], capacity - 1) + 1)
               if i not in descriptor["avoid_reference"]]
    # The witness has enough geometric room for at least five possible spans.
    assert choices
    reference = int(rng.choice(choices))
    if descriptor.get("source_scale") and 4 in choices and rng.random() < .35:
        reference = 4
    lengths = _sample_lengths(rng, count, reference, capacity, descriptor["mode"], descriptor["witness"])
    positions = [0] + list(np.cumsum(gaps))
    anchors = [(int(p * perpendicular[0]), int(p * perpendicular[1])) for p in positions]
    cells = [(r + step * direction[0], c + step * direction[1])
             for (r, c), length in zip(anchors, lengths)
             for step in range(max(length, reference))]
    lo = tuple(min(cell[k] for cell in cells) for k in (0, 1))
    box = tuple(max(cell[k] for cell in cells) - lo[k] + 1 for k in (0, 1))
    # Full canvas range, including compact arrays placed far off-center.
    canvas_mode = int(rng.integers(4))
    shapes = [(h, w) for h in range(box[0], 31) for w in range(box[1], 31)
              if (h, w) not in used_shapes and h * w > 2 * max(sum(lengths), count * reference)]
    preferred = shapes
    if descriptor.get("source_scale") and (16, 16) in shapes:
        preferred = [(16, 16)]
    elif canvas_mode == 0:
        preferred = [s for s in shapes if s[0] == s[1]] or shapes
    elif canvas_mode == 1:
        preferred = [s for s in shapes if s[0] <= box[0] + 4 and s[1] <= box[1] + 4] or shapes
    elif canvas_mode == 2:
        preferred = [s for s in shapes if max(s[0] - box[0], s[1] - box[1]) >= 10] or shapes
    assert preferred
    shape = preferred[int(rng.integers(len(preferred)))]
    margins = [shape[k] - box[k] for k in (0, 1)]
    offsets = [int(rng.choice((0, margin))) if rng.random() < .40 else int(rng.integers(margin + 1)) for margin in margins]
    anchors = [(r - lo[0] + offsets[0], c - lo[1] + offsets[1]) for r, c in anchors]
    background = descriptor["background"]
    colors = _choose_colors(rng, descriptor["pool"], count, descriptor["palette_mode"], descriptor["required_colors"])
    return _draw_pair(shape, background, anchors, direction, lengths, colors, reference)


def _directions_for_puzzle(parameters, train_count, test_count):
    directions = parameters["directions"]
    if parameters["regime"] == "common":
        return [directions[0]] * train_count, [directions[0]] * test_count
    if parameters["regime"] == "mixed":
        return [directions[i % 8] for i in range(train_count)], [directions[(i + 2) % 8] for i in range(test_count)]
    # Hold out a random reversal/rotation rather than one universal diagonal.
    base = directions[0]
    train = [base if i % 2 == 0 else directions[1] for i in range(train_count)]
    held_out = [d for d in (base ^ 1, *directions[2:]) if d not in train]
    return train, [held_out[i % len(held_out)] for i in range(test_count)]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    train_count, test_count = _resolve_pair_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng)
    train_directions, test_directions = _directions_for_puzzle(parameters, train_count, test_count)
    counts = ((3, 5), (3, 5, 7), (3, 5, 7, 9))[parameters["level"]]
    puzzle = {"train": [], "test": []}
    analyses = {"train": [], "test": []}
    seen_inputs = set()
    modes = ("mixed", "extend", "trim", "one_extend", "one_trim")
    test_equal_first = bool(rng.integers(2))
    for split, directions in (("train", train_directions), ("test", test_directions)):
        for index, direction in enumerate(directions):
            witness = split == "train" and index == 0
            eligible = [c for c in counts if (not witness or c >= 5)
                        and (index != 1 or c != analyses[split][0]["count"])]
            count = int(rng.choice(eligible))
            equal = index % 2 == 1 if split == "train" else test_equal_first != bool(index % 2) if index < 2 else bool(rng.integers(2))
            if split == "train" and index >= 2:
                equal = bool(rng.integers(2))
            background = int(rng.integers(10))
            if split == "train" and index == 1 and background == analyses[split][0]["background"]:
                background = (background + int(rng.integers(1, 10))) % 10
            pool = [c for c in parameters["pool"] if c != background]
            palette_mode = "mono" if witness else "multi" if split == "train" and index == 1 else "any"
            required_colors = ()
            if split == "test":
                familiar = sorted({c for a in analyses["train"] for c in a["palette"]})
                novel = [c for c in range(10) if c not in familiar]
                color_mode = str(rng.choice(("familiar", "novel", "mixture"), p=(.4, .2, .4)))
                if color_mode == "novel":
                    background = int(rng.choice([c for c in range(10) if c != novel[0]]))
                    pool = [c for c in novel if c != background]
                elif color_mode == "familiar":
                    pool = [c for c in familiar if c != background]
                else:
                    old_color, new_color = int(rng.choice(familiar)), int(rng.choice(novel))
                    background = int(rng.choice([c for c in range(10) if c not in (old_color, new_color)]))
                    pool = [c for c in range(10) if c != background]
                    required_colors = (old_color, new_color)
                    palette_mode = "multi"
            avoid = {analyses[split][0]["reference_length"]} if index == 1 else set()
            descriptor = {"direction": direction, "count": count, "witness": witness,
                          "equal": equal, "avoid_reference": avoid,
                          "background": background, "pool": pool, "palette_mode": palette_mode,
                          "required_colors": required_colors,
                          "mode": "mixed" if witness else str(rng.choice(modes)),
                          "source_scale": count == 5 and not witness and rng.random() < .25}
            avoid_shapes = {analyses[split][0]["shape"]} if index == 1 else set()
            pair = _generate_pair(rng, "source_train" if split == "train" else "target_test", parameters, descriptor, avoid_shapes)
            # Exact content exclusion, without banning source colors or sizes.
            if _is_source_pair(pair) or _content_fingerprint(pair["input"]) in seen_inputs:
                for shift in range(1, 10):
                    candidate = {k: ((grid.astype(np.int16) + shift) % 10).astype(np.int8) for k, grid in pair.items()}
                    if not _is_source_pair(candidate) and _content_fingerprint(candidate["input"]) not in seen_inputs:
                        pair = candidate
                        break
                else:
                    raise AssertionError("no unused recoloring")
            analysis = _analyze_input_grid(pair["input"])
            puzzle[split].append(pair)
            analyses[split].append(analysis)
            seen_inputs.add(_content_fingerprint(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle

def _line_coordinates(orientation, row, col):
    if orientation == "horizontal":
        return row, col, 1
    if orientation == "vertical":
        return col, row, 1
    if orientation == "diag_down_right":
        return row - col, row + col, 2
    if orientation == "diag_down_left":
        return row + col, row - col, 2
    raise AssertionError("unknown orientation")


def _infer_background(grid):
    values, counts = np.unique(grid, return_counts=True)
    maximum_count = int(np.max(counts))
    assert int(np.sum(counts == maximum_count)) == 1
    return int(values[int(np.argmax(counts))])


def _analyze_input_grid(grid):
    background = _infer_background(grid)
    coordinates = np.argwhere(grid != background)
    assert len(coordinates) > 0

    candidates = []

    for orientation in _CANONICAL_UNITS:
        raw_groups = {}

        for row_value, col_value in coordinates:
            row = int(row_value)
            col = int(col_value)
            line_key, axial_value, axial_step = _line_coordinates(
                orientation, row, col
            )
            raw_groups.setdefault(line_key, []).append(
                (row, col, int(grid[row, col]), axial_value)
            )

        count = len(raw_groups)
        if count not in (3, 5, 7, 9):
            continue

        keys = sorted(raw_groups)
        gaps = tuple(b - a for a, b in zip(keys, keys[1:]))
        if orientation in ("horizontal", "vertical"):
            if min(gaps) < 2:
                continue
        elif min(gaps) < 4 or any(gap % 2 for gap in gaps):
            continue

        ordered_groups = []
        valid = True

        for key in keys:
            entries = sorted(
                raw_groups[key],
                key=lambda entry: entry[3],
            )
            if len(entries) < 1:
                valid = False
                break

            colors = {entry[2] for entry in entries}
            if len(colors) != 1:
                valid = False
                break

            axial_values = [entry[3] for entry in entries]
            if any(
                axial_values[index + 1] - axial_values[index]
                != axial_step
                for index in range(len(axial_values) - 1)
            ):
                valid = False
                break

            ordered_groups.append(entries)

        if not valid:
            continue

        minima = [group[0][3] for group in ordered_groups]
        maxima = [group[-1][3] for group in ordered_groups]
        common_minimum = len(set(minima)) == 1
        common_maximum = len(set(maxima)) == 1

        # The varying input lengths must make exactly one endpoint side common.
        if common_minimum == common_maximum:
            continue

        anchor_side = "min" if common_minimum else "max"
        lengths = [len(group) for group in ordered_groups]
        reference_length = lengths[count // 2]

        if reference_length < 2:
            continue

        palette = tuple(
            sorted({entry[2] for group in ordered_groups for entry in group})
        )

        candidates.append(
            {
                "background": background,
                "orientation": orientation,
                "count": count,
                "keys": tuple(keys),
                "gaps": gaps,
                "groups": ordered_groups,
                "lengths": tuple(lengths),
                "reference_length": reference_length,
                "anchor_side": anchor_side,
                "palette": palette,
                "shape": tuple(grid.shape),
            }
        )

    assert len(candidates) == 1, (
        "input must have one unambiguous parallel-stroke interpretation"
    )
    return candidates[0]


def _construct_expected_output(input_grid, analysis):
    background = analysis["background"]
    reference_length = analysis["reference_length"]
    orientation = analysis["orientation"]
    unit_row, unit_col = _CANONICAL_UNITS[orientation]

    if analysis["anchor_side"] == "min":
        direction_sign = 1
    else:
        direction_sign = -1

    expected = np.full(
        input_grid.shape,
        background,
        dtype=input_grid.dtype,
    )

    for group in analysis["groups"]:
        anchor = group[0] if analysis["anchor_side"] == "min" else group[-1]
        anchor_row, anchor_col, color, _ = anchor

        for step in range(reference_length):
            row = anchor_row + direction_sign * step * unit_row
            col = anchor_col + direction_sign * step * unit_col
            assert 0 <= row < expected.shape[0]
            assert 0 <= col < expected.shape[1]
            assert expected[row, col] == background
            expected[row, col] = color

    return expected


def _content_fingerprint(grid):
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_fingerprint(pair):
    return (
        _content_fingerprint(pair["input"]),
        _content_fingerprint(pair["output"]),
    )


def _puzzle_fingerprint(puzzle):
    return (
        tuple(_pair_fingerprint(pair) for pair in puzzle["train"]),
        tuple(_pair_fingerprint(pair) for pair in puzzle["test"]),
    )


def _puzzles_equal(first, second):
    if set(first) != {"train", "test"}:
        return False
    if set(second) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            for grid_name in ("input", "output"):
                grid_a = pair_a[grid_name]
                grid_b = pair_b[grid_name]
                if grid_a.dtype != grid_b.dtype:
                    return False
                if not np.array_equal(grid_a, grid_b):
                    return False

    return True



# Literal source snapshots support regression and exact-content copy checks.
_SOURCE_ROWS = (('7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7727777777777717/7277777779777177/2777777797771777/7772777977717777/7727779777177777/7777797771777777/7777977717777777/7777777177717777/7777771777177777/7777777771777777/7777777717777777', '7777777777777777/7777777777777777/7777727777777777/7777277777777777/7772777277777777/7727772777777777/7277727779777777/2777277797777777/7772777977717777/7727779777177777/7777797771777177/7777977717771777/7777777177717777/7777771777177777/7777777771777777/7777777717777777'), ('7777777777777777/7777777777777777/7775577777777777/7777777777777777/7777777777777777/7777777777777777/7779999999777777/7777777777777777/7773333333333337/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777', '7777777777777777/7777777777777777/7775555555777777/7777777777777777/7777777777777777/7777777777777777/7779999999777777/7777777777777777/7773333333777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777'), ('7777777777777777/7777777777777777/7787787878777877/7787787878777877/7777787878777877/7777787877777877/7777777777777877/7777777777777877/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777', '7777777777777777/7777777777777777/7787787878777877/7787787878777877/7787787878777877/7787787878777877/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777/7777777777777777'), ('7777777777777777/7777777777777777/7777787777777777/7777877777777777/7778777877777777/7787778777777777/7777787776777777/7777777767777777/7777777677777777/7777776777776777/7777777777767777/7777777777677767/7777777776777677/7777777767776777/7777777777767777/7777777777677777', '7777777777777777/7777777777777777/7777787777777777/7777877777777777/7778777877777777/7787778777777777/7777787776777777/7777877767777777/7777777677777777/7777776777776777/7777777777767777/7777777777677767/7777777776777677/7777777777776777/7777777777767777/7777777777777777'))

def _source_pairs():
    return [{k: np.array([[int(c) for c in row] for row in text.split('/')], dtype=np.int8)
             for k, text in zip(("input", "output"), rows)} for rows in _SOURCE_ROWS]


def _is_source_pair(pair):
    if pair["input"].shape != (16, 16):
        return False
    rows = tuple('/'.join(''.join(str(int(c)) for c in row) for row in pair[k]) for k in ("input", "output"))
    return rows in _SOURCE_ROWS


def _selection_witness(analysis):
    keys = np.array(analysis["keys"])
    distances = abs(2 * keys - keys[0] - keys[-1])
    nearest = np.flatnonzero(distances == min(distances))
    return (len(nearest) == 1 and int(nearest[0]) != analysis["count"] // 2
            and analysis["lengths"][int(nearest[0])] != analysis["reference_length"])


def stroke_features(input_grid):
    """Measurements reconstructed from cells, suitable for split/seed audits."""
    a = _analyze_input_grid(input_grid)
    reference = a["reference_length"]
    delta = [reference - length for length in a["lengths"]]
    extensions = sum(d > 0 for d in delta)
    trims = sum(d < 0 for d in delta)
    coordinates = np.argwhere(input_grid != a["background"])
    # Include output extents when measuring the spare canvas.
    expected = _construct_expected_output(input_grid, a)
    union = np.argwhere((input_grid != a["background"]) | (expected != a["background"]))
    low, high = union.min(axis=0), union.max(axis=0)
    margins = (int(low[0]), int(low[1]), int(input_grid.shape[0] - high[0] - 1), int(input_grid.shape[1] - high[1] - 1))
    colors = tuple(g[0][2] for g in a["groups"])
    frequencies = tuple(sorted(colors.count(c) for c in set(colors)))
    pattern = "mixed" if extensions and trims else "extend" if extensions else "trim"
    return {"direction": (a["orientation"], a["anchor_side"]), "count": a["count"],
            "reference": reference, "lengths": a["lengths"], "gaps": a["gaps"],
            "unequal": len(set(a["gaps"])) > 1, "shape": a["shape"],
            "background": a["background"], "palette_size": len(a["palette"]),
            "palette": a["palette"], "colors": colors, "frequencies": frequencies,
            "adjacent_repeat": any(x == y for x, y in zip(colors, colors[1:])),
            "separated_repeat": any(colors[i] == colors[j] for i in range(len(colors)) for j in range(i + 2, len(colors))),
            "pattern": pattern, "changed_strokes": extensions + trims,
            "unchanged_noncentral": a["count"] - 1 - extensions - trims,
            "extension": max(delta), "truncation": -min(delta),
            "nonmedian": sorted(a["lengths"])[a["count"] // 2] != reference,
            "geometric_witness": bool(_selection_witness(a)), "margins": margins,
            "spare": (margins[0] + margins[2], margins[1] + margins[3]),
            "foreground_box": tuple(int(x) for x in coordinates.max(axis=0) - coordinates.min(axis=0) + 1)}


def _validate_pair_transformation(pair, regime):
    assert regime in ("source_train", "target_test")
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    analysis = _analyze_input_grid(inp)
    expected = _construct_expected_output(inp, analysis)
    assert np.array_equal(out, expected), "incorrect stroke normalization"
    assert not np.array_equal(inp, out)
    assert not _is_source_pair(pair)
    background = analysis["background"]
    assert np.count_nonzero(inp == background) > inp.size / 2
    assert np.count_nonzero(out == background) > out.size / 2
    # Check the shared part of each stroke separately: only free tails change.
    reference = analysis["reference_length"]
    for group in analysis["groups"]:
        ordered = group if analysis["anchor_side"] == "min" else group[::-1]
        for r, c, color, _ in ordered[:reference]:
            assert inp[r, c] == out[r, c] == color
        for r, c, _, _ in ordered[reference:]:
            assert out[r, c] == background
    return analysis


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key, both = _content_fingerprint(pair["input"]), _pair_fingerprint(pair)
            assert key not in seen_inputs and both not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(both)
            analyses[split].append(_validate_pair_transformation(pair, "source_train" if split == "train" else "target_test"))
    train = analyses["train"]
    assert len({a["count"] for a in train}) >= 2
    assert len({a["reference_length"] for a in train}) >= 2
    assert len({a["background"] for a in train}) >= 2
    assert any(len(a["palette"]) == 1 for a in train)
    assert any(len(a["palette"]) >= 2 for a in train)
    assert any(sorted(a["lengths"])[a["count"] // 2] != a["reference_length"] for a in train)
    assert any(_selection_witness(a) for a in train)
    assert any(min(a["lengths"]) < a["reference_length"] for a in train)
    assert any(max(a["lengths"]) > a["reference_length"] for a in train)
    # Training always reveals both operations and both spacing classes. Tests
    # recombine visible geometry; no palette, direction or length is a secret.
    for split in ("train", "test"):
        examples = analyses[split]
        if len(examples) > 1:
            assert len({len(set(a["gaps"])) > 1 for a in examples}) == 2
            for field in ("count", "reference_length", "shape"):
                assert len({a[field] for a in examples}) > 1


def _check_equivariance(pair, rng):
    inp, out = pair["input"], pair["output"]
    palette = rng.permutation(10).astype(np.int8)
    recolored = palette[inp]
    assert np.array_equal(_construct_expected_output(recolored, _analyze_input_grid(recolored)), palette[out])
    # Both diagonal slopes and either endpoint are exercised by the D4 group.
    for turns in range(4):
        for mirror in (False, True):
            left, right = np.rot90(inp, turns), np.rot90(out, turns)
            if mirror:
                left, right = np.fliplr(left), np.fliplr(right)
            assert np.array_equal(_construct_expected_output(left, _analyze_input_grid(left)), right)


def self_test() -> None:
    rng = np.random.default_rng(99173)
    # All four exact source pairs, including three unequal-gap geometries.
    for pair in _source_pairs():
        assert np.array_equal(_construct_expected_output(pair["input"], _analyze_input_grid(pair["input"])), pair["output"])
        _check_equivariance(pair, rng)
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(314159, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(314159, train_count, test_count))
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, puzzles = set(), set(), set()
    records = {"train": [], "test": []}
    regimes, color_modes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 3:
            assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_fingerprint(puzzle))
        directions = {}
        familiar = {c for pair in puzzle["train"] for c in _analyze_input_grid(pair["input"])["palette"]}
        for split in ("train", "test"):
            directions[split] = set()
            for index, pair in enumerate(puzzle[split]):
                a = _validate_pair_transformation(pair, "source_train" if split == "train" else "target_test")
                f = stroke_features(pair["input"])
                records[split].append(f)
                directions[split].add(f["direction"])
                if split == "test":
                    colors = set(f["palette"])
                    color_modes.add("familiar" if colors <= familiar else "novel" if colors.isdisjoint(familiar) else "mixture")
                if index == 0:
                    _check_equivariance(pair, rng)
                # Changing only the visible central length must change the
                # inferred target. No fixed four-cell predictor can do this.
                if (split == "test" and f["reference"] > 2
                        and any(length != f["reference"] - 1 for i, length in enumerate(f["lengths"]) if i != f["count"] // 2)):
                    altered = pair["input"].copy()
                    middle = a["groups"][a["count"] // 2]
                    free = middle[-1] if a["anchor_side"] == "min" else middle[0]
                    altered[free[0], free[1]] = a["background"]
                    changed = _analyze_input_grid(altered)
                    assert changed["reference_length"] == f["reference"] - 1
                    assert not np.array_equal(_construct_expected_output(altered, changed), pair["output"])
        regimes.add("common" if len(directions["train"] | directions["test"]) == 1 else "transfer" if directions["train"].isdisjoint(directions["test"]) else "mixed")
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(puzzles) == 42
    assert regimes == {"common", "mixed", "transfer"}
    assert color_modes == {"familiar", "novel", "mixture"}
    for split, rows in records.items():
        assert len({f["direction"] for f in rows}) == 8
        assert {f["count"] for f in rows} == {3, 5, 7, 9}
        assert len({f["reference"] for f in rows}) >= 9
        assert min(f["reference"] for f in rows) == 2
        assert max(max(f["lengths"]) for f in rows) >= 20
        assert min(min(f["lengths"]) for f in rows) == 1
        assert max(f["extension"] for f in rows) >= 5
        assert max(f["truncation"] for f in rows) >= 5
        assert {f["pattern"] for f in rows} == {"mixed", "extend", "trim"}
        assert any(f["changed_strokes"] == 1 for f in rows)
        assert any(f["unchanged_noncentral"] > 0 for f in rows)
        assert len({f["shape"] for f in rows}) >= 40
        assert len({(f["count"], f["reference"], f["gaps"], f["lengths"]) for f in rows}) >= 60
        assert {f["background"] for f in rows} == set(range(10))
        assert max(f["palette_size"] for f in rows) >= 4
        assert any(max(f["frequencies"]) - min(f["frequencies"]) >= 2 and len(f["frequencies"]) > 1 for f in rows)
        assert any(f["adjacent_repeat"] for f in rows) and any(f["separated_repeat"] for f in rows)
        assert any(max(f["spare"]) >= 10 for f in rows)
        assert any(f["shape"] == (16, 16) and f["count"] == 5 for f in rows)
        assert any(f["count"] == 9 and f["unequal"] and f["palette_size"] > 1 for f in rows)
    # Validator must reject a changed reference pixel in a purported output.
    broken = {split: [{k: g.copy() for k, g in p.items()} for p in puzzle[split]] for split in puzzle}
    pair = broken["test"][0]
    a = _analyze_input_grid(pair["input"])
    r, c, _, _ = a["groups"][a["count"] // 2][0]
    pair["output"][r, c] = a["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged reference stroke")
