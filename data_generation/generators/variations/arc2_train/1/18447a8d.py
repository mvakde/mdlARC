"""Generate contour-matching ARC puzzles with translated, colored complements."""

import numpy as np


_SOURCE_INPUTS = {
    "77777777777/88777779999/88777777799/88777779999/77777777777/87777777733/88877773333/87777777733/77777777777/88877777111/87777777111/88877777111/77777777777",
    "77777777777/88877777799/87777779999/88777777999/77777777777/88877777722/88777777222/87777772222/77777777777",
    "77777777777/87777772222/87777772222/88777777222/77777777777/88777774444/88877777744/88777777774/77777777777/87777777333/88887777733/88877777333/77777777777/87777776666/88877777776/88887777766/77777777777",
    "77777777777/88777777711/88887777771/87777771111/77777777777/88777777722/88877777772/87777777222/77777777777/88877777999/88887777779/87777779999/77777777777/88877777555/88887777755/88777775555/77777777777",
}


def _integer_count(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if not _integer_count(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer_count(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _orient(grid, code):
    if code >= 4:
        grid = np.fliplr(grid)
    return np.rot90(grid, code % 4).copy()


def _permutation(rng, count, kind):
    order = list(range(count))
    if kind == "identity":
        return tuple(order)
    if kind == "swap":
        a, b = [int(x) for x in rng.choice(count, 2, replace=False)]
        order[a], order[b] = order[b], order[a]
    elif kind == "cycle":
        nodes = [int(x) for x in rng.permutation(count)]
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            order[a] = b
    elif kind == "partial_cycle":
        length = int(rng.integers(3, count)) if count > 3 else count
        nodes = [int(x) for x in rng.choice(count, length, replace=False)]
        for a, b in zip(nodes, nodes[1:] + nodes[:1]):
            order[a] = b
    elif kind == "double_swap":
        nodes = [int(x) for x in rng.choice(count, 4, replace=False)]
        order[nodes[0]], order[nodes[1]] = nodes[1], nodes[0]
        order[nodes[2]], order[nodes[3]] = nodes[3], nodes[2]
    else:
        raise ValueError("invalid permutation kind")
    return tuple(order)


def _choose_kind(rng, count, regime, forced=None):
    if forced is not None:
        return forced
    kinds = ["identity", "swap", "cycle"]
    weights = [2, 4, 3] if regime == "source_train" else [1, 4, 4]
    if count >= 4:
        kinds += ["partial_cycle", "double_swap"]
        weights += [4, 3]
    return str(rng.choice(kinds, p=np.asarray(weights) / sum(weights)))


def _profiles(rng, count, height, packed, history):
    # Positive width in every row makes both silhouettes connected. Include
    # constant profiles: the original first pair has a rectangular match.
    vocabulary = [tuple(int(x) for x in item)
                  for item in np.array(np.meshgrid(*([np.arange(1, packed)] * height))).T.reshape(-1, height)]
    available = set(vocabulary)
    selected = []
    reusable = list(set(history.get((height, packed), ())) & available)
    rng.shuffle(reusable)
    if reusable and rng.random() < 0.55:
        reuse_count = min(len(reusable), count, int(rng.integers(1, max(2, count // 2 + 1))))
        selected.extend(reusable[:reuse_count])
        available.difference_update(selected)
    constants = [tuple([width] * height) for width in range(1, packed) if tuple([width] * height) in available]
    if constants and len(selected) < count and rng.random() < 0.38:
        profile = constants[int(rng.integers(len(constants)))]
        selected.append(profile)
        available.remove(profile)
    remaining = sorted(available)
    indices = rng.choice(len(remaining), count - len(selected), replace=False)
    selected.extend(remaining[int(index)] for index in indices)
    rng.shuffle(selected)
    history.setdefault((height, packed), []).extend(selected)
    return selected


def _geometry(rng, count, history, regime):
    valid = [(h, w) for h in (2, 3, 4) for w in range(4, 10)
             if count * h + count - 1 <= 30]
    reusable = [key for key in history if key in valid]
    if reusable and rng.random() < (0.52 if regime == "target_test" else 0.35):
        height, packed = reusable[int(rng.integers(len(reusable)))]
    else:
        height, packed = valid[int(rng.integers(len(valid)))]
    spacings = [(separator, top, bottom)
                for separator in (1, 2, 3)
                for top in range(3) for bottom in range(3)
                if count * height + (count - 1) * separator + top + bottom <= 30]
    separator, top, bottom = spacings[int(rng.integers(len(spacings)))]
    return height, packed, separator, top, bottom


def _colors(rng, roles, count, regime, seen_accents, previous_role, fixed_role):
    # Literal colors are pair-level; the shared structural role and moving
    # payload role, demonstrated by train, are the puzzle-level convention.
    if fixed_role is None:
        choices = [(int(a), int(b)) for a in roles for b in roles if a != b and (a, b) != previous_role]
        background, structural = choices[int(rng.integers(len(choices)))]
    else:
        background, structural = fixed_role
    accent_pool = roles[:8] if regime == "source_train" else roles
    available = [int(x) for x in accent_pool if x not in (background, structural)]
    unique_count = count if count == 2 or rng.random() >= 0.36 else count - 1
    # Even with repeated colors, at least two accents identify the anchor bank.
    unique_count = min(unique_count, len(available))
    if regime == "target_test" and seen_accents:
        familiar = [x for x in available if x in seen_accents]
        novel = [x for x in available if x not in seen_accents]
        mode = int(rng.integers(3))
        if mode == 0:
            preferred = familiar
        elif mode == 1:
            preferred = novel
        else:
            preferred = available
        take = min(len(preferred), unique_count)
        chosen = [int(x) for x in rng.choice(preferred, take, replace=False)] if take else []
        rest = [x for x in available if x not in chosen]
        chosen += [int(x) for x in rng.choice(rest, unique_count - take, replace=False)]
    else:
        chosen = [int(x) for x in rng.choice(available, unique_count, replace=False)]
    colors = chosen + [chosen[int(rng.integers(len(chosen)))] for _ in range(count - unique_count)]
    rng.shuffle(colors)
    return background, structural, colors


def _make_pair(rng, regime, count, forced_kind, roles, history, seen_accents, previous_role, fixed_role, orientation):
    band_height, packed, separator, top, bottom = _geometry(rng, count, history, regime)
    anchors = _profiles(rng, count, band_height, packed, history)
    kind = _choose_kind(rng, count, regime, forced_kind)
    source_to_target = _permutation(rng, count, kind)
    background, structural, colors = _colors(rng, roles, count, regime, seen_accents, previous_role, fixed_role)
    if regime == "source_train":
        seen_accents.update(colors)
    left = int(rng.integers(0, 4))
    maximum_combined = max(anchors[source][row] + packed - anchors[target][row]
                           for source, target in enumerate(source_to_target)
                           for row in range(band_height))
    detached_end = left + max(packed + int(rng.integers(2, 11)),
                              maximum_combined + int(rng.integers(1, 5)))
    width = detached_end + int(rng.integers(0, 4))
    height = top + count * band_height + (count - 1) * separator + bottom
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full_like(input_grid, background)
    for source, target in enumerate(source_to_target):
        row0 = top + source * (band_height + separator)
        for local_row in range(band_height):
            row = row0 + local_row
            anchor_width = anchors[source][local_row]
            payload_width = packed - anchors[target][local_row]
            input_grid[row, left:left + anchor_width] = structural
            input_grid[row, detached_end - payload_width:detached_end] = colors[target]
    for target in range(count):
        row0 = top + target * (band_height + separator)
        for local_row in range(band_height):
            row = row0 + local_row
            anchor_width = anchors[target][local_row]
            output_grid[row, left:left + anchor_width] = structural
            output_grid[row, left + anchor_width:left + packed] = colors[target]
    return {"input": _orient(input_grid, orientation), "output": _orient(output_grid, orientation)}, (background, structural)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_string(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def _parse_canonical(input_grid, output_grid, orientation):
    height, width = input_grid.shape
    for background in [int(x) for x in np.unique(input_grid)]:
        active = np.flatnonzero(np.any(input_grid != background, axis=1))
        if len(active) == 0:
            continue
        runs = []
        for row in active:
            if not runs or row != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(int(row))
        count = len(runs)
        if not 2 <= count <= 7 or len(set(map(len, runs))) != 1:
            continue
        band_height = len(runs[0])
        if not 2 <= band_height <= 4:
            continue
        separators = [runs[i + 1][0] - runs[i][-1] - 1 for i in range(count - 1)]
        if len(set(separators)) != 1 or not 1 <= separators[0] <= 3:
            continue
        if not 0 <= runs[0][0] <= 2 or not 0 <= height - 1 - runs[-1][-1] <= 2:
            continue
        anchor_profiles, payload_profiles, payload_colors = [], [], []
        left_positions, right_positions, structural_values = set(), set(), set()
        valid = True
        for band in runs:
            anchors, payloads, colors = [], [], []
            for row_index in band:
                row = input_grid[row_index]
                occupied = np.flatnonzero(row != background)
                if len(occupied) < 2:
                    valid = False
                    break
                boundaries = np.flatnonzero(np.diff(occupied) > 1)
                if len(boundaries) != 1:
                    valid = False
                    break
                first = occupied[:boundaries[0] + 1]
                second = occupied[boundaries[0] + 1:]
                first_colors = set(int(x) for x in row[first])
                second_colors = set(int(x) for x in row[second])
                if len(first_colors) != 1 or len(second_colors) != 1 or first_colors == second_colors:
                    valid = False
                    break
                anchors.append(len(first))
                payloads.append(len(second))
                colors.append(next(iter(second_colors)))
                structural_values.update(first_colors)
                left_positions.add(int(first[0]))
                right_positions.add(int(second[-1]) + 1)
            if not valid or len(set(colors)) != 1:
                valid = False
                break
            anchor_profiles.append(tuple(anchors))
            payload_profiles.append(tuple(payloads))
            payload_colors.append(colors[0])
        if not valid or len(structural_values) != 1 or len(left_positions) != 1 or len(right_positions) != 1:
            continue
        structural = next(iter(structural_values))
        if len(set(payload_colors)) < 2 or structural in payload_colors:
            continue
        if len(set(anchor_profiles)) != count or len(set(payload_profiles)) != count:
            continue
        left = next(iter(left_positions))
        detached_end = next(iter(right_positions))
        sums = [sum(anchor_profiles[b][row] + payload_profiles[b][row] for b in range(count))
                for row in range(band_height)]
        if len(set(sums)) != 1 or sums[0] % count:
            continue
        packed = sums[0] // count
        if not 4 <= packed <= 9 or not 2 <= detached_end - (left + packed) <= 15:
            continue
        if any(not all(1 <= value < packed for value in profile)
               for profile in anchor_profiles + payload_profiles):
            continue
        target_to_source = []
        for anchor in anchor_profiles:
            candidates = [source for source, payload in enumerate(payload_profiles)
                          if all(a + p == packed for a, p in zip(anchor, payload))]
            if len(candidates) != 1:
                valid = False
                break
            target_to_source.append(candidates[0])
        if not valid or len(set(target_to_source)) != count:
            continue
        expected = np.full(input_grid.shape, background, dtype=np.int8)
        for target, band in enumerate(runs):
            source = target_to_source[target]
            for local_row, row_index in enumerate(band):
                anchor_width = anchor_profiles[target][local_row]
                expected[row_index, left:left + anchor_width] = structural
                expected[row_index, left + anchor_width:left + packed] = payload_colors[source]
        if not np.array_equal(output_grid, expected) or np.array_equal(input_grid, output_grid):
            continue
        permutation = [0] * count
        for target, source in enumerate(target_to_source):
            permutation[source] = target
        return {"band_count": count, "band_height": band_height, "packed_width": packed,
                "canvas_shape": input_grid.shape, "separator": separators[0],
                "outer_margins": (runs[0][0], height - 1 - runs[-1][-1], left, width - detached_end),
                "translation": detached_end - (left + packed),
                "background": background, "structural": structural,
                "profiles": tuple(anchor_profiles), "accents": tuple(payload_colors),
                "permutation": tuple(permutation), "orientation": orientation}
    return None


def _analyze_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    _check_grid(input_grid)
    _check_grid(output_grid)
    assert input_grid.shape == output_grid.shape
    for code in range(8):
        result = _parse_canonical(_orient(input_grid, code), _orient(output_grid, code), code)
        if result is not None:
            return result
    raise AssertionError("pair does not follow common-width contour matching")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyzed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            item = _analyze_pair(pair)
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _source_string(pair["input"]) not in _SOURCE_INPUTS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyzed[split].append(item)
    # The rule has no hidden color or direction map. Every test color, layout,
    # and contour is visible in its own input; train demonstrates the operation.
    assert any(any(source != target for source, target in enumerate(item["permutation"]))
               for item in analyzed["train"])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    roles = [int(x) for x in rng.permutation(10)]
    # These are compositional task conventions, not a hidden literal-color map.
    history, seen_accents = {}, set()
    stable_orientation = int(rng.integers(8)) if rng.random() < 0.35 else None
    fixed_role = (roles[0], roles[1]) if rng.random() < 0.25 else None
    stable_count = int(rng.integers(2, 8)) if rng.random() < 0.18 else None
    train_counts = ([stable_count] * train_count if stable_count is not None else
                    [int(rng.integers(2, 8)) for _ in range(train_count)])
    forced = [None] * train_count
    if stable_count is not None or rng.random() < 0.7:
        identity_index = int(rng.integers(train_count))
        forced[identity_index] = "identity"
    moving_indices = [i for i in range(train_count) if forced[i] != "identity"]
    forced[moving_indices[int(rng.integers(len(moving_indices)))]] = "swap"
    puzzle = {"train": [], "test": []}
    previous_role = None
    for index, count in enumerate(train_counts):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        orientation = stable_orientation if stable_orientation is not None else int(rng.integers(8))
        pair, previous_role = _make_pair(pair_rng, "source_train", count, forced[index],
                                         roles, history, seen_accents, previous_role, fixed_role, orientation)
        puzzle["train"].append(pair)
    for _ in range(test_count):
        count = stable_count if stable_count is not None else int(rng.integers(2, 8))
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        orientation = stable_orientation if stable_orientation is not None else int(rng.integers(8))
        pair, previous_role = _make_pair(pair_rng, "target_test", count, None,
                                         roles, history, seen_accents, previous_role, fixed_role, orientation)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[key], b[key])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for key in ("input", "output")) and all(len(left[s]) == len(right[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, bands, orientations = set(), set(), set(), set(), set()
    split_bands = {"train": set(), "test": set()}
    packed_widths, heights, translations, separators = set(), set(), set(), set()
    constant_profiles, repeated_colors, one_swaps, reused_profiles = 0, 0, 0, 0
    reused_changed_colors, reused_changed_placements = 0, 0
    stable_orientations, mixed_orientations, orientation_shifts = 0, 0, 0
    stable_roles, mixed_roles, novel_accents, familiar_accents = 0, 0, 0, 0
    larger_tests, varied_margins, mixed_band_counts, stable_band_counts, mixed_widths = 0, 0, 0, 0, 0
    combined_variations = 0
    all_inputs, all_layouts = set(), set()
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        known_profiles = {}
        analyzed = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                item = _analyze_pair(pair)
                analyzed[split].append(item)
                shapes.add(pair["input"].shape)
                bands.add(item["band_count"])
                split_bands[split].add(item["band_count"])
                orientations.add(item["orientation"])
                packed_widths.add(item["packed_width"])
                heights.add(item["band_height"])
                translations.add(item["translation"])
                separators.add(item["separator"])
                all_inputs.add(_grid_key(pair["input"]))
                all_layouts.add((item["band_count"], item["profiles"], item["permutation"]))
                constant_profiles += sum(len(set(profile)) == 1 for profile in item["profiles"])
                repeated_colors += len(set(item["accents"])) < len(item["accents"])
                one_swaps += item["band_count"] == 4 and sum(i != x for i, x in enumerate(item["permutation"])) == 2
                combined_variations += (item["orientation"] % 2 == 1 and item["band_count"] >= 5
                                        and item["packed_width"] != 5 and any(item["outer_margins"]))
                for target, profile in enumerate(item["profiles"]):
                    source = item["permutation"].index(target)
                    key = (item["band_height"], item["packed_width"], profile)
                    occurrence = (item["accents"][source], source, target)
                    if key in known_profiles:
                        previous = known_profiles[key]
                        reused_profiles += 1
                        reused_changed_colors += previous[0] != occurrence[0]
                        reused_changed_placements += previous[1:] != occurrence[1:]
                    known_profiles[key] = occurrence
        all_items = analyzed["train"] + analyzed["test"]
        train_items, test_items = analyzed["train"], analyzed["test"]
        train_orientations = {item["orientation"] for item in train_items}
        role_pairs = {(item["background"], item["structural"]) for item in all_items}
        stable_orientations += len({item["orientation"] for item in all_items}) == 1
        mixed_orientations += len({item["orientation"] for item in all_items}) > 1
        orientation_shifts += any(item["orientation"] not in train_orientations for item in test_items)
        stable_roles += len(role_pairs) == 1
        mixed_roles += len(role_pairs) > 1
        train_accents = set().union(*(set(item["accents"]) for item in train_items))
        novel_accents += sum(bool(set(item["accents"]) - train_accents) for item in test_items)
        familiar_accents += sum(set(item["accents"]) <= train_accents for item in test_items)
        larger_tests += any(item["band_count"] > max(t["band_count"] for t in train_items)
                            for item in test_items)
        varied_margins += len({item["outer_margins"] for item in all_items}) > 1
        mixed_band_counts += len({item["band_count"] for item in all_items}) > 1
        stable_band_counts += len({item["band_count"] for item in all_items}) == 1
        mixed_widths += len({item["packed_width"] for item in all_items}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(bands) >= 5 and len(orientations) >= 4
    assert split_bands["train"] == set(range(2, 8)) and split_bands["test"] == set(range(2, 8))
    assert len(packed_widths) >= 5 and len(heights) == 3
    assert len(translations) >= 5 and len(separators) >= 2
    assert constant_profiles > 0 and repeated_colors > 0 and one_swaps > 0 and reused_profiles > 0
    assert reused_changed_colors > 0 and reused_changed_placements > 0
    assert stable_orientations > 0 and mixed_orientations > 0 and orientation_shifts > 0
    assert stable_roles > 0 and mixed_roles > 0 and novel_accents > 0 and familiar_accents > 0
    assert larger_tests > 0 and varied_margins > 0 and mixed_band_counts > 0 and stable_band_counts > 0 and mixed_widths > 0
    assert combined_variations > 0
    assert len(all_inputs) >= 0.98 * total_pairs
    assert len(all_layouts) >= 100
    assert _puzzles_equal(generate_puzzle(777), generate_puzzle(777))
    assert _puzzles_equal(generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3))
    assert not _puzzles_equal(generate_puzzle(123456, 7, 3), generate_puzzle(123457, 7, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, *counts)
        validate_generated_puzzle(puzzle)
        for requested, actual in zip(counts, (len(puzzle["train"]), len(puzzle["test"]))):
            assert requested is None or requested == actual
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                      {"num_train": 2.0}, {"num_test": True}):
        try:
            generate_puzzle(4000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    broken = generate_puzzle(5000, 4, 2)
    broken["test"][0]["output"][:] = broken["test"][0]["input"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
