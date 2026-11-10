import numpy as np


_SOURCE_EXAMPLES = (
    (("7071550", "7701555", "0701000", "0001505"), ("000", "000", "303", "030")),
    (("7701500", "0001000", "7701500", "0771550"), ("003", "333", "003", "000")),
    (("0771500", "0071055", "0701550", "0071000"), ("000", "300", "003", "330")),
    (("7001050", "0071500", "0001555", "7771555"), ("003", "030", "000", "000")),
    (("7701500", "7001500", "0001505", "0001550"), ("003", "033", "030", "003")),
    (("0001050", "7771550", "0001500", "7001555"), ("303", "000", "033", "000")),
)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        # The arithmetic schedule prevents a run of consecutive seeds from
        # accidentally collapsing to one pair count.
        num_test = int(seed) % 4 + 1
        minimum = max(2, num_test + 1)
        return minimum + (int(seed) // 4) % (11 - minimum), num_test
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _rng_for_pair(rng, split, index):
    words = [int(x) for x in rng.integers(0, 2**31, size=4)]
    return np.random.default_rng(words + [split, index])


def _rectangle(mask, rng, clear=False):
    h, w = mask.shape
    y0 = int(rng.integers(0, h))
    x0 = int(rng.integers(0, w))
    y1 = int(rng.integers(y0 + 1, h + 1))
    x1 = int(rng.integers(x0 + 1, w + 1))
    mask[y0:y1, x0:x1] = not clear


def _shape_mask(rng, h, w):
    kind = int(rng.integers(0, 8))
    mask = np.zeros((h, w), dtype=bool)
    if kind == 0:  # unconstrained binary texture
        mask = rng.random((h, w)) < rng.uniform(0.18, 0.82)
    elif kind == 1:  # compact horizontal or vertical bars
        for _ in range(int(rng.integers(1, 5))):
            if rng.random() < 0.5:
                y = int(rng.integers(0, h))
                x = int(rng.integers(0, w))
                length = int(rng.integers(1, w - x + 1))
                mask[y:min(h, y + int(rng.integers(1, 3))), x:x + length] = True
            else:
                y = int(rng.integers(0, h))
                x = int(rng.integers(0, w))
                length = int(rng.integers(1, h - y + 1))
                mask[y:y + length, x:min(w, x + int(rng.integers(1, 3)))] = True
    elif kind == 2:  # one to several blocks
        for _ in range(int(rng.integers(1, 5))):
            _rectangle(mask, rng)
    elif kind == 3:  # filled area with cutout holes
        mask[:] = True
        for _ in range(int(rng.integers(1, 4))):
            _rectangle(mask, rng, clear=True)
    elif kind in (4, 7):  # repeated two-dimensional motif
        mh = int(rng.integers(2, min(5, h + 1)))
        mw = int(rng.integers(2, min(5, w + 1)))
        tile = rng.random((mh, mw)) < rng.uniform(0.2, 0.8)
        mask = np.tile(tile, ((h + mh - 1) // mh, (w + mw - 1) // mw))[:h, :w]
    elif kind == 5:  # connected irregular growth
        y = int(rng.integers(0, h))
        x = int(rng.integers(0, w))
        for _ in range(int(rng.integers(max(2, h * w // 5), max(3, h * w)))):
            mask[y, x] = True
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
            y = min(h - 1, max(0, y + dy))
            x = min(w - 1, max(0, x + dx))
    else:  # sparse and dense masks, including near-empty/full panels
        probability = float(rng.choice([0.06, 0.12, 0.88, 0.94]))
        mask = rng.random((h, w)) < probability
    return mask


def _masks(rng, h, w, mode, training):
    left = _shape_mask(rng, h, w)
    if mode == "teaching":
        right = _shape_mask(rng, h, w)
        cells = rng.choice(h * w, size=4, replace=False)
        for cell, state in zip(cells, ((0, 0), (0, 1), (1, 0), (1, 1))):
            y, x = divmod(int(cell), w)
            left[y, x], right[y, x] = state
    elif mode == "identical":
        right = left.copy()
    elif mode == "contained":
        right = left & (rng.random((h, w)) < rng.uniform(0.25, 0.8))
    elif mode == "disjoint":
        right = ~left & _shape_mask(rng, h, w)
    elif mode == "complementary":
        if not left.any() or left.all():
            left.flat[0] = True
            left.flat[-1] = False
        right = ~left
    elif mode == "both_blank":
        left[:] = False
        right = left.copy()
    elif mode == "both_full":
        left[:] = True
        right = left.copy()
    elif mode == "shifted":
        shift_y = int(rng.integers(-2, 3))
        shift_x = int(rng.integers(-2, 3))
        right = np.roll(left, (shift_y, shift_x), axis=(0, 1))
    else:
        right = _shape_mask(rng, h, w)

    if mode not in ("teaching", "complementary", "both_blank", "both_full"):
        if h >= 4 and rng.random() < 0.24:
            row = int(rng.integers(0, h))
            left[row] = False
            right[row] = False
        if h >= 4 and mode not in ("disjoint",) and rng.random() < 0.16:
            row = int(rng.integers(0, h))
            left[row] = True
            right[row] = True

    if training and mode != "teaching":
        # All demonstrations display both output colors. Extreme outputs
        # occur in tests after the marker and background have been taught.
        if not np.any(~left & ~right):
            y, x = divmod(int(rng.integers(0, h * w)), w)
            left[y, x] = right[y, x] = False
        if not np.any(left | right):
            y, x = divmod(int(rng.integers(0, h * w)), w)
            left[y, x] = True

    if h >= 4 and mode not in ("teaching", "both_blank", "both_full") and rng.random() < 0.2:
        src, dst = rng.choice(h, size=2, replace=False)
        left[dst] = left[src]
        right[dst] = right[src]
    if training:
        if not np.any(~left & ~right):
            left[0, 0] = right[0, 0] = False
        if not np.any(left | right):
            left[-1, -1] = True
            if mode in ("identical", "contained"):
                right[-1, -1] = True
    return left, right


def _choose_size(rng, geometry, previous=None, harder=False):
    if geometry == "source":
        return 4, 3
    for _ in range(8):
        if harder and rng.random() < 0.55:
            size = (int(rng.integers(6, 15)), int(rng.integers(5, 15)))
        else:
            size = (int(rng.integers(3, 15)), int(rng.integers(2, 15)))
        if size != previous:
            return size
    return (3, 2) if previous != (3, 2) else (4, 3)


def _pair_colors(rng, convention, index, training):
    marker = convention["marker"]
    if convention["palette"] == "source":
        return convention["source_colors"]
    background = convention["background"]
    if convention["palette"] == "variable_background":
        candidates = [c for c in range(10) if c != marker]
        if training and index == 1:
            candidates.remove(convention["first_background"])
        background = int(rng.choice(candidates))
        if training and index == 0:
            convention["first_background"] = background
    available = [c for c in range(10) if c not in (marker, background)]
    divider = int(rng.choice(available))
    available.remove(divider)
    left = int(rng.choice(available))
    if convention["palette"] != "variable_background" and rng.random() < 0.3:
        right = left
    else:
        right = int(rng.choice([c for c in available if c != left]))
    return background, left, divider, right, marker


def _build_pair(left, right, colors, orientation):
    background, left_color, divider, right_color, marker = colors
    h, w = left.shape
    inp = np.full((h, 2 * w + 1), background, dtype=np.int8)
    inp[:, :w] = np.where(left, left_color, background)
    inp[:, w] = divider
    inp[:, w + 1:] = np.where(right, right_color, background)
    out = np.where(~left & ~right, marker, background).astype(np.int8)
    if orientation == "horizontal":
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _source_pair(pair):
    inp, out = pair["input"], pair["output"]
    for src_in, src_out in _SOURCE_EXAMPLES:
        if inp.shape != (len(src_in), len(src_in[0])) or out.shape != (len(src_out), len(src_out[0])):
            continue
        if all("".join(str(int(v)) for v in row) == src for row, src in zip(inp, src_in)) and all(
            "".join(str(int(v)) for v in row) == src for row, src in zip(out, src_out)
        ):
            return True
    return False


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    colors = [int(c) for c in rng.choice(10, size=5, replace=False)]
    palette = str(rng.choice(["source", "fixed_background", "variable_background"], p=[0.22, 0.43, 0.35]))
    orientation_policy = str(rng.choice(["vertical", "horizontal", "mixed"], p=[0.56, 0.19, 0.25]))
    convention = {
        "marker": colors[4],
        "background": colors[0],
        "source_colors": (colors[0], colors[1], colors[2], colors[3], colors[4]),
        "palette": palette,
    }
    geometry = "source" if rng.random() < 0.16 else "variable"
    pairs = {"train": [], "test": []}
    records = []
    used_inputs = set()
    previous_size = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            prng = _rng_for_pair(rng, 0 if split == "train" else 1, index)
            if orientation_policy == "mixed":
                orientation = ("vertical", "horizontal")[index] if split == "train" and index < 2 else str(prng.choice(["vertical", "horizontal"]))
            else:
                orientation = orientation_policy
            mode = "teaching" if split == "train" and index == 0 else str(prng.choice(
                ["identical", "contained", "disjoint", "shifted", "free"] if split == "train"
                else ["identical", "contained", "disjoint", "shifted", "free", "complementary", "both_blank", "row_mix"]
                + ([] if palette == "variable_background" or split == "train" else ["both_full"])
            ))
            reused = None
            if split == "test" and mode == "row_mix":
                reused = records[int(prng.integers(0, len(records)))]
                h, w = reused["size"]
            else:
                h, w = _choose_size(prng, geometry, previous_size, harder=split == "test")
            previous_size = (h, w)
            left, right = _masks(prng, h, w, "free" if mode == "row_mix" else mode, split == "train")
            if reused is not None:
                rows = prng.choice(h, size=int(prng.integers(1, h)), replace=False)
                for row in rows:
                    left[row] = reused["left"][row]
                    right[row] = reused["right"][row]
            if palette == "variable_background" and mode != "both_blank":
                # The varying background is the unique color present in both
                # panels; retain at least one empty cell in each panel.
                if left.all():
                    left[0, 0] = False
                if right.all():
                    right[0, 0] = False
            pair_colors = _pair_colors(prng, convention, index, split == "train")
            pair = _build_pair(left, right, pair_colors, orientation)
            training_rows = {
                (record["size"][1], tuple(a), tuple(b))
                for record in records for a, b in zip(record["left"], record["right"])
            }

            def is_new_example():
                if _grid_key(pair["input"]) in used_inputs:
                    return False
                if split == "test" and index == 0:
                    return any(
                        (w, tuple(a), tuple(b)) not in training_rows
                        for a, b in zip(left, right)
                    )
                return True

            for attempt in range(24):
                if is_new_example():
                    break
                if attempt == 20:
                    # A new panel width guarantees a row unseen in training;
                    # the many legal shapes also give a distinct input grid.
                    trained_widths = {record["size"][1] for record in records}
                    candidates = [size for size in ((a, b) for a in range(3, 15) for b in range(2, 15))
                                  if size[1] not in trained_widths]
                    h, w = candidates[0]
                left, right = _masks(prng, h, w, "free", split == "train")
                if palette == "variable_background":
                    if left.all():
                        left[0, 0] = False
                    if right.all():
                        right[0, 0] = False
                pair = _build_pair(left, right, pair_colors, orientation)
            assert is_new_example()
            used_inputs.add(_grid_key(pair["input"]))
            pairs[split].append(pair)
            if split == "train":
                records.append({"size": (h, w), "left": left.copy(), "right": right.copy()})
    validate_generated_puzzle(pairs)
    return pairs


def _grid_key(grid):
    return (tuple(int(v) for v in grid.shape), grid.tobytes())


def _inspect_pair(pair, marker):
    inp, out = pair["input"], pair["output"]
    h, w = out.shape
    vertical = inp.shape == (h, 2 * w + 1)
    horizontal = inp.shape == (2 * h + 1, w)
    assert vertical != horizontal
    normalized = inp if vertical else inp.T
    normalized_out = out if vertical else out.T
    height, width = normalized_out.shape
    left, right = normalized[:, :width], normalized[:, width + 1:]
    line = normalized[:, width]
    assert len(set(int(c) for c in line)) == 1
    divider = int(line[0])
    assert divider not in left and divider not in right and divider != marker
    output_colors = set(int(c) for c in np.unique(normalized_out))
    assert len(output_colors) <= 2
    if output_colors == {marker}:
        common = set(int(c) for c in np.unique(left)) & set(int(c) for c in np.unique(right))
        assert len(common) == 1
        background = next(iter(common))
    else:
        others = output_colors - {marker}
        assert len(others) == 1
        background = next(iter(others))
    assert background != divider and background != marker
    for panel in (left, right):
        assert len(np.unique(panel)) <= 2
        assert len(set(int(c) for c in np.unique(panel)) - {background}) <= 1
        assert marker not in panel
    left_bits, right_bits = left != background, right != background
    expected = np.where(~left_bits & ~right_bits, marker, background)
    assert np.array_equal(normalized_out, expected)
    assert not np.array_equal(inp, out)
    return {
        "orientation": "vertical" if vertical else "horizontal",
        "size": (height, width),
        "background": background,
        "divider": divider,
        "left": left_bits,
        "right": right_bits,
        "left_colors": set(int(c) for c in np.unique(left)),
        "right_colors": set(int(c) for c in np.unique(right)),
        "output_colors": output_colors,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    inputs, pair_keys = set(), set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert bool(np.all((0 <= grid) & (grid <= 9)))
        key = _grid_key(pair["input"])
        pair_key = (key, _grid_key(pair["output"]))
        assert key not in inputs and pair_key not in pair_keys
        inputs.add(key)
        pair_keys.add(pair_key)
        assert not _source_pair(pair)
    input_colors = set(int(c) for pair in pairs for c in np.unique(pair["input"]))
    train_output_colors = set(int(c) for pair in puzzle["train"] for c in np.unique(pair["output"]))
    marker_candidates = train_output_colors - input_colors
    assert len(marker_candidates) == 1
    marker = next(iter(marker_candidates))
    infos = [_inspect_pair(pair, marker) for pair in pairs]
    train_infos = infos[:len(puzzle["train"])]
    test_infos = infos[len(puzzle["train"]):]
    for info in train_infos:
        assert info["output_colors"] == {marker, info["background"]}
    backgrounds = {info["background"] for info in train_infos}
    orientations = {info["orientation"] for info in train_infos}
    if len(backgrounds) > 1:
        for info in infos:
            assert info["background"] in info["left_colors"] & info["right_colors"]
            assert info["left_colors"] & info["right_colors"] == {info["background"]}
    else:
        assert all(info["background"] in backgrounds for info in infos)
    assert all(info["orientation"] in orientations for info in test_infos)
    states = set()
    for info in train_infos:
        states.update(zip(info["left"].flat, info["right"].flat))
    assert states == {(False, False), (False, True), (True, False), (True, True)}
    for info in test_infos:
        assert 2 <= info["size"][0] <= 14 and 2 <= info["size"][1] <= 14
    for info in infos:
        assert 2 <= info["size"][0] <= 14 and 2 <= info["size"][1] <= 14


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        ) for split in ("train", "test")
    )


def _has_enclosed_hole(mask):
    h, w = mask.shape
    seen = set()
    queue = [(y, x) for y in range(h) for x in range(w)
             if not mask[y, x] and (y in (0, h - 1) or x in (0, w - 1))]
    while queue:
        y, x = queue.pop()
        if (y, x) in seen:
            continue
        seen.add((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and (ny, nx) not in seen:
                queue.append((ny, nx))
    return any(not mask[y, x] and (y, x) not in seen for y in range(h) for x in range(w))


def _is_repeated_tile(mask):
    h, w = mask.shape
    if mask.all() or not mask.any():
        return False
    for th in range(1, min(5, h // 2 + 1)):
        for tw in range(1, min(5, w // 2 + 1)):
            tiled = np.tile(mask[:th, :tw], ((h + th - 1) // th, (w + tw - 1) // tw))[:h, :w]
            if np.array_equal(mask, tiled):
                return True
    return False


def _is_irregular(mask):
    points = np.argwhere(mask)
    if len(points) < 5 or _is_repeated_tile(mask):
        return False
    box = mask[points[:, 0].min():points[:, 0].max() + 1,
               points[:, 1].min():points[:, 1].max() + 1]
    return (0.3 < float(box.mean()) < 0.85
            and len(set(box.sum(axis=0))) >= 3
            and len(set(box.sum(axis=1))) >= 3)


def _has_old_anchor_recipe(left, right):
    width = left.shape[1]
    for first, second in ((left, right), (right, left)):
        first_counts = first.sum(axis=1)
        second_counts = second.sum(axis=1)
        partial = (second_counts > 0) & (second_counts < width)
        if np.any((first_counts == 0) & partial) and np.any((first_counts == width) & partial):
            return True
    return False


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, palettes, layouts, structures = set(), set(), set(), set()
    orientations, relationships = set(), set()
    novel_tests = blank_rows = full_rows = repeated_rows = 0
    empty_outputs = full_outputs = 0
    varied_sizes = varied_palettes = 0
    varied_backgrounds = varied_dividers = mixed_orientations = 0
    source_sizes = matching_foregrounds = reused_tests = novel_puzzles = 0
    fewer_states = compact_blocks = 0
    enclosed_holes = repeated_tiles = irregular_masks = 0
    sparse_panels = dense_panels = without_anchor_recipe = 0
    marker_colors = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pairs = puzzle["train"] + puzzle["test"]
        input_colors = set(int(c) for pair in pairs for c in np.unique(pair["input"]))
        marker = next(iter(set(int(c) for pair in puzzle["train"] for c in np.unique(pair["output"])) - input_colors))
        marker_colors.add(marker)
        infos = [_inspect_pair(pair, marker) for pair in pairs]
        varied_backgrounds += len({info["background"] for info in infos}) > 1
        varied_dividers += len({info["divider"] for info in infos}) > 1
        mixed_orientations += len({info["orientation"] for info in infos}) > 1
        if len({info["size"] for info in infos}) > 1:
            varied_sizes += 1
        if len({info["background"] for info in infos}) > 1 or len({info["divider"] for info in infos}) > 1:
            varied_palettes += 1
        train_rows = set()
        for info in infos[:len(puzzle["train"])]:
            for a, b in zip(info["left"], info["right"]):
                train_rows.add((len(a), tuple(a), tuple(b)))
        puzzle_has_new_test_row = False
        for pair_index, (pair, info) in enumerate(zip(pairs, infos)):
            h, w = info["size"]
            sizes.add((h, w))
            source_sizes += (h, w) == (4, 3)
            orientations.add(info["orientation"])
            palettes.add(tuple(sorted(int(c) for c in np.unique(pair["input"]))))
            layout = (info["orientation"], (h, w), tuple(info["left"].flat), tuple(info["right"].flat))
            layouts.add(layout)
            structures.add((h, w, tuple(info["left"].sum(axis=1)), tuple(info["right"].sum(axis=1))))
            left, right = info["left"], info["right"]
            enclosed_holes += any(_has_enclosed_hole(mask) for mask in (left, right))
            repeated_tiles += any(_is_repeated_tile(mask) for mask in (left, right))
            irregular_masks += any(_is_irregular(mask) for mask in (left, right))
            sparse_panels += any(mask.mean() <= 0.1 for mask in (left, right))
            dense_panels += any(mask.mean() >= 0.9 for mask in (left, right))
            if pair_index < len(puzzle["train"]):
                without_anchor_recipe += not _has_old_anchor_recipe(left, right)
            fewer_states += len(set(zip(left.flat, right.flat))) < 4
            matching_foregrounds += bool(
                (info["left_colors"] - {info["background"]})
                and (info["left_colors"] - {info["background"]})
                == (info["right_colors"] - {info["background"]})
            )
            compact_blocks += any(
                np.all(mask[y:y + 2, x:x + 2])
                for mask in (left, right)
                for y in range(h - 1) for x in range(w - 1)
            )
            if np.array_equal(left, right):
                relationships.add("identical")
            if np.array_equal(left, ~right):
                relationships.add("complementary")
            if not np.any(left & right):
                relationships.add("disjoint")
            if np.all(right <= left) or np.all(left <= right):
                relationships.add("contained")
            if any(not a.any() and not b.any() for a, b in zip(left, right)):
                blank_rows += 1
            if any(a.all() and b.all() for a, b in zip(left, right)):
                full_rows += 1
            rows = [(tuple(a), tuple(b)) for a, b in zip(left, right)]
            if len(rows) > len(set(rows)):
                repeated_rows += 1
            if pair_index >= len(puzzle["train"]):
                if any((len(a), tuple(a), tuple(b)) not in train_rows for a, b in zip(left, right)):
                    novel_tests += 1
                    puzzle_has_new_test_row = True
                if any((len(a), tuple(a), tuple(b)) in train_rows for a, b in zip(left, right)):
                    reused_tests += 1
                if np.all(pair["output"] == info["background"]):
                    empty_outputs += 1
                if np.all(pair["output"] == marker):
                    full_outputs += 1
            expected = np.where(~left & ~right, marker, info["background"])
            if info["orientation"] == "horizontal":
                expected = expected.T
            assert np.array_equal(pair["output"], expected)
        novel_puzzles += puzzle_has_new_test_row
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 30 and len(palettes) >= 30
    assert len(layouts) >= 180 and len(structures) >= 120
    assert orientations == {"vertical", "horizontal"}
    assert {"identical", "complementary", "disjoint", "contained"} <= relationships
    assert varied_sizes >= 25 and varied_palettes >= 20
    assert varied_backgrounds >= 5 and varied_dividers >= 10 and mixed_orientations >= 3
    assert source_sizes >= 10 and matching_foregrounds >= 5 and reused_tests >= 4
    assert novel_puzzles == 42 and fewer_states >= 20 and compact_blocks >= 30
    assert enclosed_holes >= 5 and repeated_tiles >= 2 and irregular_masks >= 10
    assert sparse_panels >= 10 and dense_panels >= 5 and without_anchor_recipe >= 30
    assert len(marker_colors) >= 8
    assert novel_tests >= 25 and blank_rows >= 5 and full_rows >= 2 and repeated_rows >= 5, (
        novel_tests, blank_rows, full_rows, repeated_rows
    )
    assert empty_outputs >= 2 and full_outputs >= 2, (empty_outputs, full_outputs)
    assert _puzzles_equal(generate_puzzle(987654, 7, 3), generate_puzzle(987654, 7, 3))
    assert not _puzzles_equal(generate_puzzle(0), generate_puzzle(1))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + ntrain * 10 + ntest, ntrain, ntest)
        assert len(puzzle["train"]) == ntrain and len(puzzle["test"]) == ntest
    for ntrain, ntest in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (ntrain or 0) + (ntest or 0), ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
