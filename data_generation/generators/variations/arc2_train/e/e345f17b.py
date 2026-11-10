"""Two aligned panels: mark the cells empty in both panels."""

import numpy as np


_SOURCE_ROWS = (
    (("60600055", "06665055", "66065055", "66005000"), ("0400", "0000", "0000", "0044")),
    (("06605550", "06065005", "06665555", "60000505"), ("0004", "0040", "0000", "0040")),
    (("66605055", "60000555", "60000000", "06665500"), ("0000", "0000", "0444", "0000")),
    (("66665050", "60005500", "60660055", "00600550"), ("0000", "0044", "0400", "4004")),
    (("06000505", "06000005", "60005500", "66060005"), ("4040", "4040", "0044", "0040")),
    (("60665005", "00065555", "06605505", "66005550"), ("0400", "0000", "0000", "0004")),
)


def _rows_array(rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_KEYS = frozenset(
    ((tuple(_rows_array(i).shape), _rows_array(i).tobytes()),
     (tuple(_rows_array(o).shape), _rows_array(o).tobytes()))
    for i, o in _SOURCE_ROWS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    array = grid.astype(np.int8, copy=False)
    return tuple(array.shape), array.tobytes()


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
    choices = [(a, b) for a, b in legal
               if (train is None or a == train) and (test is None or b == test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _candidate_panels(grid):
    """Find equal adjacent panels from the input colors alone."""
    height, width = grid.shape
    candidates = []
    if width % 2 == 0:
        candidates.append(("side", grid[:, :width // 2], grid[:, width // 2:]))
    if height % 2 == 0:
        candidates.append(("stack", grid[:height // 2], grid[height // 2:]))
    valid = []
    for direction, first, second in candidates:
        colors_a = set(int(x) for x in np.unique(first))
        colors_b = set(int(x) for x in np.unique(second))
        common = colors_a & colors_b
        if len(common) == 1:
            valid.append((direction, first, second, next(iter(common))))
    return valid


def _shape(rng):
    direction = "side" if rng.integers(2) == 0 else "stack"
    # The original 4 by 4 panels remain possible. Other shapes use parity
    # to make orientation apparent even when a panel is nearly blank.
    if direction == "side" and rng.random() < 0.11:
        return 4, 4, direction
    if direction == "side":
        return int(rng.choice((3, 5, 7, 9, 11))), int(rng.integers(3, 13)), direction
    return int(rng.integers(3, 13)), int(rng.choice((3, 5, 7, 9, 11))), direction


def _empty_count(rng, area, regime, first_train):
    if first_train:
        return int(rng.integers(2, max(3, area // 3 + 1)))
    choice = int(rng.integers(10))
    if regime == "target_test" and choice == 0:
        return 0
    if choice == 1:
        return 1
    if choice == 2:
        return int(rng.integers(max(2, area * 2 // 3), max(3, area * 9 // 10)))
    return int(rng.integers(2, max(3, area * 3 // 5 + 1)))


def _empty_mask(rng, height, width, count, style_hint):
    if count == 0:
        return np.zeros((height, width), dtype=bool)
    rr, cc = np.indices((height, width))
    mode = int(rng.integers(6)) if style_hint is None else style_hint
    if mode == 0:  # scattered
        score = rng.random((height, width))
    elif mode == 1:  # a line, possibly with a second nearby row/column
        if rng.integers(2):
            score = abs(rr - int(rng.integers(height))) + rng.random((height, width)) * .25
        else:
            score = abs(cc - int(rng.integers(width))) + rng.random((height, width)) * .25
    elif mode == 2:  # corner growth
        r0 = int(rng.choice((0, height - 1)))
        c0 = int(rng.choice((0, width - 1)))
        score = abs(rr - r0) + abs(cc - c0) + rng.random((height, width)) * .5
    elif mode == 3:  # compact patch
        r0, c0 = int(rng.integers(height)), int(rng.integers(width))
        score = np.maximum(abs(rr - r0), abs(cc - c0)) + rng.random((height, width)) * .4
    elif mode == 4:  # border
        score = np.minimum.reduce((rr, cc, height - 1 - rr, width - 1 - cc))
        score = score + rng.random((height, width)) * .7
    else:  # separated islands
        r0, c0 = int(rng.integers(max(1, height // 3))), int(rng.integers(max(1, width // 3)))
        r1 = int(rng.integers(max(0, height * 2 // 3), height))
        c1 = int(rng.integers(max(0, width * 2 // 3), width))
        score = np.minimum(abs(rr - r0) + abs(cc - c0), abs(rr - r1) + abs(cc - c1))
        score = score + rng.random((height, width)) * .3
    order = np.argsort(score.ravel(), kind="stable")
    mask = np.zeros(height * width, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(height, width)


def _masks(rng, empty, relation):
    height, width = empty.shape
    occupied = np.flatnonzero(~empty.ravel())
    size = len(occupied)
    rng.shuffle(occupied)
    if relation == "disjoint":
        categories = rng.choice((1, 2), size=size)
        categories[:2] = (1, 2)
    elif relation == "subset":
        one = int(rng.choice((1, 2)))
        categories = rng.choice((one, 3), size=size)
        categories[:2] = (one, 3)
    elif relation == "near_identical":
        categories = np.full(size, 3, dtype=np.int8)
        categories[:2] = (1, 2)
        if size > 8:
            categories[2:int(rng.integers(3, min(6, size)))] = 1
    elif relation == "unequal":
        if rng.integers(2):
            probabilities = (.76, .08, .16)
        else:
            probabilities = (.08, .76, .16)
        categories = rng.choice((1, 2, 3), size=size, p=probabilities)
        categories[:2] = (1, 2)
    elif relation == "blank":
        categories = np.full(size, int(rng.choice((1, 2))), dtype=np.int8)
    else:
        categories = rng.choice((1, 2, 3), size=size)
        categories[:min(3, size)] = (1, 2, 3)[:min(3, size)]
    left = np.zeros(height * width, dtype=bool)
    right = np.zeros(height * width, dtype=bool)
    left[occupied] = (categories == 1) | (categories == 3)
    right[occupied] = (categories == 2) | (categories == 3)
    return left.reshape(height, width), right.reshape(height, width)


def _colored_panel(rng, mask, background, colors, minimum_colors):
    panel = np.full(mask.shape, background, dtype=np.int8)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return panel
    maximum = min(3, count, len(colors))
    if maximum < minimum_colors:
        number = maximum
    elif maximum == minimum_colors:
        number = minimum_colors
    else:
        number = int(rng.integers(minimum_colors, maximum + 1))
    chosen = colors[:number]
    values = np.resize(np.asarray(chosen, dtype=np.int8), count)
    rng.shuffle(values)
    panel[mask] = values
    return panel


def _make_pair(rng, marker, background, regime, first_train, direction_hint,
               shape_hint, style_hint):
    height, width, direction = _shape(rng) if shape_hint is None else shape_hint
    if direction_hint is not None:
        direction = direction_hint
        if direction == "side" and height % 2 == 0 and (height, width) != (4, 4):
            height = int(rng.choice((3, 5, 7, 9, 11)))
        if direction == "stack" and width % 2 == 0:
            width = int(rng.choice((3, 5, 7, 9, 11)))
    area = height * width
    empty = _empty_mask(rng, height, width,
                        _empty_count(rng, area, regime, first_train), style_hint)
    relation = "balanced" if first_train else str(rng.choice(
        ("balanced", "disjoint", "subset", "near_identical", "unequal", "blank"),
        p=(.28, .16, .18, .16, .17, .05)))
    if not np.any(empty) and relation in ("subset", "blank"):
        relation = "disjoint"
    left_mask, right_mask = _masks(rng, empty, relation)
    available = [x for x in range(10) if x not in (marker, background)]
    rng.shuffle(available)
    left_count = int(np.count_nonzero(left_mask))
    right_count = int(np.count_nonzero(right_mask))
    left_min = 2 if first_train and left_count >= 2 else 1
    right_min = 2 if first_train and right_count >= 2 else 1
    left_colors = available[:min(3, max(1, left_count))]
    right_colors = available[3:3 + min(3, max(1, right_count))]
    left = _colored_panel(rng, left_mask, background, left_colors, left_min)
    right = _colored_panel(rng, right_mask, background, right_colors, right_min)
    input_grid = np.concatenate((left, right), axis=1 if direction == "side" else 0)
    output_grid = np.full((height, width), background, dtype=np.int8)
    output_grid[empty] = marker
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    marker = int(rng.integers(10))
    backgrounds = [x for x in range(10) if x != marker]
    rng.shuffle(backgrounds)
    # Only the marker is a hidden puzzle convention. Background and panel
    # foreground identities are visible local choices in each input.
    layout_mode = int(rng.choice((0, 1, 2), p=(.15, .25, .60)))
    fixed_shape = _shape(rng) if layout_mode == 0 else None
    uniform_direction = (fixed_shape[2] if fixed_shape is not None else
                         str(rng.choice(("side", "stack"))) if layout_mode == 1 else None)
    test_direction_change = layout_mode == 1 and rng.random() < .40
    pair_rngs = [np.random.default_rng(seq) for seq in
                 np.random.SeedSequence(int(rng.integers(2**63))).spawn(train_count + test_count)]
    used = set()
    puzzle = {"train": [], "test": []}
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            absolute = index if split == "train" else train_count + index
            if split == "train" and uniform_direction is None and index < 2:
                direction_hint = ("side", "stack")[index]
            elif split == "test" and test_direction_change and index == 0:
                direction_hint = "stack" if uniform_direction == "side" else "side"
            else:
                direction_hint = uniform_direction
            background = backgrounds[index] if split == "train" and index < 2 else int(rng.choice(backgrounds))
            for _ in range(120):
                pair = _make_pair(pair_rngs[absolute], marker, background,
                                  "source_train" if split == "train" else "target_test",
                                  split == "train" and index == 0, direction_hint,
                                  fixed_shape,
                                  int(pair_rngs[absolute].integers(1, 6)) if split == "train" and index == 0
                                  else 0 if split == "train" and index == 1 else None)
                key = (_key(pair["input"]), _key(pair["output"]))
                if key in _SOURCE_KEYS or key[0] in used or len(_candidate_panels(pair["input"])) != 1:
                    continue
                used.add(key[0])
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a unique unambiguous pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")), "pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    marker = None
    seen_inputs, seen_pairs = set(), set()
    categories_seen = set()
    training_multi = False
    test_multi = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            inp, out = pair["input"], pair["output"]
            key = (_key(inp), _key(out))
            _require(key[0] not in seen_inputs and key not in seen_pairs, "duplicate pair")
            _require(key not in _SOURCE_KEYS, "source pair copied")
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            candidates = _candidate_panels(inp)
            _require(len(candidates) == 1, "panel split is ambiguous")
            _, left, right, background = candidates[0]
            _require(out.shape == left.shape == right.shape, "output must match one panel")
            left_colors = set(int(x) for x in np.unique(left)) - {background}
            right_colors = set(int(x) for x in np.unique(right)) - {background}
            _require(not (left_colors & right_colors), "panel colors overlap")
            _require(background in set(int(x) for x in np.unique(out)), "output background")
            markers = set(int(x) for x in np.unique(out)) - {background}
            _require(len(markers) <= 1, "one output marker")
            if markers:
                this_marker = next(iter(markers))
                _require(this_marker not in set(int(x) for x in np.unique(inp)), "marker in input")
                if marker is None:
                    _require(split == "train", "marker missing from training")
                    marker = this_marker
                else:
                    _require(marker == this_marker, "marker changed")
            empty = (left == background) & (right == background)
            if marker is not None:
                expected = np.where(empty, marker, background)
                _require(np.array_equal(out, expected), "incorrect shared-empty output")
            else:
                _require(not np.any(empty), "unmarked shared empty cell")
            local_categories = {
                0 if a == background and b == background else
                1 if a != background and b == background else
                2 if a == background and b != background else 3
                for a, b in zip(left.flat, right.flat)
            }
            if split == "train":
                categories_seen |= local_categories
                training_multi |= len(left_colors) > 1 or len(right_colors) > 1
            else:
                test_multi |= len(left_colors) > 1 or len(right_colors) > 1
    _require(marker is not None, "training must establish the marker")
    _require(categories_seen == {0, 1, 2, 3}, "training must demonstrate all four occupancy cases")
    _require(not test_multi or training_multi, "test uses unseen multicolor occupancy")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            # Recompute the displayed rule directly from the returned arrays.
            inp, out = pair["input"], pair["output"]
            direction, a, b, bg = _candidate_panels(inp)[0]
            _require(direction in ("side", "stack"), "direction")
            marked = out != bg
            _require(np.array_equal(marked, (a == bg) & (b == bg)), "independent NOR check")
    for nt, ns in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "explicit counts")
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (nt or 0) + (ns or 0), nt, ns)
        _require(nt is None or len(puzzle["train"]) == nt, "partial train count")
        _require(ns is None or len(puzzle["test"]) == ns, "partial test count")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(3000, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(90210, 8, 3), generate_puzzle(90210, 8, 3)
    _require(all(_key(x[k]) == _key(y[k]) for s in ("train", "test")
                 for x, y in zip(a[s], b[s]) for k in ("input", "output")), "determinism")
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, directions, backgrounds, marker_counts = set(), set(), set(), set()
    profiles, structures, palettes = set(), set(), set()
    zero_test = multicolor = blank = no_overlap = subset = unequal = near_identical = 0
    line = block = border = islands = many_markers = 0
    changed_bg = mixed_training = changed_test_orientation = changed_test_size = 0
    for puzzle in puzzles:
        train_bgs, train_dirs, train_shapes = set(), set(), set()
        test_dirs, test_shapes = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                direction, a, b, bg = _candidate_panels(inp)[0]
                shapes.add(out.shape)
                directions.add(direction)
                backgrounds.add(bg)
                marker_counts.add(int(np.count_nonzero(out != bg)))
                left, right = a != bg, b != bg
                profile = (int(np.count_nonzero(~left & ~right)),
                           int(np.count_nonzero(left & ~right)),
                           int(np.count_nonzero(~left & right)),
                           int(np.count_nonzero(left & right)))
                profiles.add(profile)
                structures.add((inp.shape, tuple((inp == bg).flat)))
                palettes.add(tuple(sorted(int(c) for c in np.unique(inp))))
                multicolor += int(len(np.unique(a[a != bg])) > 1 or len(np.unique(b[b != bg])) > 1)
                blank += int(not np.any(left) or not np.any(right))
                no_overlap += int(not np.any(left & right))
                only_left = int(np.count_nonzero(left & ~right))
                only_right = int(np.count_nonzero(~left & right))
                overlap = int(np.count_nonzero(left & right))
                subset += int(only_left == 0 or only_right == 0)
                unequal += int(3 * min(np.count_nonzero(left), np.count_nonzero(right))
                               <= max(np.count_nonzero(left), np.count_nonzero(right)))
                near_identical += int(overlap > 0 and only_left + only_right
                                      <= max(2, int(.1 * np.count_nonzero(left | right))))
                empty = ~(left | right)
                number = int(np.count_nonzero(empty))
                many_markers += int(number >= .6 * empty.size)
                if number >= 2:
                    rows, cols = np.where(empty)
                    line += int(len(set(rows)) == 1 or len(set(cols)) == 1)
                    if number >= 3:
                        border += int(all(r in (0, empty.shape[0] - 1)
                                          or c in (0, empty.shape[1] - 1)
                                          for r, c in zip(rows, cols)))
                    if number >= 4:
                        block += int((rows.max() - rows.min() + 1)
                                     * (cols.max() - cols.min() + 1) == number)
                    islands += int(not np.any(empty[1:, :] & empty[:-1, :])
                                   and not np.any(empty[:, 1:] & empty[:, :-1]))
                if split == "train":
                    train_bgs.add(bg)
                    train_dirs.add(direction)
                    train_shapes.add(out.shape)
                else:
                    zero_test += int(not np.any(out != bg))
                    test_dirs.add(direction)
                    test_shapes.add(out.shape)
        changed_bg += int(len(train_bgs) > 1)
        mixed_training += int(len(train_dirs) > 1)
        changed_test_orientation += int(bool(test_dirs - train_dirs))
        changed_test_size += int(bool(test_shapes - train_shapes))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(shapes) >= 30 and len(directions) == 2, "geometry diversity")
    _require(len(backgrounds) >= 8 and changed_bg == 42, "local background variation")
    _require(len(marker_counts) >= 20 and zero_test > 0, "shared-empty count diversity")
    _require(len(profiles) >= 100 and len(structures) >= 150, "mask diversity")
    _require(len(palettes) >= 30 and multicolor > 50, "color variation")
    _require(blank > 0 and no_overlap > 0 and subset > 0 and unequal > 0
             and near_identical > 0, "relationship variation")
    _require(line > 0 and block > 0 and border > 0 and islands > 0,
             "empty-space geometry variation")
    _require(many_markers > 0 and mixed_training > 0 and changed_test_orientation > 0
             and changed_test_size > 0, "within-puzzle and train/test variation")
