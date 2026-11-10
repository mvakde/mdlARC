import numpy as np


_SOURCE_PAIRS = (
    (("000808000", "000888000", "000880000", "000004000", "000444000", "000040000"),
     ("000808808", "000888888", "000880088", "000004000", "000444000", "000040000")),
    (("000808000", "000088000", "000008000", "000400000", "000444000", "000040000"),
     ("808808000", "880088000", "800008000", "000400000", "000444000", "000040000")),
    (("000800000", "000088000", "000800000", "000400000", "000444000", "000040000"),
     ("008800000", "880088000", "008800000", "000400000", "000444000", "000040000")),
    (("000808000", "000088000", "000800000", "000004000", "000444000", "000040000"),
     ("000808808", "000088880", "000800008", "000004000", "000444000", "000040000")),
)
_SOURCE_KEYS = {
    (tuple(tuple(int(c) for c in row) for row in before),
     tuple(tuple(int(c) for c in row) for row in after))
    for before, after in _SOURCE_PAIRS
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    n = _count_value("num_train", num_train, 2, 10)
    m = _count_value("num_test", num_test, 1, 4)
    if n is None and m is None:
        m = 1 + seed % 4
        n = m + 1 + (seed // 4) % (10 - m)
    elif n is None:
        n = int(rng.integers(m + 1, 11))
    elif m is None:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    return n, m


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_copy(pair):
    return (tuple(map(tuple, pair["input"].tolist())),
            tuple(map(tuple, pair["output"].tolist()))) in _SOURCE_KEYS


def _cue(size, style, direction, background, color):
    cue = np.full((size, size), background, dtype=np.int8)
    center = [size // 2] if size % 2 else [size // 2 - 1, size // 2]
    if style == 0:
        cue[size - 2, :] = color
        cue[size - 1, center] = color
    elif style == 1:
        cue[1, :] = color
        cue[2:, center] = color
    elif style == 2:
        cue[size - 2, :] = color
        cue[size - 1, 0] = color
        cue[size - 1, -1] = color
    else:
        raise ValueError("unknown cue style")
    cue[0, 0 if direction == "left" else size - 1] = color
    return cue


def _new_mask(rng, size, kind, asymmetric=False):
    for _ in range(80):
        mask = np.zeros((size, size), dtype=bool)
        if kind == "full":
            mask[:] = True
        elif kind == "symmetric":
            half = (size + 1) // 2
            core = rng.random((size, half)) < float(rng.uniform(.25, .8))
            mask[:, :half] = core
            mask[:, size // 2 + 1 if size % 2 else half:] = core[:, :size // 2][:, ::-1]
        elif kind == "dense_holes":
            mask[:] = True
            interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
            count = int(rng.integers(1, min(len(interior), max(2, size)) + 1))
            for i in rng.choice(len(interior), size=count, replace=False):
                mask[interior[int(i)]] = False
            if rng.random() < .6:
                mask[int(rng.integers(size)), int(rng.integers(size))] = False
        elif kind == "connected":
            count = int(rng.integers(max(3, size), min(size * size - 1, 2 * size + 5) + 1))
            r, c = int(rng.integers(size)), int(rng.integers(size))
            mask[r, c] = True
            for _ in range(count - 1):
                border = set()
                for rr, cc in zip(*np.where(mask)):
                    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        nr, nc = int(rr) + dr, int(cc) + dc
                        if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                            border.add((nr, nc))
                choice = sorted(border)[int(rng.integers(len(border)))]
                mask[choice] = True
        else:
            if kind == "sparse":
                count = int(rng.integers(1, min(size + 1, size * size - 1) + 1))
            else:
                density = float(rng.uniform(.23, .78))
                count = max(2, min(size * size - 1, int(round(density * size * size))))
            mask.flat[rng.choice(size * size, size=count, replace=False)] = True
        if not mask.any():
            continue
        if asymmetric and np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask
    # A bounded fallback for the very small symmetric vocabulary.
    mask = np.zeros((size, size), dtype=bool)
    mask[0, 0] = True
    mask[size - 1, size - 1] = True
    if not asymmetric:
        mask[0, size - 1] = True
        mask[size - 1, 0] = True
    return mask


def _source_panel(rng, mask, background, payload_colors):
    source = np.full(mask.shape, background, dtype=np.int8)
    positions = rng.permutation(np.flatnonzero(mask))
    if len(payload_colors) == 1 or len(positions) < 2:
        source.flat[positions] = payload_colors[0]
    else:
        source.flat[positions[0]] = payload_colors[0]
        source.flat[positions[1]] = payload_colors[1]
        for position in positions[2:]:
            source.flat[position] = payload_colors[int(rng.integers(len(payload_colors)))]
    return source


def _palette(rng, multicolor, swap=None):
    if swap is None:
        colors = [int(v) for v in rng.choice(10, size=4 if multicolor else 3, replace=False)]
        background, payload, cue_color = colors[:3]
        payload_colors = [payload] + (colors[3:4] if multicolor else [])
    else:
        old_background, old_payload, old_cue = swap
        background = int(rng.choice([c for c in range(10) if c not in (old_payload, old_cue)]))
        payload_colors = [old_cue]
        cue_color = old_payload
        if multicolor:
            payload_colors.append(int(rng.choice([c for c in range(10) if c not in (background, old_payload, old_cue)])))
    return background, payload_colors, cue_color


def _make_pair(source, direction, style, palette, margins):
    background, _, cue_color = palette
    size = source.shape[0]
    top, left, bottom, right = margins
    height, width = top + 2 * size + bottom, left + 3 * size + right
    grid = np.full((height, width), background, dtype=np.int8)
    x = left + size
    grid[top:top + size, x:x + size] = source
    grid[top + size:top + 2 * size, x:x + size] = _cue(size, style, direction, background, cue_color)
    result = grid.copy()
    dest_x = x - size if direction == "left" else x + size
    result[top:top + size, dest_x:dest_x + size] = source[:, ::-1]
    return {"input": grid, "output": result}


def _choose_layout(rng, style, split):
    size = int(rng.integers(3 if style != 1 else 4, 7 if split == "train" else 8))
    limit = 3 if split == "train" else 4
    margins = tuple(int(v) for v in rng.integers(0, limit + 1, size=4))
    return size, margins


def _analyze_pair(pair):
    before, after = pair["input"], pair["output"]
    height, width = before.shape
    background = int(np.bincount(before.ravel().astype(int), minlength=10).argmax())
    results = {}
    for size in range(3, 8):
        for left in range(5):
            right = width - 3 * size - left
            if not 0 <= right <= 4:
                continue
            x = left + size
            for top in range(5):
                bottom = height - 2 * size - top
                if not 0 <= bottom <= 4:
                    continue
                source = before[top:top + size, x:x + size]
                cue = before[top + size:top + 2 * size, x:x + size]
                source_colors = set(int(c) for c in np.unique(source)) - {background}
                cue_colors = set(int(c) for c in np.unique(cue)) - {background}
                if not source_colors or len(source_colors) > 3 or len(cue_colors) != 1:
                    continue
                cue_color = next(iter(cue_colors))
                if cue_color in source_colors:
                    continue
                outside = before.copy()
                outside[top:top + size, x:x + size] = background
                outside[top + size:top + 2 * size, x:x + size] = background
                if not np.all(outside == background):
                    continue
                for direction in ("left", "right"):
                    dest_x = x - size if direction == "left" else x + size
                    expected = before.copy()
                    expected[top:top + size, dest_x:dest_x + size] = source[:, ::-1]
                    if not np.array_equal(after, expected) or np.array_equal(before, after):
                        continue
                    for style in range(3):
                        if np.array_equal(cue, _cue(size, style, direction, background, cue_color)):
                            key = (size, top, left, bottom, right, direction, background, cue_color)
                            if key not in results:
                                results[key] = {"size": size, "top": top, "left": left,
                                                "bottom": bottom, "right": right,
                                                "direction": direction, "background": background,
                                                "cue_color": cue_color, "source": source.copy(),
                                                "styles": set()}
                            results[key]["styles"].add(style)
    _require(len(results) == 1, "Pair must have one clear source, cue, and reflected destination")
    return next(iter(results.values()))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, rng, num_train, num_test)
    style = int(rng.integers(3))  # One cue grammar for this puzzle.
    left_count = int(rng.integers(1, num_train))
    train_directions = ["left"] * left_count + ["right"] * (num_train - left_count)
    rng.shuffle(train_directions)
    # Put opposite directions first, with randomized anchor order.
    first_direction = "left" if rng.random() < .5 else "right"
    first_index = train_directions.index(first_direction)
    train_directions[0], train_directions[first_index] = train_directions[first_index], train_directions[0]
    second_direction = "right" if first_direction == "left" else "left"
    second_index = train_directions.index(second_direction)
    train_directions[1], train_directions[second_index] = train_directions[second_index], train_directions[1]
    if num_test > 1 and rng.random() < .65:
        test_directions = ["left", "right"] + ["left" if rng.random() < .5 else "right" for _ in range(num_test - 2)]
        rng.shuffle(test_directions)
    else:
        test_directions = ["left" if rng.random() < .5 else "right" for _ in range(num_test)]
    repeat_anchors = num_train >= 3 and rng.random() < .32
    train_multi = rng.random() < .7
    pairs = {"train": [], "test": []}
    used_inputs = set()
    first_context = None
    first_palette = None
    training_rows = {}
    for split, directions in (("train", train_directions), ("test", test_directions)):
        for index, direction in enumerate(directions):
            if split == "train" and index == 1 and repeat_anchors:
                size, margins, source, palette = first_context
                pair = _make_pair(source, direction, style, palette, margins)
            else:
                for _ in range(120):
                    size, margins = _choose_layout(rng, style, split)
                    if split == "train" and index < 2:
                        kind = "random"
                        asymmetric = True
                    else:
                        kinds = ("random", "sparse", "dense_holes", "connected", "symmetric", "full")
                        weights = (.30, .08, .25, .18, .12, .07) if split == "test" else (.34, .17, .17, .16, .11, .05)
                        kind = str(rng.choice(kinds, p=weights))
                        asymmetric = False
                    if split == "test" and rng.random() < .20 and training_rows:
                        kind = "familiar"
                        size = int(rng.choice(sorted(training_rows)))
                        vocabulary = sorted(training_rows[size])
                        mask = np.asarray(
                            [vocabulary[int(rng.integers(len(vocabulary)))] for _ in range(size)],
                            dtype=bool,
                        )
                        if not mask.any():
                            continue
                    else:
                        mask = _new_mask(rng, size, kind, asymmetric)
                    if split == "train" and index < 2:
                        asymmetric_rows = sum(not np.array_equal(row, row[::-1]) for row in mask)
                        if int(mask.sum()) < max(3, size) or asymmetric_rows < 2:
                            continue
                    multi = (train_multi and index == 0) if split == "train" and index < 2 else (train_multi and rng.random() < (.5 if split == "test" else .4))
                    if split == "train" and index == 1 and not repeat_anchors:
                        palette = _palette(rng, multi, swap=first_palette)
                    elif split == "train" and index == 2 and repeat_anchors:
                        palette = _palette(rng, multi, swap=first_palette)
                    else:
                        palette = _palette(rng, multi)
                    source = _source_panel(rng, mask, palette[0], palette[1])
                    if split == "test" and kind == "random" and rng.random() < .65:
                        old_rows = training_rows.get(size, set())
                        if old_rows and all(tuple(row.tolist()) in old_rows for row in mask):
                            continue
                    pair = _make_pair(source, direction, style, palette, margins)
                    if _grid_key(pair["input"]) not in used_inputs and not _source_copy(pair):
                        break
                else:
                    raise RuntimeError("Unable to construct a unique pair")
                if split == "train" and index == 0:
                    first_context = (size, margins, source.copy(), palette)
                    first_palette = (palette[0], palette[1][0], palette[2])
            if _grid_key(pair["input"]) in used_inputs or _source_copy(pair):
                raise RuntimeError("Repeated anchor input or source pair")
            used_inputs.add(_grid_key(pair["input"]))
            pairs[split].append(pair)
            if split == "train":
                training_rows.setdefault(size, set()).update(
                    tuple(row.tolist()) for row in source != palette[0]
                )
    validate_generated_puzzle(pairs)
    return pairs


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "Pair counts")
    input_keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid must be a 2D NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape), "Grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid dtype")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "ARC colors")
            key = _pair_key(pair)
            _require(key[0] not in input_keys and key not in pair_keys, "Duplicate input or pair")
            input_keys.add(key[0]); pair_keys.add(key)
            _require(not _source_copy(pair), "Source task copy")
            analyses[split].append(_analyze_pair(pair))
            _require(np.array_equal(_predict_from_input(pair["input"]), pair["output"]),
                     "Input-only cue reading must determine the output")
    common_styles = set.intersection(*(a["styles"] for split in ("train", "test") for a in analyses[split]))
    _require(bool(common_styles), "Cue grammar must be consistent")
    for direction in ("left", "right"):
        _require(any(a["direction"] == direction and not np.array_equal(a["source"], a["source"][:, ::-1])
                     for a in analyses["train"]), "Both cue directions need asymmetric training evidence")
    multi_test = any(len(set(int(c) for c in np.unique(a["source"])) - {a["background"]}) > 1 for a in analyses["test"])
    if multi_test:
        _require(any(len(set(int(c) for c in np.unique(a["source"])) - {a["background"]}) > 1
                     for a in analyses["train"]), "Multicolor test needs a training demonstration")
    for a in analyses["test"]:
        _require(a["size"] >= 3 and a["direction"] in ("left", "right"), "Test regime")


def _predict_from_input(input_grid):
    """Read the cue geometry from an input, without using any output or metadata."""
    height, width = input_grid.shape
    background = int(np.bincount(input_grid.ravel().astype(int), minlength=10).argmax())
    predictions = []
    for size in range(3, 8):
        for left in range(5):
            if not 0 <= width - 3 * size - left <= 4:
                continue
            x = left + size
            for top in range(5):
                if not 0 <= height - 2 * size - top <= 4:
                    continue
                source = input_grid[top:top + size, x:x + size]
                cue = input_grid[top + size:top + 2 * size, x:x + size]
                source_colors = set(int(c) for c in np.unique(source)) - {background}
                cue_colors = set(int(c) for c in np.unique(cue)) - {background}
                if not source_colors or len(cue_colors) != 1 or source_colors & cue_colors:
                    continue
                cue_mask = cue == next(iter(cue_colors))
                if cue_mask[0].sum() != 1 or not (cue_mask[0, 0] or cue_mask[0, -1]):
                    continue
                if not np.array_equal(cue_mask[1:], cue_mask[1:, ::-1]):
                    continue
                if not any(cue_mask[row].all() for row in range(1, size)):
                    continue
                outside = input_grid.copy()
                outside[top:top + size, x:x + size] = background
                outside[top + size:top + 2 * size, x:x + size] = background
                if not np.all(outside == background):
                    continue
                dest_x = x - size if cue_mask[0, 0] else x + size
                prediction = input_grid.copy()
                prediction[top:top + size, dest_x:dest_x + size] = source[:, ::-1]
                predictions.append(prediction)
    _require(len(predictions) == 1, "Input must independently determine one reflection")
    return predictions[0]


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "Train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "Test count diversity")
    _require(len({tuple(_pair_key(q) for split in ("train", "test") for q in p[split]) for p in puzzles}) == 42,
             "Cross-seed puzzle diversity")
    all_analyses = []
    measures = {name: 0 for name in (
        "mixed_tests", "repeat_sources", "swapped_roles", "varied_sizes",
        "varied_backgrounds", "left_majority", "right_majority", "equal_balance",
        "left_tests", "right_tests", "novel_test_rows", "familiar_test_rows",
        "multicolor_tests", "symmetric_tests", "sparse_tests", "full_tests",
        "complex_tests", "connected_tests", "disconnected_tests", "combinations",
    )}
    observed_styles = set()
    distinct_masks = set()
    distinct_layouts = set()

    def component_count(mask):
        seen = set()
        count = 0
        for row, col in zip(*np.where(mask)):
            start = (int(row), int(col))
            if start in seen:
                continue
            count += 1
            frontier = [start]
            seen.add(start)
            while frontier:
                r, c = frontier.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (r + dr, c + dc)
                    if (0 <= neighbor[0] < mask.shape[0]
                            and 0 <= neighbor[1] < mask.shape[1]
                            and mask[neighbor] and neighbor not in seen):
                        seen.add(neighbor)
                        frontier.append(neighbor)
        return count

    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                _require(np.array_equal(_predict_from_input(pair["input"]), pair["output"]),
                         "Input-only transformation prediction mismatch")
        train = [_analyze_pair(q) for q in p["train"]]
        test = [_analyze_pair(q) for q in p["test"]]
        all_analyses.extend(train + test)
        measures["mixed_tests"] += len({a["direction"] for a in test}) == 2
        measures["varied_sizes"] += len({a["size"] for a in train + test}) > 1
        measures["varied_backgrounds"] += len({a["background"] for a in train + test}) > 1
        left_train = sum(a["direction"] == "left" for a in train)
        balance = "left_majority" if left_train * 2 > len(train) else "right_majority" if left_train * 2 < len(train) else "equal_balance"
        measures[balance] += 1
        measures["repeat_sources"] += any(np.array_equal(a["source"], b["source"]) and a["direction"] != b["direction"]
                                        for i, a in enumerate(train) for b in train[i + 1:])
        measures["swapped_roles"] += any(a["cue_color"] in (set(int(c) for c in np.unique(b["source"])) - {b["background"]})
                             and b["cue_color"] in (set(int(c) for c in np.unique(a["source"])) - {a["background"]})
                             for i, a in enumerate(train) for b in train[i + 1:])
        common_styles = set.intersection(*(a["styles"] for a in train + test))
        observed_styles.update(common_styles)
        training_rows = {}
        for a in train:
            training_rows.setdefault(a["size"], set()).update(
                tuple(row.tolist()) for row in (a["source"] != a["background"])
            )
        for a in train + test:
            mask = a["source"] != a["background"]
            distinct_masks.add((a["size"], tuple(mask.ravel().tolist())))
            distinct_layouts.add((a["size"], a["top"], a["left"], a["bottom"], a["right"], a["direction"]))
        for a in test:
            mask = a["source"] != a["background"]
            colors = set(int(c) for c in np.unique(a["source"])) - {a["background"]}
            rows = {tuple(row.tolist()) for row in mask}
            old_rows = training_rows.get(a["size"], set())
            measures["left_tests" if a["direction"] == "left" else "right_tests"] += 1
            measures["novel_test_rows"] += any(row not in old_rows for row in rows)
            measures["familiar_test_rows"] += rows.issubset(old_rows)
            measures["multicolor_tests"] += len(colors) > 1
            measures["symmetric_tests"] += np.array_equal(a["source"], a["source"][:, ::-1])
            measures["sparse_tests"] += int(mask.sum()) <= 2
            measures["full_tests"] += bool(mask.all())
            asymmetric_rows = sum(not np.array_equal(row, row[::-1]) for row in a["source"])
            measures["complex_tests"] += int(mask.sum()) >= 5 and asymmetric_rows >= 3 and not mask[1:-1, 1:-1].all()
            components = component_count(mask)
            measures["connected_tests" if components == 1 else "disconnected_tests"] += 1
            measures["combinations"] += (len(colors) > 1 and a["direction"] == "left"
                                         and a["size"] >= 5 and (a["top"] > 0 or a["left"] > 0))
    floors = {"mixed_tests": 3, "repeat_sources": 2, "swapped_roles": 10,
              "varied_sizes": 35, "varied_backgrounds": 35,
              "left_majority": 5, "right_majority": 5, "equal_balance": 2,
              "left_tests": 20, "right_tests": 20, "novel_test_rows": 25,
              "familiar_test_rows": 5, "multicolor_tests": 10,
              "symmetric_tests": 5, "sparse_tests": 1, "full_tests": 1,
              "complex_tests": 20, "connected_tests": 10,
              "disconnected_tests": 10, "combinations": 5}
    for name, minimum in floors.items():
        _require(measures[name] >= minimum, f"Insufficient {name}: {measures[name]}")
    _require(observed_styles == {0, 1, 2}, "Cue-body diversity")
    _require(len(distinct_masks) >= 200 and len(distinct_layouts) >= 200, "Source and layout diversity")
    _require(len({a["size"] for a in all_analyses}) >= 5, "Panel-size diversity")
    _require(len({(a["top"], a["left"], a["bottom"], a["right"]) for a in all_analyses}) >= 30, "Placement diversity")
    _require(len({a["background"] for a in all_analyses}) >= 8, "Background diversity")
    _require(any(np.all(a["source"] != a["background"]) for a in all_analyses), "Full-panel examples")
    _require(any(np.array_equal(a["source"], a["source"][:, ::-1]) for a in all_analyses), "Symmetric examples")
    _require(any(len(set(int(c) for c in np.unique(a["source"])) - {a["background"]}) > 1 for a in all_analyses), "Multicolor examples")
    for i, (n, m) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        p = generate_puzzle(1000 + i, n, m)
        _require((len(p["train"]), len(p["test"])) == (n, m), "Explicit counts")
    for i, (n, m) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, n, m)
        _require((n is None or len(p["train"]) == n) and (m is None or len(p["test"]) == m), "Partial counts")
    for i, (n, m) in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4))):
        try:
            generate_puzzle(3000 + i, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    p = generate_puzzle(12345, 7, 3)
    q = generate_puzzle(12345, 7, 3)
    _require(all(np.array_equal(a[k], b[k]) for split in ("train", "test") for a, b in zip(p[split], q[split]) for k in ("input", "output")), "Determinism")
