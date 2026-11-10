"""Generate paired patterns that slide to opposite horizontal edges."""

import hashlib

import numpy as np


_SOURCE_HASHES = {
    "098af55f1d894eb1a05cef76148fcdee3cd95b2ede12debd40e735e5094d0d03",
    "f3bcb0cdb3c33a8878248d37be843405cd8d6893a52f74bea6ac0edde2e64d9b",
    "bd5416308a550fdcb996ee550341122f0caf645c0dcc21a6df365b0b4b7d1344",
    "6375292f1e62a28715bf063477cbb3ade295c5e9df7bb1f1b792350beb0dda22",
    "efb38d861634de2df14d49c244855c035d43b72b0bd90f3dfd20679fb4045c85",
}


def _counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _object_height(rng, canvas_height):
    if rng.random() < 0.20:
        return int(rng.integers(max(2, canvas_height - 3), canvas_height + 1))
    return int(rng.integers(2, min(canvas_height, 14) + 1))


def _vertical(rng, canvas_height, relation):
    if relation == "disjoint":
        h1 = int(rng.integers(2, max(2, (canvas_height - 1) // 2) + 1))
        h2 = int(rng.integers(2, canvas_height - h1))
        gap = int(rng.integers(1, canvas_height - h1 - h2 + 1))
        t1 = int(rng.integers(0, canvas_height - h1 - h2 - gap + 1))
        t2 = t1 + h1 + gap
        return (h2, h1, t2, t1) if rng.random() < 0.5 else (h1, h2, t1, t2)
    h1 = _object_height(rng, canvas_height)
    h2 = h1 if relation == "aligned" else _object_height(rng, canvas_height)
    if relation in ("common_top", "common_bottom") and h1 == h2:
        h2 = h1 - 1 if h1 == canvas_height else h1 + 1
    if relation == "aligned":
        top = int(rng.integers(0, canvas_height - h1 + 1))
        return h1, h2, top, top
    if relation == "common_top":
        top = int(rng.integers(0, canvas_height - max(h1, h2) + 1))
        return h1, h2, top, top
    if relation == "common_bottom":
        bottom = int(rng.integers(max(h1, h2), canvas_height + 1))
        return h1, h2, bottom - h1, bottom - h2
    for _ in range(32):
        t1 = int(rng.integers(0, canvas_height - h1 + 1))
        t2 = int(rng.integers(0, canvas_height - h2 + 1))
        overlap = min(t1 + h1, t2 + h2) - max(t1, t2)
        if overlap > 0 and t1 != t2 and t1 + h1 != t2 + h2:
            return h1, h2, t1, t2
    top = int(rng.integers(0, canvas_height - max(h1, h2) + 1))
    if h1 == h2:
        h2 = h1 - 1 if h1 == canvas_height else h1 + 1
    return h1, h2, top, top


def _mask(rng, height, width, force_gap):
    mask = np.zeros((height, width), dtype=bool)
    kind = int(rng.integers(7))
    if kind == 0:
        mask[:] = True
        if width >= 3 and height >= 3 and rng.random() < 0.7:
            mask[1:-1, 1:-1] = rng.random((height - 2, width - 2)) < 0.65
    elif kind == 1:
        rows = rng.random(height) < float(rng.uniform(0.35, 0.8))
        rows[0] = rows[-1] = True
        mask[rows, :] = True
        if width >= 3 and rng.random() < 0.6:
            mask[rows, 1:-1] &= rng.random((int(rows.sum()), width - 2)) < 0.8
    elif kind == 2:
        columns = rng.random(width) < float(rng.uniform(0.35, 0.8))
        columns[0] = columns[-1] = True
        mask[:, columns] = True
        if height >= 3 and rng.random() < 0.6:
            mask[1:-1, :] &= rng.random((height - 2, width)) < 0.8
    elif kind == 3:
        mask[int(rng.integers(height)), :] = True
        stems = rng.choice(width, size=int(rng.integers(1, min(4, width) + 1)), replace=False)
        mask[:, stems] = True
    elif kind == 4:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if height >= 3 and width >= 3:
            mask[1:-1, 1:-1] = rng.random((height - 2, width - 2)) < 0.13
        mask &= rng.random((height, width)) < float(rng.uniform(0.7, 0.95))
    elif kind == 5:
        for _ in range(int(rng.integers(2, 6))):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            if rng.random() < 0.5:
                mask[row, col:col + int(rng.integers(1, width - col + 1))] = True
            else:
                mask[row:row + int(rng.integers(1, height - row + 1)), col] = True
    else:
        mask[:] = rng.random((height, width)) < float(rng.uniform(0.24, 0.58))
    mask[0, int(rng.integers(width))] = True
    mask[-1, int(rng.integers(width))] = True
    mask[int(rng.integers(height)), 0] = True
    mask[int(rng.integers(height)), -1] = True
    if force_gap:
        gap_col = int(rng.integers(1, width - 1))
        for col in range(width):
            if col != gap_col and not np.any(mask[:, col]):
                mask[int(rng.integers(height)), col] = True
        mask[:, gap_col] = False
        mask[0, int(rng.choice([0, width - 1]))] = True
        mask[-1, int(rng.choice([0, width - 1]))] = True
    else:
        for col in range(1, width - 1):
            if not np.any(mask[:, col - 1:col + 2]):
                mask[int(rng.integers(height)), col] = True
    return mask


def _dimensions(rng, left_gap, right_gap):
    height = int(rng.integers(6, 31))
    width = int(rng.integers(11 if left_gap and right_gap else 10, 31))
    left_minimum = 3 if left_gap else 2
    right_minimum = 3 if right_gap else 2
    left = int(rng.integers(left_minimum, min(9, width - right_minimum - 5) + 1))
    if not left_gap and rng.random() < 0.08:
        left = 1
    right = int(rng.integers(right_minimum, min(9, width - left - 5) + 1))
    if not right_gap and rng.random() < 0.08:
        right = 1
    return height, width, left, right


def _colors(rng, background, reserved, same, required=None, exclude=()):
    choices = [c for c in range(1, 10) if c != background and c not in exclude]
    if required is None:
        choices = [c for c in choices if c != reserved]
    if same:
        color = int(required if required is not None else rng.choice(choices))
        return color, color
    if required is not None:
        other = int(rng.choice([c for c in choices if c != required and c != reserved]))
        return (required, other) if rng.random() < 0.5 else (other, required)
    selected = rng.choice(choices, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def _pair(rng, spec, background, colors):
    if spec == "group":
        left_gap = bool(rng.integers(2))
        right_gap = not left_gap or rng.random() < 0.5
    else:
        left_gap = bool(rng.random() < 0.35)
        right_gap = bool(rng.random() < 0.35)
    for _ in range(80):
        height, width, left_width, right_width = _dimensions(rng, left_gap, right_gap)
        relation = ("aligned", "common_top", "common_bottom", "partial", "disjoint")[int(rng.integers(5))]
        left_height, right_height, left_top, right_top = _vertical(rng, height, relation)
        edge = "neither" if spec == "anchor" else str(rng.choice(["neither", "neither", "left", "right"]))
        free = width - left_width - right_width - 3
        if edge == "neither":
            left_inset = int(rng.integers(1, free))
            right_inset = int(rng.integers(1, free - left_inset + 1))
        elif edge == "left":
            left_inset = 0
            right_inset = int(rng.integers(1, free + 1))
        else:
            right_inset = 0
            left_inset = int(rng.integers(1, free + 1))
        left_mask = _mask(rng, left_height, left_width, left_gap)
        right_mask = _mask(rng, right_height, right_width, right_gap)
        if int(left_mask.sum() + right_mask.sum()) >= height * width / 2:
            continue
        inp = np.full((height, width), background, dtype=np.int8)
        out = np.full_like(inp, background)
        for mask, top, col, output_col, color in (
            (left_mask, left_top, left_inset, 0, colors[0]),
            (right_mask, right_top, width - right_inset - right_width, width - right_width, colors[1]),
        ):
            rows, cols = np.nonzero(mask)
            inp[top + rows, col + cols] = color
            out[top + rows, output_col + cols] = color
        return {"input": inp, "output": out}
    raise RuntimeError("could not fit two objects on a background-majority canvas")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    reserved = int(rng.integers(1, 10))
    mixed_background = bool(rng.random() < 0.28)
    train_backgrounds = [0] * train_count
    if mixed_background:
        train_backgrounds[1] = int(rng.choice([c for c in range(1, 10) if c != reserved]))
    rng.shuffle(train_backgrounds)
    test_backgrounds = [
        int(rng.choice([c for c in range(1, 10) if c != reserved]))
        if mixed_background and rng.random() < 0.5 else 0
        for _ in range(test_count)
    ]
    anchor_bg, group_bg = train_backgrounds[:2]
    anchor_colors = _colors(rng, anchor_bg, reserved, False)
    group_colors = _colors(rng, group_bg, reserved, True, exclude=anchor_colors)
    train_specs = [("anchor", anchor_colors, anchor_bg), ("group", group_colors, group_bg)]
    for index in range(2, train_count):
        bg = train_backgrounds[index]
        train_specs.append(("random", _colors(rng, bg, reserved, rng.random() < 0.32), bg))
    rng.shuffle(train_specs)
    novel_index = int(rng.integers(test_count)) if rng.random() < 0.72 else -1
    test_specs = []
    for index, bg in enumerate(test_backgrounds):
        required = reserved if index == novel_index else None
        test_specs.append(("random", _colors(rng, bg, reserved, rng.random() < 0.4, required), bg))
    puzzle = {"train": [], "test": []}
    seen = set()
    for key, specs in (("train", train_specs), ("test", test_specs)):
        for spec, colors, background in specs:
            for _ in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _pair(pair_rng, spec, background, colors)
                inp = pair["input"]
                signature = (inp.shape, inp.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[key].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape
    frequencies = np.bincount(inp.ravel(), minlength=10)
    background = int(np.argmax(frequencies))
    assert frequencies[background] > inp.size / 2
    occupied = np.flatnonzero(np.any(inp != background, axis=0))
    assert len(occupied) >= 2
    gaps = np.diff(occupied) - 1
    assert len(gaps) and int(gaps.max()) >= 3
    assert int(np.count_nonzero(gaps == gaps.max())) == 1
    split = int(np.argmax(gaps))
    assert all(int(gap) <= 2 for j, gap in enumerate(gaps) if j != split)
    bounds = ((int(occupied[0]), int(occupied[split])),
              (int(occupied[split + 1]), int(occupied[-1])))
    regions = [inp[:, start:end + 1] for start, end in bounds]
    colors, vertical, internal_gaps = [], [], []
    for region in regions:
        content = region != background
        values = np.unique(region[content])
        assert len(values) == 1
        colors.append(int(values[0]))
        rows = np.flatnonzero(np.any(content, axis=1))
        vertical.append((int(rows[0]), int(rows[-1])))
        internal_gaps.append(bool(np.any(~np.any(content, axis=0))))
    expected = np.full_like(inp, background)
    expected[:, :regions[0].shape[1]] = regions[0]
    expected[:, -regions[1].shape[1]:] = regions[1]
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    payload = bytes((inp.shape[0], inp.shape[1])) + inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()
    assert hashlib.sha256(payload).hexdigest() not in _SOURCE_HASHES
    return bounds, colors, vertical, internal_gaps, background


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, training_colors = set(), set(), set()
    anchor_seen = grouping_seen = False
    backgrounds = set()
    for key in ("train", "test"):
        for pair in puzzle[key]:
            bounds, colors, vertical, internal_gaps, background = _analyze_pair(pair)
            inp, out = pair["input"], pair["output"]
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert all(color != background and color != 0 for color in colors)
            left_moves = bounds[0][0] > 0
            right_moves = bounds[1][1] < inp.shape[1] - 1
            assert left_moves or right_moves
            if key == "train":
                training_colors.update(colors)
                backgrounds.add(background)
                anchor_seen |= left_moves and right_moves and colors[0] != colors[1]
                grouping_seen |= colors[0] == colors[1] and any(internal_gaps)
    assert anchor_seen and grouping_seen and len(training_colors) >= 3
    if any(_analyze_pair(p)[4] != 0 for p in puzzle["test"]):
        assert any(bg != 0 for bg in backgrounds)


def _same_puzzle(a, b):
    return all(
        len(a[key]) == len(b[key])
        and all(np.array_equal(x[name], y[name])
                for x, y in zip(a[key], b[key]) for name in ("input", "output"))
        for key in ("train", "test")
    )


def _component_count(mask):
    height, width = mask.shape
    seen = set()
    count = 0
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or (row, col) in seen:
                continue
            count += 1
            queue = [(row, col)]
            seen.add((row, col))
            for r, c in queue:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= rr < height and 0 <= cc < width
                            and mask[rr, cc] and (rr, cc) not in seen):
                        seen.add((rr, cc))
                        queue.append((rr, cc))
    return count


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, patterns, color_sets = set(), set(), set(), set()
    regimes = {"train": set(), "test": set()}
    combinations = set()
    relations = {"train": set(), "test": set()}
    axes = {name: 0 for name in (
        "train_edge", "test_edge", "train_unequal", "test_unequal",
        "test_same", "test_both_move", "one_sided_gap", "colored_background",
        "small_canvas", "large_canvas", "tight_gap", "roomy_gap",
    )}
    same_sparse = partial_top = interior_gap = near_full = one_column = 0
    disconnected = enclosed_holes = 0
    novel_color_puzzles = 0
    first_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first = puzzle["train"][0]["input"]
        first_inputs.add((first.shape, first.tobytes()))
        train_colors = set()
        for pair in puzzle["train"]:
            train_colors.update(_analyze_pair(pair)[1])
        if any(set(_analyze_pair(pair)[1]) - train_colors for pair in puzzle["test"]):
            novel_color_puzzles += 1
        for key in ("train", "test"):
            for pair in puzzle[key]:
                bounds, colors, vertical, gaps, bg = _analyze_pair(pair)
                inp = pair["input"]
                height, width = inp.shape
                shapes.add(inp.shape)
                color_sets.add((bg, *colors))
                widths = tuple(end - start + 1 for start, end in bounds)
                left_move = bounds[0][0] > 0
                right_move = bounds[1][1] < width - 1
                unequal = vertical[0] != vertical[1]
                same = colors[0] == colors[1]
                relation = (vertical[0][0] == vertical[1][0],
                            vertical[0][1] == vertical[1][1],
                            max(vertical[0][0], vertical[1][0]) <= min(vertical[0][1], vertical[1][1]))
                if relation[:2] == (True, True):
                    relation_name = "aligned"
                elif relation[0]:
                    relation_name = "common_top"
                elif relation[1]:
                    relation_name = "common_bottom"
                elif relation[2]:
                    relation_name = "partial"
                else:
                    relation_name = "disjoint"
                relations[key].add(relation_name)
                axes[f"{key}_edge"] += int(not left_move or not right_move)
                axes[f"{key}_unequal"] += int(unequal)
                axes["one_sided_gap"] += int(gaps[0] != gaps[1])
                axes["colored_background"] += int(bg != 0)
                axes["small_canvas"] += int(height <= 9 and width <= 14)
                axes["large_canvas"] += int(height >= 25 or width >= 25)
                between_gap = bounds[1][0] - bounds[0][1] - 1
                axes["tight_gap"] += int(between_gap <= 4)
                axes["roomy_gap"] += int(between_gap >= 10)
                if key == "test":
                    axes["test_same"] += int(same)
                    axes["test_both_move"] += int(left_move and right_move)
                regimes[key].add((same, unequal, left_move, right_move, relation))
                combinations.add((key, same, unequal, left_move, right_move, any(gaps), bg != 0))
                layouts.add((inp.shape, widths, bounds, tuple(vertical), bg))
                for start, end in bounds:
                    crop = inp[:, start:end + 1] != bg
                    rows = np.flatnonzero(np.any(crop, axis=1))
                    crop = crop[rows[0]:rows[-1] + 1]
                    patterns.add((crop.shape, crop.tobytes()))
                    partial_top += int(not np.all(crop[0]))
                    near_full += int(crop.shape[0] >= height - 2)
                    one_column += int(crop.shape[1] == 1)
                    disconnected += int(_component_count(crop) > 1)
                    enclosed_holes += int(_component_count(np.pad(~crop, 1, constant_values=True)) > 1)
                interior_gap += sum(gaps)
                same_sparse += int(same and any(gaps))
        validate_generated_puzzle(puzzle)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 90 and len(layouts) >= 200 and len(patterns) >= 120
    assert len(color_sets) >= 35 and len(first_inputs) >= 40
    assert len(regimes["train"]) >= 12 and len(regimes["test"]) >= 10
    assert len(combinations) >= 30
    assert all(len(relations[key]) == 5 for key in ("train", "test"))
    assert all(axes[name] >= floor for name, floor in (
        ("train_edge", 20), ("test_edge", 10),
        ("train_unequal", 30), ("test_unequal", 10),
        ("test_same", 8), ("test_both_move", 8),
        ("one_sided_gap", 15), ("colored_background", 3),
        ("small_canvas", 3), ("large_canvas", 20),
        ("tight_gap", 30), ("roomy_gap", 5),
    ))
    assert same_sparse >= 42 and partial_top >= 100 and interior_gap >= 50
    assert near_full >= 15 and one_column >= 5 and novel_color_puzzles >= 15
    assert disconnected >= 50 and enclosed_holes >= 5
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        seed = 100 + train_count * 10 + test_count
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _same_puzzle(puzzle, generate_puzzle(seed, train_count, test_count))
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(271, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (2, 3)):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
