"""Compositional rectangular band-transfer puzzles."""

import hashlib
import itertools

import numpy as np


# Bit order: top, bottom, left, right.
_MASKS = (3, 12, 7, 11, 13, 14, 15)
_SOURCE_DIGESTS = frozenset({
    "5a4d95985d5bccb1c362df55f08f058a94e9e0eccce15c62a031a15c34b53150",
    "df2cabbfd07f97048ffad6b2c9f1d8af4fdd14b04ce8a9a43a751720e9a35753",
    "0d7e257adfd8d789b61ab4c47e81070bb30580d0175c38de6813d046fe59792b",
    "f3fb284105019c241f7237838b7058a3ebee666105509cb97c33c0854e0a402e",
})


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, num_train, num_test):
    if num_train is not None:
        if not _is_int(num_train) or not 2 <= num_train <= 10:
            raise ValueError("num_train must be 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_int(num_test) or not 1 <= num_test <= 4:
            raise ValueError("num_test must be 1 through 4")
        num_test = int(num_test)
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


def _depth(height, width, mask):
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    distances = []
    if mask & 1:
        distances.append(np.broadcast_to(rows, (height, width)))
    if mask & 2:
        distances.append(np.broadcast_to(height - 1 - rows, (height, width)))
    if mask & 4:
        distances.append(np.broadcast_to(cols, (height, width)))
    if mask & 8:
        distances.append(np.broadcast_to(width - 1 - cols, (height, width)))
    return np.minimum.reduce(distances)


def _paint(height, width, mask, colors, widths):
    depth = _depth(height, width, mask)
    result = np.full((height, width), colors[-1], dtype=np.int8)
    start = 0
    for color, thickness in zip(colors[:-1], widths):
        result[(depth >= start) & (depth < start + thickness)] = color
        start += thickness
    return result


def _minimum_size(mask, finite_depth):
    def axis(first, second):
        if first and second:
            return 2 * finite_depth + 1
        if first or second:
            return finite_depth + 1
        return 4
    return axis(bool(mask & 1), bool(mask & 2)), axis(bool(mask & 4), bool(mask & 8))


def _profile(rng, background, mask, target_count, compact):
    layer_count = int(rng.integers(3, 6))
    depth_limit = {2: 8, 3: 7, 4: 6}[mask.bit_count()]
    if target_count >= 4:
        depth_limit = min(depth_limit, 4)
    elif target_count == 3:
        depth_limit = min(depth_limit, 5)
    if compact:
        layer_count = int(rng.integers(3, 5))
        widths = (1,) * (layer_count - 1)
    elif target_count == 1 and rng.random() < 0.20:
        layer_count = 4
        widths = (2, 2, 2) if mask == 15 else tuple(int(x) for x in rng.choice((2, 2, 3), size=3))
        if sum(widths) > depth_limit:
            widths = (2, 2, 2)
    else:
        for _ in range(40):
            widths = tuple(int(x) for x in rng.choice((1, 1, 2, 2, 3), size=layer_count - 1))
            if sum(widths) <= depth_limit:
                break
        else:
            widths = (1,) * (layer_count - 1)
    choices = [color for color in range(10) if color != background]
    colors = tuple(int(x) for x in rng.choice(choices, size=layer_count, replace=False))
    return colors, widths


def _size(rng, mask, finite_depth, target_count, compact):
    minimum_height, minimum_width = _minimum_size(mask, finite_depth)
    cap = {1: 25, 2: 17, 3: 12, 4: 10}[min(target_count, 4)]
    cap = max(cap, minimum_height, minimum_width)
    def dimension(minimum):
        if compact:
            return minimum + int(rng.integers(0, 4))
        if finite_depth >= 6 and mask == 15:
            return minimum + int(rng.integers(0, 3))
        # An independent long tail admits large and slender rectangles.
        extra = int(rng.integers(0, 7)) if rng.random() < 0.78 else int(rng.integers(7, 17))
        return min(cap, minimum + extra)
    return dimension(minimum_height), dimension(minimum_width)


def _compositions(n):
    for cuts in itertools.product((False, True), repeat=n - 1):
        result = []
        current = 1
        for cut in cuts:
            if cut:
                result.append(current)
                current = 1
            else:
                current += 1
        result.append(current)
        yield tuple(result)


def _pack(rng, sizes, compact):
    count = len(sizes)
    # Random horizontal or vertical shelves, with independent internal gaps,
    # margins, row offsets and alignment. No object is rotated.
    for trial in range(90):
        transpose = bool(rng.integers(0, 2))
        dimensions = [(w, h) if transpose else (h, w) for h, w in sizes]
        order = tuple(int(x) for x in rng.permutation(count))
        cuts = tuple(bool(x) for x in rng.integers(0, 2, size=count - 1))
        groups = []
        cursor = 0
        for length in _compositions(count):
            if tuple(sum(length[:i + 1]) - 1 for i in range(len(length) - 1)) == tuple(i for i, cut in enumerate(cuts) if cut):
                for group_size in length:
                    groups.append(order[cursor:cursor + group_size])
                    cursor += group_size
                break
        gap_stop = 3 if compact else 5
        vertical_gaps = [int(rng.integers(1, gap_stop)) for _ in range(len(groups) - 1)]
        horizontal_gaps = [[int(rng.integers(1, gap_stop)) for _ in range(len(group) - 1)] for group in groups]
        heights = [max(dimensions[i][0] for i in group) for group in groups]
        widths = [sum(dimensions[i][1] for i in group) + sum(gaps) for group, gaps in zip(groups, horizontal_gaps)]
        needed_h = sum(heights) + sum(vertical_gaps)
        needed_w = max(widths)
        if needed_h > 28 or needed_w > 28:
            continue
        padding = 3 if compact else 7
        canvas_h = int(rng.integers(needed_h + 2, min(30, needed_h + padding) + 1))
        canvas_w = int(rng.integers(needed_w + 2, min(30, needed_w + padding) + 1))
        top = int(rng.integers(1, canvas_h - needed_h))
        left = int(rng.integers(1, canvas_w - needed_w))
        positions = [None] * count
        row_top = 0
        for group_index, (group, group_h, group_w, gaps) in enumerate(zip(groups, heights, widths, horizontal_gaps)):
            current_x = int(rng.integers(0, needed_w - group_w + 1))
            for j, index in enumerate(group):
                h, w = dimensions[index]
                positions[index] = (top + row_top + int(rng.integers(0, group_h - h + 1)), left + current_x)
                current_x += w + (gaps[j] if j < len(gaps) else 0)
            row_top += group_h
            if group_index < len(vertical_gaps):
                row_top += vertical_gaps[group_index]
        if transpose:
            return canvas_w, canvas_h, [(x, y) for y, x in positions]
        return canvas_h, canvas_w, positions
    return None


def _candidate(rng, mask, background, target_count, require_expansion):
    compact = bool(rng.random() < 0.25)
    colors, widths = _profile(rng, background, mask, target_count, compact)
    finite_depth = sum(widths)
    sizes = [_size(rng, mask, finite_depth, target_count, compact) for _ in range(target_count + 1)]
    # A minority of exemplars expose a one-cell core, but most do not.
    if rng.random() < 0.18:
        sizes[0] = _minimum_size(mask, finite_depth)
    if require_expansion:
        exemplar_depth = int(_depth(*sizes[0], mask).max())
        if max(int(_depth(*s, mask).max()) for s in sizes[1:]) <= exemplar_depth:
            h, w = sizes[1]
            if mask & 3:
                h += 2 if mask & 3 == 3 else 1
            if mask & 12:
                w += 2 if mask & 12 == 12 else 1
            sizes[1] = (h, w)
    packed = _pack(rng, sizes, compact)
    if packed is None:
        return None
    height, width, positions = packed
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    for index, ((h, w), (top, left)) in enumerate(zip(sizes, positions)):
        painted = _paint(h, w, mask, colors, widths)
        output_grid[top:top + h, left:left + w] = painted
        input_grid[top:top + h, left:left + w] = painted if index == 0 else colors[0]
    return input_grid, output_grid


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or grid[row, col] == background:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny, nx] and grid[ny, nx] != background:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            piece = grid[y0:y1 + 1, x0:x1 + 1]
            assert len(cells) == piece.size and np.all(piece != background)
            components.append(((y0, y1, x0, x1), piece))
    return components


def _infer_profile(exemplar):
    candidates = []
    for mask in _MASKS:
        depth = _depth(*exemplar.shape, mask)
        sequence = []
        for level in range(int(depth.max()) + 1):
            colors = np.unique(exemplar[depth == level])
            if len(colors) != 1:
                break
            sequence.append(int(colors[0]))
        else:
            run_colors, run_lengths = [], []
            for color in sequence:
                if not run_colors or run_colors[-1] != color:
                    run_colors.append(color)
                    run_lengths.append(1)
                else:
                    run_lengths[-1] += 1
            if len(run_colors) >= 3 and len(run_colors) <= 5 and len(set(run_colors)) == len(run_colors):
                candidates.append((mask, tuple(run_colors), tuple(run_lengths[:-1])))
    assert len(candidates) == 1
    return candidates[0]


def _analyze(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background = int(input_grid[0, 0])
    for grid in (input_grid, output_grid):
        assert np.all(grid[0] == background) and np.all(grid[-1] == background)
        assert np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)
    components = _components(input_grid, background)
    assert 2 <= len(components) <= 5
    multicolored = [(box, piece) for box, piece in components if len(np.unique(piece)) > 1]
    assert len(multicolored) == 1
    exemplar_box, exemplar = multicolored[0]
    mask, colors, widths = _infer_profile(exemplar)
    assert background not in colors
    finite_depth = sum(widths)
    assert int(_depth(*exemplar.shape, mask).max()) >= finite_depth
    expected = input_grid.copy()
    target_shapes, target_boxes, target_depths = [], [], []
    for box, piece in components:
        if box == exemplar_box:
            continue
        assert np.all(piece == colors[0])
        h, w = piece.shape
        depth = _depth(h, w, mask)
        assert int(depth.max()) >= finite_depth
        y0, y1, x0, x1 = box
        expected[y0:y1 + 1, x0:x1 + 1] = _paint(h, w, mask, colors, widths)
        target_shapes.append((h, w))
        target_boxes.append(box)
        target_depths.append(int(depth.max()))
    assert target_shapes and np.array_equal(expected, output_grid)
    assert not np.array_equal(input_grid, output_grid)
    return {
        "background": background, "mask": mask, "colors": colors, "widths": widths,
        "exemplar_shape": exemplar.shape, "exemplar_box": exemplar_box,
        "exemplar_depth": int(_depth(*exemplar.shape, mask).max()),
        "target_shapes": tuple(target_shapes), "target_boxes": tuple(target_boxes),
        "target_depths": tuple(target_depths), "shape": input_grid.shape,
    }


def _pair_digest(pair):
    source = pair["input"].astype(np.int8, copy=False)
    target = pair["output"].astype(np.int8, copy=False)
    return hashlib.sha256(bytes(source.shape) + source.tobytes() + target.tobytes()).hexdigest()


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                _validate_grid(grid)
            assert pair["input"].shape == pair["output"].shape
            input_key = (pair["input"].shape, pair["input"].tobytes())
            pair_key = (input_key, pair["output"].tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            analyses[split].append(_analyze(pair["input"], pair["output"]))
    # The global rule needs no hidden color or orientation convention: every
    # exemplar reveals its own profile. Training must show fixed-width transfer
    # to a target with a deeper core than its exemplar.
    assert any(max(a["widths"]) >= 2 and max(a["target_depths"]) > a["exemplar_depth"] for a in analyses["train"])
    # A test pair must be a new visible configuration, regardless of regime.
    train_signatures = {(a["mask"], a["colors"], a["widths"], a["exemplar_shape"], a["target_shapes"]) for a in analyses["train"]}
    for a in analyses["test"]:
        assert (a["mask"], a["colors"], a["widths"], a["exemplar_shape"], a["target_shapes"]) not in train_signatures


def _mask_schedule(rng, num_train, num_test):
    mode = int(rng.choice((0, 1, 2), p=(0.22, 0.16, 0.62)))
    if mode == 0:
        train = [int(rng.choice((7, 11, 13, 14))), 15]
        train.extend(int(rng.choice((7, 11, 13, 14, 15))) for _ in range(num_train - 2))
        test = [int(rng.choice((3, 12))) for _ in range(num_test)]
    elif mode == 1:
        mask = int(rng.choice(_MASKS))
        train = [mask] * num_train
        test = [mask] * num_test
    else:
        train = [int(rng.choice(_MASKS)) for _ in range(num_train)]
        if num_train >= 3 and len(set(train)) == 1:
            train[-1] = int(rng.choice([m for m in _MASKS if m != train[0]]))
        test = [int(rng.choice(_MASKS)) for _ in range(num_test)]
    rng.shuffle(train)
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    train_masks, test_masks = _mask_schedule(rng, num_train, num_test)
    # Palette stability is a puzzle-level style choice. Color roles and side
    # direction are still inferred locally from every visible exemplar.
    stable_background = bool(rng.random() < 0.24)
    background_order = [int(x) for x in rng.permutation(10)]
    evidence_index = int(rng.integers(0, num_train))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    all_masks = (("train", train_masks), ("test", test_masks))
    pair_number = 0
    for split, masks in all_masks:
        for index, mask in enumerate(masks):
            background = background_order[0] if stable_background else background_order[pair_number % 10]
            pair_number += 1
            require_expansion = split == "train" and index == evidence_index
            for _ in range(240):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                probabilities = (0.28, 0.38, 0.25, 0.09) if split == "train" else (0.22, 0.43, 0.25, 0.10)
                target_count = int(pair_rng.choice((1, 2, 3, 4), p=probabilities))
                candidate = _candidate(pair_rng, mask, background, target_count, require_expansion)
                if candidate is None:
                    continue
                input_grid, output_grid = candidate
                input_key = (input_grid.shape, input_grid.tobytes())
                if input_key in used_inputs:
                    continue
                pair = {"input": input_grid, "output": output_grid}
                if _pair_digest(pair) in _SOURCE_DIGESTS:
                    continue
                try:
                    analysis = _analyze(input_grid, output_grid)
                    if require_expansion:
                        assert max(analysis["widths"]) >= 2
                        assert max(analysis["target_depths"]) > analysis["exemplar_depth"]
                except AssertionError:
                    continue
                used_inputs.add(input_key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct an unambiguous pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, tuple(_pair_digest(pair) for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    masks, widths, shapes, layouts, backgrounds, target_counts = set(), set(), set(), set(), set(), set()
    thick = large = full_canvas = small_canvas = train_stripes = test_frames = single_tests = many_targets = 0
    even_core = odd_core = loose = no_one_gap = nonminimal_test = even_two_target_test = 0
    same_height_different_width = same_width_different_height = 0
    first_recipe = 0
    varying_background = consistent_mask = varying_counts = varying_widths = varying_shapes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"], pair["output"])
                local.append(a)
                masks.add(a["mask"])
                widths.add(a["widths"])
                shapes.add(a["shape"])
                backgrounds.add(a["background"])
                target_counts.add(len(a["target_shapes"]))
                layouts.add((a["shape"], a["mask"], a["exemplar_box"], a["target_boxes"]))
                thick += sum(w >= 2 for w in a["widths"]) >= 2
                large += max(max(a["exemplar_shape"]), *(max(s) for s in a["target_shapes"])) >= 16
                full_canvas += 30 in a["shape"]
                small_canvas += max(a["shape"]) <= 18
                train_stripes += split == "train" and a["mask"] in (3, 12)
                test_frames += split == "test" and a["mask"] not in (3, 12)
                single_tests += split == "test" and len(a["target_shapes"]) == 1
                many_targets += len(a["target_shapes"]) == 4
                nonminimal_test += split == "test" and min(a["target_depths"]) > sum(a["widths"])
                first_recipe += split == "train" and len(local) == 1 and a["exemplar_depth"] == sum(a["widths"]) and a["exemplar_shape"] in a["target_shapes"]
                same_height_different_width += any(x[0] == y[0] and x[1] != y[1] for x, y in itertools.combinations(a["target_shapes"], 2))
                same_width_different_height += any(x[1] == y[1] and x[0] != y[0] for x, y in itertools.combinations(a["target_shapes"], 2))
                for shape in a["target_shapes"]:
                    depth = int(_depth(*shape, a["mask"]).max())
                    # At maximal depth the remaining core span can be even or odd.
                    constrained = shape[0] if a["mask"] & 3 == 3 else shape[1] if a["mask"] & 12 == 12 else None
                    if constrained is not None:
                        even_core += (constrained - 2 * sum(a["widths"])) % 2 == 0
                        odd_core += (constrained - 2 * sum(a["widths"])) % 2 == 1
                        even_two_target_test += split == "test" and len(a["target_shapes"]) == 2 and constrained - 2 * sum(a["widths"]) >= 2 and (constrained - 2 * sum(a["widths"])) % 2 == 0
                boxes = [a["exemplar_box"], *a["target_boxes"]]
                gaps = [max(0, b[0] - c[1] - 1, c[0] - b[1] - 1, b[2] - c[3] - 1, c[2] - b[3] - 1) for b, c in itertools.combinations(boxes, 2)]
                loose += any(gap >= 2 for gap in gaps)
                no_one_gap += all(gap >= 2 for gap in gaps)
        varying_background += len({a["background"] for a in local}) > 1
        consistent_mask += len({a["mask"] for a in local}) == 1
        varying_counts += len({len(a["target_shapes"]) for a in local}) > 1
        varying_widths += len({a["widths"] for a in local}) > 1
        varying_shapes += len({a["shape"] for a in local}) > 1
    assert masks == set(_MASKS)
    assert len(widths) >= 15 and len(shapes) >= 35 and len(layouts) >= 150
    assert len(backgrounds) == 10 and target_counts == {1, 2, 3, 4}
    assert (2, 2, 2) in widths and (2, 2, 1) in widths
    assert min(thick, large, full_canvas, small_canvas, train_stripes, test_frames, single_tests, many_targets, even_core, odd_core, loose, no_one_gap, nonminimal_test, even_two_target_test, same_height_different_width, same_width_different_height) > 0
    assert first_recipe < 10
    assert varying_background >= 15 and consistent_mask >= 2
    assert min(varying_counts, varying_widths, varying_shapes) >= 35
    assert _puzzle_key(generate_puzzle(90210, 6, 3)) == _puzzle_key(generate_puzzle(90210, 6, 3))
    assert _puzzle_key(generate_puzzle(90210, 6, 3)) != _puzzle_key(generate_puzzle(90211, 6, 3))
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
