"""Generators for clockwise, block-enlarged template color transfer."""

import hashlib

import numpy as np


# SHA-256 of the three source input grids as row-major int8 arrays.  Exact
# content, rather than canvas size or palette, is excluded.
_SOURCE_INPUTS = {
    ((22, 12), "6a9b60aea018adcebc564b7de83694fb23db0ed395ed25b50c0609e96945b56e"),
    ((24, 13), "8138c7ad3c4842eea4aecd1b32faba705d6758d6070b2c18ba2f25d786181430"),
    ((16, 26), "bcba1def7f2d18f8f93ed824e7ff4493da129922053fdc8c95f6178a69c7d8e4"),
}

# Index 3 is the clockwise quarter turn.  The other seven transformations
# are considered when checking what the demonstrations actually teach.
_CLOCKWISE = 3
_LAYOUTS = ("above", "below", "left", "right", "nw", "ne", "sw", "se")


def _is_integral(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        if not _is_integral(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_integral(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lo = max(2, num_test + 1)
        num_train = lo + (seed // 4) % (11 - lo)
    elif num_train is None:
        lo = max(2, num_test + 1)
        num_train = lo + (seed // 7) % (11 - lo)
    elif num_test is None:
        num_test = 1 + (seed // 11) % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _bbox(mask):
    rows, columns = np.nonzero(mask)
    assert len(rows)
    return int(rows.min()), int(rows.max()) + 1, int(columns.min()), int(columns.max()) + 1


def _transform(template, orientation):
    rotated = np.rot90(template, k=orientation % 4)
    return np.fliplr(rotated) if orientation >= 4 else rotated


def _sample_mask(rng, height, width, style):
    if min(height, width) == 1:
        # A thin template can still use several colors and demonstrate order.
        mask = np.ones((height, width), dtype=bool)
        if max(height, width) >= 5 and style == "sparse":
            mask.flat[2:-1:3] = False
        return mask
    if style == "full":
        return np.ones((height, width), dtype=bool)
    if style == "hole" and min(height, width) >= 3:
        mask = np.ones((height, width), dtype=bool)
        mask[int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))] = False
        return mask
    if style == "near_full":
        mask = rng.random((height, width)) >= float(rng.uniform(0.05, 0.25))
    elif style == "symmetric":
        half = rng.random((height, (width + 1) // 2)) < float(rng.uniform(0.45, 0.8))
        mask = np.concatenate((half, np.fliplr(half[:, :width // 2])), axis=1)
    elif style == "sparse":
        mask = rng.random((height, width)) < float(rng.uniform(0.25, 0.48))
    else:
        mask = rng.random((height, width)) < float(rng.uniform(0.4, 0.78))
    if style == "symmetric":
        # Paired boundary additions retain the left-right symmetry.
        if not mask[0].any():
            j = int(rng.integers(width))
            mask[0, j] = mask[0, width - 1 - j] = True
        if not mask[-1].any():
            j = int(rng.integers(width))
            mask[-1, j] = mask[-1, width - 1 - j] = True
        if not mask[:, 0].any():
            i = int(rng.integers(height))
            mask[i, 0] = mask[i, -1] = True
    else:
        if not mask[0].any():
            mask[0, int(rng.integers(width))] = True
        if not mask[-1].any():
            mask[-1, int(rng.integers(width))] = True
        if not mask[:, 0].any():
            mask[int(rng.integers(height)), 0] = True
        if not mask[:, -1].any():
            mask[int(rng.integers(height)), -1] = True
    return mask


def _relative_positions(rng, layout, th, tw, ph, pw):
    gap = int(rng.integers(1, 7))
    ty = tx = 0
    if layout == "above":
        py = th + gap
        px = int(rng.integers(-pw + 1, tw))
    elif layout == "below":
        py = -ph - gap
        px = int(rng.integers(-pw + 1, tw))
    elif layout == "left":
        px = tw + gap
        py = int(rng.integers(-ph + 1, th))
    elif layout == "right":
        px = -pw - gap
        py = int(rng.integers(-ph + 1, th))
    else:
        py = th + gap if layout[0] == "n" else -ph - gap
        px = tw + int(rng.integers(1, 7)) if layout[1] == "w" else -pw - int(rng.integers(1, 7))
    return ty, tx, py, px


def _place(rng, template, scale, background, temporary, layout):
    th, tw = template.shape
    rotated = _transform(template, _CLOCKWISE)
    ph, pw = rotated.shape[0] * scale, rotated.shape[1] * scale
    ty, tx, py, px = _relative_positions(rng, layout, th, tw, ph, pw)
    min_y, min_x = min(ty, py), min(tx, px)
    ty, py = ty - min_y, py - min_y
    tx, px = tx - min_x, px - min_x
    need_h, need_w = max(ty + th, py + ph), max(tx + tw, px + pw)
    if need_h > 30 or need_w > 30:
        return None

    canvas_style = int(rng.integers(5))
    margin_max = (1, 3, 7, 2, 4)[canvas_style]
    top, bottom, left, right = [int(rng.integers(margin_max + 1)) for _ in range(4)]
    if canvas_style == 3:
        # Make one outer side touch the canvas boundary.
        edge = int(rng.integers(4))
        if edge == 0:
            top = 0
        elif edge == 1:
            bottom = 0
        elif edge == 2:
            left = 0
        else:
            right = 0
    height, width = need_h + top + bottom, need_w + left + right
    if height > 30 or width > 30:
        return None
    if canvas_style == 4:
        if rng.random() < 0.65:
            target = max(height, width)
            if target <= 30:
                height = width = target
        else:
            if rng.random() < 0.5:
                height = 30
            else:
                width = 30
    ty, py = ty + top, py + top
    tx, px = tx + left, px + left
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    input_grid[ty:ty + th, tx:tx + tw] = template
    output_grid[ty:ty + th, tx:tx + tw] = template
    for row in range(rotated.shape[0]):
        for col in range(rotated.shape[1]):
            color = int(rotated[row, col])
            if color != background:
                block = np.s_[py + row * scale:py + (row + 1) * scale,
                              px + col * scale:px + (col + 1) * scale]
                input_grid[block] = temporary
                output_grid[block] = color
    counts = np.bincount(input_grid.ravel(), minlength=10)
    if counts[background] <= max(counts[i] for i in range(10) if i != background):
        return None
    return input_grid, output_grid


def _sample_pair(rng, layout, first_training=False, regime="source_train"):
    for _ in range(500):
        if first_training:
            height = int(rng.integers(2, 5))
            width = int(rng.integers(2, 5))
            if height * width < 5:
                continue
            style = "full"
            scale = int(rng.integers(2, 5))
        else:
            height = int(rng.integers(1, 7))
            width = int(rng.integers(1, 7))
            if height * width < 2 or height * width > 25:
                continue
            style = str(rng.choice(("full", "hole", "near_full", "symmetric", "sparse", "random")))
            scale = int(rng.integers(2, 7))
            if regime == "target_test" and rng.random() < 0.35:
                scale = int(rng.integers(4, 7))
        if scale * max(height, width) > 29:
            continue
        colors = rng.permutation(10)
        background, temporary = int(colors[0]), int(colors[1])
        mask = _sample_mask(rng, height, width, style)
        occupied = int(mask.sum())
        if first_training:
            palette_size = min(4, occupied)
        else:
            palette_size = int(rng.integers(1, min(7, occupied) + 1))
        palette = tuple(int(v) for v in colors[2:2 + palette_size])
        # Reuse the sampled mask rather than making its density and palette
        # accidentally depend on a second random mask.
        template = np.full((height, width), background, dtype=np.int8)
        coordinates = np.argwhere(mask)
        rng.shuffle(coordinates)
        values = list(palette)
        values += [int(v) for v in rng.choice(palette, size=occupied - palette_size)]
        rng.shuffle(values)
        for (row, col), color in zip(coordinates, values):
            template[int(row), int(col)] = color
        pair = _place(rng, template, scale, background, temporary, layout)
        if pair is None:
            continue
        input_grid, output_grid = pair
        if _source_copy(input_grid):
            continue
        if first_training:
            matches = [i for i in range(8) if _predict(input_grid, i) is not None
                       and np.array_equal(_predict(input_grid, i), output_grid)]
            if matches != [_CLOCKWISE]:
                continue
        return pair
    raise RuntimeError("could not construct a valid pair")


def _source_copy(grid):
    return (grid.shape, hashlib.sha256(grid.tobytes()).hexdigest()) in _SOURCE_INPUTS


def _background(grid):
    counts = np.bincount(grid.astype(np.int64).ravel(), minlength=10)
    biggest = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[biggest]) != 1:
        return None
    return biggest


def _parse_candidate(grid, temporary, orientation):
    background = _background(grid)
    if background is None or temporary == background:
        return None
    placeholder_mask = grid == temporary
    template_mask = (grid != background) & ~placeholder_mask
    if not placeholder_mask.any() or not template_mask.any():
        return None
    pr0, pr1, pc0, pc1 = _bbox(placeholder_mask)
    tr0, tr1, tc0, tc1 = _bbox(template_mask)
    if not (tr1 < pr0 or pr1 < tr0 or tc1 < pc0 or pc1 < tc0):
        return None
    template = grid[tr0:tr1, tc0:tc1]
    transformed = _transform(template, orientation)
    sh, sw = transformed.shape
    ph, pw = pr1 - pr0, pc1 - pc0
    if ph % sh or pw % sw or ph // sh != pw // sw:
        return None
    scale = ph // sh
    if scale < 2:
        return None
    expected_mask = np.kron(transformed != background, np.ones((scale, scale), dtype=bool))
    if not np.array_equal(placeholder_mask[pr0:pr1, pc0:pc1], expected_mask):
        return None
    predicted = grid.copy()
    for row in range(sh):
        for col in range(sw):
            color = int(transformed[row, col])
            if color != background:
                predicted[pr0 + row * scale:pr0 + (row + 1) * scale,
                          pc0 + col * scale:pc0 + (col + 1) * scale] = color
    return predicted, {
        "background": background, "temporary": temporary,
        "payload": tuple(sorted(int(v) for v in np.unique(template) if v != background)),
        "template": template, "template_bbox": (tr0, tr1, tc0, tc1),
        "placeholder_bbox": (pr0, pr1, pc0, pc1), "scale": scale,
    }


def _predict(grid, orientation):
    background = _background(grid)
    if background is None:
        return None
    predictions = []
    for color in np.unique(grid):
        if int(color) == background:
            continue
        result = _parse_candidate(grid, int(color), orientation)
        if result is not None:
            predictions.append(result[0])
    if len(predictions) != 1:
        return None
    return predictions[0]


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert changed.any()
    temporaries = np.unique(input_grid[changed])
    assert len(temporaries) == 1
    result = _parse_candidate(input_grid, int(temporaries[0]), _CLOCKWISE)
    assert result is not None
    predicted, features = result
    assert np.array_equal(predicted, output_grid)
    assert np.all(input_grid[changed] == features["temporary"])
    assert np.all(output_grid[changed] != features["temporary"])
    return features


def _learn_orientation(train):
    compatible = set(range(8))
    for pair in train:
        matching = {i for i in range(8) if (prediction := _predict(pair["input"], i)) is not None
                    and np.array_equal(prediction, pair["output"])}
        compatible &= matching
    return compatible


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs = set()
    features = []
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((0 <= grid) & (grid <= 9))
        key = (pair["input"].shape, pair["input"].tobytes())
        assert key not in inputs
        inputs.add(key)
        assert not _source_copy(pair["input"])
        features.append(_analyze_pair(pair["input"], pair["output"]))
    # This inference sees only demonstrations, and then predicts each test
    # output from its input.  It does not receive a test output or metadata.
    assert _learn_orientation(train) == {_CLOCKWISE}
    for pair in test:
        predicted = _predict(pair["input"], _CLOCKWISE)
        assert predicted is not None and np.array_equal(predicted, pair["output"])
    assert all(feature["scale"] >= 2 for feature in features)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_integral(seed) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Clockwise is the sole learned convention in the source family.  Layout
    # policies shape examples but have no effect on the answer rule.
    policy = int(rng.integers(4))
    if policy == 0:  # source's above-to-left layout shift
        anchor_train, anchor_test = "above", "left"
    elif policy == 1:  # its reverse
        anchor_train, anchor_test = "left", "above"
    elif policy == 2:  # unchanged relative placement
        anchor_train = anchor_test = str(rng.choice(_LAYOUTS))
    else:  # independent layouts in both splits
        anchor_train, anchor_test = (str(rng.choice(_LAYOUTS)), str(rng.choice(_LAYOUTS)))
    puzzle = {"train": [], "test": []}
    used = set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if index == 0:
                layout = anchor_train if section == "train" else anchor_test
            elif section == "train" and index == 1:
                layout = str(rng.choice([v for v in _LAYOUTS if v != anchor_train]))
            else:
                layout = str(rng.choice(_LAYOUTS))
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                input_grid, output_grid = _sample_pair(
                    pair_rng, layout, first_training=section == "train" and index == 0,
                    regime="source_train" if section == "train" else "target_test")
                key = (input_grid.shape, input_grid.tobytes())
                if key not in used:
                    used.add(key)
                    puzzle[section].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("could not generate distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[section]) == len(b[section]) and all(
        np.array_equal(x[part], y[part])
        for x, y in zip(a[section], b[section]) for part in ("input", "output"))
        for section in ("train", "test"))


def _component_count(mask):
    seen = set()
    count = 0
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        start = (int(row), int(col))
        if start in seen:
            continue
        count += 1
        pending = [start]
        seen.add(start)
        while pending:
            y, x = pending.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                yy, xx = neighbor
                if (0 <= yy < height and 0 <= xx < width and mask[yy, xx]
                        and neighbor not in seen):
                    seen.add(neighbor)
                    pending.append(neighbor)
    return count


def _layout(features):
    tr0, tr1, tc0, tc1 = features["template_bbox"]
    pr0, pr1, pc0, pc1 = features["placeholder_bbox"]
    vertical = "n" if tr1 < pr0 else "s" if pr1 < tr0 else ""
    horizontal = "w" if tc1 < pc0 else "e" if pc1 < tc0 else ""
    if vertical and horizontal:
        return vertical + horizontal
    return {"n": "above", "s": "below", "w": "left", "e": "right"}[vertical or horizontal]


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    assert not _puzzles_equal(generate_puzzle(12345), generate_puzzle(12346))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(20000 + (train_count or 0) * 10 + (test_count or 0),
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(21000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")

    train_counts, test_counts = set(), set()
    shapes, scales, palettes, layouts, canvas_shapes, role_pairs = set(), set(), set(), set(), set(), set()
    shape_scales = set()
    full = thin = symmetric = edge = square = thirty = 0
    disconnected = empty_band = enclosed_hole = sparse = 0
    monochrome = rich = rectangular_train = large_train = unseen_test = 0
    combined = source_sized = 0
    mixed_roles = mixed_scales = mixed_palettes = mixed_shapes = mixed_layouts = 0
    above_to_left = left_to_above = unchanged_layout = other_shift = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_roles, puzzle_scales, puzzle_palettes, puzzle_shapes, puzzle_layouts = set(), set(), set(), set(), set()
        train_colors, test_colors = set(), set()
        first_train_layout = first_test_layout = None
        for section, pairs in (("train", puzzle["train"]), ("test", puzzle["test"])):
          for pair_index, pair in enumerate(pairs):
            f = _analyze_pair(pair["input"], pair["output"])
            mask = f["template"] != f["background"]
            shapes.add(mask.shape)
            scales.add(f["scale"])
            palettes.add(len(f["payload"]))
            shape_scales.add((mask.shape, f["scale"]))
            role = (f["background"], f["temporary"])
            role_pairs.add(role)
            puzzle_roles.add(role)
            puzzle_scales.add(f["scale"])
            puzzle_palettes.add(len(f["payload"]))
            puzzle_shapes.add(mask.shape)
            canvas_shapes.add(pair["input"].shape)
            tr0, tr1, tc0, tc1 = f["template_bbox"]
            pr0, pr1, pc0, pc1 = f["placeholder_bbox"]
            placement = _layout(f)
            layouts.add(placement)
            puzzle_layouts.add(placement)
            if pair_index == 0 and section == "train":
                first_train_layout = placement
            if pair_index == 0 and section == "test":
                first_test_layout = placement
            if section == "train":
                train_colors.update(f["payload"])
                rectangular_train += mask.shape[0] != mask.shape[1]
                large_train += max(mask.shape) >= 5 and f["scale"] == 2
            else:
                test_colors.update(f["payload"])
            full += bool(mask.all())
            thin += min(mask.shape) == 1
            clockwise_mask = np.rot90(mask, k=-1)
            symmetric += sum(
                candidate.shape == clockwise_mask.shape
                and np.array_equal(candidate, clockwise_mask)
                for candidate in (_transform(mask, orientation) for orientation in range(8))
            ) > 1
            edge += min(tr0, tc0, pr0, pc0,
                        pair["input"].shape[0] - max(tr1, pr1),
                        pair["input"].shape[1] - max(tc1, pc1)) == 0
            square += pair["input"].shape[0] == pair["input"].shape[1]
            thirty += 30 in pair["input"].shape
            disconnected += _component_count(mask) > 1
            empty_band += (any(not row.any() for row in mask[1:-1])
                           or any(not col.any() for col in mask.T[1:-1]))
            enclosed_hole += any(
                not mask[row, col] and all(mask[y, x] for y, x in
                    ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
                for row in range(1, mask.shape[0] - 1)
                for col in range(1, mask.shape[1] - 1)
            )
            sparse += float(mask.mean()) <= 0.5
            monochrome += len(f["payload"]) == 1
            rich += len(f["payload"]) >= 6
            combined += (mask.shape[0] != mask.shape[1]
                         and f["scale"] >= 4 and len(f["payload"]) >= 5)
            source_sized += pair["input"].shape in {(22, 12), (24, 13), (16, 26)}
        mixed_roles += len(puzzle_roles) > 1
        mixed_scales += len(puzzle_scales) > 1
        mixed_palettes += len(puzzle_palettes) > 1
        mixed_shapes += len(puzzle_shapes) > 1
        mixed_layouts += len(puzzle_layouts) > 1
        unseen_test += bool(test_colors - train_colors)
        above_to_left += (first_train_layout, first_test_layout) == ("above", "left")
        left_to_above += (first_train_layout, first_test_layout) == ("left", "above")
        unchanged_layout += first_train_layout == first_test_layout
        other_shift += (first_train_layout != first_test_layout
                        and (first_train_layout, first_test_layout)
                        not in {("above", "left"), ("left", "above")})
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 12 and len(scales) >= 4 and len(palettes) >= 5
    assert len(layouts) == 8 and len(canvas_shapes) >= 45 and len(role_pairs) >= 30
    assert len(shape_scales) >= 60 and rectangular_train >= 30 and large_train >= 5
    assert full >= 20 and thin >= 5 and symmetric >= 20
    assert edge >= 5 and square >= 5 and thirty >= 5
    assert disconnected >= 5 and empty_band >= 5 and enclosed_hole >= 5 and sparse >= 5
    assert monochrome >= 5 and rich >= 5 and combined >= 5 and unseen_test >= 3
    assert source_sized >= 1
    assert mixed_roles >= 35 and mixed_scales >= 25
    assert mixed_palettes >= 35 and mixed_shapes >= 35 and mixed_layouts >= 35
    assert min(above_to_left, left_to_above, unchanged_layout, other_shift) >= 1
