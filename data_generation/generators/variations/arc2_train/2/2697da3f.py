"""Generate four-arm ARC puzzles from a tightly cropped symmetric motif."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("00000000", "00000000", "00000000", "00040400", "00444400", "00040400", "00000000"),
    ("0000000", "0000000", "0044000", "0404400", "0044000", "0000000", "0000000"),
    ("00000000", "00000000", "00000000", "40004000", "40444000", "04400000", "40444000", "40004000"),
    ("0000000", "0444040", "0004440", "0000400", "0004440", "0444040", "0000000", "0000000"),
    ("0000000000", "0040444000", "0044004000", "0400040000", "0044004000", "0040444000", "0000000000", "0000000000"),
)


def _check_optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _check_optional_count("num_train", num_train, 2, 10)
    num_test = _check_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _connected8(mask):
    locations = np.argwhere(mask)
    if not len(locations):
        return False
    start = tuple(locations[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _connected4(mask):
    locations = np.argwhere(mask)
    if not len(locations):
        return False
    start = tuple(locations[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                    and mask[nr, nc] and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mirror_upper(upper, h):
    if h % 2:
        return np.concatenate((upper, upper[-2::-1]), axis=0)
    return np.concatenate((upper, upper[::-1]), axis=0)


def _valid_mask(mask, asymmetric):
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if not np.array_equal(mask, mask[::-1, :]) or not _connected8(mask):
        return False
    if asymmetric and np.array_equal(mask, mask[:, ::-1]):
        return False
    # A square with two symmetry axes does not reveal which arm is original.
    if mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask[:, ::-1]):
        return False
    return True


def _walk_mask(rng, h, w, density):
    half = (h + 1) // 2
    upper = np.zeros((half, w), dtype=bool)
    row = 0
    for col in range(w):
        remaining = w - col - 1
        if row < half - 1 and (half - 1 - row > remaining or rng.random() < 0.48):
            row += 1
        elif row > 0 and half - 1 - row < remaining and rng.random() < 0.19:
            row -= 1
        upper[row, col] = True
    upper[half - 1, w - 1] = True
    target = int(round(density * h * w))
    for _ in range(h * w * 3):
        if int(_mirror_upper(upper, h).sum()) >= target:
            break
        candidates = []
        for r in range(half):
            for c in range(w):
                if not upper[r, c] and upper[max(0, r-1):min(half, r+2), max(0, c-1):min(w, c+2)].any():
                    candidates.append((r, c))
        if not candidates:
            break
        r, c = candidates[int(rng.integers(len(candidates)))]
        upper[r, c] = True
    return _mirror_upper(upper, h)


def _sample_mask(rng, h, w, used, asymmetric=False):
    half = (h + 1) // 2
    for _ in range(120):
        style = int(rng.choice(4, p=[0.60, 0.19, 0.13, 0.08]))
        density = float(rng.uniform(0.28, 0.64) if style == 0 else rng.uniform(0.25, 0.75))
        if style == 0:
            mask = _walk_mask(rng, h, w, density)
        elif style == 1:
            upper = rng.random((half, w)) < density
            mask = _mirror_upper(upper, h)
        elif style == 2:
            upper = rng.random((half, (w + 1) // 2)) < density
            full = np.concatenate((upper, upper[:, -2::-1] if w % 2 else upper[:, ::-1]), axis=1)
            mask = _mirror_upper(full, h)
        else:
            upper = np.ones((half, w), dtype=bool) if rng.random() < 0.16 else rng.random((half, w)) < density
            mask = _mirror_upper(upper, h)
        key = (h, w, mask.tobytes())
        if _valid_mask(mask, asymmetric) and (h == 2 or key not in used):
            used.add(key)
            return mask
    # Connected construction is a bounded fallback even at awkward dimensions.
    for _ in range(120):
        mask = _walk_mask(rng, h, w, float(rng.uniform(0.25, 0.8)))
        key = (h, w, mask.tobytes())
        if _valid_mask(mask, asymmetric) and (h == 2 or key not in used):
            used.add(key)
            return mask
    raise ValueError("unable to construct a distinct motif")


def _paint_crop(rng, mask, background, multicolor):
    colors = [int(x) for x in rng.choice([x for x in range(10) if x != background], size=2 if multicolor else 1, replace=False)]
    crop = np.full(mask.shape, background, dtype=np.int8)
    orbits = [(r, c) for r in range((mask.shape[0] + 1) // 2) for c in range(mask.shape[1]) if mask[r, c]]
    assignments = rng.integers(0, len(colors), size=len(orbits))
    if multicolor:
        assignments[0] = 0
        assignments[-1] = 1
    for (r, c), index in zip(orbits, assignments):
        crop[r, c] = colors[index]
        crop[mask.shape[0] - 1 - r, c] = colors[index]
    return crop


def _construct_output(horizontal_crop, background):
    h, w = horizontal_crop.shape
    side = 2 * w + h
    result = np.full((side, side), background, dtype=np.int8)
    result[w:w+h, :w] = horizontal_crop
    result[:w, w:w+h] = horizontal_crop.T
    result[w:w+h, w+h:] = horizontal_crop[:, ::-1]
    result[w+h:, w:w+h] = horizontal_crop.T[::-1, :]
    return result


def _sample_shape(rng, familiar=False):
    if familiar:
        return ((3, 4), (5, 5))[int(rng.integers(2))]
    h = int(rng.integers(2, 9))
    w = int(rng.integers(max(3, h), min(13, (30-h)//2) + 1))
    return h, w


def _extra_padding(rng, size):
    maximum = 30 - size
    if maximum == 0:
        return 0
    band = int(rng.integers(3))
    if band == 0:
        low, high = 1, min(2, maximum)
    elif band == 1:
        low, high = 1, min(8, maximum)
    else:
        low, high = min(4, maximum), maximum
    return int(rng.integers(low, high + 1))


def _make_pair(rng, shape, vertical, background, multicolor, tight, asymmetric, used_inputs, used_masks):
    h, w = shape
    for _ in range(30):
        mask = _sample_mask(rng, h, w, used_masks, asymmetric)
        horizontal_crop = _paint_crop(rng, mask, background, multicolor)
        crop = horizontal_crop.T.copy() if vertical else horizontal_crop
        ch, cw = crop.shape
        extra_h = 0 if tight else _extra_padding(rng, ch)
        extra_w = 0 if tight else _extra_padding(rng, cw)
        # Non-tight cases expose the background through a full row and column.
        if not tight and (extra_h == 0 or extra_w == 0):
            continue
        top = int(rng.integers(extra_h + 1))
        left = int(rng.integers(extra_w + 1))
        input_grid = np.full((ch + extra_h, cw + extra_w), background, dtype=np.int8)
        input_grid[top:top+ch, left:left+cw] = crop
        key = _grid_key(input_grid)
        if key in used_inputs or tuple(''.join(map(str, row)) for row in input_grid) in _SOURCE_INPUT_ROWS:
            continue
        used_inputs.add(key)
        return {"input": input_grid, "output": _construct_output(horizontal_crop, background)}
    raise ValueError("unable to construct a distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    fixed_background = bool(rng.random() < 0.32)
    background = int(rng.integers(10))
    mixed_orientation = bool(rng.random() < 0.48)
    familiar = bool(rng.random() < 0.35)
    shapes = [_sample_shape(rng, familiar), _sample_shape(rng, familiar)]
    for _ in range(100):
        if shapes[0][0] != shapes[1][0] and shapes[0][1] != shapes[1][1] and shapes[0][0] >= 3 and shapes[1][0] >= 3:
            break
        shapes[0] = _sample_shape(rng, familiar)
        shapes[1] = _sample_shape(rng, familiar)
    else:
        shapes = [(3, 4), (5, 5)]
    shapes += [_sample_shape(rng, bool(rng.random() < 0.18)) for _ in range(num_train - 2)]
    orientations = [False, mixed_orientation] + [bool(mixed_orientation and rng.random() < 0.5) for _ in range(num_train - 2)]
    multi_flags = [True, False] + [bool(rng.random() < 0.4) for _ in range(num_train - 2)]
    tight_flags = [fixed_background] + [bool(fixed_background and rng.random() < 0.12) for _ in range(num_train - 1)]
    order = list(rng.permutation(2)) + [2 + int(i) for i in rng.permutation(num_train - 2)]
    shapes = [shapes[int(i)] for i in order]
    orientations = [orientations[int(i)] for i in order]
    multi_flags = [multi_flags[int(i)] for i in order]
    tight_flags = [tight_flags[int(i)] for i in order]
    used_inputs, used_masks = set(), set()
    train = []
    train_backgrounds = []
    seen_orientations = set()
    for i in range(num_train):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        bg = background if fixed_background else int(pair_rng.integers(10))
        if i == 1 and not fixed_background and bg == train_backgrounds[0]:
            bg = (bg + 1) % 10
        need_asymmetry = orientations[i] not in seen_orientations
        train.append(_make_pair(pair_rng, shapes[i], orientations[i], bg, multi_flags[i], tight_flags[i], need_asymmetry, used_inputs, used_masks))
        train_backgrounds.append(bg)
        seen_orientations.add(orientations[i])
    test = []
    for _ in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        kind = int(pair_rng.choice(4, p=[0.18, 0.30, 0.34, 0.18]))
        if kind == 0:
            shape = shapes[int(pair_rng.integers(num_train))]
        elif kind == 1:
            base = shapes[int(pair_rng.integers(num_train))]
            cap = min(13, (30-base[0])//2)
            shape = (base[0], int(pair_rng.integers(base[1]+1, cap+1))) if base[1] < cap else _sample_shape(pair_rng)
        elif kind == 2:
            shape = _sample_shape(pair_rng)
        else:
            shape = _sample_shape(pair_rng, bool(pair_rng.random() < 0.45))
        vertical = bool(mixed_orientation and pair_rng.random() < 0.5)
        bg = background if fixed_background else int(pair_rng.integers(10))
        multi = bool(pair_rng.random() < 0.45)
        tight = bool(fixed_background and pair_rng.random() < 0.17)
        test.append(_make_pair(pair_rng, shape, vertical, bg, multi, tight, False, used_inputs, used_masks))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _extract_crop(grid, background):
    positions = np.argwhere(grid != background)
    assert len(positions)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0) + 1
    return grid[lo[0]:hi[0], lo[1]:hi[1]], tuple(int(x) for x in (*lo, *hi))


def _axis(crop, background):
    mask = crop != background
    tb = np.array_equal(crop, crop[::-1, :])
    lr = np.array_equal(crop, crop[:, ::-1])
    if tb and not lr:
        return "horizontal"
    if lr and not tb:
        return "vertical"
    if tb and lr and crop.shape[0] != crop.shape[1]:
        return "horizontal" if crop.shape[1] > crop.shape[0] else "vertical"
    raise AssertionError("motif axis is ambiguous")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs, masks = set(), set(), set()
    train_shapes, train_axes, test_axes = [], set(), set()
    train_bgs, test_bgs = [], []
    multi_train, mono_train = False, False
    tight_train, tight_test = False, False
    asymmetric_axes = set()
    for split, collection in (("train", train), ("test", test)):
        for pair in collection:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            ikey, pkey = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            assert tuple(''.join(map(str, row)) for row in inp) not in _SOURCE_INPUT_ROWS
            bg = int(out[0, 0])
            assert np.all(out[:1, :1] == bg)
            crop, (r0, c0, r1, c1) = _extract_crop(inp, bg)
            tight = inp.shape == crop.shape
            if split == "train":
                train_bgs.append(bg)
                tight_train |= tight
            else:
                test_bgs.append(bg)
                tight_test |= tight
            if not tight:
                assert inp.shape[0] > crop.shape[0] and inp.shape[1] > crop.shape[1]
                assert any(np.all(row == bg) for row in inp)
                assert any(np.all(col == bg) for col in inp.T)
            assert np.all(inp[:r0] == bg) and np.all(inp[r1:] == bg)
            assert np.all(inp[:, :c0] == bg) and np.all(inp[:, c1:] == bg)
            axis = _axis(crop, bg)
            horizontal = crop if axis == "horizontal" else crop.T
            h, w = horizontal.shape
            assert 2 <= h <= 8 and h <= w <= 13 and 2*w+h <= 30
            mask = horizontal != bg
            assert _valid_mask(mask, False)
            assert np.array_equal(horizontal, horizontal[::-1, :])
            mkey = (mask.shape, mask.tobytes())
            assert h == 2 or mkey not in masks
            masks.add(mkey)
            assert np.array_equal(out, _construct_output(horizontal, bg))
            assert set(np.unique(out)) == set(np.unique(inp)) | {bg}
            assert out.shape == (2*w+h, 2*w+h)
            if split == "train":
                train_shapes.append((h, w))
                train_axes.add(axis)
                foreground_colors = set(np.unique(inp).tolist()) - {bg}
                multi_train |= len(foreground_colors) >= 2
                mono_train |= len(foreground_colors) == 1
                if not np.array_equal(mask, mask[:, ::-1]):
                    asymmetric_axes.add(axis)
            else:
                test_axes.add(axis)
    assert len({h for h, _ in train_shapes}) >= 2
    assert len({w for _, w in train_shapes}) >= 2
    assert multi_train and mono_train
    assert test_axes <= train_axes and train_axes <= asymmetric_axes
    if len(set(train_bgs)) > 1:
        assert not tight_train and not tight_test
    else:
        if tight_test:
            assert tight_train
        if tight_train or tight_test:
            assert set(test_bgs) == set(train_bgs)
    assert len({tuple(sorted(np.unique(p["input"]).tolist())) for p in train}) >= 2


def _predict_from_training(train, test_input):
    backgrounds = {int(pair["output"][0, 0]) for pair in train}
    row_colors = {int(row[0]) for row in test_input if np.all(row == row[0])}
    col_colors = {int(col[0]) for col in test_input.T if np.all(col == col[0])}
    candidates = row_colors & col_colors
    if len(candidates) == 1 and (len(backgrounds) > 1 or next(iter(candidates)) in backgrounds):
        bg = candidates.pop()
    else:
        assert len(backgrounds) == 1
        bg = next(iter(backgrounds))
    crop, _ = _extract_crop(test_input, bg)
    axis = _axis(crop, bg)
    a = crop if axis == "horizontal" else crop.T
    h, w = a.shape
    result = np.full((2*w+h, 2*w+h), bg, dtype=np.int8)
    result[w:w+h, :w] = a
    result[:w, w:w+h] = np.rot90(a, 3)
    result[w:w+h, w+h:] = np.rot90(a, 2)
    result[w+h:, w:w+h] = np.rot90(a, 1)
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_grid_key(x["input"]) for s in ("train", "test") for x in p[s]) for p in puzzles}) >= 40
    sizes, colors, densities, axes, canvases = set(), set(), set(), set(), set()
    coverage = {name: 0 for name in (
        "diagonal", "multicolor", "sparse", "dense", "solid", "doubly_symmetric",
        "tight", "large_canvas", "rectangular_canvas", "vertical", "combined",
        "within_shapes", "within_palettes", "new_test_shape", "length_extension",
    )}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_shapes, puzzle_palettes = set(), set()
        train_shapes, test_shapes = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])
                bg = int(pair["output"][0, 0])
                crop, _ = _extract_crop(pair["input"], bg)
                axis = _axis(crop, bg)
                a = crop if axis == "horizontal" else crop.T
                mask = a != bg
                foreground_colors = set(np.unique(pair["input"]).tolist()) - {bg}
                multicolor = len(foreground_colors) >= 2
                sizes.add(a.shape)
                colors.add(tuple(sorted(np.unique(pair["input"]).tolist())))
                densities.add(round(float(np.mean(mask)), 1))
                axes.add(axis)
                canvases.add(pair["input"].shape)
                puzzle_shapes.add(a.shape)
                puzzle_palettes.add(tuple(sorted(np.unique(pair["input"]).tolist())))
                (train_shapes if split == "train" else test_shapes).append(a.shape)
                coverage["diagonal"] += not _connected4(mask)
                coverage["multicolor"] += multicolor
                coverage["sparse"] += float(mask.mean()) < 0.5
                coverage["dense"] += float(mask.mean()) > 0.75
                coverage["solid"] += bool(mask.all())
                coverage["doubly_symmetric"] += bool(np.array_equal(mask, mask[:, ::-1]))
                coverage["tight"] += pair["input"].shape == crop.shape
                coverage["large_canvas"] += max(pair["input"].shape) >= 20
                coverage["rectangular_canvas"] += max(pair["input"].shape) >= 2 * min(pair["input"].shape)
                coverage["vertical"] += axis == "vertical"
                coverage["combined"] += axis == "vertical" and multicolor and pair["output"].shape[0] >= 24
        coverage["within_shapes"] += len(puzzle_shapes) >= 2
        coverage["within_palettes"] += len(puzzle_palettes) >= 2
        coverage["new_test_shape"] += any(shape not in train_shapes for shape in test_shapes)
        coverage["length_extension"] += any(
            test_shape[0] == train_shape[0] and test_shape[1] > train_shape[1]
            for test_shape in test_shapes for train_shape in train_shapes
        )
    assert len(sizes) >= 25 and len(canvases) >= 50
    assert len(colors) >= 25 and len(densities) >= 6
    assert axes == {"horizontal", "vertical"}
    assert coverage["within_shapes"] == 42 and coverage["within_palettes"] == 42
    assert coverage["new_test_shape"] >= 10 and coverage["length_extension"] >= 5
    for name in ("diagonal", "multicolor", "sparse", "large_canvas", "rectangular_canvas", "vertical"):
        assert coverage[name] >= 20, (name, coverage[name])
    for name in ("dense", "solid", "doubly_symmetric", "tight"):
        assert coverage[name] >= 5, (name, coverage[name])
    assert coverage["combined"] >= 1
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + index, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(17, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    for args in ((123456, 7, 3), (654321, None, None)):
        first, second = generate_puzzle(*args), generate_puzzle(*args)
        for split in ("train", "test"):
            for a, b in zip(first[split], second[split]):
                assert np.array_equal(a["input"], b["input"])
                assert np.array_equal(a["output"], b["output"])
