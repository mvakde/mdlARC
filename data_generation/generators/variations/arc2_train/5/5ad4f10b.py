"""Generate ARC puzzles that compress a colored block motif to a 3-by-3 ink mask."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = {
    ((22, 23), "4b676c1312331ec3fcdcbb713e2b02b2ba8ca7064042cbcbc0fd30709b9c7f4c"),
    ((20, 24), "b5cb93fdf611e038e46247f3088b840cbf4e595e65cc664aac9ae229d3f75e21"),
    ((24, 23), "e35b656da27c3c4597bba42eb07534b1471dced50d55394caf11bd333d891a2f"),
    ((22, 26), "fbb83c0da36a73b80cc716814f9f66012cea0fae0b958e9cc8931263d94e0627"),
}


_SHAPES = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)), ((0, 0), (1, 0))),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
}


def _checked_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _array_key(array):
    return tuple(array.shape), array.astype(np.int8, copy=False).tobytes()


def _source_key(array):
    return tuple(array.shape), hashlib.sha256(array.astype(np.int8, copy=False).tobytes()).hexdigest()


def _component_sizes(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack = [(row, col)]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        sizes.append(size)
    return sizes


def _symmetric(mask):
    reflected = np.fliplr(mask)
    return any(np.array_equal(mask, other) for other in (
        np.rot90(mask, 1), np.rot90(mask, 2), np.rot90(mask, 3),
        reflected, np.rot90(reflected, 1), np.rot90(reflected, 2), np.rot90(reflected, 3),
    ))


def _all_masks():
    masks = []
    for code in range(512):
        mask = np.array([(code >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if 3 <= int(mask.sum()) <= 8 and mask.any(axis=0).all() and mask.any(axis=1).all():
            masks.append(mask)
    return masks


_MASKS = _all_masks()


def _sample_mask(rng, used, wanted_connected=None, test=False):
    candidates = [m for m in _MASKS if tuple(m.ravel()) not in used]
    if wanted_connected is not None:
        filtered = [m for m in candidates if (len(_component_sizes(m)) == 1) == wanted_connected]
        if filtered:
            candidates = filtered
    if test:
        filtered = [m for m in candidates if m.sum() >= 4]
        if filtered:
            candidates = filtered
    desired_symmetry = bool(rng.random() < 0.32)
    filtered = [m for m in candidates if _symmetric(m) == desired_symmetry]
    if filtered:
        candidates = filtered
    occupancies = sorted(set(int(m.sum()) for m in candidates))
    occupancy = int(rng.choice(occupancies))
    candidates = [m for m in candidates if int(m.sum()) == occupancy]
    return candidates[int(rng.integers(len(candidates)))].copy()


def _sample_dimensions(rng, scale, occupied):
    lattice = 3 * scale
    minimum_area = 2 * occupied * scale * scale + 12
    candidates = [(h, w) for h in range(lattice, 31) for w in range(lattice, 31)
                  if h * w >= minimum_area]
    aspect = str(rng.choice(np.array(["wide", "tall", "square", "any"]), p=[0.30, 0.30, 0.18, 0.22]))
    if aspect == "wide":
        filtered = [(h, w) for h, w in candidates if w > h]
    elif aspect == "tall":
        filtered = [(h, w) for h, w in candidates if h > w]
    elif aspect == "square":
        filtered = [(h, w) for h, w in candidates if h == w]
    else:
        filtered = candidates
    if filtered:
        candidates = filtered
    mode = int(rng.integers(3))
    areas = sorted(h * w for h, w in candidates)
    low = areas[int((0, 0.34, 0.69)[mode] * (len(areas) - 1))]
    high = areas[int((0.33, 0.68, 1)[mode] * (len(areas) - 1))]
    filtered = [(h, w) for h, w in candidates if low <= h * w <= high]
    return filtered[int(rng.integers(len(filtered)))]


def _sample_origin(rng, height, width, scale):
    limits = (height - 3 * scale, width - 3 * scale)
    result = []
    for limit in limits:
        if limit == 0:
            result.append(0)
        elif rng.random() < 0.30:
            result.append(int(rng.choice([0, limit])))
        else:
            result.append(int(rng.integers(limit + 1)))
    return tuple(result)


def _clutter_target(rng, available, area, mode):
    maximum = min(55, available // 3, max(8, area // 8))
    minimum = min(maximum, max(5, area // 65))
    if mode == 0:
        high = min(maximum, max(minimum, 18))
        return int(rng.integers(minimum, high + 1))
    if mode == 1:
        low = min(maximum, max(minimum, 16))
        high = min(maximum, max(low, 35))
        return int(rng.integers(low, high + 1))
    low = min(maximum, max(minimum, 30))
    return int(rng.integers(low, maximum + 1))


def _place_clutter(rng, grid, background, ink, row, col, scale):
    height, width = grid.shape
    lattice = np.zeros(grid.shape, dtype=bool)
    lattice[row:row + 3 * scale, col:col + 3 * scale] = True
    original_background = grid == background
    region_mode = str(rng.choice(np.array(["mixed", "outside", "inside"]), p=[0.55, 0.30, 0.15]))
    if region_mode == "inside" and int((original_background & lattice).sum()) < 12:
        region_mode = "mixed"
    allowed = original_background.copy()
    if region_mode == "outside":
        allowed &= ~lattice
    elif region_mode == "inside":
        allowed &= lattice
    if int(allowed.sum()) < 6:
        allowed = original_background.copy()
    target = _clutter_target(rng, int(allowed.sum()), height * width, int(rng.integers(3)))
    target = min(target, max(5, int(allowed.sum()) // 3))
    ink_mask = np.zeros(grid.shape, dtype=bool)
    singleton_only = rng.random() < 0.12
    first_shape = 2 if rng.random() < 0.70 and not singleton_only else 1
    count = 0
    for step in range(target * 4):
        if count >= target:
            break
        if step == 0:
            size = first_shape
        elif singleton_only:
            size = 1
        else:
            size = int(rng.choice([1, 2, 3, 4], p=[0.48, 0.27, 0.17, 0.08]))
        size = min(size, target - count)
        shape = _SHAPES[size][int(rng.integers(len(_SHAPES[size])))]
        for _ in range(90):
            r = int(rng.integers(height))
            c = int(rng.integers(width))
            cells = [(r + dr, c + dc) for dr, dc in shape]
            if any(rr >= height or cc >= width or not allowed[rr, cc] for rr, cc in cells):
                continue
            cell_set = set(cells)
            if any(ink_mask[rr, cc] or any(
                0 <= rr + dr < height and 0 <= cc + dc < width
                and ink_mask[rr + dr, cc + dc] and (rr + dr, cc + dc) not in cell_set
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))) for rr, cc in cells):
                continue
            for rr, cc in cells:
                ink_mask[rr, cc] = True
            count += size
            break
    if count < 4:
        raise RuntimeError("Insufficient room for ink cue")
    grid[ink_mask] = ink


def _decode_input(grid):
    values = [int(x) for x in np.unique(grid)]
    assert len(values) == 3, "Each input needs background, block color, and ink"
    candidates = []
    for background in values:
        for template in values:
            if template == background:
                continue
            ink = next(x for x in values if x not in (background, template))
            coordinates = np.argwhere(grid == template)
            top, left = coordinates.min(axis=0)
            bottom, right = coordinates.max(axis=0)
            side_h, side_w = int(bottom - top + 1), int(right - left + 1)
            if side_h != side_w or side_h % 3:
                continue
            scale = side_h // 3
            if not 2 <= scale <= 6:
                continue
            mask = np.zeros((3, 3), dtype=bool)
            valid = True
            for mr in range(3):
                for mc in range(3):
                    tile = grid[top + mr * scale:top + (mr + 1) * scale,
                                left + mc * scale:left + (mc + 1) * scale]
                    filled = tile == template
                    if filled.all():
                        mask[mr, mc] = True
                    elif filled.any():
                        valid = False
            if not valid or not (3 <= mask.sum() <= 8):
                continue
            if not mask.any(axis=0).all() or not mask.any(axis=1).all():
                continue
            if int((grid == template).sum()) != int(mask.sum()) * scale * scale:
                continue
            ink_mask = grid == ink
            ink_count = int(ink_mask.sum())
            background_count = int((grid == background).sum())
            template_count = int((grid == template).sum())
            if ink_count < 4 or background_count <= max(ink_count, template_count):
                continue
            output = np.full((3, 3), background, dtype=np.int8)
            output[mask] = ink
            inside = int(ink_mask[top:top + side_h, left:left + side_w].sum())
            candidates.append({
                "background": background, "template": template, "ink": ink,
                "scale": scale, "mask": mask, "output": output,
                "row": int(top), "col": int(left), "ink_count": ink_count,
                "ink_inside": inside, "ink_outside": ink_count - inside,
                "ink_components": _component_sizes(ink_mask),
                "connected": len(_component_sizes(mask)) == 1,
                "symmetric": _symmetric(mask),
            })
    assert len(candidates) == 1, "Input must have one clear block-template interpretation"
    return candidates[0]


def _make_pair(rng, mask, scale, palette):
    background, template, ink = palette
    for _ in range(30):
        height, width = _sample_dimensions(rng, scale, int(mask.sum()))
        top, left = _sample_origin(rng, height, width, scale)
        grid = np.full((height, width), background, dtype=np.int8)
        for mr, mc in np.argwhere(mask):
            r, c = top + int(mr) * scale, left + int(mc) * scale
            grid[r:r + scale, c:c + scale] = template
        try:
            _place_clutter(rng, grid, background, ink, top, left, scale)
            decoded = _decode_input(grid)
        except (AssertionError, RuntimeError):
            continue
        if not np.array_equal(decoded["mask"], mask):
            continue
        if _source_key(grid) in _SOURCE_INPUT_DIGESTS:
            continue
        output = np.full((3, 3), background, dtype=np.int8)
        output[mask] = ink
        return {"input": grid, "output": output}
    raise RuntimeError("Could not construct an unambiguous pair")


def _palette(rng, color_order, base_palettes, split, index):
    if split == "train":
        if index == 0:
            return tuple(int(x) for x in color_order[:3])
        if index == 1:
            background = int(color_order[3])
            if rng.random() < 0.55:
                return background, base_palettes[0][2], base_palettes[0][1]
            return background, int(color_order[4]), int(color_order[5])
        background = int(rng.choice(color_order))
        foregrounds = [int(x) for x in color_order if int(x) != background]
        chosen = rng.choice(foregrounds, size=2, replace=False)
        return background, int(chosen[0]), int(chosen[1])
    mode = int(rng.integers(3))
    source = base_palettes[int(rng.integers(len(base_palettes)))]
    if mode == 0:
        return source
    if mode == 1:
        return source[0], source[2], source[1]
    background = int(rng.choice(color_order))
    foregrounds = [int(x) for x in color_order if int(x) != background]
    chosen = rng.choice(foregrounds, size=2, replace=False)
    return background, int(chosen[0]), int(chosen[1])


def _predict_from_demonstrations(train_pairs, test_input):
    for pair in train_pairs:
        prediction = _decode_input(pair["input"])["output"]
        assert np.array_equal(prediction, pair["output"])
    return _decode_input(test_input)["output"]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    color_order = [int(x) for x in rng.permutation(10)]
    scales = [int(x) for x in rng.choice(np.arange(2, 7), size=2, replace=False)]
    connected_plan = [True, False]
    rng.shuffle(connected_plan)
    used_masks = set()
    train, test, palettes = [], [], []
    for index in range(train_count):
        scale = scales[index] if index < 2 else int(rng.choice([2, 3, 4, 5, 6], p=[0.12, 0.27, 0.28, 0.24, 0.09]))
        mask = _sample_mask(rng, used_masks, connected_plan[index] if index < 2 else None)
        used_masks.add(tuple(mask.ravel()))
        palette = _palette(rng, color_order, palettes, "train", index)
        palettes.append(palette)
        train.append(_make_pair(rng, mask, scale, palette))
    trained_scales = {_decode_input(pair["input"])["scale"] for pair in train}
    for index in range(test_count):
        if rng.random() < 0.62:
            scale = int(rng.choice(sorted(trained_scales)))
        else:
            scale = int(rng.integers(2, 7))
        mask = _sample_mask(rng, used_masks, test=True)
        used_masks.add(tuple(mask.ravel()))
        palette = _palette(rng, color_order, palettes, "test", index)
        test.append(_make_pair(rng, mask, scale, palette))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, seen_masks = set(), set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert pair["output"].shape == (3, 3)
            assert pair["input"].shape != pair["output"].shape
            assert _source_key(pair["input"]) not in _SOURCE_INPUT_DIGESTS
            input_key = _array_key(pair["input"])
            pair_key = input_key, _array_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            decoded = _decode_input(pair["input"])
            assert np.array_equal(decoded["output"], pair["output"])
            mask_key = tuple(decoded["mask"].ravel())
            assert mask_key not in seen_masks
            seen_masks.add(mask_key)
            assert max(decoded["ink_components"]) <= 4
            information[split].append(decoded)
    train_info = information["train"]
    assert len({x["scale"] for x in train_info}) >= 2
    assert {x["connected"] for x in train_info} == {False, True}
    assert len({x["ink"] for x in train_info}) >= 2
    assert len({x["template"] for x in train_info}) >= 2
    assert len({x["background"] for x in train_info}) >= 2
    for pair in puzzle["test"]:
        prediction = _predict_from_demonstrations(
            [{"input": p["input"].copy(), "output": p["output"].copy()} for p in puzzle["train"]],
            pair["input"].copy(),
        )
        assert np.array_equal(prediction, pair["output"])


def _puzzles_equal(first, second):
    return all(np.array_equal(a[name], b[name])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for name in ("input", "output")) and all(
                   len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(781, 6, 3), generate_puzzle(781, 6, 3))
    assert not _puzzles_equal(generate_puzzle(781, 6, 3), generate_puzzle(782, 6, 3))
    for seed, nt, ne in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True},
                      {"num_test": False}):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    infos = [_decode_input(pair["input"]) for puzzle in puzzles
             for split in ("train", "test") for pair in puzzle[split]]
    shapes = {tuple(pair["input"].shape) for puzzle in puzzles
              for split in ("train", "test") for pair in puzzle[split]}
    masks = {tuple(info["mask"].ravel()) for info in infos}
    assert len(shapes) >= 45 and len(masks) >= 40
    assert {info["scale"] for info in infos} == {2, 3, 4, 5, 6}
    assert {info["symmetric"] for info in infos} == {False, True}
    assert max(info["ink_count"] for info in infos) >= 40
    assert any(max(info["ink_components"]) >= 2 for info in infos)
    assert any(info["row"] == 0 or info["col"] == 0 for info in infos)
    assert any(info["ink_inside"] == 0 for info in infos)
    assert any(info["ink_outside"] == 0 for info in infos)
    assert len({info["background"] for info in infos}) >= 7
    assert any(info["mask"].sum() == 8 for info in infos)
    assert any(info["mask"].sum() == 3 for info in infos)
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
