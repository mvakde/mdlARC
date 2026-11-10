"""Selector-panel ARC puzzles: markers retrieve pair-local reference motifs."""

import hashlib
import itertools

import numpy as np


# (lattice rows, lattice columns, motif side). Geometry is fixed per puzzle.
_LAYOUTS = ((2, 2, 3), (1, 3, 3), (2, 3, 3), (2, 2, 5))
_SOURCE_DIGESTS = {
    "e47425b0f1029d0b5fa1aaffb7594151e39fb96f49874e6c99b7949a621241b9",
    "f50b217c3146d02e907b1d87ae033648fa77a39a820b16ad78b5af99556950e4",
    "cadf59255266c5e21442672a4096396bf8156379338a7997724b24b9ff497023",
    "32854be57f6a4469251969610e87e29cf8a0e0e89af79cf371c624f144b9ab23",
}


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
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (int(seed) // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _geometry(layout):
    rows, cols, tile = layout
    return rows * tile + rows - 1, cols * tile + cols - 1


def _slots(layout):
    rows, cols, tile = layout
    for row in range(rows):
        for col in range(cols):
            top, left = row * (tile + 1), col * (tile + 1)
            yield top, left, top + tile // 2, left + tile // 2


def _separated(first, second):
    ar, ac, ah, aw = first
    br, bc, bh, bw = second
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _tight(mask):
    return (bool(mask[0].any()) and bool(mask[-1].any())
            and bool(mask[:, 0].any()) and bool(mask[:, -1].any()))


def _mask(rng, side, center_absent=False):
    low, high = (3, 8) if side == 3 else (6, 18)
    middle = (side * side) // 2
    choices = [i for i in range(side * side) if not (center_absent and i == middle)]
    for _ in range(160):
        count = int(rng.integers(low, high + 1))
        chosen = rng.choice(choices, size=count, replace=False)
        flat = np.zeros(side * side, dtype=bool)
        flat[chosen] = True
        candidate = flat.reshape(side, side)
        if _tight(candidate):
            return candidate
    fallback = np.zeros((side, side), dtype=bool)
    fallback[0, 0] = fallback[0, -1] = fallback[-1, 0] = True
    fallback[-1, -1] = True
    if not center_absent:
        fallback[side // 2, side // 2] = True
    return fallback


def _origin(rng, height, width, panel_h, panel_w, location):
    if location == "top_left":
        return 0, 0
    if location == "top_right":
        return 0, width - panel_w
    if location == "bottom_left":
        return height - panel_h, 0
    if location == "bottom_right":
        return height - panel_h, width - panel_w
    if height - panel_h < 4 or width - panel_w < 4:
        return None
    return (int(rng.integers(2, height - panel_h - 1)),
            int(rng.integers(2, width - panel_w - 1)))


def _near_panel(box, panel):
    r, c, h, w = box
    pr, pc, ph, pw = panel
    row_overlap = r < pr + ph and pr < r + h
    col_overlap = c < pc + pw and pc < c + w
    return ((row_overlap and (c == pc + pw + 1 or pc == c + w + 1))
            or (col_overlap and (r == pr + ph + 1 or pr == r + h + 1)))


def _pack(rng, height, width, panel, side, count, close):
    candidates = [(r, c, side, side)
                  for r in range(height - side + 1)
                  for c in range(width - side + 1)
                  if _separated((r, c, side, side), panel)]
    if len(candidates) < count:
        return None
    near = [box for box in candidates if _near_panel(box, panel)]
    for _ in range(24):
        chosen = []
        if close and near:
            chosen.append(near[int(rng.integers(len(near)))])
        for index in rng.permutation(len(candidates)):
            box = candidates[int(index)]
            if all(_separated(box, other) for other in chosen):
                chosen.append(box)
                if len(chosen) == count:
                    return chosen
    return None


def _sample_canvas(rng, layout, count, location, close, regime):
    panel_h, panel_w = _geometry(layout)
    side = layout[2]
    min_h = 14 if side == 3 else 22
    min_w = 14 if side == 3 else 22
    if layout[1] == 3:
        min_w = max(min_w, 18)
    if layout[0] == 2 and layout[1] == 3:
        min_h = max(min_h, 17)
    for trial in range(26):
        if trial == 25:
            height = width = 30
        else:
            upper = 31 if regime == "target_test" else 29
            height = int(rng.integers(min_h, upper))
            width = int(rng.integers(min_w, upper))
        origin = _origin(rng, height, width, panel_h, panel_w, location)
        if origin is None:
            continue
        panel = origin[0], origin[1], panel_h, panel_w
        boxes = _pack(rng, height, width, panel, side, count, close)
        if boxes is not None:
            return height, width, origin, boxes
    raise RuntimeError("No bounded canvas packing found")


def _generate_pair(rng, layout, config):
    regime, location, distractor_count, force_absent, force_duplicate = config
    panel_h, panel_w = _geometry(layout)
    side = layout[2]
    marker_count = layout[0] * layout[1]
    roles = rng.permutation(10).tolist()
    background, panel_color = int(roles[0]), int(roles[1])
    marker_colors = [int(x) for x in roles[2:2 + marker_count]]
    distractors = [int(x) for x in roles[2 + marker_count:2 + marker_count + distractor_count]]
    colors = marker_colors + distractors
    masks = [_mask(rng, side, force_absent and i == 0) for i in range(len(colors))]
    if force_duplicate and distractors:
        masks[-1] = masks[0].copy()
    close = bool(rng.integers(4) == 0)
    height, width, origin, boxes = _sample_canvas(
        rng, layout, len(colors), location, close, regime)
    grid = np.full((height, width), background, dtype=np.int8)
    pr, pc = origin
    grid[pr:pr + panel_h, pc:pc + panel_w] = panel_color
    for color, (_, _, center_r, center_c) in zip(marker_colors, _slots(layout)):
        grid[pr + center_r, pc + center_c] = color
    for color, mask, (row, col, _, _) in zip(colors, masks, boxes):
        patch = grid[row:row + side, col:col + side]
        patch[mask] = color
    output = np.full((panel_h, panel_w), background, dtype=np.int8)
    for color, mask, (top, left, _, _) in zip(marker_colors, masks, _slots(layout)):
        output[top:top + side, left:left + side][mask] = color
    return {"input": grid, "output": output}


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _pair_digest(pair):
    digest = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name].astype(np.int8, copy=False)
        digest.update(np.asarray(grid.shape, dtype=np.int16).tobytes())
        digest.update(grid.tobytes())
    return digest.hexdigest()


def _inspect_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    candidates = []
    for layout in _LAYOUTS:
        panel_h, panel_w = _geometry(layout)
        centers = [(r, c) for _, _, r, c in _slots(layout)]
        expected_field = panel_h * panel_w - len(centers)
        for value, count in zip(values, counts):
            panel_color = int(value)
            if panel_color == background or int(count) != expected_field:
                continue
            coordinates = np.argwhere(grid == panel_color)
            top, left = coordinates.min(axis=0).tolist()
            bottom, right = coordinates.max(axis=0).tolist()
            if (bottom - top + 1, right - left + 1) != (panel_h, panel_w):
                continue
            local = grid[top:top + panel_h, left:left + panel_w]
            field = np.ones((panel_h, panel_w), dtype=bool)
            for row, col in centers:
                field[row, col] = False
            if not np.all(local[field] == panel_color):
                continue
            markers = tuple(int(local[row, col]) for row, col in centers)
            if (len(set(markers)) != len(markers)
                    or background in markers or panel_color in markers):
                continue
            candidates.append((layout, background, panel_color, (top, left), markers))
    assert len(candidates) == 1
    layout, background, panel_color, origin, markers = candidates[0]
    panel_h, panel_w = _geometry(layout)
    side = layout[2]
    outside = grid.copy()
    pr, pc = origin
    outside[pr:pr + panel_h, pc:pc + panel_w] = background
    external = {int(x) for x in np.unique(outside)} - {background}
    assert panel_color not in external and set(markers) <= external
    assert len(external) - len(markers) in (0, 1, 2)
    boxes, masks = {}, {}
    for color in external:
        positions = np.argwhere(outside == color)
        row, col = positions.min(axis=0).tolist()
        bottom, right = positions.max(axis=0).tolist()
        assert (bottom - row + 1, right - col + 1) == (side, side)
        patch = outside[row:row + side, col:col + side]
        assert set(int(x) for x in np.unique(patch)) <= {background, color}
        mask = patch == color
        assert _tight(mask)
        if side == 3:
            assert 3 <= int(mask.sum()) <= 8
        else:
            assert 6 <= int(mask.sum()) <= 18
        boxes[color] = row, col, side, side
        masks[color] = mask
    panel = pr, pc, panel_h, panel_w
    assert all(_separated(box, panel) for box in boxes.values())
    assert all(_separated(a, b) for a, b in itertools.combinations(boxes.values(), 2))
    if origin == (0, 0):
        location = "top_left"
    elif origin == (0, grid.shape[1] - panel_w):
        location = "top_right"
    elif origin == (grid.shape[0] - panel_h, 0):
        location = "bottom_left"
    elif origin == (grid.shape[0] - panel_h, grid.shape[1] - panel_w):
        location = "bottom_right"
    else:
        assert 2 <= pr <= grid.shape[0] - panel_h - 2
        assert 2 <= pc <= grid.shape[1] - panel_w - 2
        location = "inset"
    return {"layout": layout, "background": background, "panel": panel_color,
            "origin": origin, "location": location, "markers": markers,
            "external": external, "boxes": boxes, "masks": masks,
            "distractors": external - set(markers)}


def _predict_from_input(grid):
    # Only visible input information is used here.
    info = _inspect_input(grid)
    output = np.full(_geometry(info["layout"]), info["background"], dtype=np.int8)
    side = info["layout"][2]
    for color, (row, col, _, _) in zip(info["markers"], _slots(info["layout"])):
        output[row:row + side, col:col + side][info["masks"][color]] = color
    return output, info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    train_info, test_info = [], []
    for section, info_list in (("train", train_info), ("test", test_info)):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            key = _grid_key(pair["input"])
            pair_key = key, _grid_key(pair["output"])
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            predicted, info = _predict_from_input(pair["input"])
            assert np.array_equal(pair["output"], predicted)
            info_list.append(info)
    infos = train_info + test_info
    assert len({info["layout"] for info in infos}) == 1
    assert len({info["background"] for info in train_info}) >= 2
    assert len({info["panel"] for info in train_info}) >= 2
    assert len({info["location"] for info in train_info}) >= 2
    assert len({tuple(pair["input"].shape) for pair in puzzle["train"]}) >= 2
    assert {len(info["distractors"]) for info in train_info} >= {0, 2}
    assert any(not info["masks"][color][info["layout"][2] // 2,
                                       info["layout"][2] // 2]
               for info in train_info for color in info["markers"])
    assert any(any(np.array_equal(info["masks"][d], info["masks"][m])
                   for d in info["distractors"] for m in info["markers"])
               for info in train_info)
    assert all(len(info["markers"]) == len(train_info[0]["markers"]) for info in infos)


def _same_puzzle(a, b):
    return (all(len(a[s]) == len(b[s]) for s in ("train", "test"))
            and all(np.array_equal(x[k], y[k])
                    for s in ("train", "test")
                    for x, y in zip(a[s], b[s]) for k in ("input", "output")))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    layout = _LAYOUTS[(0, 0, 0, 0, 1, 2, 3)[int(seed) % 7]]
    locations = ["top_left", "top_right", "bottom_left", "bottom_right", "inset"]
    first, second = rng.choice(len(locations), size=2, replace=False).tolist()
    train_locations = [locations[first], locations[second]]
    train_locations += [locations[int(rng.integers(5))] for _ in range(train_count - 2)]
    distractors = [0, 2] + ([1] if train_count >= 3 else [])
    distractors += [int(rng.integers(3)) for _ in range(train_count - len(distractors))]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    train, test, used = [], [], set()
    first_background, first_panel = None, None
    for index in range(train_count + test_count):
        local = np.random.default_rng(int(pair_seeds[index]))
        if index < train_count:
            config = ("source_train", train_locations[index], distractors[index],
                      index == 0, index == 1)
        else:
            loc = locations[int(local.choice(5, p=(.12, .12, .17, .27, .32)))]
            config = ("target_test", loc, int(local.choice(3, p=(.26, .40, .34))),
                      False, False)
        for _ in range(32):
            pair = _generate_pair(local, layout, config)
            key = _grid_key(pair["input"])
            if key in used:
                continue
            info = _inspect_input(pair["input"])
            if index == 0:
                first_background, first_panel = info["background"], info["panel"]
            elif index == 1 and (info["background"] == first_background
                                  or info["panel"] == first_panel):
                continue
            if index == 1 and pair["input"].shape == train[0]["input"].shape:
                continue
            used.add(key)
            (train if index < train_count else test).append(pair)
            break
        else:
            raise RuntimeError("Could not make a unique, varied pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    all_pairs = [pair for puzzle in puzzles
                 for pair in puzzle["train"] + puzzle["test"]]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            assert np.array_equal(_predict_from_input(pair["input"])[0], pair["output"])
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    infos = [_inspect_input(pair["input"]) for pair in all_pairs]
    assert {info["layout"] for info in infos} == set(_LAYOUTS)
    assert {info["location"] for info in infos} == {
        "top_left", "top_right", "bottom_left", "bottom_right", "inset"}
    assert {len(info["distractors"]) for info in infos} == {0, 1, 2}
    assert len({tuple(sorted((m.shape, m.tobytes()) for m in info["masks"].values()))
                for info in infos}) > 100
    assert len({tuple(pair["input"].shape) for pair in all_pairs}) > 100
    assert len({_pair_digest(pair) for pair in all_pairs}) > 300
    assert len({(info["background"], info["panel"]) for info in infos}) > 50
    assert min(pair["input"].shape[0] for pair in all_pairs) <= 16
    assert max(pair["input"].shape[0] for pair in all_pairs) >= 28
    assert min(pair["input"].shape[1] for pair in all_pairs) <= 16
    assert max(pair["input"].shape[1] for pair in all_pairs) >= 28
    assert len({int(mask.sum()) for info in infos
                for mask in info["masks"].values()}) >= 12
    absent_pairs = sum(
        any(not info["masks"][color][info["layout"][2] // 2,
                                     info["layout"][2] // 2]
            for color in info["markers"])
        for info in infos)
    assert absent_pairs > 150
    repeated_density_pairs = sum(
        len({int(mask.sum()) for mask in info["masks"].values()})
        < len(info["masks"]) for info in infos)
    assert repeated_density_pairs > 150
    near_references = sum(
        _near_panel(box, (*info["origin"], *_geometry(info["layout"])))
        for info in infos for box in info["boxes"].values())
    assert near_references > 50
    novel_test_masks = new_test_marker_sets = new_test_locations = 0
    new_test_marker_colors = cross_role_puzzles = 0
    for puzzle in puzzles:
        train = [_inspect_input(pair["input"]) for pair in puzzle["train"]]
        tests = [_inspect_input(pair["input"]) for pair in puzzle["test"]]
        vocabulary = {(mask.shape, mask.tobytes())
                      for info in train for mask in info["masks"].values()}
        marker_sets = {frozenset(info["markers"]) for info in train}
        marker_colors = {color for info in train for color in info["markers"]}
        locations = {info["location"] for info in train}
        roles = {}
        for info in train + tests:
            role_items = [(info["background"], "background"),
                          (info["panel"], "panel")]
            role_items += [(color, "marker") for color in info["markers"]]
            role_items += [(color, "distractor") for color in info["distractors"]]
            for color, role in role_items:
                roles.setdefault(color, set()).add(role)
        cross_role_puzzles += any(len(role_set) > 1 for role_set in roles.values())
        for info in tests:
            novel_test_masks += any(
                (mask.shape, mask.tobytes()) not in vocabulary
                for mask in info["masks"].values())
            new_test_marker_sets += frozenset(info["markers"]) not in marker_sets
            new_test_locations += info["location"] not in locations
            new_test_marker_colors += any(
                color not in marker_colors for color in info["markers"])
    assert novel_test_masks > 70
    assert new_test_marker_sets > 70
    assert new_test_locations > 15
    assert new_test_marker_colors > 10
    assert cross_role_puzzles > 30
    assert _same_puzzle(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert not _same_puzzle(puzzles[0], puzzles[1])
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                       (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for i, kwargs in enumerate(({"num_train": 1}, {"num_train": 11},
                                {"num_test": 0}, {"num_test": 5},
                                {"num_train": 2, "num_test": 2},
                                {"num_train": 3, "num_test": 4},
                                {"num_train": True}, {"num_test": False})):
        try:
            generate_puzzle(2000 + i, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
