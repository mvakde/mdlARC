"""Generate ARC puzzles about matching a clipped twofold template."""

import hashlib
import json

import numpy as np


_SHAPES = tuple(
    (h, w) for h in range(3, 7) for w in range(3, 7)
    if h * w <= 30
)
_SOURCE_PAIR_DIGESTS = {
    "98840fbd100d1abc8bfefb33de2950e96783721c14820b277019f1046ce0b674",
    "0302df3864bde96c8e5039408c2f052d47319cea0428301e0a3c37332f24ac2f",
    "dbf3be73bd6dd9864f9d8f8344fd8f07f48e7d8cd29a82dd730edba51e7f9ffd",
    "5eb1f4fe965a23618c7986f171c2343f623a0e9dd0090d1170e2d70f6c3c3f84",
    "947fd715207f8c69d7efe5398748ec2c209b1f9841f01308f2fff68269200d7d",
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


def _choose_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
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


def _connected(mask):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(int(v) for v in points[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                    and mask[q] and q not in seen):
                seen.add(q)
                stack.append(q)
    return len(seen) == int(mask.sum())


def _pattern(rng, shape, mode):
    h, w = shape
    if mode == "bars":
        mask = np.zeros(shape, bool)
        spine = int(rng.integers(0, w))
        mask[:, spine] = True
        mask[0, :] = True
        mask[-1, :] = True
        if h >= 4 and rng.random() < .65:
            mask[int(rng.integers(1, h - 1)), :] = True
        if rng.random() < .4:
            mask[:, w - 1 - spine] = True
    elif mode == "cross":
        mask = np.zeros(shape, bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
        if rng.random() < .5:
            mask[(h - 1) // 2, :] = True
        if rng.random() < .3:
            mask[:, (w - 1) // 2] = True
    elif mode == "repeat":
        row = rng.random(w) < float(rng.uniform(.35, .75))
        row[int(rng.integers(w))] = True
        mask = np.tile(row, (h, 1))
        mask[:, int(rng.integers(w))] = True
        if rng.random() < .5:
            mask[int(rng.integers(h)), :] = True
    elif mode == "solid":
        mask = np.ones(shape, bool)
        holes = int(rng.integers(1, min(5, h * w // 4) + 1))
        for _ in range(holes):
            mask[int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))] = False
    else:
        for _ in range(500):
            mask = rng.random(shape) < float(rng.uniform(.38, .72))
            if _connected(mask) and 4 <= int(mask.sum()) < h * w - 1:
                break
        else:
            return _pattern(rng, shape, "bars")
    # A tight crop is defined by foreground at all four sides.
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return _pattern(rng, shape, "cross")
    return mask


def _bbox(grid, color):
    points = np.argwhere(grid == color)
    if not len(points):
        return None
    return (int(points[:, 0].min()), int(points[:, 0].max()),
            int(points[:, 1].min()), int(points[:, 1].max()))


def _candidate(grid, color, background):
    box = _bbox(grid, color)
    if box is None:
        return None
    r0, r1, c0, c1 = box
    crop = grid[r0:r1 + 1, c0:c1 + 1].copy()
    if not np.all((crop == color) | (crop == background)):
        return None
    if crop.shape not in _SHAPES or int((crop == color).sum()) < 4:
        return None
    return {"color": color, "bbox": box, "crop": crop, "mask": crop == color}


def _gap(a, b):
    return (a[1] + 1 < b[0] or b[1] + 1 < a[0]
            or a[3] + 1 < b[2] or b[3] + 1 < a[2])


def _offsets(grid, mask, background, selector):
    points = np.argwhere(grid == selector)
    if not len(points):
        return []
    enlarged = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    eh, ew = enlarged.shape
    h, w = grid.shape
    min_r, max_r = int(points[:, 0].min()), int(points[:, 0].max())
    min_c, max_c = int(points[:, 1].min()), int(points[:, 1].max())
    found = []
    for top in range(max_r - eh + 1, min_r + 1):
        for left in range(max_c - ew + 1, min_c + 1):
            r0, r1 = max(0, top), min(h, top + eh)
            c0, c1 = max(0, left), min(w, left + ew)
            if r0 >= r1 or c0 >= c1:
                continue
            visible = enlarged[r0 - top:r1 - top, c0 - left:c1 - left]
            if int(visible.sum()) != len(points):
                continue
            target = grid[r0:r1, c0:c1]
            if np.all((target == selector) == visible) and np.all(
                (target == selector) | (target == background)
            ):
                found.append((top, left))
    return found


def _solutions(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or np.count_nonzero(counts == max(counts)) != 1:
        return []
    background = int(colors[int(np.argmax(counts))])
    foreground = [int(c) for c in colors if c != background]
    solutions = []
    for selector in foreground:
        other = [c for c in foreground if c != selector]
        candidates = [_candidate(grid, c, background) for c in other]
        if any(c is None for c in candidates):
            continue
        selector_box = _bbox(grid, selector)
        if not (_gap(candidates[0]["bbox"], candidates[1]["bbox"])
                and all(_gap(c["bbox"], selector_box) for c in candidates)):
            continue
        matches = [_offsets(grid, c["mask"], background, selector) for c in candidates]
        if bool(matches[0]) == bool(matches[1]):
            continue
        index = 0 if matches[0] else 1
        for offset in matches[index]:
            solutions.append({"background": background, "selector": selector,
                              "selected": candidates[index],
                              "distractor": candidates[1 - index],
                              "selector_bbox": selector_box, "offset": offset})
    return solutions


def _paint(grid, mask, top, left, color):
    h, w = grid.shape
    r0, r1 = max(0, top), min(h, top + mask.shape[0])
    c0, c1 = max(0, left), min(w, left + mask.shape[1])
    if r0 < r1 and c0 < c1:
        view = grid[r0:r1, c0:c1]
        view[mask[r0 - top:r1 - top, c0 - left:c1 - left]] = color


def _pair_digest(pair):
    data = []
    for key in ("input", "output"):
        grid = pair[key]
        data.append([*grid.shape, [int(v) for v in grid.flat]])
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _make_pair(rng, role_colors, shape_kind, clip_kind, regime):
    for _ in range(250):
        size_class = int(rng.choice(3, p=(.42, .36, .22)))
        if size_class == 0:
            h, w = int(rng.integers(15, 20)), int(rng.integers(15, 20))
            shapes = [s for s in _SHAPES if max(s) <= 5]
        elif size_class == 1:
            h, w = int(rng.integers(20, 25)), int(rng.integers(19, 25))
            shapes = [s for s in _SHAPES if max(s) <= 6]
        else:
            h, w = int(rng.integers(25, 31)), int(rng.integers(24, 31))
            shapes = list(_SHAPES)
        chosen_shape = shapes[int(rng.integers(len(shapes)))]
        if shape_kind == "same":
            other_shape = chosen_shape
        elif shape_kind == "transpose":
            if chosen_shape[0] == chosen_shape[1]:
                continue
            other_shape = chosen_shape[::-1]
        else:
            alternatives = [s for s in shapes if s != chosen_shape]
            other_shape = alternatives[int(rng.integers(len(alternatives)))]
        selected_mode = ("bars", "cross", "repeat", "solid", "irregular")[int(rng.integers(5))]
        selected = _pattern(rng, chosen_shape, selected_mode)
        if selected.all():
            continue
        near_edit = shape_kind == "same" and rng.random() < .55
        if near_edit:
            other = selected.copy()
            positions = rng.permutation(chosen_shape[0] * chosen_shape[1])
            for flat in positions[:int(rng.integers(1, 4))]:
                other.flat[int(flat)] = ~other.flat[int(flat)]
            if not (other[0].any() and other[-1].any() and other[:, 0].any()
                    and other[:, -1].any() and int(other.sum()) >= 4):
                continue
        else:
            other_mode = ("bars", "cross", "repeat", "solid", "irregular")[int(rng.integers(5))]
            other = _pattern(rng, other_shape, other_mode)
        if chosen_shape == other_shape and np.array_equal(selected, other):
            continue
        eh, ew = 2 * chosen_shape[0], 2 * chosen_shape[1]
        vertical, horizontal = clip_kind
        if vertical == "bottom":
            clip = int(rng.integers(1, min(5, eh - 3)))
            top = h - eh + clip
        elif vertical == "top":
            clip = int(rng.integers(1, min(4, eh - 3)))
            top = -clip
        else:
            if h - eh < 2:
                continue
            top = int(rng.integers(1, h - eh + 1))
        if horizontal == "left":
            left = -int(rng.integers(1, min(4, ew - 3)))
        elif horizontal == "right":
            left = w - ew + int(rng.integers(1, min(4, ew - 3)))
        else:
            if w - ew < 2:
                continue
            left = int(rng.integers(1, w - ew + 1))
        visible_box = (max(0, top), min(h, top + eh) - 1,
                       max(0, left), min(w, left + ew) - 1)
        masks = [selected, other]
        rng.shuffle(masks)
        positions = []
        for mask in masks:
            available = [(r, c) for r in range(h - mask.shape[0] + 1)
                         for c in range(w - mask.shape[1] + 1)
                         if _gap((r, r + mask.shape[0] - 1, c, c + mask.shape[1] - 1), visible_box)
                         and all(_gap((r, r + mask.shape[0] - 1, c, c + mask.shape[1] - 1),
                                      (rr, rr + mm.shape[0] - 1, cc, cc + mm.shape[1] - 1))
                                 for (rr, cc), mm in zip(positions, masks))]
            if not available:
                break
            positions.append(available[int(rng.integers(len(available)))])
        if len(positions) != 2:
            continue
        background, selected_color, distractor_color, selector_color = role_colors
        color0 = selected_color if masks[0] is selected else distractor_color
        color1 = distractor_color if masks[0] is selected else selected_color
        grid = np.full((h, w), background, dtype=np.int8)
        _paint(grid, masks[0], *positions[0], color0)
        _paint(grid, masks[1], *positions[1], color1)
        _paint(grid, np.repeat(np.repeat(selected, 2, axis=0), 2, axis=1),
               top, left, selector_color)
        output = np.where(selected, selected_color, background).astype(np.int8)
        solutions = _solutions(grid)
        if (len(solutions) != 1 or solutions[0]["selector"] != selector_color
                or solutions[0]["offset"] != (top, left)
                or not np.array_equal(solutions[0]["selected"]["crop"], output)):
            continue
        pair = {"input": grid, "output": output}
        if _pair_digest(pair) in _SOURCE_PAIR_DIGESTS:
            continue
        return pair
    raise RuntimeError(f"unable to construct {regime} pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    # The rule and the color pool are puzzle-wide. Literal roles are local to each pair.
    pool = tuple(int(v) for v in rng.choice(10, size=int(rng.integers(6, 9)), replace=False))
    shape_kinds = ["same", "transpose" if rng.random() < .6 else "different"]
    clip_kinds = [("none", "none"),
                  ("bottom", "left" if rng.random() < .5 else "right")]
    for _ in range(2, train_count):
        shape_kinds.append(("same", "transpose", "different")[int(rng.integers(3))])
        vertical = ("none", "bottom", "bottom", "top")[int(rng.integers(4))]
        horizontal = ("none", "left", "right")[int(rng.integers(3))]
        clip_kinds.append((vertical, horizontal))
    # Shuffle the teaching cases so the selected shape and clipping do not encode order.
    order = rng.permutation(train_count)
    shape_kinds = [shape_kinds[int(i)] for i in order]
    clip_kinds = [clip_kinds[int(i)] for i in order]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_training_roles = None
    for index in range(train_count + test_count):
        if index < train_count:
            section, regime = "train", "source_train"
            shape_kind, clip_kind = shape_kinds[index], clip_kinds[index]
        else:
            section, regime = "test", "target_test"
            shape_kind = ("same", "transpose", "different")[int(rng.choice(3, p=(.4, .3, .3)))]
            if rng.random() < .72:
                vertical = "bottom"
            else:
                vertical = "none" if rng.random() < .8 else "top"
            demonstrated_horiz = {style[1] for style in clip_kinds}
            horizontal = ("none", "left", "right")[int(rng.choice(3, p=(.6, .2, .2)))]
            if horizontal not in demonstrated_horiz:
                horizontal = "none"
            if vertical == "top" and not any(style[0] == "top" for style in clip_kinds):
                vertical = "bottom"
            clip_kind = (vertical, horizontal)
        for _ in range(100):
            role_colors = tuple(int(v) for v in rng.choice(pool, size=4, replace=False))
            if (index != 1 or
                    all(role_colors[j] != first_training_roles[j] for j in (0, 1, 3))):
                break
        else:
            raise RuntimeError("unable to vary training color roles")
        for _ in range(30):
            child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            pair = _make_pair(child, role_colors, shape_kind, clip_kind, regime)
            key = _grid_key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[section].append(pair)
                if index == 0:
                    first_training_roles = role_colors
                break
        else:
            raise RuntimeError("unable to construct distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    training_styles = set()
    training_same = training_different = False
    train_colors = set()
    training_backgrounds, training_selected_colors, training_selectors = set(), set(), set()
    test_records = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape != pair["output"].shape
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
            solutions = _solutions(pair["input"])
            assert len(solutions) == 1
            sol = solutions[0]
            assert np.array_equal(pair["output"], sol["selected"]["crop"])
            assert set(int(v) for v in np.unique(pair["output"])) == {
                sol["background"], sol["selected"]["color"]}
            h, w = pair["input"].shape
            top, left = sol["offset"]
            sh, sw = sol["selected"]["mask"].shape
            vertical = "top" if top < 0 else "bottom" if top + 2 * sh > h else "none"
            horizontal = "left" if left < 0 else "right" if left + 2 * sw > w else "none"
            if section == "train":
                training_styles.add((vertical, horizontal))
                training_same |= sol["selected"]["mask"].shape == sol["distractor"]["mask"].shape
                training_different |= sol["selected"]["mask"].shape != sol["distractor"]["mask"].shape
                train_colors.update(int(v) for v in np.unique(pair["input"]))
                training_backgrounds.add(sol["background"])
                training_selected_colors.add(sol["selected"]["color"])
                training_selectors.add(sol["selector"])
            else:
                test_records.append((vertical, horizontal))
    assert ("none", "none") in training_styles
    assert training_same and training_different
    assert any(v == "bottom" for v, _ in training_styles)
    demonstrated_horiz = {hor for _, hor in training_styles}
    demonstrated_vert = {ver for ver, _ in training_styles}
    for ver, hor in test_records:
        assert hor in demonstrated_horiz
        assert ver != "top" or "top" in demonstrated_vert
    assert len(train_colors) >= 4
    assert len(training_backgrounds) >= 2
    assert len(training_selected_colors) >= 2
    assert len(training_selectors) >= 2


def self_test() -> None:
    samples = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in samples:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in samples}) >= 4
    assert len({len(p["test"]) for p in samples}) >= 3
    for seed, ntrain, ntest in ((101, 3, 2), (102, 6, 3), (103, 9, 4),
                                (200, 2, 1), (201, 10, 1), (202, 10, 4),
                                (300, None, 1), (301, None, 4),
                                (302, 2, None), (303, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    a = generate_puzzle(8128, 6, 3)
    b = generate_puzzle(8128, 6, 3)
    for section in ("train", "test"):
        for x, y in zip(a[section], b[section]):
            assert np.array_equal(x["input"], y["input"])
            assert np.array_equal(x["output"], y["output"])
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(400, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    grids, layouts, masks, palettes, shapes, clips, puzzle_keys = (set() for _ in range(7))
    symmetric = same_test = near_test = compact = large = 0
    square_test = portrait_test = landscape_test = 0
    severe_bottom = severe_left = full_selector = side_by_side = 0
    repeated_rows = dense = role_swaps = 0
    for puzzle in samples:
        puzzle_keys.add(tuple(tuple(
            (_grid_key(pair["input"]), _grid_key(pair["output"]))
            for pair in puzzle[section]) for section in ("train", "test")))
        local_palettes = set()
        train_roles = {}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                sol = _solutions(grid)[0]
                selected, other = sol["selected"]["mask"], sol["distractor"]["mask"]
                assert np.array_equal(pair["output"], sol["selected"]["crop"])
                local_palettes.add(tuple(sorted(int(v) for v in np.unique(grid))))
                if section == "train":
                    train_roles.setdefault(sol["background"], set()).add("background")
                    train_roles.setdefault(sol["selector"], set()).add("selector")
                    for item in (sol["selected"], sol["distractor"]):
                        train_roles.setdefault(item["color"], set()).add("candidate")
                grids.add(_grid_key(grid))
                layouts.add((grid.shape, sol["selected"]["bbox"],
                             sol["distractor"]["bbox"], sol["selector_bbox"]))
                masks.add((selected.shape, selected.tobytes()))
                palettes.add(tuple(sorted(int(v) for v in np.unique(grid))))
                shapes.add(grid.shape)
                h, w = grid.shape
                top, left = sol["offset"]
                clips.add((top < 0, top + 2 * selected.shape[0] > h,
                           left < 0, left + 2 * selected.shape[1] > w))
                symmetric += int(np.array_equal(selected, selected[:, ::-1])
                                 or np.array_equal(selected, selected[::-1]))
                repeated_rows += int(len({row.tobytes() for row in selected}) < selected.shape[0])
                dense += int(selected.mean() >= .8)
                compact += int(h <= 19 and w <= 19)
                large += int(h >= 25 and w >= 24)
                severe_bottom += int(top + 2 * selected.shape[0] - h >= 3)
                severe_left += int(left <= -3)
                full_selector += int(top >= 0 and left >= 0 and
                                     top + 2 * selected.shape[0] <= h and
                                     left + 2 * selected.shape[1] <= w)
                selector_box = sol["selector_bbox"]
                side_by_side += int(any(
                    candidate["bbox"][0] <= selector_box[1] and
                    candidate["bbox"][1] >= selector_box[0]
                    for candidate in (sol["selected"], sol["distractor"])))
                if section == "test" and selected.shape == other.shape:
                    same_test += 1
                    near_test += int(np.count_nonzero(selected != other) <= 3)
                if section == "test":
                    square_test += int(selected.shape[0] == selected.shape[1])
                    portrait_test += int(selected.shape[0] > selected.shape[1])
                    landscape_test += int(selected.shape[0] < selected.shape[1])
        assert len(local_palettes) >= 2
        role_swaps += int(any(len(roles) >= 2 for roles in train_roles.values()))
    assert len(grids) > 200 and len(layouts) > 150 and len(masks) > 100
    assert len(palettes) > 20 and len(shapes) > 35 and len(clips) >= 5
    assert len(puzzle_keys) == 42
    assert symmetric > 20 and same_test > 10 and near_test > 3
    assert compact > 20 and large > 20
    assert min(square_test, portrait_test, landscape_test) > 5
    assert severe_bottom > 15 and severe_left > 5 and full_selector > 15
    assert side_by_side > 20 and repeated_rows > 50 and dense > 20
    assert role_swaps >= 35
