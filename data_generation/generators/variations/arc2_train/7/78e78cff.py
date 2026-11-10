"""Broken-frame ARC puzzle: fill the cavity and extrude its four outlets."""

import numpy as np


_SIDES = ("top", "bottom", "left", "right")
_SOURCE_ROW_PAIRS = (
    (("33333333333", "33333333333", "33113113333", "33133333333",
      "33333333333", "33336333333", "33133333333", "33113113333",
      "33333333333", "33333333333", "33333333333"),
     ("33336333333", "33336333333", "33116113333", "33166666666",
      "66666666666", "66666666666", "33166666666", "33116113333",
      "33336333333", "33336333333", "33336333333")),
    (("111111111111", "111111111111", "111211112111", "112111111211",
      "111111111111", "112113111211", "111211112111", "111111111111",
      "111111111111", "111111111111", "111111111111"),
     ("111133331111", "111133331111", "111233332111", "112333333211",
      "333333333333", "112333333211", "111233332111", "111133331111",
      "111133331111", "111133331111", "111133331111")),
    (("444444444444", "444444444444", "444244424444", "442444442444",
      "442444442444", "444444444444", "442444842444", "442444442444",
      "444224224444", "444444444444", "444444444444", "444444444444"),
     ("444488844444", "444488844444", "444288824444", "442888882444",
      "442888882444", "888888888888", "442888882444", "442888882444",
      "444228224444", "444448444444", "444448444444", "444448444444")),
)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside [{low}, {high}]")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train = int(rng.integers(2, 11))
        test = int(rng.integers(1, min(4, train - 1) + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _palette(rng, anchor, relationship):
    if anchor is None:
        return tuple(int(x) for x in rng.choice(10, 3, replace=False))
    if relationship == "stable":
        keep = (0, 1, 2) if rng.random() < 0.7 else (int(rng.integers(0, 3)),)
    elif relationship == "partial":
        k = int(rng.integers(1, 3))
        keep = tuple(int(x) for x in rng.choice(3, k, replace=False))
    else:
        k = int(rng.choice((0, 0, 1, 2, 3)))
        keep = tuple(int(x) for x in rng.choice(3, k, replace=False))
    result = [None, None, None]
    for role in keep:
        result[role] = anchor[role]
    remaining = [c for c in range(10) if c not in result]
    rng.shuffle(remaining)
    for role in range(3):
        if result[role] is None:
            result[role] = remaining.pop()
    return tuple(result)


def _gap_widths(rng, cavity_height, cavity_width, style):
    lengths = {"top": cavity_width, "bottom": cavity_width,
               "left": cavity_height, "right": cavity_height}
    # Partial openings keep structural cells at both ends. A joined frame
    # can also have one fully open side, bracketed by its two corners.
    limits = {side: max(1, length - 2) for side, length in lengths.items()}
    mode = int(rng.choice(4, p=(0.19, 0.18, 0.24, 0.39)))
    if mode == 0:
        widths = {side: 1 for side in _SIDES}
    elif mode == 1:
        widths = {side: int(rng.integers(min(2, limits[side]),
                                          min(4, limits[side]) + 1)) for side in _SIDES}
    else:
        widths = {side: int(rng.integers(1, limits[side] + 1)) for side in _SIDES}
        if mode == 2:
            axis = ("top", "bottom") if rng.random() < 0.5 else ("left", "right")
            for side in axis:
                widths[side] = int(rng.integers(max(1, limits[side] - 1), limits[side] + 1))
    if style == "joined" and mode != 0 and rng.random() < 0.14:
        side = str(rng.choice(_SIDES))
        widths[side] = lengths[side]
    return widths


def _sample_gaps(rng, top, left, cavity_height, cavity_width, style):
    widths = _gap_widths(rng, cavity_height, cavity_width, style)
    gaps = {}
    for side in _SIDES:
        length = cavity_width if side in ("top", "bottom") else cavity_height
        width = widths[side]
        low = 0 if style == "joined" and width == length else 1
        high = length - width if style == "joined" and width == length else length - width - 1
        offset = int(rng.integers(low, high + 1))
        gaps[side] = ((left + 1 if side in ("top", "bottom") else top + 1) + offset,
                      width)
    return gaps


def _construct(height, width, top, left, cavity_height, cavity_width,
               style, gaps, marker, palette):
    background, frame, fill = palette
    bottom, right = top + cavity_height + 1, left + cavity_width + 1
    inp = np.full((height, width), background, dtype=np.int8)
    inp[top, left:right + 1] = frame
    inp[bottom, left:right + 1] = frame
    inp[top:bottom + 1, left] = frame
    inp[top:bottom + 1, right] = frame
    if style == "chamfered":
        for y in (top, bottom):
            for x in (left, right):
                inp[y, x] = background
    for side in ("top", "bottom"):
        start, n = gaps[side]
        inp[top if side == "top" else bottom, start:start + n] = background
    for side in ("left", "right"):
        start, n = gaps[side]
        inp[start:start + n, left if side == "left" else right] = background
    inp[marker] = fill
    out = inp.copy()
    out[top + 1:bottom, left + 1:right] = fill
    start, n = gaps["top"]
    out[:top + 1, start:start + n] = fill
    start, n = gaps["bottom"]
    out[bottom:, start:start + n] = fill
    start, n = gaps["left"]
    out[start:start + n, :left + 1] = fill
    start, n = gaps["right"]
    out[start:start + n, right:] = fill
    return {"input": inp, "output": out}


def _sample_pair(rng, style, palette, regime):
    # Height, width, and the four ray lengths are independent draws. The
    # test distribution gives somewhat more weight to long or thin cavities.
    if regime == "target_test" and rng.random() < 0.35:
        if rng.random() < 0.5:
            ch = int(rng.choice((3, 4, 11, 12, 13)))
            cw = int(rng.integers(3, 14))
        else:
            ch = int(rng.integers(3, 14))
            cw = int(rng.choice((3, 4, 11, 12, 13)))
    else:
        ch = int(rng.integers(3, 12 if regime == "source_train" else 14))
        cw = int(rng.integers(3, 12 if regime == "source_train" else 14))
    margin_top = int(rng.integers(1, 8))
    margin_bottom = int(rng.integers(1, 8))
    margin_left = int(rng.integers(1, 8))
    margin_right = int(rng.integers(1, 8))
    height = ch + 2 + margin_top + margin_bottom
    width = cw + 2 + margin_left + margin_right
    top, left = margin_top, margin_left
    gaps = _sample_gaps(rng, top, left, ch, cw, style)
    marker_mode = int(rng.choice(3, p=(0.23, 0.27, 0.50)))
    if marker_mode == 0:
        my = top + 1 + (ch - 1) // 2
        mx = left + 1 + (cw - 1) // 2
    elif marker_mode == 1:
        edge = int(rng.integers(0, 4))
        my = top + 1 + (0 if edge == 0 else ch - 1 if edge == 1 else int(rng.integers(ch)))
        mx = left + 1 + (0 if edge == 2 else cw - 1 if edge == 3 else int(rng.integers(cw)))
    else:
        my = int(rng.integers(top + 1, top + ch + 1))
        mx = int(rng.integers(left + 1, left + cw + 1))
    return _construct(height, width, top, left, ch, cw, style, gaps, (my, mx), palette)


def _one_run(side):
    positions = np.flatnonzero(side)
    assert positions.size > 0 and np.all(np.diff(positions) == 1), "one contiguous opening per side"
    return int(positions[0]), int(positions[-1])


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    single = np.flatnonzero(counts == 1)
    assert len(single) == 1, "one marker cell"
    fill = int(values[single[0]])
    assert np.sum(counts == np.max(counts)) == 1
    background = int(values[np.argmax(counts)])
    frame = int(next(v for v in values if int(v) not in (fill, background)))
    fy, fx = np.where(inp == frame)
    top, bottom, left, right = int(fy.min()), int(fy.max()), int(fx.min()), int(fx.max())
    height, width = inp.shape
    assert 0 < top < bottom < height - 1 and 0 < left < right < width - 1
    ch, cw = bottom - top - 1, right - left - 1
    assert 3 <= ch <= 13 and 3 <= cw <= 13
    corners = (inp[top, left], inp[top, right], inp[bottom, left], inp[bottom, right])
    assert all(v == frame for v in corners) or all(v == background for v in corners)
    style = "joined" if all(v == frame for v in corners) else "chamfered"
    sides = {"top": inp[top, left + 1:right], "bottom": inp[bottom, left + 1:right],
             "left": inp[top + 1:bottom, left], "right": inp[top + 1:bottom, right]}
    gaps = {}
    for side, cells in sides.items():
        assert np.all((cells == frame) | (cells == background))
        a, b = _one_run(cells == background)
        start = (left + 1 if side in ("top", "bottom") else top + 1) + a
        gaps[side] = (start, b - a + 1)
        line = top if side == "top" else bottom if side == "bottom" else left if side == "left" else right
        if side in ("top", "bottom"):
            assert inp[line, start - 1] == frame and inp[line, start + b - a + 1] == frame
        else:
            assert inp[start - 1, line] == frame and inp[start + b - a + 1, line] == frame
        if style == "chamfered":
            assert a >= 1 and b <= len(cells) - 2
    marker_positions = np.argwhere(inp == fill)
    assert marker_positions.shape == (1, 2)
    my, mx = (int(x) for x in marker_positions[0])
    assert top < my < bottom and left < mx < right
    expected_in = np.full(inp.shape, background, dtype=np.int8)
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            on_horizontal = y in (top, bottom)
            on_vertical = x in (left, right)
            if not (on_horizontal or on_vertical):
                continue
            if on_horizontal and on_vertical and style == "chamfered":
                continue
            if on_horizontal:
                side = "top" if y == top else "bottom"
                start, n = gaps[side]
                if start <= x < start + n:
                    continue
            if on_vertical:
                side = "left" if x == left else "right"
                start, n = gaps[side]
                if start <= y < start + n:
                    continue
            expected_in[y, x] = frame
    expected_in[my, mx] = fill
    expected_out = expected_in.copy()
    for y in range(height):
        for x in range(width):
            if expected_in[y, x] != background:
                continue
            cavity = top < y < bottom and left < x < right
            top_ray = y <= top and gaps["top"][0] <= x < sum(gaps["top"])
            bottom_ray = y >= bottom and gaps["bottom"][0] <= x < sum(gaps["bottom"])
            left_ray = x <= left and gaps["left"][0] <= y < sum(gaps["left"])
            right_ray = x >= right and gaps["right"][0] <= y < sum(gaps["right"])
            if cavity or top_ray or bottom_ray or left_ray or right_ray:
                expected_out[y, x] = fill
    assert np.array_equal(inp, expected_in), "malformed input frame"
    assert np.array_equal(out, expected_out), "incorrect cavity fill or outlet ray"
    assert not np.array_equal(inp, out)
    return {"height": height, "width": width, "cavity_height": ch,
            "cavity_width": cw, "top": top, "bottom": bottom, "left": left,
            "right": right, "style": style, "marker": (my - top - 1, mx - left - 1),
            "palette": (background, frame, fill),
            "gaps": gaps, "gap_widths": tuple(gaps[s][1] for s in _SIDES)}


def _normalized_key(inp, palette):
    norm = np.empty(inp.shape, dtype=np.int8)
    for role, color in enumerate(palette):
        norm[inp == color] = role
    return _grid_key(norm)


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        nt, ns = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
        seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
        styles = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = pair["input"], pair["output"]
                for grid in (inp, out):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert all(1 <= n <= 30 for n in grid.shape)
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert np.all((0 <= grid) & (grid <= 9))
                assert inp.shape == out.shape
                ik, pk = _grid_key(inp), (_grid_key(inp), _grid_key(out))
                assert ik not in seen_inputs and pk not in seen_pairs
                seen_inputs.add(ik)
                seen_pairs.add(pk)
                assert (_rows(inp), _rows(out)) not in _SOURCE_ROW_PAIRS
                meta = _analyze_pair(inp, out)
                styles[split].add(meta["style"])
                nk = _normalized_key(inp, meta["palette"])
                assert nk not in seen_layouts
                seen_layouts.add(nk)
                if split == "train":
                    assert 3 <= meta["cavity_height"] <= 11
                    assert 3 <= meta["cavity_width"] <= 11
        assert styles["train"] == {"joined", "chamfered"}
        assert styles["test"] <= styles["train"]
    except AssertionError:
        raise
    except (TypeError, ValueError, IndexError, KeyError, StopIteration) as exc:
        raise AssertionError("malformed generated puzzle") from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    relationship = str(rng.choice(("stable", "partial", "varied")))
    anchor = _palette(rng, None, relationship)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            for _ in range(120):
                style = ("joined" if index == 0 else "chamfered") if split == "train" and index < 2 else str(rng.choice(("joined", "chamfered")))
                palette = anchor if split == "train" and index == 0 else _palette(rng, anchor, relationship)
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                candidate = _sample_pair(np.random.default_rng(child_seed), style, palette,
                                         "source_train" if split == "train" else "target_test")
                key = _normalized_key(candidate["input"], palette)
                if key in used or (_rows(candidate["input"]), _rows(candidate["output"])) in _SOURCE_ROW_PAIRS:
                    continue
                used.add(key)
                puzzle[split].append(candidate)
                break
            else:
                raise ValueError("could not construct distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, cavities, margins, profiles, palettes, layouts = set(), set(), set(), set(), set(), set()
    styles_by_height = {}
    all_narrow = moderate = opposing_wide = other_side_dominates = centered = 0
    full_joined = near_side = offset_openings = tall = broad = 0
    repeat_roles = full_reassign = test_smaller = test_larger = 0
    within_size_change = within_cavity_change = within_palette_change = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_meta = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_meta = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        within_size_change += len({(m["height"], m["width"]) for m in train_meta}) > 1
        within_cavity_change += len({(m["cavity_height"], m["cavity_width"]) for m in train_meta}) > 1
        within_palette_change += len({m["palette"] for m in train_meta + test_meta}) > 1
        anchor = train_meta[0]["palette"]
        repeat_roles += any(any(a == b for a, b in zip(anchor, m["palette"])) for m in train_meta[1:] + test_meta)
        full_reassign += any(all(a != b for a, b in zip(anchor, m["palette"])) for m in train_meta[1:] + test_meta)
        max_train_area = max(m["height"] * m["width"] for m in train_meta)
        test_smaller += any(m["height"] * m["width"] < max_train_area for m in test_meta)
        test_larger += any(m["height"] * m["width"] > max_train_area for m in test_meta)
        for m in train_meta + test_meta:
            sizes.add((m["height"], m["width"]))
            cavities.add((m["cavity_height"], m["cavity_width"]))
            margins.add((m["top"], m["height"] - m["bottom"] - 1,
                         m["left"], m["width"] - m["right"] - 1))
            profiles.add((m["style"], m["gap_widths"]))
            palettes.add(m["palette"])
            layouts.add((m["height"], m["width"], m["top"], m["left"],
                         m["cavity_height"], m["cavity_width"], m["style"],
                         m["marker"], tuple(m["gaps"][s] for s in _SIDES)))
            styles_by_height.setdefault(m["cavity_height"], set()).add(m["style"])
            widths = m["gap_widths"]
            all_narrow += all(w == 1 for w in widths)
            moderate += all(2 <= w <= 4 for w in widths)
            opposing_wide += (widths[0] >= 3 and widths[1] >= 3) or (widths[2] >= 3 and widths[3] >= 3)
            other_side_dominates += max(widths[1:]) > widths[0]
            full_joined += m["style"] == "joined" and any(
                widths[i] == (m["cavity_width"] if i < 2 else m["cavity_height"])
                for i in range(4))
            near_side += m["marker"][0] in (0, m["cavity_height"] - 1) or m["marker"][1] in (0, m["cavity_width"] - 1)
            offset_openings += m["gaps"]["top"] != m["gaps"]["bottom"]
            tall += m["cavity_height"] >= 2 * m["cavity_width"]
            broad += m["cavity_width"] >= 2 * m["cavity_height"]
            centered += m["marker"] == ((m["cavity_height"] - 1) // 2,
                                         (m["cavity_width"] - 1) // 2)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 75 and len(cavities) >= 45 and len(margins) >= 80
    assert len(profiles) >= 35 and len(palettes) >= 35 and len(layouts) >= 100
    assert sum(len(v) == 2 for v in styles_by_height.values()) >= 4
    assert all_narrow >= 5 and opposing_wide >= 5 and other_side_dominates >= 10 and centered >= 5
    assert moderate >= 5 and full_joined >= 2 and near_side >= 10
    assert offset_openings >= 10 and tall >= 5 and broad >= 5
    assert repeat_roles >= 5 and full_reassign >= 5
    assert test_smaller >= 5 and test_larger >= 3
    assert within_size_change >= 30 and within_cavity_change >= 30
    assert within_palette_change >= 35
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        p = generate_puzzle(3000 + i, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
        validate_generated_puzzle(p)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(4000 + i, train, test)
        assert (train is None or len(p["train"]) == train)
        assert (test is None or len(p["test"]) == test)
    for i, args in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                              {"num_test": 5}, {"num_train": 2, "num_test": 2},
                              {"num_train": 2, "num_test": 3},
                              {"num_train": 3.5}, {"num_test": True})):
        try:
            generate_puzzle(5000 + i, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid count: {args}")
    assert len({_signature(generate_puzzle(s, 6, 2)) for s in range(210, 215)}) >= 4
