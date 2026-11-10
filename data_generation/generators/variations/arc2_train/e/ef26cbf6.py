"""Aligned-compartment color transfer puzzles."""

import numpy as np


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        draw = rng.random()
        if draw < 0.24:
            return 2, 1
        if draw < 0.42:
            return 3, int(rng.integers(1, 3))
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(4, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _sample_roles(rng, reserved):
    return tuple(int(x) for x in rng.choice(
        [c for c in range(10) if c != reserved], size=3, replace=False
    ))


def _sample_geometry(rng, first=False):
    # Geometry is independent of the orientation and the side carrying keys.
    if first and rng.random() < 0.22:
        return (3, (3, 3, 3), 3, 3)
    choices = np.arange(1, 9)
    weights = np.array([7, 13, 23, 21, 13, 12, 7, 4], dtype=float)
    if first:
        lanes = int(rng.integers(3, 6))
    else:
        lanes = int(rng.choice(choices, p=weights / weights.sum()))
    spans = [2] * lanes
    budget = 30 - (3 * lanes - 1)
    for index in rng.permutation(lanes):
        extra = int(rng.integers(0, min(5, budget) + 1))
        spans[int(index)] += extra
        budget -= extra
    first_depth = int(rng.integers(2, 8))
    second_depth = int(rng.integers(2, 8))
    if first_depth == second_depth and rng.random() < 0.72:
        second_depth = 2 + (second_depth - 1) % 6
    return lanes, tuple(spans), first_depth, second_depth


def _pattern_mask(rng, rows, cols, allow_single):
    area = rows * cols
    draw = rng.random()
    mask = np.zeros((rows, cols), dtype=bool)
    if allow_single and draw < 0.12:
        mask.flat[int(rng.integers(area))] = True
        return mask
    if draw < 0.20:
        mask[:, :] = True
        return mask
    if draw < 0.34 and rows >= 3 and cols >= 3:
        # A rectangular ring keeps a visible enclosed background hole.
        top = int(rng.integers(rows - 2))
        left = int(rng.integers(cols - 2))
        bottom = int(rng.integers(top + 3, rows + 1))
        right = int(rng.integers(left + 3, cols + 1))
        mask[top:bottom, left:right] = True
        mask[top + 1:bottom - 1, left + 1:right - 1] = False
        return mask
    if draw < 0.51:
        if rng.random() < 0.5:
            row = int(rng.integers(rows))
            length = int(rng.integers(2, cols + 1))
            start = int(rng.integers(cols - length + 1))
            mask[row, start:start + length] = True
        else:
            col = int(rng.integers(cols))
            length = int(rng.integers(2, rows + 1))
            start = int(rng.integers(rows - length + 1))
            mask[start:start + length, col] = True
        return mask
    target = int(rng.integers(2, area + 1))
    if draw < 0.76:
        mask.flat[rng.choice(area, size=target, replace=False)] = True
        return mask
    # Connected growth; this complements the disconnected subset mode.
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    chosen = {start}
    while len(chosen) < target:
        frontier = set()
        for r, c in chosen:
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= point[0] < rows and 0 <= point[1] < cols and point not in chosen:
                    frontier.add(point)
        ordered = sorted(frontier)
        chosen.add(ordered[int(rng.integers(len(ordered)))])
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _draw_key_colors(rng, lanes, roles, reserved, split, first):
    pool = [c for c in range(10) if c not in roles and (split == "test" or c != reserved)]
    if first or rng.random() < 0.56:
        colors = list(int(c) for c in rng.choice(pool, size=lanes, replace=lanes > len(pool)))
    else:
        colors = list(int(c) for c in rng.choice(pool, size=lanes, replace=True))
        if lanes >= 2:
            colors[1] = colors[0]
    if split == "test" and rng.random() < 0.53:
        colors[int(rng.integers(lanes))] = reserved
    return colors


def _generate_pair(rng, layout, geometry, roles, reserved, split, first=False):
    lanes, spans, near_depth, far_depth = geometry
    background, divider, placeholder = roles
    length = sum(spans) + lanes - 1
    cross = near_depth + 1 + far_depth
    vertical = layout in ("left", "right")
    inp = np.full((length, cross) if vertical else (cross, length), background, dtype=np.int8)
    if vertical:
        inp[:, near_depth] = divider
    else:
        inp[near_depth, :] = divider
    first_bank, second_bank = [], []
    offset = 0
    for index, span in enumerate(spans):
        if vertical:
            first_bank.append((slice(offset, offset + span), slice(0, near_depth)))
            second_bank.append((slice(offset, offset + span), slice(near_depth + 1, cross)))
            if index < lanes - 1:
                inp[offset + span, :] = divider
        else:
            first_bank.append((slice(0, near_depth), slice(offset, offset + span)))
            second_bank.append((slice(near_depth + 1, cross), slice(offset, offset + span)))
            if index < lanes - 1:
                inp[:, offset + span] = divider
        offset += span + 1
    if layout in ("left", "top"):
        keys, patterns = first_bank, second_bank
    else:
        keys, patterns = second_bank, first_bank
    colors = _draw_key_colors(rng, lanes, roles, reserved, split, first)
    for index, (color, key_slice, pattern_slice) in enumerate(zip(colors, keys, patterns)):
        key = inp[key_slice]
        key[int(rng.integers(key.shape[0])), int(rng.integers(key.shape[1]))] = color
        pattern = inp[pattern_slice]
        pattern[_pattern_mask(rng, *pattern.shape, allow_single=index > 0)] = placeholder
    out = inp.copy()
    for color, pattern_slice in zip(colors, patterns):
        view = out[pattern_slice]
        view[view == placeholder] = color
    return inp, out


def _lattice_candidate(inp, orientation, main):
    height, width = inp.shape
    vertical = orientation == "vertical"
    cross = width if vertical else height
    length = height if vertical else width
    if not 2 <= main <= cross - 3:
        return None
    divider = int(inp[0, main] if vertical else inp[main, 0])
    if vertical:
        if not np.all(inp[:, main] == divider):
            return None
        separators = [r for r in range(height) if np.all(inp[r, :] == divider)]
    else:
        if not np.all(inp[main, :] == divider):
            return None
        separators = [c for c in range(width) if np.all(inp[:, c] == divider)]
    bounds = [-1] + separators + [length]
    spans = tuple(bounds[i + 1] - bounds[i] - 1 for i in range(len(bounds) - 1))
    if not 1 <= len(spans) <= 9 or min(spans) < 2:
        return None
    divider_mask = np.zeros(inp.shape, dtype=bool)
    if vertical:
        divider_mask[:, main] = True
        divider_mask[separators, :] = True
    else:
        divider_mask[main, :] = True
        divider_mask[:, separators] = True
    if not np.array_equal(inp == divider, divider_mask):
        return None
    first_bank, second_bank = [], []
    for i in range(len(spans)):
        segment = slice(bounds[i] + 1, bounds[i + 1])
        if vertical:
            first_bank.append((segment, slice(0, main)))
            second_bank.append((segment, slice(main + 1, width)))
        else:
            first_bank.append((slice(0, main), segment))
            second_bank.append((slice(main + 1, height), segment))
    return divider, spans, main, cross - main - 1, first_bank, second_bank, divider_mask


def _infer_input(inp):
    candidates = []
    for orientation in ("vertical", "horizontal"):
        cross = inp.shape[1] if orientation == "vertical" else inp.shape[0]
        for main in range(2, cross - 2):
            lattice = _lattice_candidate(inp, orientation, main)
            if lattice is None:
                continue
            divider, spans, near_depth, far_depth, first_bank, second_bank, mask = lattice
            for background in range(10):
                if background == divider or not np.any(inp[~mask] == background):
                    continue
                for key_side, (keys, patterns) in enumerate(((first_bank, second_bank), (second_bank, first_bank))):
                    key_colors = []
                    for section in keys:
                        marks = inp[section][inp[section] != background]
                        if len(marks) != 1:
                            break
                        key_colors.append(int(marks[0]))
                    if len(key_colors) != len(spans):
                        continue
                    placeholder = None
                    counts = []
                    for section in patterns:
                        marks = inp[section][inp[section] != background]
                        unique = np.unique(marks)
                        if len(marks) < 1 or len(unique) != 1:
                            break
                        value = int(unique[0])
                        if placeholder is None:
                            placeholder = value
                        if value != placeholder:
                            break
                        counts.append(int(len(marks)))
                    if len(counts) != len(spans) or max(counts) < 2:
                        continue
                    if placeholder in (background, divider) or any(c in (background, divider, placeholder) for c in key_colors):
                        continue
                    layout = ("left", "right")[key_side] if orientation == "vertical" else ("top", "bottom")[key_side]
                    candidates.append({
                        "orientation": orientation, "layout": layout, "background": background,
                        "divider": divider, "placeholder": placeholder, "lanes": len(spans),
                        "spans": spans, "near_depth": near_depth, "far_depth": far_depth,
                        "key_colors": tuple(key_colors), "pattern_counts": tuple(counts),
                        "pattern_slices": patterns,
                    })
    assert len(candidates) == 1, f"input has {len(candidates)} interpretations"
    return candidates[0]


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    analysis = _infer_input(inp)
    expected = inp.copy()
    for color, section in zip(analysis["key_colors"], analysis["pattern_slices"]):
        view = expected[section]
        view[view == analysis["placeholder"]] = color
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return analysis


def _value_signature(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


_SOURCE_PAIRS = (
    (("0004100", "0704011", "0004010", "4444444", "0004110", "0304010", "0004111", "4444444", "0004110", "0804011", "0004101"),
     ("0004700", "0704077", "0004070", "4444444", "0004330", "0304030", "0004333", "4444444", "0004880", "0804088", "0004808")),
    (("00040004000", "03040204060", "00040004000", "44444444444", "10040104101", "01041114101", "11141014010"),
     ("00040004000", "03040204060", "00040004000", "44444444444", "30040204606", "03042224606", "33342024060")),
    (("1004000", "0104060", "1104000", "4444444", "0014000", "0114020", "1004000", "4444444", "1104000", "0104080", "1114000"),
     ("6004000", "0604060", "6604000", "4444444", "0024000", "0224020", "2004000", "4444444", "8804000", "0804080", "8884000")),
)


def _matches_source_pair(inp, out):
    rows = tuple("".join(str(int(c)) for c in row) for row in inp)
    result = tuple("".join(str(int(c)) for c in row) for row in out)
    return (rows, result) in _SOURCE_PAIRS


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not _matches_source_pair(inp, out)
            signature = (_value_signature(inp), _value_signature(out))
            assert signature[0] not in seen_inputs and signature not in seen_pairs
            seen_inputs.add(signature[0])
            seen_pairs.add(signature)
            analyses[split].append(_analyze_pair(inp, out))
    # A multi-lane demonstration with distinct key colors establishes aligned
    # transfer, including when another pair has a one-cell pattern or one lane.
    assert any(a["lanes"] >= 2 and len(set(a["key_colors"])) >= 2 for a in analyses["train"])
    assert len({a["layout"] for a in analyses["train"]}) >= 2


def _new_unique_pair(master_rng, layout, geometry, roles, reserved, split, first, used):
    for _ in range(200):
        pair_rng = np.random.default_rng(int(master_rng.integers(0, np.iinfo(np.int64).max)))
        inp, out = _generate_pair(pair_rng, layout, geometry, roles, reserved, split, first)
        if _value_signature(inp) in used or _matches_source_pair(inp, out):
            continue
        try:
            _analyze_pair(inp, out)
        except AssertionError:
            continue
        used.add(_value_signature(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct an unambiguous, unique pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    reserved = int(rng.integers(10))
    stable_roles = rng.random() < 0.56
    initial_roles = _sample_roles(rng, reserved)
    roles = initial_roles
    layouts = list(rng.choice(("left", "right", "top", "bottom"), size=4, replace=False))
    train_layouts = layouts[:2] + [str(rng.choice(layouts)) for _ in range(num_train - 2)]
    puzzle = {"train": [], "test": []}
    geometries = []
    used = set()
    for index, layout in enumerate(train_layouts):
        if index and not stable_roles:
            for _ in range(20):
                next_roles = _sample_roles(rng, reserved)
                if next_roles != roles:
                    roles = next_roles
                    break
        geometry = _sample_geometry(rng, first=index == 0)
        pair = _new_unique_pair(rng, layout, geometry, roles, reserved, "train", index == 0, used)
        puzzle["train"].append(pair)
        geometries.append(geometry)
    reverse = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    for _ in range(num_test):
        if not stable_roles:
            for _ in range(20):
                next_roles = _sample_roles(rng, reserved)
                if next_roles != roles:
                    roles = next_roles
                    break
        anchor = str(rng.choice(train_layouts))
        mode = rng.random()
        if mode < 0.27:
            layout = anchor
        elif mode < 0.67:
            layout = reverse[anchor]
        else:
            layout = str(rng.choice(("top", "bottom") if anchor in ("left", "right") else ("left", "right")))
        geometry = geometries[int(rng.integers(len(geometries)))] if rng.random() < 0.28 else _sample_geometry(rng)
        puzzle["test"].append(_new_unique_pair(rng, layout, geometry, roles, reserved, "test", False, used))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple(tuple((_value_signature(p["input"]), _value_signature(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (201, None, 1), (202, None, 4), (203, 2, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed, train, test))
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": True}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    layouts, counts, depths, spans, role_modes, repeats, singles, fulls = set(), set(), set(), set(), set(), 0, 0, 0
    coverage = {name: 0 for name in ("small", "one_lane", "six_lanes", "bottom_test", "shared_layout", "new_layout", "novel_key", "new_geometry", "reused_geometry")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        analyses = train + test
        train_layouts = {a["layout"] for a in train}
        train_keys = {c for a in train for c in a["key_colors"]}
        train_geometries = {(a["lanes"], a["spans"], a["near_depth"], a["far_depth"]) for a in train}
        coverage["small"] += len(train) <= 3
        coverage["one_lane"] += any(a["lanes"] == 1 for a in analyses)
        coverage["six_lanes"] += any(a["lanes"] >= 6 for a in analyses)
        coverage["bottom_test"] += any(a["layout"] == "bottom" for a in test)
        coverage["shared_layout"] += any(a["layout"] in train_layouts for a in test)
        coverage["new_layout"] += any(a["layout"] not in train_layouts for a in test)
        coverage["novel_key"] += any(set(a["key_colors"]) - train_keys for a in test)
        coverage["new_geometry"] += any((a["lanes"], a["spans"], a["near_depth"], a["far_depth"]) not in train_geometries for a in test)
        coverage["reused_geometry"] += any((a["lanes"], a["spans"], a["near_depth"], a["far_depth"]) in train_geometries for a in test)
        role_modes.add(len({(a["background"], a["divider"], a["placeholder"]) for a in analyses}) == 1)
        for a in analyses:
            layouts.add(a["layout"])
            depths.add((a["near_depth"], a["far_depth"]))
            spans.add(a["spans"])
            repeats += len(set(a["key_colors"])) < a["lanes"]
            singles += 1 in a["pattern_counts"]
            fulls += any(c == s * (a["far_depth"] if a["layout"] in ("left", "top") else a["near_depth"]) for c, s in zip(a["pattern_counts"], a["spans"]))
    assert layouts == {"left", "right", "top", "bottom"}
    assert len(counts) >= 8 and len(depths) >= 15 and len(spans) >= 25
    assert role_modes == {True, False} and repeats > 10 and singles > 5 and fulls > 5
    assert all(value >= 5 for value in coverage.values()), coverage
