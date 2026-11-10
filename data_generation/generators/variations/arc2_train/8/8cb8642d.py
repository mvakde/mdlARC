"""Framed rectangles whose inner-corner marker supplies a stretched-X ink."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = {
    "4b8085d5be87dad52ad0f49c1523a8be8b2f9af12bd7eb2d0794af8473b2c91e",
    "b8d7d57b7368496a429d453f7edd6b5e018736abcf665b2d0a4ad0002a06b783",
    "a4e8b2b17b59201e59e528064f3349af1bab794530fab6444c9e15f1bf7e029e",
    "4542ba91101b8b1e9069c7b4ea1d4c82c00a7c502b912efe0ab0247beb97d428",
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_input(grid):
    h, w = grid.shape
    content = f"{h}x{w}:" + "".join(str(int(v)) for v in grid.flat)
    return hashlib.sha256(content.encode()).hexdigest() in _SOURCE_INPUT_DIGESTS


def _extract_panels(grid, background):
    """Find complete four-connected rectangles, including their lone marker."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    panels = []
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or int(grid[sy, sx]) == background:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and int(grid[ny, nx]) != background):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            y0, y1 = min(y for y, _ in cells), max(y for y, _ in cells) + 1
            x0, x1 = min(x for _, x in cells), max(x for _, x in cells) + 1
            rect = grid[y0:y1, x0:x1]
            rh, rw = rect.shape
            _require(rh >= 5 and rw >= 5 and len(cells) == rh * rw,
                     "panel must be a filled rectangle with a 3x3 interior")
            values, counts = np.unique(rect, return_counts=True)
            _require(len(values) == 2 and sorted(int(v) for v in counts) == [1, rh * rw - 1],
                     "panel needs exactly one exceptional cell")
            marker_index = int(np.argmin(counts))
            ink = int(values[marker_index])
            frame = int(values[1 - marker_index])
            _require(ink != frame and ink != background and frame != background,
                     "panel colors must have distinct roles")
            _require(np.all(rect[0] == frame) and np.all(rect[-1] == frame)
                     and np.all(rect[:, 0] == frame) and np.all(rect[:, -1] == frame),
                     "one-cell perimeter must be intact")
            my, mx = (int(v) for v in np.argwhere(rect == ink)[0])
            _require((my, mx) in ((1, 1), (1, rw - 2),
                                  (rh - 2, 1), (rh - 2, rw - 2)),
                     "marker must lie at an inner corner")
            panels.append((y0, y1, x0, x1, frame, ink, my, mx))
    _require(panels, "pair must have at least one panel")
    return sorted(panels)


def _draw_stretched_x(interior, ink):
    h, w = interior.shape
    if h <= w:
        for r in range(h):
            d = min(r, h - 1 - r)
            interior[r, d] = ink
            interior[r, w - 1 - d] = ink
        for r in {(h - 1) // 2, h // 2}:
            d = min(r, h - 1 - r)
            interior[r, d:w - d] = ink
    else:
        for c in range(w):
            d = min(c, w - 1 - c)
            interior[d, c] = ink
            interior[h - 1 - d, c] = ink
        for c in {(w - 1) // 2, w // 2}:
            d = min(c, w - 1 - c)
            interior[d:h - d, c] = ink


def _transform_input(input_grid, background):
    output = input_grid.copy()
    for y0, y1, x0, x1, _, ink, _, _ in _extract_panels(input_grid, background):
        interior = output[y0 + 1:y1 - 1, x0 + 1:x1 - 1]
        interior[:, :] = background
        _draw_stretched_x(interior, ink)
    return output


def _valid_count(value, name):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")


def _resolve_counts(rng, num_train, num_test):
    _valid_count(num_train, "num_train")
    _valid_count(num_test, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _randint(rng, low, high):
    return int(rng.integers(low, high + 1))


def _partition(rng, count):
    """Random shelves; transposition later gives column-based layouts too."""
    if count == 1:
        return [1]
    if count == 2:
        return [2] if _randint(rng, 0, 1) else [1, 1]
    possible = []
    for a in range(1, min(4, count) + 1):
        for b in range(0, min(4, count - a) + 1):
            for c in range(0, min(4, count - a - b) + 1):
                for d in range(0, min(4, count - a - b - c) + 1):
                    parts = [v for v in (a, b, c, d) if v]
                    if sum(parts) == count and len(parts) <= 4:
                        possible.append(parts)
    return possible[_randint(rng, 0, len(possible) - 1)]


def _dimension(rng, cap_h, cap_w, profile, large):
    """Select a square, near oblong, or strongly oblong interior when feasible."""
    cap_h, cap_w = max(3, cap_h), max(3, cap_w)
    if profile == "square":
        maximum = min(cap_h, cap_w)
        low = max(3, maximum - 5) if large else 3
        side = _randint(rng, low, maximum)
        return side, side
    vertical = profile in ("tall", "near_tall", "far_tall", "thin_tall")
    a, b = (cap_h, cap_w) if vertical else (cap_w, cap_h)
    gap = 1 if profile.startswith("near") else (4 if profile.startswith("far") else 2)
    gap = min(gap, a - 3)
    if gap < 1:
        return _dimension(rng, cap_h, cap_w, "square", large)
    short_max = min(b, a - gap)
    if large:
        short = _randint(rng, max(3, short_max - 4), short_max)
    else:
        short = _randint(rng, 3, short_max)
    if profile.startswith("near"):
        long = min(a, short + _randint(rng, 1, min(2, a - short)))
    elif profile.startswith("thin"):
        short = min(short, 4, a - 1)
        long = _randint(rng, max(short + 1, a - 4), a)
    else:
        long = _randint(rng, short + gap, a)
    return (long, short) if vertical else (short, long)


def _assign_colors(rng, specs, background, colors):
    available = [c for c in colors if c != background]
    _require(len(available) >= 2, "insufficient local colors")
    mode = _randint(rng, 0, 3)
    common = available[_randint(rng, 0, len(available) - 1)]
    for spec in specs:
        frame = available[_randint(rng, 0, len(available) - 1)]
        ink_options = [c for c in available if c != frame]
        ink = ink_options[_randint(rng, 0, len(ink_options) - 1)]
        if mode == 1 and len(specs) > 1:
            ink = common
            if frame == ink:
                other = [c for c in available if c != ink]
                frame = other[_randint(rng, 0, len(other) - 1)]
        if mode == 2 and len(specs) > 1:
            frame = common
            ink = [c for c in available if c != frame][
                _randint(rng, 0, len(available) - 2)]
        spec["frame"], spec["ink"] = frame, ink
    if mode == 3 and len(specs) > 1:
        a = specs[0]
        b = specs[1]
        b["frame"], b["ink"] = a["ink"], a["frame"]


def _make_pair(rng, count, profiles, colors, background, target,
               novel=None, protect_shapes=False):
    groups = _partition(rng, count)
    if len(groups) == 4 or max(groups) == 4:
        gap_y = _randint(rng, 1, 2)
        gap_x = _randint(rng, 1, 2)
    else:
        gap_y = _randint(rng, 1, 4)
        gap_x = _randint(rng, 1, 4)
    rows = len(groups)
    cap_h = (29 - (rows - 1) * gap_y) // rows - 2
    compact = target and _randint(rng, 0, 3) == 0
    large = target and not compact and (_randint(rng, 0, 2) != 0)
    specs = []
    row_specs = []
    shuffled_profiles = list(profiles)
    rng.shuffle(shuffled_profiles)
    k = 0
    for group_size in groups:
        cap_w = (30 - (group_size - 1) * gap_x) // group_size - 2
        row = []
        for _ in range(group_size):
            profile = shuffled_profiles[k]
            limit = 8 if compact else (26 if target else 10)
            ih, iw = _dimension(rng, min(cap_h, limit),
                                min(cap_w, limit), profile, large)
            spec = {"ih": ih, "iw": iw, "corner": _randint(rng, 0, 3)}
            specs.append(spec)
            row.append(spec)
            k += 1
        row_specs.append(row)
    # Sometimes the marker corner is shared; otherwise it is chosen independently.
    if count >= 2 and _randint(rng, 0, 4) == 0:
        corner = _randint(rng, 0, 3)
        for spec in specs:
            spec["corner"] = corner
    if count >= 3 and _randint(rng, 0, 4) == 0:
        corners = [int(v) for v in rng.choice(4, size=count, replace=True)]
        corners[:3] = [int(v) for v in rng.permutation(4)[:3]]
        for spec, corner in zip(specs, corners):
            spec["corner"] = corner
    _assign_colors(rng, specs, background, colors)
    if novel is not None:
        spec = specs[_randint(rng, 0, count - 1)]
        if _randint(rng, 0, 1):
            spec["ink"] = novel
        else:
            spec["frame"] = novel
    if (not protect_shapes and novel is None and len(row_specs[0]) >= 2
            and _randint(rng, 0, 5) == 0):
        first, second = row_specs[0][:2]
        for field in ("ih", "iw", "frame", "ink"):
            second[field] = first[field]
        second["corner"] = (first["corner"] + _randint(rng, 1, 3)) % 4
    row_heights = [max(s["ih"] + 2 for s in row) for row in row_specs]
    row_widths = [sum(s["iw"] + 2 for s in row) + (len(row) - 1) * gap_x
                  for row in row_specs]
    content_h = sum(row_heights) + (rows - 1) * gap_y
    content_w = max(row_widths)
    if target and not compact and _randint(rng, 0, 1):
        height = _randint(rng, max(content_h + 1, 23), 30)
        width = _randint(rng, max(content_w, 23), 30)
    else:
        extra = 3 if compact else 5
        height = min(30, content_h + 1 + _randint(rng, 0, min(extra, 29 - content_h)))
        width = min(30, content_w + _randint(rng, 0, min(extra, 30 - content_w)))
    grid = np.full((height, width), background, dtype=np.int8)
    y = _randint(rng, 1, height - content_h)
    align_column = all(group == 1 for group in groups) and _randint(rng, 0, 1)
    column_x = _randint(rng, 0, width - content_w) if align_column else None
    for row, row_h, row_w in zip(row_specs, row_heights, row_widths):
        x = column_x if align_column else _randint(rng, 0, width - row_w)
        align_row = len(row) >= 3 and _randint(rng, 0, 1)
        for spec in row:
            py = y if align_row else y + _randint(rng, 0, row_h - spec["ih"] - 2)
            oh, ow = spec["ih"] + 2, spec["iw"] + 2
            grid[py:py + oh, x:x + ow] = spec["frame"]
            corner = spec["corner"]
            my = py + (1 if corner < 2 else spec["ih"])
            mx = x + (1 if corner % 2 == 0 else spec["iw"])
            grid[my, mx] = spec["ink"]
            x += ow + gap_x
        y += row_h + gap_y
    if _randint(rng, 0, 1):
        grid = grid.T.copy()
    return {"input": grid, "output": _transform_input(grid, background)}


def _profile(rng):
    options = ("square", "near_tall", "near_wide", "tall", "wide",
               "far_tall", "far_wide", "thin_tall", "thin_wide")
    return options[_randint(rng, 0, len(options) - 1)]


def _pair_plan(rng, train_count, test_count):
    plans = {"train": [], "test": []}
    # The first two demonstrations jointly show a square and both long axes.
    first = _randint(rng, 2, 3)
    if _randint(rng, 0, 1):
        second = _randint(rng, 1, 2)
        first_profiles = ["far_tall", "far_wide"] + [
            _profile(rng) for _ in range(first - 2)]
        second_profiles = ["square"] + [_profile(rng) for _ in range(second - 1)]
    else:
        second = 2
        near = "near_tall" if _randint(rng, 0, 1) else "near_wide"
        first_profiles = ["square", near] + [_profile(rng) for _ in range(first - 2)]
        second_profiles = ["far_tall", "far_wide"]
    rng.shuffle(first_profiles)
    rng.shuffle(second_profiles)
    plans["train"].append((first, first_profiles))
    plans["train"].append((second, second_profiles))
    for _ in range(2, train_count):
        count = _randint(rng, 1, 5)
        style = _randint(rng, 0, 5)
        if style == 0:
            shapes = ["square"] * count
        elif style == 1:
            shapes = [p for p in (_profile(rng) for _ in range(count))]
            shapes = [p if p != "square" else "near_wide" for p in shapes]
        else:
            shapes = [_profile(rng) for _ in range(count)]
        plans["train"].append((count, shapes))
    for _ in range(test_count):
        count = _randint(rng, 1, 5)
        style = _randint(rng, 0, 5)
        if style == 0:
            shapes = ["square"] * count
        elif style == 1:
            shapes = [rng.choice(("far_tall", "far_wide", "thin_tall", "thin_wide"))
                      for _ in range(count)]
        else:
            shapes = [_profile(rng) for _ in range(count)]
        plans["test"].append((count, shapes))
    return plans


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    palette_size = _randint(rng, 4, 7)
    train_colors = [int(v) for v in rng.choice(10, size=palette_size, replace=False)]
    bg_base = train_colors[0]
    variable_background = bool(_randint(rng, 0, 2) == 0)
    plans = _pair_plan(rng, train_count, test_count)
    child_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    new_color = next((c for c in range(10) if c not in train_colors), None)
    new_color_test = new_color is not None and _randint(rng, 0, 1) == 1
    index = 0
    for split in ("train", "test"):
        for count, profiles in plans[split]:
            local_rng = np.random.default_rng(int(child_seeds[index]))
            index += 1
            for attempt in range(12):
                familiar = sorted({int(v) for pair in puzzle["train"]
                                   for v in np.unique(pair["input"])})
                color_pool = (familiar if split == "test" and not new_color_test
                              else train_colors)
                if variable_background:
                    bg = color_pool[_randint(local_rng, 0, len(color_pool) - 1)]
                else:
                    bg = bg_base
                colors = color_pool.copy()
                novel = None
                if split == "test" and new_color_test:
                    novel = new_color
                pair = _make_pair(local_rng, count, profiles, colors, bg,
                                  split == "test", novel,
                                  split == "train" and len(puzzle["train"]) < 2)
                key = _key(pair["input"])
                if key not in seen and not _source_input(pair["input"]):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _oracle_output(input_grid, panels, background):
    """Independent distance-to-edges construction used by validation."""
    result = input_grid.copy()
    for y0, y1, x0, x1, _, ink, _, _ in panels:
        h, w = y1 - y0 - 2, x1 - x0 - 2
        for r in range(h):
            for c in range(w):
                if h <= w:
                    depth = min(r, h - 1 - r)
                    marked = c == depth or c == w - 1 - depth
                    marked |= r in ((h - 1) // 2, h // 2) and depth <= c < w - depth
                else:
                    depth = min(c, w - 1 - c)
                    marked = r == depth or r == h - 1 - depth
                    marked |= c in ((w - 1) // 2, w // 2) and depth <= r < h - depth
                result[y0 + 1 + r, x0 + 1 + c] = ink if marked else background
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle requires exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_shapes = set()
    train_stretch = {"tall": False, "wide": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair requires input and output")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D array")
                _require(all(1 <= v <= 30 for v in grid.shape)
                         and np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((0 <= grid) & (grid <= 9))),
                         "invalid grid size, type, or color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "output dimensions must match and drawing must change input")
            ikey, okey = _key(inp), _key(out)
            _require(ikey not in seen_inputs and (ikey, okey) not in seen_pairs,
                     "duplicate pair")
            _require(not _source_input(inp), "original source input copied")
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            bg = int(inp[0, 0])
            _require((np.all(inp[0, :] == bg) or np.all(inp[:, 0] == bg))
                     and (np.all(out[0, :] == bg) or np.all(out[:, 0] == bg)),
                     "an exterior side must identify the local background")
            panels = _extract_panels(inp, bg)
            _require(len(panels) <= 6, "too many panels")
            _require(np.array_equal(out, _oracle_output(inp, panels, bg)),
                     "output is not the local stretched X")
            if split == "train":
                for y0, y1, x0, x1, _, _, _, _ in panels:
                    ih, iw = y1 - y0 - 2, x1 - x0 - 2
                    _require(ih <= 10 and iw <= 10,
                             "training panel exceeds demonstration scale")
                    train_shapes.add("square" if ih == iw else
                                     ("tall" if ih > iw else "wide"))
                    if ih - iw >= 2:
                        train_stretch["tall"] = True
                    if iw - ih >= 2:
                        train_stretch["wide"] = True
    _require(train_shapes == {"square", "tall", "wide"},
             "training must demonstrate squares and both oblong directions")
    _require(all(train_stretch.values()),
             "training must show a central stroke on both long axes")


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(314159)
    _require(_signature(a) == _signature(generate_puzzle(314159)),
             "automatic generation is not deterministic")
    b = generate_puzzle(271828, 7, 3)
    _require(_signature(b) == _signature(generate_puzzle(271828, 7, 3)),
             "explicit generation is not deterministic")
    requests = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                (2, None), (10, None))
    for i, (nt, nq) in enumerate(requests):
        p = generate_puzzle(100 + i, nt, nq)
        validate_generated_puzzle(p)
        _require(nt is None or len(p["train"]) == nt, "train count ignored")
        _require(nq is None or len(p["test"]) == nq, "test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(55, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    signatures, shapes, counts, train_counts, test_counts = set(), set(), set(), set(), set()
    backgrounds, corners, color_counts = set(), set(), set()
    review = {name: 0 for name in (
        "single_test", "four_test", "all_square", "all_oblong",
        "near_square", "long_thin", "large_interior", "compact_test",
        "large_test", "horizontal", "vertical", "staggered", "edge",
        "repeated_ink", "repeated_frame", "cross_role", "shared_corner",
        "three_corners", "marker_only_change", "variable_background",
        "novel_test", "familiar_test", "mixed_axes", "first_near",
        "same_frame_different_inks", "same_ink_different_frames",
        "tight_gap", "wide_gap",
    )}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        first_panels = _extract_panels(p["train"][0]["input"],
                                        int(p["train"][0]["input"][0, 0]))
        review["first_near"] += any(
            abs((a[1] - a[0]) - (a[3] - a[2])) == 1 for a in first_panels)
        train_colors = {int(v) for pair in p["train"] for v in np.unique(pair["input"])}
        pair_backgrounds = {int(pair["input"][0, 0])
                            for split in ("train", "test") for pair in p[split]}
        review["variable_background"] += len(pair_backgrounds) > 1
        novel = any({int(v) for v in np.unique(pair["input"])} - train_colors
                    for pair in p["test"])
        review["novel_test"] += novel
        review["familiar_test"] += not novel
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                backgrounds.add(int(inp[0, 0]))
                shapes.add(inp.shape)
                color_counts.add(len(np.unique(inp)))
                panels = _extract_panels(inp, int(inp[0, 0]))
                counts.add(len(panels))
                panel_shapes = []
                positions = []
                frames = []
                inks = []
                marker_corners = []
                for y0, y1, x0, x1, frame, ink, my, mx in panels:
                    ih, iw = y1 - y0 - 2, x1 - x0 - 2
                    panel_shapes.append("square" if ih == iw else
                                        ("tall" if ih > iw else "wide"))
                    positions.append((y0, y1, x0, x1))
                    frames.append(frame)
                    inks.append(ink)
                    marker_corners.append(((1, 1), (1, iw), (ih, 1),
                                           (ih, iw)).index((my, mx)))
                    review["near_square"] += abs(ih - iw) == 1
                    review["large_interior"] += max(ih, iw) > 11
                    review["long_thin"] += min(ih, iw) <= 4 and max(ih, iw) >= 10
                review["all_square"] += len(panels) >= 2 and all(
                    shape == "square" for shape in panel_shapes)
                review["all_oblong"] += len(panels) >= 2 and all(
                    shape != "square" for shape in panel_shapes)
                review["mixed_axes"] += len(panels) >= 3 and len(set(panel_shapes)) >= 2
                review["edge"] += any(y0 == 0 or x0 == 0 or y1 == inp.shape[0]
                                      or x1 == inp.shape[1] for y0, y1, x0, x1 in positions)
                review["repeated_ink"] += len(panels) >= 2 and len(set(inks)) < len(inks)
                review["repeated_frame"] += len(panels) >= 2 and len(set(frames)) < len(frames)
                review["cross_role"] += bool(set(frames) & set(inks))
                review["shared_corner"] += len(panels) >= 2 and len(set(marker_corners)) == 1
                review["three_corners"] += len(set(marker_corners)) >= 3
                if len(panels) >= 3:
                    review["horizontal"] += len({a[0] for a in positions}) == 1
                    review["vertical"] += len({a[2] for a in positions}) == 1
                    review["staggered"] += (len({a[0] for a in positions}) > 1
                                            and len({a[2] for a in positions}) > 1)
                for i, a in enumerate(panels):
                    for b in panels[i + 1:]:
                        same = (a[1] - a[0], a[3] - a[2], a[4], a[5]) == (
                            b[1] - b[0], b[3] - b[2], b[4], b[5])
                        review["marker_only_change"] += same and a[6:] != b[6:]
                        review["same_frame_different_inks"] += a[4] == b[4] and a[5] != b[5]
                        review["same_ink_different_frames"] += a[5] == b[5] and a[4] != b[4]
                        dx = max(0, a[2] - b[3], b[2] - a[3])
                        dy = max(0, a[0] - b[1], b[0] - a[1])
                        if dx > 0 and dy == 0:
                            review["tight_gap"] += dx == 1
                            review["wide_gap"] += dx >= 4
                        if dy > 0 and dx == 0:
                            review["tight_gap"] += dy == 1
                            review["wide_gap"] += dy >= 4
                if split == "test":
                    review["single_test"] += len(panels) == 1
                    review["four_test"] += len(panels) >= 4
                    review["compact_test"] += all(v < 24 for v in inp.shape)
                    review["large_test"] += all(v >= 24 for v in inp.shape)
                corners.update(((1, 1), (1, x1 - x0 - 2),
                                (y1 - y0 - 2, 1),
                                (y1 - y0 - 2, x1 - x0 - 2)).index((my, mx))
                               for y0, y1, x0, x1, _, _, my, mx in panels)
    _require(len(signatures) >= 40 and len(shapes) >= 30,
             "insufficient puzzle or canvas diversity")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts insufficiently diverse")
    _require({1, 2, 3, 4}.issubset(counts) and len(corners) == 4,
             "panel count or marker corners insufficiently diverse")
    _require(len(backgrounds) >= 5 and len(color_counts) >= 4,
             "color diversity insufficient")
    _require(all(value > 0 for value in review.values()),
             "a requested review regime was not observed")
    source_dimensions = {(13, 14), (12, 15), (17, 18), (19, 24)}
    _require(any(pair["input"].shape in source_dimensions
                 for split in ("train", "test")
                 for pair in generate_puzzle(59)[split]),
             "original canvas dimensions were accidentally banned")
