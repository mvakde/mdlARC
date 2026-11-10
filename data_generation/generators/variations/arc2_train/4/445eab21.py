import numpy as np


_SOURCE_ROWS = (
    ("6666600000", "6000600000", "6000600000", "6666600000", "0000000000", "0077777700", "0070000700", "0070000700", "0077777700", "0000000000"),
    ("0777700000", "0700700000", "0700700000", "0777700000", "0000000000", "0008888800", "0008000800", "0008000800", "0008888800", "0000000000"),
    ("0444444000", "0400004000", "0400004000", "0400004000", "0400004000", "0400004000", "0444444000", "0000000222", "0000000202", "0000000222"),
    ("3333309999", "3000309009", "3000309009", "3000309009", "3000309009", "3000309009", "3000309009", "3000309009", "3333309009", "0000009999"),
)
_SOURCE_OUTPUT_COLORS = (7, 8, 4, 3)


def _checked_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    if num_train is not None:
        num_train = _checked_count("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + ((seed + seed // 4) % (11 - low))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _interior(dims):
    return (dims[0] - 2) * (dims[1] - 2)


def _border(dims):
    return 2 * (dims[0] + dims[1]) - 4


def _dimension_pools():
    dims = [(a, b) for a in range(3, 21) for b in range(a, 21) if a * b <= 180]
    pools = {name: [] for name in ("border_foil", "shortside_foil", "equal_perimeter", "easy", "close", "large_gap", "mixed")}
    for winner in dims:
        for loser in dims:
            aw, al = winner[0] * winner[1], loser[0] * loser[1]
            if aw <= al or _interior(winner) <= _interior(loser):
                continue
            if winner[0] + loser[0] > 29:
                continue
            crossed = (winner[0] - loser[0]) * (winner[1] - loser[1]) < 0
            if crossed and _border(winner) < _border(loser):
                pools["border_foil"].append((winner, loser))
            if crossed and winner[0] < loser[0]:
                pools["shortside_foil"].append((winner, loser))
            if _border(winner) == _border(loser):
                pools["equal_perimeter"].append((winner, loser))
            if winner[0] >= loser[0] and winner[1] >= loser[1]:
                pools["easy"].append((winner, loser))
            if aw - al <= 3:
                pools["close"].append((winner, loser))
            if aw >= 2 * al:
                pools["large_gap"].append((winner, loser))
            if crossed:
                pools["mixed"].append((winner, loser))
    return pools


_DIMENSION_POOLS = _dimension_pools()


def _choose_dimensions(rng, category, multi_witness=False):
    candidates = _DIMENSION_POOLS[category]
    if multi_witness:
        candidates = [
            pair for pair in candidates
            if pair[0][1] <= 14 and pair[1][1] <= 14
            and 40 <= pair[0][0] * pair[0][1] <= 70
            and pair[1][0] * pair[1][1] > 30
            and _border(pair[0]) < 34
            and _interior(pair[0]) > 12
        ]
    choice = candidates[int(rng.integers(len(candidates)))]
    return tuple(choice[0]), tuple(choice[1])


def _oriented(rng, dims):
    return dims[::-1] if bool(rng.integers(2)) else dims


def _boxes_separated(a, b, same_color=False):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    row_gap = max(br - (ar + ah), ar - (br + bh))
    col_gap = max(bc - (ac + aw), ac - (bc + bw))
    # A gap of zero means the borders touch; allow only diagonal corner contact.
    return (row_gap >= 1) or (col_gap >= 1) or (not same_color and row_gap == 0 and col_gap == 0)


def _draw_frame(grid, row, col, height, width, color):
    grid[row, col:col + width] = color
    grid[row + height - 1, col:col + width] = color
    grid[row:row + height, col] = color
    grid[row:row + height, col + width - 1] = color


def _place_frames(rng, specs, background):
    n = len(specs)
    heights = [dims[0] for dims, _ in specs]
    widths = [dims[1] for dims, _ in specs]
    for attempt in range(22):
        if attempt >= 18:
            canvas_h = canvas_w = 30
        elif n == 2:
            possible = []
            if sum(widths) + 1 <= 30:
                possible.append((max(heights), sum(widths) + 1))
            if sum(heights) + 1 <= 30:
                possible.append((sum(heights) + 1, max(widths)))
            if not possible:
                raise ValueError("frames cannot share an ARC canvas")
            base_h, base_w = possible[int(rng.integers(len(possible)))]
            canvas_h = min(30, base_h + int(rng.integers(0, 10)))
            canvas_w = min(30, base_w + int(rng.integers(0, 10)))
        else:
            canvas_h = int(rng.integers(max(19, max(heights)), 31))
            canvas_w = int(rng.integers(max(19, max(widths)), 31))
        boxes = []
        order = list(rng.permutation(n))
        for index in order:
            h, w = specs[index][0]
            if h > canvas_h or w > canvas_w:
                break
            for _ in range(180):
                r = int(rng.integers(0, canvas_h - h + 1))
                c = int(rng.integers(0, canvas_w - w + 1))
                proposal = (r, c, h, w)
                if all(_boxes_separated(proposal, old[:4], specs[index][1] == specs[old[4]][1]) for old in boxes):
                    boxes.append(proposal + (index,))
                    break
            else:
                break
        if len(boxes) != n:
            continue
        r0 = min(b[0] for b in boxes)
        c0 = min(b[1] for b in boxes)
        r1 = max(b[0] + b[2] for b in boxes)
        c1 = max(b[1] + b[3] for b in boxes)
        top = int(rng.integers(0, min(7, 31 - (r1 - r0))))
        left = int(rng.integers(0, min(7, 31 - (c1 - c0))))
        bottom = int(rng.integers(0, min(7, 31 - (r1 - r0) - top)))
        right = int(rng.integers(0, min(7, 31 - (c1 - c0) - left)))
        grid = np.full((top + r1 - r0 + bottom, left + c1 - c0 + right), background, dtype=np.int8)
        placed = []
        for r, c, h, w, index in boxes:
            color = specs[index][1]
            rr, cc = top + r - r0, left + c - c0
            _draw_frame(grid, rr, cc, h, w, color)
            placed.append((index, rr, cc, h, w))
        return grid, placed
    raise ValueError("could not place frames")


def _pair(rng, background, category, count, position_preference=None, multi_witness=False):
    if category == "compact":
        available = [x for x in range(10) if x != background]
        colors = rng.permutation(available)
        grid = np.full((10, 10), background, dtype=np.int8)
        winner_above = bool(rng.integers(2))
        if winner_above:
            winner_row, loser_row = 0, 6
        else:
            winner_row, loser_row = 5, 0
        winner_col = int(rng.integers(0, 6))
        loser_col = int(rng.integers(0, 5))
        _draw_frame(grid, winner_row, winner_col, 5, 5, int(colors[0]))
        _draw_frame(grid, loser_row, loser_col, 4, 6, int(colors[1]))
        return grid, np.full((2, 2), int(colors[0]), dtype=np.int8)
    for _ in range(60):
        canonical_winner, canonical_loser = _choose_dimensions(rng, category, multi_witness)
        winner_dims = _oriented(rng, canonical_winner)
        loser_dims = _oriented(rng, canonical_loser)
        if count == 2 and winner_dims[0] + loser_dims[0] > 29 and winner_dims[1] + loser_dims[1] > 29:
            continue
        available = [x for x in range(10) if x != background]
        colors = list(rng.permutation(available))
        winner_color, loser_color = int(colors[0]), int(colors[1])
        specs = [(winner_dims, winner_color), (loser_dims, loser_color)]
        if count >= 3:
            for extra in range(count - 2):
                if multi_witness and extra < 2:
                    extra_dims = (5, 5) if extra == 0 else (5, 6)
                    extra_color = int(colors[2])
                else:
                    limit = max(3, min(7, int(np.sqrt(canonical_loser[0] * canonical_loser[1] - 1))))
                    h = int(rng.integers(3, limit + 1))
                    w = int(rng.integers(3, max(3, min(8, (canonical_loser[0] * canonical_loser[1] - 1) // h)) + 1))
                    if (h - 2) * (w - 2) >= _interior(canonical_winner):
                        h, w = 3, 3
                    extra_dims = _oriented(rng, (h, w))
                    extra_color = int(colors[2 + extra]) if extra + 2 < len(colors) and bool(rng.integers(2)) else int(colors[int(rng.integers(1, min(3 + extra, len(colors))))])
                specs.append((extra_dims, extra_color))
        try:
            grid, placed = _place_frames(rng, specs, background)
        except ValueError:
            continue
        by_index = {index: (r, c, h, w) for index, r, c, h, w in placed}
        transforms = list(rng.permutation(4))
        for transform in transforms:
            flip_rows, flip_cols = bool(transform & 1), bool(transform & 2)
            positions = {}
            for index in (0, 1):
                r, c, h, w = by_index[index]
                positions[index] = (
                    grid.shape[0] - r - h if flip_rows else r,
                    grid.shape[1] - c - w if flip_cols else c,
                )
            if position_preference is None or (positions[0] < positions[1]) == position_preference:
                if flip_rows:
                    grid = np.flip(grid, axis=0)
                if flip_cols:
                    grid = np.flip(grid, axis=1)
                grid = grid.copy()
                break
        else:
            continue
        return grid, np.full((2, 2), winner_color, dtype=np.int8)
    raise ValueError("could not construct comparison")


def _fingerprint(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    # Output color has no fixed identity. Only the invariant comparison and 2x2 output persist.
    backgrounds = list(rng.permutation(10))
    variable_background = bool(rng.integers(2))
    profile = int(rng.integers(3))
    witnesses = ["border_foil", "shortside_foil"]
    rng.shuffle(witnesses)
    first_position = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    total = num_train + num_test
    for index in range(total):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        if split == "train" and local_index < 2:
            category = witnesses[local_index]
            count = 4 if local_index == 0 else 2
            multi_witness = local_index == 0
            position = first_position if local_index == 0 else not first_position
        else:
            if split == "train":
                choices = ("easy", "easy", "close", "mixed", "large_gap") if profile == 0 else ("easy", "close", "mixed", "equal_perimeter", "large_gap")
            else:
                choices = ("equal_perimeter", "equal_perimeter", "mixed", "close") if profile == 0 else (("mixed", "border_foil", "shortside_foil", "close") if profile == 1 else ("easy", "close", "large_gap", "equal_perimeter"))
            category = choices[int(rng.integers(len(choices)))]
            count = int(rng.choice([2, 2, 2, 3, 4, 5]))
            multi_witness = False
            position = None
            if split == "test" and profile == 2 and local_index == 0:
                category, count = "compact", 2
        background = int(backgrounds[index % 10]) if variable_background else int(backgrounds[0])
        for _ in range(50):
            child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            input_grid, output_grid = _pair(child_rng, background, category, count, position, multi_witness)
            fingerprint = _fingerprint(input_grid)
            if split == "train" and local_index == 1 and output_grid[0, 0] == puzzle["train"][0]["output"][0, 0]:
                continue
            if fingerprint not in seen:
                break
        else:
            raise ValueError("could not generate distinct pairs")
        seen.add(fingerprint)
        puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask)))
    result = []
    while pending:
        start = pending.pop()
        component = [start]
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _analyze_input(grid):
    colors = [int(x) for x in np.unique(grid)]
    candidates = []
    for background in colors:
        frames = []
        valid = True
        for color in colors:
            if color == background:
                continue
            for component in _components(grid == color):
                rows = [p[0] for p in component]
                cols = [p[1] for p in component]
                r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
                h, w = r1 - r0 + 1, c1 - c0 + 1
                if h < 3 or w < 3 or len(component) != _border((h, w)):
                    valid = False
                    break
                crop = grid[r0:r1 + 1, c0:c1 + 1]
                if not (np.all(crop[0] == color) and np.all(crop[-1] == color) and np.all(crop[:, 0] == color) and np.all(crop[:, -1] == color) and np.all(crop[1:-1, 1:-1] == background)):
                    valid = False
                    break
                frames.append({"color": color, "height": h, "width": w, "area": h * w, "interior": _interior((h, w)), "border": _border((h, w)), "bbox": (r0, c0, r1, c1)})
            if not valid:
                break
        if valid and 2 <= len(frames) <= 5:
            candidates.append({"background": background, "frames": frames})
    assert len(candidates) == 1
    return candidates[0]


def _duplicates_source_pair(input_grid, output_grid):
    if input_grid.shape != (10, 10) or output_grid.shape != (2, 2):
        return False
    rows = tuple("".join(str(int(x)) for x in row) for row in input_grid)
    return any(rows == original and np.all(output_grid == color) for original, color in zip(_SOURCE_ROWS, _SOURCE_OUTPUT_COLORS))


def _top_two(frames):
    ordered = sorted(frames, key=lambda x: x["area"], reverse=True)
    assert ordered[0]["area"] > ordered[1]["area"]
    return ordered[:2]


def _is_border_foil(winner, loser):
    return winner["border"] < loser["border"] and winner["height"] != loser["height"] and winner["width"] != loser["width"]


def _is_shortside_foil(winner, loser):
    return min(winner["height"], winner["width"]) < min(loser["height"], loser["width"])


def _has_aggregate_color_foil(frames):
    winner, _ = _top_two(frames)
    by_color = {}
    for frame in frames:
        if frame is not winner:
            by_color.setdefault(frame["color"], []).append(frame)
    return any(len(group) >= 2 and sum(f["border"] for f in group) > winner["border"] for group in by_color.values())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_analyses = []
    all_analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            identity = _fingerprint(input_grid)
            pair_identity = (identity, _fingerprint(output_grid))
            assert identity not in seen_inputs and pair_identity not in seen_pairs
            seen_inputs.add(identity)
            seen_pairs.add(pair_identity)
            assert not _duplicates_source_pair(input_grid, output_grid)
            analysis = _analyze_input(input_grid)
            frames = analysis["frames"]
            for i, first in enumerate(frames):
                for second in frames[i + 1:]:
                    if first["color"] == second["color"]:
                        ar0, ac0, ar1, ac1 = first["bbox"]
                        br0, bc0, br1, bc1 = second["bbox"]
                        assert max(br0 - ar1 - 1, ar0 - br1 - 1) >= 1 or max(bc0 - ac1 - 1, ac0 - bc1 - 1) >= 1
            winner, runner_up = _top_two(frames)
            assert winner["interior"] > max(f["interior"] for f in frames if f is not winner)
            assert output_grid.shape == (2, 2) and np.all(output_grid == winner["color"])
            assert not np.array_equal(input_grid, output_grid)
            all_analyses.append(analysis)
            if split == "train":
                train_analyses.append(analysis)
    comparisons = [_top_two(a["frames"]) for a in train_analyses]
    assert any(_is_border_foil(*pair) for pair in comparisons)
    assert any(_is_shortside_foil(*pair) for pair in comparisons)
    assert len({int(np.unique(pair["output"])[0]) for pair in puzzle["train"]}) >= 2
    positions = [w["bbox"][:2] < l["bbox"][:2] for w, l in comparisons]
    assert any(positions) and not all(positions)
    assert any(len(a["frames"]) >= 4 and len({f["color"] for f in a["frames"]}) < len(a["frames"]) for a in train_analyses)
    assert any(_has_aggregate_color_foil(a["frames"]) for a in train_analyses)
    # All comparisons, even when extra frames occur, use the same observable outer-area rule.
    assert all(2 <= len(a["frames"]) <= 5 for a in all_analyses)
    assert len({len(a["frames"]) for a in all_analyses}) >= 2


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (6, 3), (None, None)):
        puzzle = generate_puzzle(101, *args)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(101, *args))
        if args[0] is not None:
            assert len(puzzle["train"]) == args[0]
        if args[1] is not None:
            assert len(puzzle["test"]) == args[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(101, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    canvas_shapes, frame_shapes, backgrounds, counts = set(), set(), set(), set()
    border_foils = shortside_foils = mixed_backgrounds = repeated_colors = 0
    equal_perimeters = one_cell_gaps = large_gaps = mixed_orientations = 0
    aggregate_foils = large_frames = outer_margins = compact_canvases = 0
    role_changes = varying_counts = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_backgrounds = set()
        local_foregrounds = set()
        local_frame_counts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_input(pair["input"])
                w, l = _top_two(analysis["frames"])
                assert pair["output"].shape == (2, 2) and np.all(pair["output"] == w["color"])
                assert not _duplicates_source_pair(pair["input"], pair["output"])
                canvas_shapes.add(pair["input"].shape)
                backgrounds.add(analysis["background"])
                local_backgrounds.add(analysis["background"])
                counts.add(len(analysis["frames"]))
                local_frame_counts.add(len(analysis["frames"]))
                for frame in analysis["frames"]:
                    frame_shapes.add((frame["height"], frame["width"]))
                    local_foregrounds.add(frame["color"])
                border_foils += _is_border_foil(w, l)
                shortside_foils += _is_shortside_foil(w, l)
                repeated_colors += len({f["color"] for f in analysis["frames"]}) < len(analysis["frames"])
                equal_perimeters += w["border"] == l["border"]
                one_cell_gaps += w["area"] - l["area"] == 1
                large_gaps += w["area"] >= 2 * l["area"]
                mixed_orientations += (w["height"] > w["width"]) != (l["height"] > l["width"]) and w["height"] != w["width"] and l["height"] != l["width"]
                aggregate_foils += _has_aggregate_color_foil(analysis["frames"])
                large_frames += any(max(f["height"], f["width"]) > 14 for f in analysis["frames"])
                compact_canvases += pair["input"].shape == (10, 10)
                margins = (
                    min(f["bbox"][0] for f in analysis["frames"]),
                    min(f["bbox"][1] for f in analysis["frames"]),
                    pair["input"].shape[0] - 1 - max(f["bbox"][2] for f in analysis["frames"]),
                    pair["input"].shape[1] - 1 - max(f["bbox"][3] for f in analysis["frames"]),
                )
                outer_margins += max(margins) >= 3
        mixed_backgrounds += len(local_backgrounds) > 1
        role_changes += bool(local_backgrounds & local_foregrounds)
        varying_counts += len(local_frame_counts) > 1
    assert len(canvas_shapes) >= 30 and len(frame_shapes) >= 60
    assert len(backgrounds) == 10 and counts == {2, 3, 4, 5}
    assert border_foils >= 42 and shortside_foils >= 42
    assert mixed_backgrounds >= 10 and repeated_colors >= 42
    assert role_changes >= 8 and varying_counts == 42
    assert equal_perimeters >= 20 and one_cell_gaps >= 8 and large_gaps >= 20
    assert mixed_orientations >= 30 and aggregate_foils >= 42
    assert large_frames >= 30 and outer_margins >= 30 and compact_canvases >= 8
