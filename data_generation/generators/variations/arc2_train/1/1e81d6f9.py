"""Corner-framed color deletion puzzles."""

import hashlib

import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")
_SOURCE_INPUT_HASHES = {
    "e1b03005a399ee0aa959d3b7bbb07defeac9f6466bcf8bee9b458f0661f7d717",
    "07349a8b0fe7a513fdaf64fecd367f34a419e82a32f34ffc26a8403b74cce1e3",
    "b4a858a21ff488398322bc3faad039c1fe93ce4836c196d57f11f66275754852",
    "d3d1cf6242ab713ce7d85a76ab8ff6c280a52a4fcf20a49c040b16e683841a1d",
}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _map(r, c, height, width, corner):
    if corner[0] == "B":
        r = height - r - 1
    if corner[1] == "R":
        c = width - c - 1
    return r, c


def _geometry(shape, corner, chamber_h, chamber_w):
    height, width = shape
    frame = [(chamber_h, c) for c in range(chamber_w + 1)]
    frame.extend((r, chamber_w) for r in range(chamber_h))
    chamber = [(r, c) for r in range(chamber_h) for c in range(chamber_w)]
    return (
        tuple(_map(r, c, height, width, corner) for r, c in frame),
        tuple(_map(r, c, height, width, corner) for r, c in chamber),
    )


def _background(grid):
    counts = np.bincount(grid.astype(np.uint8).ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= grid.size // 2:
        return None
    return background


def _detect_selector_frames(grid):
    background = _background(grid)
    if background is None:
        return []
    height, width = grid.shape
    found = []
    for corner in _CORNERS:
        for chamber_h in range(2, min(7, height - 2) + 1):
            for chamber_w in range(2, min(7, width - 2) + 1):
                frame, chamber = _geometry(
                    grid.shape, corner, chamber_h, chamber_w
                )
                frame_color = int(grid[frame[0]])
                if frame_color == background:
                    continue
                if any(int(grid[pos]) != frame_color for pos in frame):
                    continue
                keys = [pos for pos in chamber if int(grid[pos]) != background]
                if len(keys) != 1:
                    continue
                key_pos = keys[0]
                key_color = int(grid[key_pos])
                if key_color == frame_color:
                    continue
                found.append({
                    "background": background,
                    "frame_color": frame_color,
                    "selector_color": key_color,
                    "selector_pos": key_pos,
                    "corner": corner,
                    "chamber_h": chamber_h,
                    "chamber_w": chamber_w,
                    "frame_cells": frame,
                    "chamber_cells": chamber,
                })
    return found


def _apply_rule(grid, details):
    output = grid.copy()
    mask = output == details["selector_color"]
    mask[details["selector_pos"]] = False
    output[mask] = details["background"]
    return output


def _source_copy(pair):
    grid = pair["input"]
    if grid.shape != (15, 15):
        return False
    digest = hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest()
    return digest in _SOURCE_INPUT_HASHES


def _neighbors(pos, shape):
    r, c = pos
    height, width = shape
    return [
        (r + dr, c + dc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= r + dr < height and 0 <= c + dc < width
    ]


_PATTERNS = {
    "single": (((0, 0),), ()),
    "bar": (((0, 0), (0, 1), (0, 2)), ()),
    "block": (((0, 0), (0, 1), (1, 0), (1, 1)), ()),
    "irregular": (((0, 0), (1, 0), (1, 1), (2, 1)), ()),
    "mixed": (((0, 1),), ((0, 0), (0, 2))),
    "hole": (((1, 1),), ((0, 1), (1, 0), (1, 2), (2, 1))),
    "split": (((0, 1),), ((0, 0), (0, 2))),
}


def _rotated(points, turns):
    transformed = list(points)
    for _ in range(turns):
        transformed = [(c, -r) for r, c in transformed]
    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    return [(r - min_r, c - min_c) for r, c in transformed]


def _place_pattern(rng, grid, allowed, key_color, companion, name):
    selected, others = _PATTERNS[name]
    joined = [(p, True) for p in selected] + [(p, False) for p in others]
    turns = int(rng.integers(4))
    positions = _rotated([p for p, _ in joined], turns)
    marked = list(zip(positions, [is_key for _, is_key in joined]))
    choices = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cells = [(r + dr, c + dc) for (dr, dc), _ in marked]
            if all(pos in allowed and grid[pos] == grid.dtype.type(-1)
                   for pos in cells):
                choices.append(cells)
    if not choices:
        return False
    cells = _pick(rng, choices)
    for pos, (_, is_key) in zip(cells, marked):
        grid[pos] = key_color if is_key else companion
    return True


def _sample_dimension(rng):
    if rng.random() < 0.13:
        return 15
    return int(rng.integers(8, 31))


def _sample_key_count(rng, regime, area):
    if regime == "target_test" and rng.random() < 0.11:
        return 0
    band = int(rng.choice(3, p=[0.29, 0.44, 0.27]))
    ranges = ((1, 2), (3, 6), (7, 16))
    low, high = ranges[band]
    return min(int(rng.integers(low, high + 1)), max(1, area // 8))


def _generate_pair(rng, regime, key_color, forced_distractor=None,
                   different_background=None, different_frame=None,
                   corner=None, different_chamber=None, force_mixed=False):
    for _ in range(100):
        height, width = _sample_dimension(rng), _sample_dimension(rng)
        max_h, max_w = min(7, height - 3), min(7, width - 3)
        chamber_h = int(rng.integers(2, max_h + 1))
        chamber_w = int(rng.integers(2, max_w + 1))
        if different_chamber == (chamber_h, chamber_w):
            continue
        actual_corner = corner or _pick(rng, _CORNERS)

        role_exclusions = {key_color}
        if forced_distractor is not None:
            role_exclusions.add(forced_distractor)
        background_choices = [
            color for color in range(10)
            if color not in role_exclusions and color != different_background
        ]
        if 0 in background_choices and rng.random() < 0.42:
            background = 0
        else:
            background = _pick(rng, background_choices)
        frame_choices = [
            color for color in range(10)
            if color not in role_exclusions
            and color != background and color != different_frame
        ]
        frame_color = _pick(rng, frame_choices)
        palette_pool = [
            color for color in range(10)
            if color not in (background, frame_color, key_color)
        ]
        palette_size = int(rng.integers(1, len(palette_pool) + 1))
        distractors = list(rng.permutation(palette_pool)[:palette_size])
        if forced_distractor is not None and forced_distractor not in distractors:
            distractors[0] = forced_distractor
        companion = _pick(rng, distractors)

        grid = np.full((height, width), -1, dtype=np.int16)
        frame, chamber = _geometry(grid.shape, actual_corner,
                                   chamber_h, chamber_w)
        protected = set(frame) | set(chamber)
        grid[:, :] = background
        for pos in frame:
            grid[pos] = frame_color
        if chamber_h > 2 and chamber_w > 2 and rng.random() < 0.65:
            local_r = int(rng.integers(1, chamber_h - 1))
            local_c = int(rng.integers(1, chamber_w - 1))
        else:
            local_r = int(rng.integers(chamber_h))
            local_c = int(rng.integers(chamber_w))
        key_pos = _map(local_r, local_c, height, width, actual_corner)
        grid[key_pos] = key_color

        allowed = set()
        for r in range(height):
            for c in range(width):
                pos = (r, c)
                if pos in protected:
                    continue
                if any(max(abs(r - fr), abs(c - fc)) <= 1
                       for fr, fc in frame):
                    continue
                allowed.add(pos)
        if len(allowed) < 8:
            continue
        # Pattern placement uses -1 as the free-cell marker.
        for pos in allowed:
            grid[pos] = -1
        target = _sample_key_count(rng, regime, len(allowed))
        if force_mixed:
            pattern = _pick(rng, ["mixed", "hole", "split"])
        else:
            options = ["single", "bar", "block", "irregular",
                       "mixed", "hole", "split"]
            if target < 4:
                options = [x for x in options if x not in ("block", "irregular")]
            if target < 3:
                options.remove("bar")
            pattern = _pick(rng, options) if target else None
        if pattern is not None:
            needed = len(_PATTERNS[pattern][0])
            if target < needed:
                target = needed
            if not _place_pattern(rng, grid, allowed, key_color,
                                  companion, pattern):
                continue
        placed_keys = sum(int(grid[pos] == key_color) for pos in allowed)
        free = [pos for pos in allowed if grid[pos] == -1]
        rng.shuffle(free)
        for pos in free[:max(0, target - placed_keys)]:
            grid[pos] = key_color
        key_count = sum(int(grid[pos] == key_color) for pos in allowed)

        # Exterior density, palette breadth, and key count are sampled
        # separately. A rival color controls the relative frequency regime.
        frequency = _pick(rng, ("rarer", "tied", "common"))
        if frequency == "common" and key_count <= 1:
            frequency = _pick(rng, ("rarer", "tied"))
        if frequency == "rarer":
            rival_goal = key_count + int(rng.integers(1, 6))
        elif frequency == "tied":
            rival_goal = max(1, key_count)
        else:
            rival_goal = max(1, key_count - int(rng.integers(1, 4)))
        rival = forced_distractor if forced_distractor is not None else companion
        if rival not in distractors:
            rival = companion
        current_rival = sum(int(grid[pos] == rival) for pos in allowed)
        free = [pos for pos in allowed if grid[pos] == -1]
        rng.shuffle(free)
        for pos in free[:max(0, rival_goal - current_rival)]:
            grid[pos] = rival
        density_goal = int(round(len(allowed) * rng.uniform(0.025, 0.20)))
        density_goal = max(density_goal, key_count + 2)
        occupied = sum(int(grid[pos] != -1) for pos in allowed)
        free = [pos for pos in allowed if grid[pos] == -1]
        rng.shuffle(free)
        other_palette = [color for color in distractors if color != rival]
        if not other_palette:
            other_palette = distractors
        color_counts = {
            color: sum(int(grid[pos] == color) for pos in allowed)
            for color in distractors
        }
        limit = (key_count - 1 if frequency == "common"
                 else key_count if frequency == "tied" else len(allowed))
        for pos in free[:max(0, density_goal - occupied)]:
            eligible = [color for color in other_palette
                        if color_counts[color] < limit]
            if not eligible:
                break
            color = _pick(rng, eligible)
            grid[pos] = color
            color_counts[color] += 1
        # Include occasional frame-colored dots, without requiring them.
        if rng.random() < 0.42:
            free = [pos for pos in allowed if grid[pos] == -1]
            rng.shuffle(free)
            for pos in free[:int(rng.integers(1, 4))]:
                grid[pos] = frame_color
        grid[grid == -1] = background
        if _background(grid) != background:
            continue
        grid = grid.astype(np.int8)
        candidates = _detect_selector_frames(grid)
        if len(candidates) != 1:
            continue
        details = candidates[0]
        if (details["corner"], details["chamber_h"], details["chamber_w"],
                details["frame_color"], details["selector_color"],
                details["selector_pos"]) != (
                actual_corner, chamber_h, chamber_w, frame_color,
                key_color, key_pos):
            continue
        pair = {"input": grid, "output": _apply_rule(grid, details)}
        if _source_copy(pair):
            continue
        return pair
    raise RuntimeError("could not construct an unambiguous pair")


def _count_value(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _count_value("num_train", num_train)
    num_test = _count_value("num_test", num_test)
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        num_train = lower + (seed // 4) % (11 - lower)
    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = lower + seed % (11 - lower)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _identity(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _mixed_witness(grid, details):
    key = details["selector_color"]
    bg = details["background"]
    pos = details["selector_pos"]
    for row, col in np.argwhere(grid == key):
        cell = (int(row), int(col))
        if cell == pos:
            continue
        if any(int(grid[neighbor]) not in (bg, key)
               for neighbor in _neighbors(cell, grid.shape)):
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen = set()
    records = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            _validate_grid(grid)
            _validate_grid(output)
            assert grid.shape == output.shape
            identity = _identity(grid)
            assert identity not in seen
            seen.add(identity)
            assert not _source_copy(pair)
            candidates = _detect_selector_frames(grid)
            assert len(candidates) == 1
            details = candidates[0]
            assert np.array_equal(output, _apply_rule(grid, details))
            assert int(output[details["selector_pos"]]) == details["selector_color"]
            if section == "train":
                assert not np.array_equal(grid, output)
            records[section].append((pair, details))
    train_keys = [d["selector_color"] for _, d in records["train"]]
    assert len(set(train_keys)) >= 2
    assert any(_mixed_witness(p["input"], d) for p, d in records["train"])
    # A literal key is preserved as an ordinary color while another key is active.
    assert any(
        np.any((p["input"] == other) & (p["output"] == other))
        for p, d in records["train"]
        for other in set(train_keys) - {d["selector_color"]}
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)

    # The invariant convention is geometric selection followed by deletion.
    # Literal colors, corner, chamber, density, and canvas are pair properties.
    color_order = list(int(x) for x in rng.permutation(10))
    train_keys = color_order[:2]
    train_keys += [_pick(rng, color_order[:min(6, ntrain + 1)])
                   for _ in range(ntrain - 2)]
    regime = _pick(rng, ("held_out", "familiar", "mixed"))
    held = [c for c in color_order if c not in set(train_keys)]
    test_keys = []
    for index in range(ntest):
        if regime == "held_out" or (regime == "mixed" and index % 2 == 0):
            test_keys.append(_pick(rng, held))
        else:
            test_keys.append(_pick(rng, train_keys))

    fixed_corner = bool(rng.random() < 0.20)
    first_corner = _pick(rng, _CORNERS)
    corners = [first_corner]
    for index in range(1, ntrain + ntest):
        if fixed_corner:
            corners.append(first_corner if index < ntrain or rng.random() < 0.5
                           else _pick(rng, [c for c in _CORNERS
                                            if c != first_corner]))
        elif index == 1:
            corners.append(_pick(rng, [c for c in _CORNERS
                                       if c != first_corner]))
        else:
            corners.append(_pick(rng, _CORNERS))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=ntrain + ntest, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_details = None
    for index, (key, corner, pair_seed) in enumerate(
            zip(train_keys + test_keys, corners, pair_seeds)):
        section = "train" if index < ntrain else "test"
        forced = (train_keys[1] if index == 0 else
                  train_keys[0] if index == 1 else None)
        pair_rng = np.random.default_rng(int(pair_seed))
        for _ in range(20):
            pair = _generate_pair(
                pair_rng, "source_train" if section == "train" else "target_test",
                key, forced_distractor=forced,
                different_background=(first_details["background"]
                                      if index == 1 else None),
                different_frame=(first_details["frame_color"]
                                 if index == 1 else None),
                corner=corner,
                different_chamber=((first_details["chamber_h"],
                                    first_details["chamber_w"])
                                   if index == 1 else None),
                force_mixed=(index == 0),
            )
            identity = _identity(pair["input"])
            if identity not in seen:
                break
        else:
            raise RuntimeError("could not make distinct pairs")
        seen.add(identity)
        puzzle[section].append(pair)
        if index == 0:
            first_details = _detect_selector_frames(pair["input"])[0]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(
        np.array_equal(pa[side], pb[side])
        for section in ("train", "test")
        for pa, pb in zip(a[section], b[section])
        for side in ("input", "output")
    ) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(9876, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(9876, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(9877, 7, 3))
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4), (6, 2))):
        puzzle = generate_puzzle(2000 + offset, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        validate_generated_puzzle(puzzle)
    for offset, (train, test) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + offset, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, sizes = set(), set(), set()
    corners, chambers, backgrounds, frames, key_counts = (
        set(), set(), set(), set(), set())
    within_background, within_frame, within_corner = 0, 0, 0
    within_chamber, fixed_train_corner, new_test_corner = 0, 0, 0
    held_out, familiar, zero_test = 0, 0, 0
    mixed, grouped, frame_dots, no_frame_dots = 0, 0, 0, 0
    palettes, frequency_statuses = set(), set()
    compact, long_canvas, full_size, source_size = 0, 0, 0, 0
    key_at_edge, key_interior, isolated_only = 0, 0, 0
    selected_group_three, few_targets, large_targets = 0, 0, 0
    bars, blocks, irregulars, holes, splits = 0, 0, 0, 0, 0
    zero_key, zero_frame, nonzero_background = 0, 0, 0
    sparse_population, large_population = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        details = [_detect_selector_frames(p["input"])[0]
                   for s in ("train", "test") for p in puzzle[s]]
        train_details = details[:len(puzzle["train"])]
        test_details = details[len(puzzle["train"]):]
        within_background += len({d["background"] for d in details}) > 1
        within_frame += len({d["frame_color"] for d in details}) > 1
        within_corner += len({d["corner"] for d in details}) > 1
        within_chamber += len({(d["chamber_h"], d["chamber_w"])
                               for d in details}) > 1
        fixed_train_corner += len({d["corner"] for d in train_details}) == 1
        new_test_corner += any(
            d["corner"] not in {t["corner"] for t in train_details}
            for d in test_details)
        train_key_set = {d["selector_color"] for d in train_details}
        held_out += any(d["selector_color"] not in train_key_set
                        for d in test_details)
        familiar += any(d["selector_color"] in train_key_set
                        for d in test_details)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                d = _detect_selector_frames(grid)[0]
                sizes.add(grid.shape)
                palettes.add(len(np.unique(grid)))
                corners.add(d["corner"])
                chambers.add((d["chamber_h"], d["chamber_w"]))
                backgrounds.add(d["background"])
                frames.add(d["frame_color"])
                zero_key += d["selector_color"] == 0
                zero_frame += d["frame_color"] == 0
                nonzero_background += d["background"] != 0
                compact += max(grid.shape) <= 11
                long_canvas += max(grid.shape) >= 2 * min(grid.shape)
                full_size += 30 in grid.shape
                source_size += grid.shape == (15, 15)
                key_at_edge += (d["selector_pos"][0] in (0, grid.shape[0]-1)
                                or d["selector_pos"][1] in (0, grid.shape[1]-1))
                key_interior += (0 < d["selector_pos"][0] < grid.shape[0]-1
                                 and 0 < d["selector_pos"][1] < grid.shape[1]-1)
                selected = [
                    (int(r), int(c))
                    for r, c in np.argwhere(grid == d["selector_color"])
                    if (int(r), int(c)) != d["selector_pos"]
                ]
                for row, col in selected:
                    cardinal = ((row-1, col), (row+1, col),
                                (row, col-1), (row, col+1))
                    if all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                           and int(grid[r, c]) not in
                           (d["background"], d["selector_color"])
                           for r, c in cardinal):
                        holes += 1
                    horizontal = ((row, col-1), (row, col+1))
                    if all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                           and int(grid[r, c]) not in
                           (d["background"], d["selector_color"])
                           for r, c in horizontal):
                        splits += 1
                key_counts.add(len(selected))
                few_targets += len(selected) in (1, 2)
                large_targets += len(selected) >= 8
                grouped += any(n in selected for pos in selected
                               for n in _neighbors(pos, grid.shape))
                isolated_only += bool(selected) and all(
                    n not in selected for pos in selected
                    for n in _neighbors(pos, grid.shape))
                mixed += _mixed_witness(grid, d)
                has_frame_dots = (np.count_nonzero(grid == d["frame_color"])
                                  > len(d["frame_cells"]))
                frame_dots += has_frame_dots
                no_frame_dots += not has_frame_dots
                exterior_counts = {
                    color: int(np.count_nonzero(grid == color))
                    for color in range(10)
                    if color not in (d["background"], d["frame_color"])
                }
                exterior_counts[d["selector_color"]] -= 1
                rivals = [count for color, count in exterior_counts.items()
                          if color != d["selector_color"]]
                if any(count > len(selected) for count in rivals):
                    frequency_statuses.add("rarer")
                elif any(count == len(selected) for count in rivals):
                    frequency_statuses.add("tied")
                else:
                    frequency_statuses.add("most")
                occupied = int(np.count_nonzero(grid != d["background"]))
                sparse_population += occupied <= 20
                large_population += occupied >= 65
                left = set(selected)
                while left:
                    start = left.pop()
                    stack = [start]
                    component = {start}
                    while stack:
                        pos = stack.pop()
                        for neighbor in _neighbors(pos, grid.shape):
                            if neighbor in left:
                                left.remove(neighbor)
                                stack.append(neighbor)
                                component.add(neighbor)
                    selected_group_three += len(component) >= 3
                    rows = [pos[0] for pos in component]
                    cols = [pos[1] for pos in component]
                    extent_h = max(rows) - min(rows) + 1
                    extent_w = max(cols) - min(cols) + 1
                    bars += (len(component) >= 3
                             and min(extent_h, extent_w) == 1)
                    blocks += (extent_h >= 2 and extent_w >= 2
                               and len(component) == extent_h * extent_w)
                    irregulars += (extent_h >= 2 and extent_w >= 2
                                   and 3 <= len(component)
                                   < extent_h * extent_w)
                assert np.array_equal(pair["output"], _apply_rule(grid, d))
                assert not _source_copy(pair)
        zero_test += any(np.array_equal(p["input"], p["output"])
                         for p in puzzle["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 45 and (15, 15) in sizes
    assert corners == set(_CORNERS) and len(chambers) >= 20
    assert len(backgrounds) == 10 and len(frames) == 10
    assert within_background >= 38 and within_frame >= 38
    assert within_corner >= 20 and held_out >= 8 and familiar >= 8
    assert within_chamber == 42 and fixed_train_corner >= 4
    assert new_test_corner >= 8
    assert min(key_counts) == 0 and max(key_counts) >= 8
    assert zero_test >= 1 and mixed >= 20 and grouped >= 20
    assert selected_group_three >= 20 and isolated_only >= 20
    assert bars >= 5 and blocks >= 5 and irregulars >= 5
    assert holes >= 5 and splits >= 5
    assert few_targets >= 20 and large_targets >= 20
    assert zero_key >= 5 and zero_frame >= 5 and nonzero_background >= 50
    assert key_at_edge >= 20 and key_interior >= 20
    assert compact >= 5 and long_canvas >= 15 and full_size >= 5
    assert source_size >= 1 and len(sizes) >= 45
    assert min(palettes) <= 4 and max(palettes) == 10
    assert frequency_statuses == {"rarer", "tied", "most"}
    assert sparse_population >= 5 and large_population >= 5
    assert frame_dots >= 15 and no_frame_dots >= 15
