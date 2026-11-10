"""Generate ARC puzzles in which paired markers cross a local scaffold run."""

import numpy as np


_SOURCE_INPUTS = {
    ("000000000000", "440444444444", "000000000000", "000000000000",
     "000008888800", "888888888888", "000008888800", "000000000000",
     "000000060000", "060000000000", "000006000606", "000000000000"),
    ("000000000000", "000070000000", "007000700000", "000000000700",
     "000070000000", "033000000330", "300330333330", "000033300020",
     "000000000000", "000000220000", "202200000000", "000000000000"),
    ("00000030000000", "00900030005000", "00000030000000",
     "00900333000000", "00000333000000", "00900333005000",
     "00000030000000", "00900333000000", "00900333000000",
     "00900030005000", "00000030000000", "00900030005000"),
    ("00000040000000", "00100440006000", "00000440000000",
     "00100444000300", "00100444030000", "00100444006000",
     "00000444000000", "01000444000000", "01000040006000",
     "00010040000000", "00010040006000"),
}


def _counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (type(value) is not int or not lo <= value <= hi):
            raise ValueError("Invalid pair count")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _put(grid, orientation, lane, pos, color):
    if orientation == "horizontal":
        grid[lane, pos] = color
    else:
        grid[pos, lane] = color


def _roles(rng, count=None, background=None, scaffold=None, pool=None):
    colors = list(range(10))
    if background is None:
        background = 0 if rng.random() < .78 else int(rng.integers(1, 10))
    colors.remove(background)
    if scaffold is None:
        scaffold = int(rng.choice([c for c in colors if c != 0]))
    colors.remove(scaffold)
    if pool is None:
        if count is None:
            count = int(rng.integers(2, 5))
        pool = tuple(int(c) for c in rng.choice(colors, size=count, replace=False))
    return background, scaffold, tuple(pool)


def _profile(rng, lane_count, size):
    mode = str(rng.choice(("thin", "blocks", "jagged", "broad", "broken")))
    span = int(rng.integers(max(6, lane_count // 2), lane_count + 1))
    offset = int(rng.integers(0, lane_count - span + 1))
    occupied = list(range(offset, offset + span))
    if mode == "broken" and span >= 8:
        gap = int(rng.integers(offset + 2, offset + span - 2))
        occupied.remove(gap)
        if span >= 13 and rng.random() < .5:
            gap2 = int(rng.integers(offset + 2, offset + span - 2))
            if gap2 in occupied:
                occupied.remove(gap2)
    base_center = int(rng.integers(3, size - 3))
    block_lo = int(rng.integers(offset, offset + max(1, span // 2)))
    block_hi = min(offset + span, block_lo + int(rng.integers(2, max(4, span // 2 + 2))))
    profile = {}
    shift = 0
    for lane in occupied:
        if mode in ("jagged", "broken") and rng.random() < .35:
            shift = int(np.clip(shift + int(rng.choice((-1, 1))), -2, 2))
        center = base_center + shift
        if mode == "thin":
            width = 1 if rng.random() < .78 else int(rng.integers(2, 5))
        elif mode == "blocks":
            width = int(rng.integers(3, 7)) if block_lo <= lane < block_hi else 1
        elif mode == "broad":
            width = int(rng.integers(3, 8))
        elif mode == "broken":
            width = int(rng.integers(1, 5))
        else:
            width = int(rng.integers(1, 7))
        width = min(width, max(2, size // 3))
        start = int(np.clip(center - width // 2, 2, size - 3))
        end = int(np.clip(start + width - 1, 2, size - 3))
        start = max(2, end - width + 1)
        profile[lane] = (start, end)
    return profile, mode


def _make_pair(rng, regime, orientation, roles, require_control=False):
    if rng.random() < .09:
        height, width = (int(rng.integers(25, 31)), int(rng.integers(20, 31)))
    else:
        height, width = (int(rng.integers(10, 24)), int(rng.integers(10, 24)))
    lane_count, size = (height, width) if orientation == "horizontal" else (width, height)
    profile, _ = _profile(rng, lane_count, size)
    bg, scaffold, palette = roles
    inp = np.full((height, width), bg, dtype=np.int8)
    for lane, (start, end) in profile.items():
        for pos in range(start, end + 1):
            _put(inp, orientation, lane, pos, scaffold)
    out = inp.copy()
    scaffold_lanes = list(profile)
    max_active = min(len(scaffold_lanes), max(3, lane_count - 2), 10)
    if regime == "target_test" and rng.random() < .57:
        low = min(4, max_active)
    else:
        low = 2
    active_count = int(rng.integers(low, max_active + 1))
    active_lanes = [int(v) for v in rng.choice(scaffold_lanes, size=active_count, replace=False)]
    unused = [v for v in range(lane_count) if v not in active_lanes]
    max_controls = min(len(unused), max(1, lane_count // 2))
    control_count = int(rng.integers(1 if require_control else 0, max_controls + 1))
    control_lanes = [int(v) for v in rng.choice(unused, size=control_count, replace=False)] if control_count else []
    for i, lane in enumerate(active_lanes):
        start, end = profile[lane]
        d1 = int(rng.integers(1, start + 1))
        d2 = int(rng.integers(1, size - end))
        c1 = int(rng.choice(palette))
        c2 = int(rng.choice(palette))
        if i == 0 and require_control:
            c2 = int(rng.choice([c for c in palette if c != c1]))
            if d1 == d2 and start >= 3:
                d1 = 3
        p1, p2 = start - d1, end + d2
        _put(inp, orientation, lane, p1, c1)
        _put(inp, orientation, lane, p2, c2)
        _put(out, orientation, lane, end + 1, c1)
        _put(out, orientation, lane, start - 1, c2)
    for j, lane in enumerate(control_lanes):
        if lane in profile:
            start, end = profile[lane]
        else:
            center = size // 2
            start = end = center
        side = int(rng.integers(0, 2))
        if require_control and j == 0:
            side = 0
        positions = list(range(0, start)) if side == 0 else list(range(end + 1, size))
        if not positions:
            continue
        limit = min(len(positions), 5)
        multiplicity = int(rng.integers(1, limit + 1))
        if require_control and j == 0 and limit >= 2:
            multiplicity = max(2, multiplicity)
        for pos in rng.choice(positions, size=multiplicity, replace=False):
            color = int(rng.choice(palette))
            _put(inp, orientation, lane, int(pos), color)
            _put(out, orientation, lane, int(pos), color)
    if np.array_equal(inp, out):
        # Bounded repair of the rare all-adjacent, same-color draw.
        lane = active_lanes[0]
        start, end = profile[lane]
        c1 = int(rng.choice(palette))
        c2 = int(rng.choice([c for c in palette if c != c1]))
        _put(inp, orientation, lane, start - 1, c1)
        _put(inp, orientation, lane, end + 1, c2)
        _put(out, orientation, lane, start - 1, c2)
        _put(out, orientation, lane, end + 1, c1)
    return inp, out


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    palette_mode = str(rng.choice(("shared", "mixed", "independent"), p=(.20, .30, .50)))
    shared = _roles(rng)
    train_orientations = [str(rng.choice(("horizontal", "vertical"))) for _ in range(train_count)]
    if rng.random() < .25:
        train_orientations = [train_orientations[0]] * train_count
    elif len(set(train_orientations)) == 1:
        train_orientations[-1] = "horizontal" if train_orientations[0] == "vertical" else "vertical"
    test_orientations = [str(rng.choice(train_orientations)) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, orientations in (("source_train", train_orientations), ("target_test", test_orientations)):
        key = "train" if regime == "source_train" else "test"
        for i, orientation in enumerate(orientations):
            if palette_mode == "shared" or (palette_mode == "mixed" and rng.random() < .42):
                roles = shared
            else:
                roles = _roles(rng)
            # Pair seeds keep retry behavior local and deterministic.
            for _ in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                inp, out = _make_pair(pair_rng, regime, orientation, roles, require_control=(regime == "source_train" and i == 0))
                if np.count_nonzero(inp == roles[0]) <= inp.size // 2:
                    continue
                if _input_candidates(inp) != [(roles[0], roles[1], orientation)]:
                    continue
                sig = (inp.shape, inp.tobytes())
                if sig not in seen:
                    seen.add(sig)
                    puzzle[key].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("Unable to make unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _line(grid, orientation, lane):
    return grid[lane, :] if orientation == "horizontal" else grid[:, lane]


def _input_candidates(inp):
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    found = []
    for scaffold in (int(v) for v in values if int(v) != bg):
        for orientation in ("horizontal", "vertical"):
            lines = inp if orientation == "horizontal" else inp.T
            active = 0
            runs = 0
            valid = True
            for line in lines:
                positions = np.flatnonzero(line == scaffold)
                if not len(positions):
                    continue
                runs += 1
                start, end = int(positions[0]), int(positions[-1])
                if start < 1 or end >= len(line) - 1 or len(positions) != end - start + 1:
                    valid = False
                    break
                markers = np.flatnonzero((line != bg) & (line != scaffold))
                first = sum(int(p < start) for p in markers)
                second = sum(int(p > end) for p in markers)
                if first + second != len(markers) or (first and second and (first != 1 or second != 1)):
                    valid = False
                    break
                active += first == second == 1
            if valid and active >= 2 and runs >= active:
                found.append((bg, scaffold, orientation))
    return found


def _inspect_pair(inp, out, bg, scaffold, orientation):
    lane_count, size = (inp.shape[0], inp.shape[1]) if orientation == "horizontal" else (inp.shape[1], inp.shape[0])
    expected = inp.copy()
    runs = []
    active = []
    controls = []
    for lane in range(lane_count):
        line = _line(inp, orientation, lane)
        positions = np.flatnonzero(line == scaffold)
        markers = np.flatnonzero((line != bg) & (line != scaffold))
        if not len(positions):
            controls.append((lane, len(markers), "blank")) if len(markers) else None
            continue
        start, end = int(positions[0]), int(positions[-1])
        if end - start + 1 != len(positions) or start < 1 or end > size - 2:
            return None
        runs.append((lane, start, end))
        first = [int(p) for p in markers if p < start]
        second = [int(p) for p in markers if p > end]
        if len(first) + len(second) != len(markers):
            return None
        if len(first) == len(second) == 1:
            p1, p2 = first[0], second[0]
            c1, c2 = int(line[p1]), int(line[p2])
            _put(expected, orientation, lane, p1, bg)
            _put(expected, orientation, lane, p2, bg)
            _put(expected, orientation, lane, start - 1, c2)
            _put(expected, orientation, lane, end + 1, c1)
            active.append((lane, start, end, start - p1, p2 - end, c1, c2))
        elif first and second:
            return None
        elif markers.size:
            controls.append((lane, len(markers), "first" if first else "second"))
    if not active or not np.array_equal(expected, out):
        return None
    return {"orientation": orientation, "runs": runs, "active": active, "controls": controls}


def _analyze(inp, out):
    inferred = _input_candidates(inp)
    assert len(inferred) == 1
    bg, scaffold, direction = inferred[0]
    assert int(np.count_nonzero(inp == bg)) > inp.size // 2
    candidate = _inspect_pair(inp, out, bg, scaffold, direction)
    assert candidate is not None
    candidate.update(background=bg, scaffold=scaffold)
    return candidate


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    analyses = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            sig = (inp.shape, inp.tobytes())
            assert sig not in seen
            seen.add(sig)
            rows = tuple("".join(str(int(v)) for v in row) for row in inp)
            assert rows not in _SOURCE_INPUTS
            analyses[regime].append(_analyze(inp, out))
    train_orientations = {a["orientation"] for a in analyses["train"]}
    assert all(a["orientation"] in train_orientations for a in analyses["test"])
    assert any(a["controls"] for a in analyses["train"])
    assert any(any(x[-2] != x[-1] for x in a["active"]) for a in analyses["train"])
    assert any(any(count >= 2 for _, count, _ in a["controls"]) for a in analyses["train"])


def _same(left, right):
    return all(len(left[k]) == len(right[k]) and all(np.array_equal(a[t], b[t]) for a, b in zip(left[k], right[k]) for t in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, orientations, palettes = set(), set(), set(), set(), set()
    features = {"thin": 0, "long_gap": 0, "multi_control": 0, "blank_lane": 0, "same_color": 0, "no_control": 0, "nonzero_bg": 0}
    diversity = {"same_direction": 0, "mixed_direction": 0, "palette_changes": 0,
                 "background_changes": 0, "scaffold_changes": 0, "new_test_marker": 0,
                 "reused_test_marker": 0, "broken": 0, "many_controls": 0,
                 "near_and_far": 0, "unequal_gaps": 0, "large_grid": 0,
                 "small_grid": 0, "thin_multi_far": 0}
    first = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first.add((puzzle["train"][0]["input"].shape, puzzle["train"][0]["input"].tobytes()))
        pair_analyses = [_analyze(p["input"], p["output"]) for p in puzzle["train"] + puzzle["test"]]
        train_directions = {a["orientation"] for a in pair_analyses[:len(puzzle["train"])]}
        diversity["same_direction" if len(train_directions) == 1 else "mixed_direction"] += 1
        diversity["palette_changes"] += len({tuple(int(c) for c in np.unique(p["input"])) for p in puzzle["train"] + puzzle["test"]}) > 1
        diversity["background_changes"] += len({a["background"] for a in pair_analyses}) > 1
        diversity["scaffold_changes"] += len({a["scaffold"] for a in pair_analyses}) > 1
        train_markers = set().union(*(set(int(c) for c in np.unique(p["input"])) - {a["background"], a["scaffold"]} for p, a in zip(puzzle["train"], pair_analyses)))
        for p, a in zip(puzzle["test"], pair_analyses[len(puzzle["train"]):]):
            test_markers = set(int(c) for c in np.unique(p["input"])) - {a["background"], a["scaffold"]}
            diversity["new_test_marker"] += bool(test_markers - train_markers)
            diversity["reused_test_marker"] += not bool(test_markers - train_markers)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            a = _analyze(inp, out)
            shapes.add(inp.shape)
            orientations.add(a["orientation"])
            palettes.add(tuple(int(v) for v in np.unique(inp)))
            features["thin"] += any(end == start for _, start, end in a["runs"])
            features["long_gap"] += any(max(d1, d2) >= 5 for _, _, _, d1, d2, _, _ in a["active"])
            features["multi_control"] += any(n >= 2 for _, n, _ in a["controls"])
            features["blank_lane"] += len(a["runs"]) < (inp.shape[0] if a["orientation"] == "horizontal" else inp.shape[1])
            features["same_color"] += any(c1 == c2 for _, _, _, _, _, c1, c2 in a["active"])
            features["no_control"] += not a["controls"]
            features["nonzero_bg"] += a["background"] != 0
            diversity["broken"] += any(a["runs"][j+1][0] - a["runs"][j][0] > 1 for j in range(len(a["runs"]) - 1))
            diversity["many_controls"] += len(a["controls"]) >= 4
            diversity["near_and_far"] += any(d1 == 1 or d2 == 1 for _, _, _, d1, d2, _, _ in a["active"]) and any(max(d1, d2) >= 5 for _, _, _, d1, d2, _, _ in a["active"])
            diversity["unequal_gaps"] += any(d1 != d2 for _, _, _, d1, d2, _, _ in a["active"])
            diversity["large_grid"] += max(inp.shape) >= 25
            diversity["small_grid"] += max(inp.shape) <= 13
            diversity["thin_multi_far"] += any(end == start for _, start, end in a["runs"]) and any(n >= 2 for _, n, _ in a["controls"]) and any(max(d1, d2) >= 5 for _, _, _, d1, d2, _, _ in a["active"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(first) >= 40 and len(shapes) >= 35 and len(palettes) >= 30
    assert orientations == {"horizontal", "vertical"}
    assert all(value >= 3 for value in features.values()), features
    assert all(value >= 3 for value in diversity.values()), diversity
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + nt + ne, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
        assert _same(p, generate_puzzle(1000 + nt + ne, nt, ne))
    for opts in ({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10}):
        p = generate_puzzle(2026, **opts)
        assert _same(p, generate_puzzle(2026, **opts))
    for opts in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7, **opts)
        except ValueError:
            continue
        raise AssertionError("Expected ValueError")
