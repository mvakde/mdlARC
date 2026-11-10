import numpy as np

_SOURCE_INPUTS = {
    ("00000000", "00000000", "00000000", "00033000", "00333300", "03333330", "33333333", "00000000"),
    ("20000000", "22000000", "22200000", "22000000", "20000000"),
    ("00000000000", "04444444440", "00000000000", "00444444400", "00000000000", "00044444000", "00000000000", "00004440000", "00000000000", "00000400000"),
    ("0000000000060", "0000000060060", "0000006060060", "0000606060060", "0000606060060", "0000006060060", "0000000060060", "0000000000060"),
    ("00000000000", "00000000000", "00000000000", "00000000000", "00000700000", "00007770000", "00000000000", "00000000000", "00000000000", "00000000000", "00000000000", "00000000000"),
    ("000000000", "000000000", "000000000", "000000000", "000000000", "077777770", "000000000"),
}


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _profile(grid, ink, orientation):
    mask = grid == ink
    if orientation == "vertical":
        mask = mask.T
    positions = np.flatnonzero(mask.any(axis=1))
    if not len(positions):
        return None
    runs = [np.flatnonzero(mask[i]) for i in positions]
    if any(len(run) != int(run[-1] - run[0] + 1) for run in runs):
        return None
    lengths = [len(run) for run in runs]
    sizes = sorted(set(lengths))
    levels = len(sizes)
    if levels == 1:
        if len(runs) != 1:
            return None
        mode, step = "single", 0
    else:
        step = sizes[1] - sizes[0]
        if any(b - a != step for a, b in zip(sizes, sizes[1:])):
            return None
        if lengths == sizes or lengths == sizes[::-1]:
            mode = "monotone"
        elif lengths == sizes + sizes[-2::-1]:
            mode = "peak"
        elif lengths == sizes + sizes[::-1]:
            mode = "plateau"
        else:
            return None
    starts = [int(run[0]) for run in runs]
    ends = [int(run[-1]) for run in runs]
    alignments = set()
    if len(set(starts)) == 1:
        alignments.add("leading")
    if len(set(ends)) == 1:
        alignments.add("trailing")
    if len(set(a + b for a, b in zip(starts, ends))) == 1:
        alignments.add("center")
    if not alignments:
        return None
    return {"orientation": orientation, "levels": levels, "mode": mode,
            "step": step, "gaps": tuple(int(v) for v in np.diff(positions)),
            "alignments": frozenset(alignments)}


def _analyze_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    background = int(colors[np.argmax(counts)])
    foreground = int(colors[np.argmin(counts)])
    profiles = [p for p in (_profile(grid, foreground, "horizontal"),
                            _profile(grid, foreground, "vertical")) if p is not None]
    assert profiles
    levels = {p["levels"] for p in profiles}
    assert len(levels) == 1
    return {"background": background, "foreground": foreground,
            "levels": levels.pop(), "profiles": profiles}


def _analyze_pair(pair):
    info = _analyze_input(pair["input"])
    assert pair["output"].shape == (1, info["levels"])
    assert np.all(pair["output"] == info["background"])
    assert not np.array_equal(pair["input"], pair["output"])
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = _grid_key(pair["input"])
            pkey = ikey, _grid_key(pair["output"])
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert _rows(pair["input"]) not in _SOURCE_INPUTS
            infos[regime].append(_analyze_pair(pair))
    train = infos["train"]
    assert len({x["levels"] for x in train}) >= 2
    assert len({x["background"] for x in train}) >= 2
    assert any(x["levels"] >= 3 for x in train)
    modes = {p["mode"] for x in train for p in x["profiles"]}
    assert "peak" in modes or "plateau" in modes
    for info in infos["test"]:
        test_modes = {p["mode"] for p in info["profiles"]}
        if "plateau" in test_modes:
            assert "plateau" in modes
        if "peak" in test_modes:
            assert "peak" in modes or "plateau" in modes


def _check_count(value, low, high):
    if value is not None and (isinstance(value, (bool, np.bool_)) or
                              not isinstance(value, (int, np.integer)) or
                              not low <= int(value) <= high):
        raise ValueError(f"Count must be an integer from {low} to {high}")


def _level_plans(rng, count):
    first = int(rng.integers(3, 7))
    second = int(rng.choice([n for n in range(1, 8) if n != first]))
    plans = [first, second]
    rng.shuffle(plans)
    plans.extend(int(rng.integers(1, 9)) for _ in range(count - 2))
    return plans


def _mode_plans(rng, levels):
    modes = [str(rng.choice(("up", "down"))) for _ in levels]
    eligible = [i for i, n in enumerate(levels) if n >= 2]
    if eligible:
        i = int(rng.choice(eligible))
        modes[i] = "plateau" if rng.random() < 0.3 else "peak"
    for i in eligible:
        if modes[i] in ("up", "down") and rng.random() < 0.15:
            modes[i] = "peak"
    return modes


def _sample_spec(rng, levels, mode, orientation):
    alignment = str(rng.choice(("leading", "center", "trailing")))
    base = int(rng.integers(1, 19 if levels == 1 else 7))
    if levels == 1:
        mode, step = "single", 0
    else:
        max_step = min(4, (26 - base) // (levels - 1))
        choices = [s for s in range(1, max_step + 1)
                   if alignment != "center" or s % 2 == 0]
        if not choices:
            alignment = str(rng.choice(("leading", "trailing")))
            choices = list(range(1, max_step + 1))
        step = int(rng.choice(choices))
    bars = levels if mode in ("single", "up", "down") else 2 * levels - (mode == "peak")
    if bars == 1:
        gaps = ()
    else:
        max_gap = 4 if bars <= 7 else (3 if bars <= 10 else 2)
        if rng.random() < 0.27:
            gaps = (int(rng.integers(1, max_gap + 1)),) * (bars - 1)
        else:
            gaps = tuple(int(v) for v in rng.integers(1, max_gap + 1, size=bars - 1))
    return {"orientation": orientation, "alignment": alignment,
            "levels": levels, "mode": mode, "base": base,
            "step": step, "gaps": gaps}


def _margin(rng, available):
    cap = min(6, available)
    if cap == 0 or rng.random() < 0.28:
        return 0
    return int(rng.integers(1, cap + 1))


def _draw_pair(rng, background, foreground, spec):
    sizes = [spec["base"] + i * spec["step"] for i in range(spec["levels"])]
    mode = spec["mode"]
    if mode == "down":
        sequence = sizes[::-1]
    elif mode == "peak":
        sequence = sizes + sizes[-2::-1]
    elif mode == "plateau":
        sequence = sizes + sizes[::-1]
    else:
        sequence = sizes
    positions = [0]
    for gap in spec["gaps"]:
        positions.append(positions[-1] + gap)
    long_side, bar_side = sizes[-1], positions[-1] + 1
    if spec["orientation"] == "horizontal":
        body_h, body_w = bar_side, long_side
    else:
        body_h, body_w = long_side, bar_side
    if body_h > 30 or body_w > 30:
        return None
    top = _margin(rng, 30 - body_h)
    bottom = _margin(rng, 30 - body_h - top)
    left = _margin(rng, 30 - body_w)
    right = _margin(rng, 30 - body_w - left)
    grid = np.full((top + body_h + bottom, left + body_w + right), background, dtype=np.int8)
    for position, length in zip(positions, sequence):
        if spec["alignment"] == "leading":
            offset = 0
        elif spec["alignment"] == "trailing":
            offset = long_side - length
        else:
            offset = (long_side - length) // 2
        if spec["orientation"] == "horizontal":
            grid[top + position, left + offset:left + offset + length] = foreground
        else:
            grid[top + offset:top + offset + length, left + position] = foreground
    return {"input": grid, "output": np.full((1, spec["levels"]), background, dtype=np.int8)}


def _choose_colors(rng, palette, prior=None):
    if prior is None:
        background = int(rng.choice(palette))
        foreground = int(rng.choice([c for c in palette if c != background]))
    else:
        background = prior[1] if rng.random() < 0.45 else int(rng.choice(
            [c for c in palette if c != prior[0]]))
        foreground = prior[0] if rng.random() < 0.45 else int(rng.choice(
            [c for c in palette if c != background]))
    assert background != foreground
    return background, foreground


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10)
    _check_count(num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    # The counting rule is the sole puzzle-wide convention. Pair color roles vary.
    palette = [int(c) for c in rng.permutation(10)[:int(rng.integers(3, 8))]]
    train_levels = _level_plans(rng, num_train)
    train_modes = _mode_plans(rng, train_levels)
    direction_regime = int(rng.integers(0, 4))
    if direction_regime == 0:
        train_directions = ["horizontal"] * num_train
        test_directions = ["vertical"] * num_test
    elif direction_regime == 1:
        train_directions = ["vertical"] * num_train
        test_directions = ["horizontal"] * num_test
    elif direction_regime == 2:
        shared = str(rng.choice(("horizontal", "vertical")))
        train_directions = [shared] * num_train
        test_directions = [shared] * num_test
    else:
        first = str(rng.choice(("horizontal", "vertical")))
        other = "vertical" if first == "horizontal" else "horizontal"
        train_directions = [first, other]
        train_directions.extend(str(rng.choice(("horizontal", "vertical")))
                                for _ in range(num_train - 2))
        test_directions = [str(rng.choice(("horizontal", "vertical")))
                           for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    prior = None
    for index, (levels, mode) in enumerate(zip(train_levels, train_modes)):
        colors = _choose_colors(rng, palette, prior if index == 1 else None)
        if index == 0:
            prior = colors
        for _ in range(350):
            spec = _sample_spec(rng, levels, mode, train_directions[index])
            pair = _draw_pair(rng, *colors, spec)
            if pair is None:
                continue
            key = _grid_key(pair["input"])
            if key in seen or _rows(pair["input"]) in _SOURCE_INPUTS:
                continue
            try:
                info = _analyze_pair(pair)
            except AssertionError:
                continue
            if info["levels"] != levels:
                continue
            if mode in ("peak", "plateau") and not any(p["mode"] == mode for p in info["profiles"]):
                continue
            seen.add(key)
            puzzle["train"].append(pair)
            break
        else:
            raise RuntimeError("Could not construct a training bar pattern")
    available = {p["mode"] for pair in puzzle["train"]
                 for p in _analyze_pair(pair)["profiles"]}
    for index in range(num_test):
        levels = int(rng.choice((1, 2, 2, 3, 6, 7, 8))) if index == 0 else int(rng.integers(1, 9))
        allowed = ["up", "down"]
        if levels >= 2 and ("peak" in available or "plateau" in available):
            allowed.extend(("peak", "peak"))
        if levels >= 2 and "plateau" in available:
            allowed.append("plateau")
        mode = str(rng.choice(allowed))
        colors = _choose_colors(rng, palette)
        for _ in range(350):
            spec = _sample_spec(rng, levels, mode, test_directions[index])
            pair = _draw_pair(rng, *colors, spec)
            if pair is None:
                continue
            key = _grid_key(pair["input"])
            if key in seen or _rows(pair["input"]) in _SOURCE_INPUTS:
                continue
            try:
                info = _analyze_pair(pair)
            except AssertionError:
                continue
            if info["levels"] != levels:
                continue
            if mode in ("peak", "plateau") and not any(p["mode"] == mode for p in info["profiles"]):
                continue
            visible_modes = {p["mode"] for p in info["profiles"]}
            if "plateau" in visible_modes and "plateau" not in available:
                continue
            if "peak" in visible_modes and not ({"peak", "plateau"} & available):
                continue
            seen.add(key)
            puzzle["test"].append(pair)
            break
        else:
            raise RuntimeError("Could not construct a test bar pattern")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[r]) == len(b[r]) and all(
        np.array_equal(pa[k], pb[k]) for pa, pb in zip(a[r], b[r])
        for k in ("input", "output")) for r in ("train", "test"))


def _independent_count(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    ink = int(values[np.argmin(counts)])
    possible = set()
    for oriented in (grid, grid.T):
        bars = []
        for row in oriented:
            points = np.flatnonzero(row == ink)
            if len(points):
                if points[-1] - points[0] + 1 != len(points):
                    break
                bars.append(len(points))
        else:
            if bars:
                possible.add(len(set(bars)))
    assert len(possible) == 1
    return possible.pop()


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, fingerprints = (set() for _ in range(5))
    features = {name: 0 for name in ("zero_top", "zero_bottom", "zero_left", "zero_right",
                 "corner_ink", "unequal_gaps", "mixed_touch_gap", "large_levels",
                 "large_step", "plateau", "peak", "vertical_small", "single_train",
                 "two_train", "peak_test", "background_changes", "role_swap",
                 "regime_h_to_v", "regime_v_to_h", "regime_shared_h",
                 "regime_shared_v", "regime_mixed_train", "within_gap_variety",
                 "within_edge_variety", "vertical_single", "short_single",
                 "edge_and_unequal_gap")}
    regimes = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_grid_key(p["input"]) for regime in ("train", "test")
                               for p in puzzle[regime]))
        features["background_changes"] += int(len({int(p["output"][0, 0]) for p in puzzle["train"]}) > 1)
        exclusive = {"train": set(), "test": set()}
        roles, gap_flags, edge_flags = [], [], []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid = pair["input"]
                info = _analyze_pair(pair)
                assert _independent_count(grid) == pair["output"].shape[1]
                shapes.add(grid.shape)
                foreground = info["foreground"]
                roles.append((info["background"], foreground))
                orientations = {p["orientation"] for p in info["profiles"]}
                if len(orientations) == 1:
                    exclusive[regime].update(orientations)
                has_unequal_gap = any(len(set(p["gaps"])) > 1 for p in info["profiles"])
                gap_flags.append(has_unequal_gap)
                touches_edge = any(np.any(side == foreground) for side in
                                   (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
                edge_flags.append(touches_edge)
                features["edge_and_unequal_gap"] += int(touches_edge and has_unequal_gap)
                if info["levels"] == 1:
                    features["short_single"] += int(np.count_nonzero(grid == foreground) <= 4)
                    features["vertical_single"] += int("vertical" in orientations)
                for side, row in (("top", grid[0]), ("bottom", grid[-1]),
                                  ("left", grid[:, 0]), ("right", grid[:, -1])):
                    features["zero_" + side] += int(np.any(row == foreground))
                features["corner_ink"] += int(grid[0, 0] == foreground)
                features["single_train"] += int(regime == "train" and info["levels"] == 1)
                features["two_train"] += int(regime == "train" and info["levels"] == 2)
                regimes[regime].add(tuple((p["orientation"], p["mode"]) for p in info["profiles"]))
                for p in info["profiles"]:
                    layouts.add((p["orientation"], p["levels"], p["mode"], p["step"],
                                 p["gaps"], tuple(sorted(p["alignments"]))))
                    gaps = p["gaps"]
                    features["unequal_gaps"] += int(len(set(gaps)) > 1)
                    features["mixed_touch_gap"] += int(1 in gaps and any(g > 1 for g in gaps))
                    features["large_levels"] += int(p["levels"] >= 6)
                    features["large_step"] += int(p["step"] >= 3)
                    features["plateau"] += int(p["mode"] == "plateau")
                    features["peak"] += int(p["mode"] == "peak")
                    features["vertical_small"] += int(regime == "test" and p["orientation"] == "vertical" and p["levels"] <= 2)
                    features["peak_test"] += int(regime == "test" and p["mode"] == "peak")
        h, v = {"horizontal"}, {"vertical"}
        features["regime_h_to_v"] += int(exclusive["train"] == h and exclusive["test"] == v)
        features["regime_v_to_h"] += int(exclusive["train"] == v and exclusive["test"] == h)
        features["regime_shared_h"] += int(exclusive["train"] == exclusive["test"] == h)
        features["regime_shared_v"] += int(exclusive["train"] == exclusive["test"] == v)
        features["regime_mixed_train"] += int(exclusive["train"] == h | v)
        features["within_gap_variety"] += int(len(set(gap_flags)) > 1)
        features["within_edge_variety"] += int(len(set(edge_flags)) > 1)
        features["role_swap"] += int(any(a[0] == b[1] and a[1] == b[0]
                                         for i, a in enumerate(roles) for b in roles[i + 1:]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(layouts) >= 100 and len(fingerprints) == 42
    assert all(value > 0 for value in features.values())
    assert len(regimes["train"]) >= 5 and len(regimes["test"]) >= 5
    for seed in (0, 7, 19, 41):
        assert _same_puzzle(generate_puzzle(seed), generate_puzzle(seed))
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)):
        a, b = generate_puzzle(seed, nt, ns), generate_puzzle(seed, nt, ns)
        assert _same_puzzle(a, b)
        validate_generated_puzzle(a)
        assert nt is None or len(a["train"]) == nt
        assert ns is None or len(a["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, None)):
        try:
            generate_puzzle(200, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
    duplicate = generate_puzzle(201, 2, 1)
    duplicate["test"][0] = duplicate["train"][0]
    try:
        validate_generated_puzzle(duplicate)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate input was accepted")
