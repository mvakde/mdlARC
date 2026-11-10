"""Generate completed point lattices selected by one visible seed cell."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("0050555050555", "5505005505555", "5050000000000",
     "5000000000000", "5550010000055", "0550000000055",
     "0550000000005", "5050000000005", "5550000000000",
     "0000000000055", "0055050505555", "5550555555555",
     "5505055550505"),
    ("5050555555505055", "0550555050050055", "5550000000000550",
     "5500000000050550", "5050000000055500", "5550000300000500",
     "0500000000055050", "0550000000005555", "0500000000055555",
     "0555055555050500", "5555050500050500", "0550050500005505",
     "5505555550555505"),
    ("55555555555050050", "05500000000000055", "05500000000000055",
     "50500000000000000", "05500000000000005", "05000000000000050",
     "55500000000000050", "55500000000000055", "55500000000000050",
     "55500000200000005", "05500000000000005", "55500000000000055",
     "00555505055555550", "05500000550550555", "00500505005055005"),
)


def _counts(rng, seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} through {hi}")
    if num_train is None and num_test is None:
        residue = int(seed) % 8
        num_test = (1, 1, 1, 2, 2, 3, 3, 4)[residue]
        num_train = (2 if residue in (0, 1)
                     else int(rng.integers(num_test + 1, 11)))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _phase_schedule(rng, train_count, test_count):
    phases = ((0, 0), (0, 1), (1, 0), (1, 1))
    first = phases[int(rng.integers(4))]
    if rng.random() < 0.35:
        second = (1 - first[0], first[1]) if rng.random() < 0.5 else (first[0], 1 - first[1])
    else:
        second = (1 - first[0], 1 - first[1])
    train = [first, second]
    train += [phases[int(rng.integers(4))] for _ in range(train_count - 2)]
    rng.shuffle(train)
    unused = [phase for phase in phases if phase not in train]
    test = []
    for _ in range(test_count):
        pool = unused if unused and rng.random() < 0.58 else phases
        test.append(pool[int(rng.integers(len(pool)))])
    return train, test


def _colors(rng, train_count, test_count):
    shuffled = [int(v) for v in rng.permutation(10)]
    structural_mode = int(rng.integers(3))
    if structural_mode == 0:  # Both structural colors stay fixed.
        backgrounds = [shuffled[0]] * (train_count + test_count)
        blockers = [shuffled[1]] * (train_count + test_count)
        marker_pool = shuffled[2:7]
    else:
        varying = shuffled[1:4]
        structural_schedule = [varying[0], varying[1]]
        structural_schedule += [varying[int(rng.integers(3))]
                                for _ in range(train_count + test_count - 2)]
        marker_pool = shuffled[4:9]
        if structural_mode == 1:  # Background anchors changing blocker colors.
            backgrounds = [shuffled[0]] * len(structural_schedule)
            blockers = structural_schedule
        else:  # Blocker anchors changing background colors.
            blockers = [shuffled[0]] * len(structural_schedule)
            backgrounds = structural_schedule
    marker_mode = int(rng.integers(3))
    if marker_mode == 0:
        markers = [marker_pool[0]] * (train_count + test_count)
    elif marker_mode == 1:
        train_markers = [marker_pool[0], marker_pool[1]]
        train_markers += [marker_pool[int(rng.integers(2))]
                          for _ in range(train_count - 2)]
        markers = train_markers + [train_markers[int(rng.integers(train_count))]
                                   for _ in range(test_count)]
    else:
        train_markers = [marker_pool[int(rng.integers(2))]
                         for _ in range(train_count)]
        markers = train_markers + [marker_pool[2 + int(rng.integers(2))]
                                   for _ in range(test_count)]
    return list(zip(backgrounds, blockers, markers))


def _dimension(rng, upper):
    category = int(rng.choice(4, p=(0.24, 0.43, 0.27, 0.06)))
    intervals = ((8, 12), (11, 18), (17, 25), (24, upper))
    lo, hi = intervals[category]
    return int(rng.integers(lo, max(lo, hi) + 1))


def _sizes(rng, train_count, test_count):
    regime = int(rng.integers(4))
    upper = 26 if regime >= 2 else 30
    train = [(_dimension(rng, upper), _dimension(rng, upper))
             for _ in range(train_count)]
    min_h, min_w = (min(v[axis] for v in train) for axis in (0, 1))
    max_h, max_w = (max(v[axis] for v in train) for axis in (0, 1))
    test = []
    for _ in range(test_count):
        if regime == 0:  # Smaller in both axes than all demonstrations.
            size = (int(rng.integers(7, min_h)), int(rng.integers(7, min_w)))
        elif regime == 1:  # Within two cells of a demonstrated shape.
            anchor = train[int(rng.integers(train_count))]
            size = tuple(int(np.clip(v + int(rng.integers(-2, 3)), 7, 30))
                         for v in anchor)
        elif regime == 2:  # Extrapolate one axis while shrinking the other.
            if rng.random() < 0.5:
                size = (int(rng.integers(max_h + 1, 31)),
                        int(rng.integers(7, min_w)))
            else:
                size = (int(rng.integers(7, min_h)),
                        int(rng.integers(max_w + 1, 31)))
        else:  # Extrapolate both axes.
            size = (int(rng.integers(max_h + 1, 31)),
                    int(rng.integers(max_w + 1, 31)))
        test.append(size)
    return train, test


def _rectangle(rng, mask, height_range, width_range, probability):
    height, width = mask.shape
    rh = int(rng.integers(height_range[0], min(height, height_range[1]) + 1))
    rw = int(rng.integers(width_range[0], min(width, width_range[1]) + 1))
    r = int(rng.integers(0, height - rh + 1))
    c = int(rng.integers(0, width - rw + 1))
    mask[r:r + rh, c:c + rw] |= rng.random((rh, rw)) < probability


def _rim(rng, mask, dense):
    height, width = mask.shape
    rows, cols = np.indices(mask.shape)
    top, bottom, left, right = (int(rng.integers(1, 4)) for _ in range(4))
    region = ((rows < top) | (rows >= height - bottom)
              | (cols < left) | (cols >= width - right))
    probability = float(rng.uniform(0.55, 0.88) if dense else rng.uniform(0.22, 0.48))
    mask |= region & (rng.random(mask.shape) < probability)
    # Clear a few irregular windows through the surround.
    for _ in range(int(rng.integers(1, 4))):
        side = int(rng.integers(4))
        if side < 2:
            start = int(rng.integers(0, width - 1))
            span = int(rng.integers(1, min(5, width - start) + 1))
            if side == 0:
                mask[:top + 1, start:start + span] = False
            else:
                mask[height - bottom - 1:, start:start + span] = False
        else:
            start = int(rng.integers(0, height - 1))
            span = int(rng.integers(1, min(5, height - start) + 1))
            if side == 2:
                mask[start:start + span, :left + 1] = False
            else:
                mask[start:start + span, width - right - 1:] = False


def _mask(rng, height, width, style):
    mask = np.zeros((height, width), dtype=bool)
    if style in (0, 4):
        _rim(rng, mask, style == 0)
        if style == 0 and rng.random() < 0.27:
            # At most two interior islands; most source-like centers stay open.
            for _ in range(int(rng.integers(1, 3))):
                rh, rw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                r = int(rng.integers(3, max(4, height - rh - 2)))
                c = int(rng.integers(3, max(4, width - rw - 2)))
                mask[r:r + rh, c:c + rw] = True
    if style in (1, 4):
        for _ in range(int(rng.integers(1, 4))):
            horizontal = rng.random() < 0.5
            thickness = int(rng.integers(1, 3))
            if horizontal:
                row = int(rng.integers(0, height - thickness + 1))
                mask[row:row + thickness, :] = True
                if rng.random() < 0.5:
                    start = int(rng.integers(0, width - 1))
                    mask[row:row + thickness, start:start + int(rng.integers(1, 4))] = False
            else:
                col = int(rng.integers(0, width - thickness + 1))
                mask[:, col:col + thickness] = True
                if rng.random() < 0.5:
                    start = int(rng.integers(0, height - 1))
                    mask[start:start + int(rng.integers(1, 4)), col:col + thickness] = False
    if style in (2, 4):
        for _ in range(int(rng.integers(2, 7))):
            _rectangle(rng, mask, (2, 7), (2, 8), float(rng.uniform(0.75, 1.0)))
        if rng.random() < 0.65:
            rh, rw = min(height - 2, int(rng.integers(2, 6))), min(width - 2, int(rng.integers(2, 6)))
            r = int(rng.integers(1, height - rh))
            c = int(rng.integers(1, width - rw))
            mask[r:r + rh, c:c + rw] = False
    if style == 3:
        mask |= rng.random(mask.shape) < float(rng.uniform(0.025, 0.14))
        if rng.random() < 0.4:
            _rectangle(rng, mask, (2, 4), (2, 4), 0.85)
    if style == 5 and rng.random() < 0.5:
        count = int(rng.integers(4, 7))
        mask[rng.integers(height, size=count), rng.integers(width, size=count)] = True
    return mask


def _ensure_visible_lattice(rng, mask, phase):
    rows, cols = np.indices(mask.shape)
    selected = (rows % 2 == phase[0]) & (cols % 2 == phase[1])
    visible = np.argwhere(selected & ~mask)
    if len(visible) < 6:
        hidden = np.argwhere(selected & mask)
        choice = rng.choice(len(hidden), size=6 - len(visible), replace=False)
        positions = hidden[choice]
        mask[positions[:, 0], positions[:, 1]] = False


def _seed_position(rng, mask, phase):
    positions = np.argwhere(~mask)
    positions = positions[((positions[:, 0] % 2 == phase[0])
                           & (positions[:, 1] % 2 == phase[1]))]
    height, width = mask.shape
    edge = ((positions[:, 0] == 0) | (positions[:, 0] == height - 1)
            | (positions[:, 1] == 0) | (positions[:, 1] == width - 1))
    if np.any(edge) and rng.random() < 0.17:
        positions = positions[edge]
    position = positions[int(rng.integers(len(positions)))]
    return int(position[0]), int(position[1])


def _ensure_witness(rng, mask, phase, seed):
    height, width = mask.shape
    candidates = []
    for row in range(phase[0], height, 2):
        for col in range(phase[1] + 2, width - 2, 2):
            triple = ((row, col - 2), (row, col), (row, col + 2))
            if seed not in triple:
                candidates.append(triple)
    for row in range(phase[0] + 2, height - 2, 2):
        for col in range(phase[1], width, 2):
            triple = ((row - 2, col), (row, col), (row + 2, col))
            if seed not in triple:
                candidates.append(triple)
    triple = candidates[int(rng.integers(len(candidates)))]
    mask[triple[0]] = False
    mask[triple[1]] = True
    mask[triple[2]] = False


def _make_pair(rng, size, phase, colors, style, witness, require_blockers=False):
    height, width = size
    background, blocker, marker = colors
    mask = _mask(rng, height, width, style)
    _ensure_visible_lattice(rng, mask, phase)
    seed = _seed_position(rng, mask, phase)
    if witness:
        _ensure_witness(rng, mask, phase, seed)
    blocker_count = int(np.count_nonzero(mask))
    if (require_blockers and blocker_count == 0) or (0 < blocker_count < 4):
        rows, cols = np.indices(mask.shape)
        candidates = np.argwhere((~mask) & ((rows % 2 != phase[0])
                                           | (cols % 2 != phase[1])))
        choice = rng.choice(len(candidates), size=4 - blocker_count,
                            replace=False)
        extra = candidates[choice]
        mask[extra[:, 0], extra[:, 1]] = True
    inp = np.full((height, width), background, dtype=np.int64)
    inp[mask] = blocker
    inp[seed] = marker
    rows, cols = np.indices(inp.shape)
    selected = ((rows % 2 == seed[0] % 2) & (cols % 2 == seed[1] % 2))
    out = inp.copy()
    out[selected & (inp == background)] = marker
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    train_phases, test_phases = _phase_schedule(rng, train_count, test_count)
    train_sizes, test_sizes = _sizes(rng, train_count, test_count)
    colors = _colors(rng, train_count, test_count)
    # At least one source-like surround in most puzzles, while other pair
    # layouts are sampled independently and can combine different components.
    train_styles = [int(rng.integers(6)) for _ in range(train_count)]
    if rng.random() < 0.68:
        train_styles[int(rng.integers(train_count))] = 0
    test_styles = [int(rng.integers(6)) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    for index in range(train_count):
        puzzle["train"].append(_make_pair(rng, train_sizes[index],
                                           train_phases[index], colors[index],
                                           train_styles[index], index == 0,
                                           require_blockers=index < 2))
    for index in range(test_count):
        puzzle["test"].append(_make_pair(rng, test_sizes[index],
                                          test_phases[index], colors[train_count + index],
                                          test_styles[index], False))
    return puzzle


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    changed = inp != out
    assert np.count_nonzero(changed) >= 4
    source = np.unique(inp[changed])
    destination = np.unique(out[changed])
    assert len(source) == len(destination) == 1
    background, marker = int(source[0]), int(destination[0])
    assert background != marker
    seed = np.argwhere(inp == marker)
    assert len(seed) == 1
    seed_row, seed_col = (int(v) for v in seed[0])
    colors = {int(v) for v in np.unique(inp)}
    assert len(colors) in (2, 3)
    assert background in colors and marker in colors
    blocker_set = colors - {background, marker}
    blocker = next(iter(blocker_set)) if blocker_set else None
    if blocker is not None:
        assert np.count_nonzero(inp == blocker) >= 4
    rows, cols = np.indices(inp.shape)
    selected = (rows % 2 == seed_row % 2) & (cols % 2 == seed_col % 2)
    expected = inp.copy()
    expected[selected & (inp == background)] = marker
    assert np.array_equal(out, expected)
    assert np.array_equal(changed, selected & (inp == background))
    assert np.count_nonzero(inp == background) >= 6
    assert {int(v) for v in np.unique(out)} == colors
    continuation = False
    if blocker is not None:
        for row in range(seed_row % 2, inp.shape[0], 2):
            for col in range(seed_col % 2 + 2, inp.shape[1] - 2, 2):
                if (inp[row, col] == blocker
                    and inp[row, col - 2] in (background, marker)
                    and inp[row, col + 2] in (background, marker)):
                    continuation = True
        for row in range(seed_row % 2 + 2, inp.shape[0] - 2, 2):
            for col in range(seed_col % 2, inp.shape[1], 2):
                if (inp[row, col] == blocker
                    and inp[row - 2, col] in (background, marker)
                    and inp[row + 2, col] in (background, marker)):
                    continuation = True
    return {"background": background, "blocker": blocker, "marker": marker,
            "phase": (seed_row % 2, seed_col % 2),
            "continuation": continuation}


def _predict_from_training(train_pairs, test_input):
    """Use demonstrations and an unlabeled input to recover a test output."""
    analyses = [_analyze(pair) for pair in train_pairs]
    values, counts = np.unique(test_input, return_counts=True)
    singleton = [int(v) for v, count in zip(values, counts) if count == 1]
    assert len(singleton) == 1
    marker = singleton[0]
    train_backgrounds = {a["background"] for a in analyses}
    train_blockers = {a["blocker"] for a in analyses if a["blocker"] is not None}
    if len(train_backgrounds) == 1 and next(iter(train_backgrounds)) in values:
        background = next(iter(train_backgrounds))
    else:
        assert len(train_blockers) == 1
        remaining = {int(v) for v in values} - {marker, next(iter(train_blockers))}
        assert len(remaining) == 1
        background = next(iter(remaining))
    seed = np.argwhere(test_input == marker)
    assert len(seed) == 1
    rows, cols = np.indices(test_input.shape)
    predicted = test_input.copy()
    predicted[(test_input == background)
              & (rows % 2 == int(seed[0, 0]) % 2)
              & (cols % 2 == int(seed[0, 1]) % 2)] = marker
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= v <= 30 for v in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair["input"])
            full_key = (key, _grid_key(pair["output"]))
            assert key not in keys and full_key not in pair_keys
            keys.add(key)
            pair_keys.add(full_key)
            rows = tuple("".join(str(int(v)) for v in row) for row in pair["input"])
            assert rows not in _SOURCE_INPUT_ROWS
            analyses[split].append(_analyze(pair))
    all_pairs = analyses["train"] + analyses["test"]
    assert len({a["phase"] for a in analyses["train"]}) >= 2
    assert any(a["continuation"] for a in analyses["train"])
    # A structural role is shared, even when the other role changes locally.
    present = [a for a in all_pairs if a["blocker"] is not None]
    assert present
    stable_background = len({a["background"] for a in all_pairs}) == 1
    stable_blocker = len({a["blocker"] for a in present}) == 1
    assert stable_background or stable_blocker
    if not stable_background:
        assert len({a["background"] for a in analyses["train"]}) >= 2
    if not stable_blocker:
        assert len({a["blocker"] for a in analyses["train"] if a["blocker"] is not None}) >= 2
    structural = {a["background"] for a in all_pairs} | {a["blocker"] for a in present}
    assert all(a["marker"] not in structural for a in all_pairs)
    for pair in puzzle["test"]:
        predicted = _predict_from_training(puzzle["train"], pair["input"])
        assert np.array_equal(predicted, pair["output"])


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    first = generate_puzzle(17321, 6, 3)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(17321, 6, 3))
    assert _puzzle_key(generate_puzzle(17322)) == _puzzle_key(generate_puzzle(17322))
    for seed, train, test in ((8001, 2, 1), (8002, 10, 1), (8003, 10, 4),
                              (8101, None, 1), (8102, None, 4),
                              (8103, 2, None), (8104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(99, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_pairs = [(split, pair) for p in puzzles for split in ("train", "test")
                 for pair in p[split]]
    analyses = [(split, pair, _analyze(pair)) for split, pair in all_pairs]
    assert {a["phase"] for split, pair, a in analyses if split == "train"} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}
    assert {a["phase"] for split, pair, a in analyses if split == "test"} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}
    assert len({pair["input"].shape for _, pair in all_pairs}) >= 35
    assert any(a["blocker"] is None for _, _, a in analyses)
    assert any(a["blocker"] is not None and not a["continuation"]
               for _, _, a in analyses)
    marker_cases = set()
    role_cases = set()
    phase_cases = set()
    for p in puzzles:
        train_a = [_analyze(pair) for pair in p["train"]]
        test_a = [_analyze(pair) for pair in p["test"]]
        train_markers = {a["marker"] for a in train_a}
        test_markers = {a["marker"] for a in test_a}
        marker_cases.add("one" if len(train_markers | test_markers) == 1 else
                         "new" if test_markers.isdisjoint(train_markers) else "reused")
        role_cases.add((len({a["background"] for a in train_a + test_a}) > 1,
                        len({a["blocker"] for a in train_a + test_a if a["blocker"] is not None}) > 1))
        phase_cases.add(any(a["phase"] not in {t["phase"] for t in train_a} for a in test_a))
    assert marker_cases == {"one", "new", "reused"}
    assert role_cases == {(False, False), (False, True), (True, False)}
    assert phase_cases == {False, True}
    measures = {name: 0 for name in (
        "two_one", "mixed_axes", "smaller", "comparable", "one_larger",
        "both_larger", "open_center", "clear", "band", "patch",
        "sparse", "dense", "no_edge_change", "no_selected_blocker",
        "long_mask", "small", "near_limit", "tall", "wide",
        "combined_shift")}
    normalized_masks = set()
    marker_positions = set()
    for puzzle in puzzles:
        training = [_analyze(pair) for pair in puzzle["train"]]
        testing = [_analyze(pair) for pair in puzzle["test"]]
        train_phases = {a["phase"] for a in training}
        mixed_axes = (len({p[0] for p in train_phases}) > 1
                      and len({p[1] for p in train_phases}) > 1)
        measures["mixed_axes"] += mixed_axes
        measures["two_one"] += (len(puzzle["train"]) == 2 and len(puzzle["test"]) == 1)
        min_h = min(pair["input"].shape[0] for pair in puzzle["train"])
        min_w = min(pair["input"].shape[1] for pair in puzzle["train"])
        max_h = max(pair["input"].shape[0] for pair in puzzle["train"])
        max_w = max(pair["input"].shape[1] for pair in puzzle["train"])
        new_marker = {a["marker"] for a in testing}.isdisjoint(
            {a["marker"] for a in training})
        for pair, analysis in zip(puzzle["test"], testing):
            h, w = pair["input"].shape
            if h < min_h and w < min_w:
                measures["smaller"] += 1
            elif h > max_h and w > max_w:
                measures["both_larger"] += 1
            elif (h > max_h and w < min_w) or (w > max_w and h < min_h):
                measures["one_larger"] += 1
            else:
                measures["comparable"] += 1
            blocker = analysis["blocker"]
            if (mixed_axes and new_marker and analysis["phase"] not in train_phases
                and blocker is not None
                and (np.any(np.all(pair["input"] == blocker, axis=0))
                     or np.any(np.all(pair["input"] == blocker, axis=1)))):
                measures["combined_shift"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                analysis = _analyze(pair)
                height, width = inp.shape
                blocker = analysis["blocker"]
                mask = inp == blocker if blocker is not None else np.zeros(inp.shape, dtype=bool)
                normalized_masks.add((inp.shape, mask.tobytes()))
                position = np.argwhere(inp == analysis["marker"])[0]
                marker_positions.add((inp.shape, int(position[0]), int(position[1])))
                rows, cols = np.indices(inp.shape)
                selected = ((rows % 2 == analysis["phase"][0])
                            & (cols % 2 == analysis["phase"][1]))
                changed = inp != out
                rim = np.ones(inp.shape, dtype=bool)
                rim[2:height - 2, 2:width - 2] = False
                center = mask[3:height - 3, 3:width - 3]
                measures["open_center"] += (center.size > 0 and center.mean() <= .05
                                            and mask[rim].mean() >= .35)
                measures["clear"] += not mask.any()
                measures["band"] += (np.any(np.all(mask, axis=0))
                                     or np.any(np.all(mask, axis=1)))
                measures["patch"] += np.any(mask[:-1, :-1] & mask[1:, :-1]
                                             & mask[:-1, 1:] & mask[1:, 1:])
                measures["sparse"] += 0 < mask.mean() < .12
                measures["dense"] += mask.mean() > .4
                measures["no_edge_change"] += not (changed[0].any() or changed[-1].any()
                                                   or changed[:, 0].any() or changed[:, -1].any())
                measures["no_selected_blocker"] += not np.any(mask & selected)
                long_mask = False
                blocked = mask & selected
                for row in range(analysis["phase"][0], height, 2):
                    points = blocked[row, analysis["phase"][1]::2]
                    long_mask |= any(points[i:i + 3].all()
                                     for i in range(max(0, len(points) - 2)))
                for col in range(analysis["phase"][1], width, 2):
                    points = blocked[analysis["phase"][0]::2, col]
                    long_mask |= any(points[i:i + 3].all()
                                     for i in range(max(0, len(points) - 2)))
                measures["long_mask"] += long_mask
                measures["small"] += height <= 10 and width <= 10
                measures["near_limit"] += height >= 27 or width >= 27
                measures["tall"] += height / width >= 1.5
                measures["wide"] += width / height >= 1.5
    assert measures["two_one"] >= 5 and measures["mixed_axes"] >= 15
    assert all(measures[name] >= 5 for name in (
        "smaller", "comparable", "one_larger", "both_larger"))
    assert all(measures[name] >= 10 for name in (
        "open_center", "clear", "band", "patch", "sparse", "dense",
        "no_edge_change", "no_selected_blocker", "long_mask",
        "small", "near_limit", "tall", "wide"))
    assert measures["combined_shift"] >= 1
    assert len(normalized_masks) >= 150 and len(marker_positions) >= 100
