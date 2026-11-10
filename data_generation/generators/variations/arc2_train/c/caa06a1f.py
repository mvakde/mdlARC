"""Generate periodic-field puzzles whose output advances one horizontal phase."""

import numpy as np


_PLACEMENTS = ("upper_left", "upper_right", "lower_left", "lower_right", "hole")
_SOURCE_SPECS = (
    (8, 8, 7, 7, 1, (6, 3)),
    (11, 11, 7, 7, 3, (6, 7)),
    (6, 6, 5, 5, 6, (5, 4)),
    (18, 18, 12, 12, 3, (8, 5, 7)),
)


def _coerce_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _coerce_count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _coerce_count("num_train", num_train)
    test = _coerce_count("num_test", num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _source_keys():
    keys = set()
    for height, width, visible_h, visible_w, mask, cycle in _SOURCE_SPECS:
        rows, cols = np.indices((height, width))
        cycle = np.asarray(cycle, dtype=np.int8)
        field = cycle[(cols + rows % 2) % len(cycle)]
        inp = np.full((height, width), mask, dtype=np.int8)
        inp[:visible_h, :visible_w] = field[:visible_h, :visible_w]
        out = cycle[(cols + rows % 2 + 1) % len(cycle)]
        keys.add((_grid_key(inp), _grid_key(out)))
    return keys


def _sample_schedule(rng, motif_size):
    period = int(rng.integers(1, 5))
    phases = [int(rng.integers(motif_size)) for _ in range(period)]
    if period > 1 and len(set(phases)) == 1:
        phases[-1] = (phases[0] + 1) % motif_size
    if period == 4 and phases[:2] == phases[2:]:
        phases[-1] = (phases[-1] + 1) % motif_size
    return tuple(phases)


def _sample_geometry(rng, placement, motif_size, period, regime):
    min_h, min_w = max(2 * period, 5), max(2 * motif_size, 5)
    if placement != "hole":
        compact = rng.random() < (.35 if regime == "source_train" else .22)
        if compact:
            visible_h = int(rng.integers(min_h, min_h + 5))
            visible_w = int(rng.integers(min_w, min_w + 5))
            max_band = 5
        else:
            max_visible = 21 if regime == "source_train" else 23
            visible_h = int(rng.integers(min_h, max_visible + 1))
            visible_w = int(rng.integers(min_w, max_visible + 1))
            max_band = 12
        bottom = int(rng.integers(1, min(max_band, 30 - visible_h) + 1))
        side = int(rng.integers(1, min(max_band, 30 - visible_w) + 1))
        height, width = visible_h + bottom, visible_w + side
        y0 = 0 if placement.startswith("upper") else bottom
        x0 = 0 if placement.endswith("left") else side
        return height, width, (y0, y0 + visible_h, x0, x0 + visible_w)
    height = int(rng.integers(min_h + 5, 31))
    width = int(rng.integers(min_w + 5, 31))
    y0 = int(rng.integers(min_h, height - 2))
    x0 = int(rng.integers(min_w, width - 2))
    hole_h = int(rng.integers(2, min(9, height - y0 - 1) + 1))
    hole_w = int(rng.integers(2, min(9, width - x0 - 1) + 1))
    return height, width, (y0, y0 + hole_h, x0, x0 + hole_w)


def generate_pair(rng, regime, placement, motif_size, mask_color):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    if placement not in _PLACEMENTS or not 2 <= motif_size <= 5:
        raise ValueError("invalid placement or motif size")
    available = np.asarray([c for c in range(10) if c != mask_color], dtype=np.int8)
    cycle = rng.choice(available, size=motif_size, replace=False)
    phases = _sample_schedule(rng, motif_size)
    height, width, box = _sample_geometry(rng, placement, motif_size, len(phases), regime)
    rows, cols = np.indices((height, width))
    row_phases = np.asarray(phases, dtype=np.int8)[rows % len(phases)]
    field = cycle[(cols + row_phases) % motif_size].astype(np.int8)
    output = cycle[(cols + row_phases + 1) % motif_size].astype(np.int8)
    inp = field.copy()
    y0, y1, x0, x1 = box
    if placement == "hole":
        inp[y0:y1, x0:x1] = mask_color
    else:
        inp[:] = mask_color
        inp[y0:y1, x0:x1] = field[y0:y1, x0:x1]
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _coerce_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # The one-cell successor step is the invariant puzzle-wide convention.
    test_places = [str(rng.choice(_PLACEMENTS, p=(.30, .18, .18, .18, .16)))
                   for _ in range(test_count)]
    train_places = list(test_places)
    for _ in range(train_count - test_count):
        if len(train_places) == test_count:
            extra = "upper_left" if "upper_left" not in train_places else str(
                rng.choice([p for p in _PLACEMENTS if p != train_places[0]]))
        else:
            extra = str(rng.choice(_PLACEMENTS))
        train_places.append(extra)
    source_keys = _source_keys()
    seen = set()
    train, test = [], []
    first_mask = int(rng.integers(10))
    for split, places in ((train, train_places), (test, test_places)):
        for index, place in enumerate(places):
            if split is train and index == 0:
                size, mask = int(rng.integers(3, 6)), first_mask
            elif split is train and index == 1:
                size = 2
                mask = int(rng.choice(np.unique(train[0]["output"])))
            else:
                size, mask = int(rng.choice((2, 3, 3, 4, 5))), int(rng.integers(10))
            for _ in range(12):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = generate_pair(child, "source_train" if split is train else "target_test",
                                     place, size, mask)
                key = _pair_key(pair)
                witness_ok = True
                if split is train and index == 0:
                    info = _analyze_pair(pair["input"], pair["output"])
                    field = info["field"]
                    witness_ok = (not np.array_equal(pair["output"], np.roll(field, 1, 0))
                                  and not np.array_equal(pair["output"], np.roll(field, -1, 0)))
                if key not in source_keys and key[0] not in seen and witness_ok:
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
            seen.add(key[0])
            split.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= dimension <= 30 for dimension in grid.shape)
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def _geometry(mask):
    height, width = mask.shape
    ys, xs = np.where(~mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    visible = np.zeros_like(mask)
    visible[y0:y1, x0:x1] = True
    if np.array_equal(~mask, visible):
        if y0 == 0 and x0 == 0 and y1 < height and x1 < width:
            return "upper_left", (y0, y1, x0, x1)
        if y0 == 0 and x0 > 0 and y1 < height and x1 == width:
            return "upper_right", (y0, y1, x0, x1)
        if y0 > 0 and x0 == 0 and y1 == height and x1 < width:
            return "lower_left", (y0, y1, x0, x1)
        if y0 > 0 and x0 > 0 and y1 == height and x1 == width:
            return "lower_right", (y0, y1, x0, x1)
    hy, hx = np.where(mask)
    if len(hy):
        a, b = int(hy.min()), int(hy.max()) + 1
        c, d = int(hx.min()), int(hx.max()) + 1
        rectangle = np.zeros_like(mask)
        rectangle[a:b, c:d] = True
        if 0 < a < b < height and 0 < c < d < width and np.array_equal(mask, rectangle):
            return "hole", (a, b, c, d)
    return None


def _analyze_pair(inp, out):
    """Recover the cycle and row phases from observed input cells."""
    assert inp.shape == out.shape
    candidates = []
    for raw in np.unique(inp):
        shape = _geometry(inp == raw)
        if shape is not None:
            candidates.append((int(raw), shape))
    assert len(candidates) == 1
    mask_color, (placement, box) = candidates[0]
    observed = inp != mask_color
    colors = set(int(c) for c in np.unique(inp[observed]))
    motif_size = len(colors)
    assert 2 <= motif_size <= 5 and mask_color not in colors
    successors = {}
    for row in range(inp.shape[0]):
        for col in range(inp.shape[1] - 1):
            if observed[row, col] and observed[row, col + 1]:
                left, right = int(inp[row, col]), int(inp[row, col + 1])
                assert left not in successors or successors[left] == right
                successors[left] = right
    assert set(successors) == colors
    cycle = [min(colors)]
    for _ in range(motif_size - 1):
        cycle.append(successors[cycle[-1]])
    assert len(set(cycle)) == motif_size and successors[cycle[-1]] == cycle[0]
    indices = {color: pos for pos, color in enumerate(cycle)}
    row_phases = {}
    for row in range(inp.shape[0]):
        columns = np.flatnonzero(observed[row])
        if len(columns):
            phase = (indices[int(inp[row, columns[0]])] - int(columns[0])) % motif_size
            assert all(cycle[(int(col) + phase) % motif_size] == inp[row, col]
                       for col in columns)
            row_phases[row] = phase
    periods = []
    for period in range(1, 5):
        groups = [[phase for row, phase in row_phases.items() if row % period == residue]
                  for residue in range(period)]
        if all(len(group) >= 2 and len(set(group)) == 1 for group in groups):
            periods.append((period, tuple(group[0] for group in groups)))
    assert periods
    period, phases = periods[0]
    rows, cols = np.indices(inp.shape)
    phase_grid = np.asarray(phases, dtype=np.int8)[rows % period]
    ordered = np.asarray(cycle, dtype=np.int8)
    field = ordered[(cols + phase_grid) % motif_size]
    expected = ordered[(cols + phase_grid + 1) % motif_size]
    assert np.array_equal(field[observed], inp[observed])
    assert np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    assert mask_color not in set(int(c) for c in np.unique(out))
    if placement == "hole":
        assert box[0] >= 2 * period and box[2] >= 2 * motif_size
    else:
        assert box[1] - box[0] >= 2 * period
        assert box[3] - box[2] >= 2 * motif_size
    return {"placement": placement, "mask": mask_color, "motif_size": motif_size,
            "period": period, "phases": phases, "cycle": tuple(cycle),
            "height": inp.shape[0], "width": inp.shape[1], "box": box, "field": field}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    source_keys = _source_keys()
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            input_key, pair_key = _grid_key(inp), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in source_keys
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[split].append(_analyze_pair(inp, out))
    witnesses = []
    for pair, info in zip(puzzle["train"], infos["train"]):
        if info["motif_size"] >= 3:
            field = info["field"]
            if (not np.array_equal(pair["output"], np.roll(field, 1, 0))
                    and not np.array_equal(pair["output"], np.roll(field, -1, 0))):
                witnesses.append(info)
    assert witnesses
    assert any(info["motif_size"] == 2 for info in infos["train"])
    assert {info["placement"] for info in infos["test"]} <= {
        info["placement"] for info in infos["train"]}
    assert len({info["mask"] for info in infos["train"]}) >= 2
    assert len({info["placement"] for info in infos["train"]}) >= 2


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_pair_key(pair) for split in ("train", "test")
                      for pair in p[split]) for p in puzzles}) == 42
    infos = [(split, _analyze_pair(pair["input"], pair["output"]))
             for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    assert {info["placement"] for _, info in infos} == set(_PLACEMENTS)
    assert {info["motif_size"] for _, info in infos} == {2, 3, 4, 5}
    assert {info["period"] for _, info in infos} == {1, 2, 3, 4}
    assert len({info["mask"] for _, info in infos}) >= 8
    assert len({(info["height"], info["width"], info["box"], info["placement"])
                for _, info in infos}) >= 70
    assert len({(info["placement"], info["motif_size"], info["period"])
                for _, info in infos}) >= 50
    assert any(info["height"] != info["width"] for _, info in infos)
    assert any(max(info["height"], info["width"]) <= 10 for _, info in infos)
    assert any(info["height"] >= 24 and info["width"] >= 24
               for split, info in infos if split == "test")
    assert any(info["motif_size"] == 2 for split, info in infos if split == "test")
    assert any(info["motif_size"] >= 4 for split, info in infos if split == "train")
    assert any(info["period"] >= 3 and len(set(info["phases"])) < info["period"]
               for _, info in infos)
    assert any(info["period"] >= 2 and any(
        (info["phases"][(i + 1) % info["period"]] - info["phases"][i])
        % info["motif_size"] not in (0, 1) for i in range(info["period"]))
               for _, info in infos)
    assert any(info["placement"] == "hole" and info["period"] >= 3
               for _, info in infos)
    assert any(info["placement"] != "hole" and
               info["height"] - (info["box"][1] - info["box"][0]) !=
               info["width"] - (info["box"][3] - info["box"][2])
               for _, info in infos)
    for puzzle in puzzles:
        train_infos = [_analyze_pair(pair["input"], pair["output"])
                       for pair in puzzle["train"]]
        assert len({info["motif_size"] for info in train_infos}) >= 2
        assert any(info["mask"] in other["cycle"]
                   for i, info in enumerate(train_infos)
                   for j, other in enumerate(train_infos) if i != j)
    assert any(max(_analyze_pair(pair["input"], pair["output"])["height"]
                   for pair in p["train"]) >= min(
                   _analyze_pair(pair["input"], pair["output"])["height"]
                   for pair in p["test"]) for p in puzzles)
    a, b = generate_puzzle(731, 7, 3), generate_puzzle(731, 7, 3)
    assert all(_pair_key(x) == _pair_key(y) for split in ("train", "test")
               for x, y in zip(a[split], b[split]))
    c = generate_puzzle(732, 7, 3)
    assert _pair_key(a["train"][0]) != _pair_key(c["train"][0])
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1101, None, 1), (1102, None, 4),
                              (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 3.0},
                   {"num_test": True}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
