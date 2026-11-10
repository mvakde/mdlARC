import numpy as np


_SOURCE_PAIRS = {
    (("060660606", "478788884", "066666660", "006066006", "487777884", "000060006", "606060060", "478878774", "660606600"),
     ("060660606", "460600004", "066666660", "006066006", "406666004", "000060006", "606060060", "460060664", "660606600")),
    (("060006600", "666666666", "066660000", "660006600", "066600606", "400666604", "066606600"),
     ("060006600", "666666666", "066660000", "660006600", "066600606", "488777784", "066606600")),
    (("606460046", "606006006", "066000060", "666000066", "600660006", "666406640"),
     ("606460046", "606806086", "066800070", "666800076", "600760086", "666406640")),
    (("046604640", "060006660", "000600666", "666000600", "066066066", "666606066", "060666666", "600060060", "040064640"),
     ("046604640", "070007670", "080608676", "676008680", "070608680", "076067076", "676607076", "070667676", "680068070", "040064640")),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
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


def _schedule(rng, n_train, n_test):
    combos = [("H", "encode"), ("H", "decode"), ("V", "encode"), ("V", "decode")]
    if int(rng.integers(2)):
        train = [("H", "encode"), ("V", "decode")]
    else:
        train = [("H", "decode"), ("V", "encode")]
    while len(train) < n_train:
        train.append(combos[int(rng.integers(4))])
    if int(rng.integers(2)):
        train[0], train[1] = train[1], train[0]
    unseen = [combo for combo in combos if combo not in train]
    test = []
    if unseen and rng.random() < 0.65:
        test.append(unseen[int(rng.integers(len(unseen)))])
    while len(test) < n_test:
        options = [combo for combo in combos if combo not in test]
        if not options:
            options = combos
        test.append(options[int(rng.integers(len(options)))])
    return train, test


def _sample_span(rng, regime):
    weights = [0.30, 0.52, 0.18] if regime == "source_train" else [0.29, 0.38, 0.33]
    band = int(rng.choice(3, p=weights))
    bounds = [(2, 5), (6, 10), (11, 24)][band]
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _bank_margins(rng, span):
    free = 29 - span
    if free == 0 or rng.random() < 0.26:
        return 0, 0
    cap = min(6, free)
    if free == 1 or rng.random() < 0.36:
        margin = int(rng.integers(1, cap + 1))
        return (margin, 0) if int(rng.integers(2)) else (0, margin)
    left = int(rng.integers(1, min(6, free - 1) + 1))
    right = int(rng.integers(1, min(6, free - left) + 1))
    return left, right


def _lanes(rng, size, count):
    if count == 1:
        return [int(rng.integers(size))]
    mode = int(rng.integers(3))
    if mode == 0:
        start = int(rng.integers(size - count + 1))
        return list(range(start, start + count))
    if mode == 1:
        # Spread endpoints, then jitter interior positions without overlap.
        selected = np.linspace(0, size - 1, count).round().astype(int)
        if count >= 3 and size >= count + 3:
            for j in range(1, count - 1):
                low = int(selected[j - 1] + 1)
                high = int(selected[j + 1] - 1)
                selected[j] = int(rng.integers(low, high + 1)) if high >= low else selected[j]
        return sorted(int(x) for x in selected)
    return sorted(int(x) for x in rng.choice(size, count, replace=False))


def _binary_field(rng, height, width, orientation, mode):
    rr, cc = np.indices((height, width))
    if mode == "iid":
        return (rng.random((height, width)) < rng.uniform(0.25, 0.75)).astype(np.int8)
    if mode == "runs":
        axial = cc if orientation == "H" else rr
        cross = rr if orientation == "H" else cc
        run = int(rng.integers(2, 6))
        palette = rng.integers(0, 2, size=(int(cross.max()) + 1, int(axial.max()) // run + 1))
        return palette[cross, axial // run].astype(np.int8)
    if mode == "patches":
        rh = int(rng.integers(2, 6))
        cw = int(rng.integers(2, 6))
        coarse = rng.integers(0, 2, size=((height - 1) // rh + 1, (width - 1) // cw + 1))
        return coarse[rr // rh, cc // cw].astype(np.int8)
    if mode == "stripes":
        coordinate = rr if int(rng.integers(2)) else cc
        period = int(rng.integers(1, 5))
        phase = int(rng.integers(2 * period))
        return (((coordinate + phase) // period) % 2).astype(np.int8)
    period = int(rng.integers(1, 3))
    phase = int(rng.integers(2))
    return (((rr // period + cc // period + phase) % 2)).astype(np.int8)


def _make_pair(rng, colors, regime, orientation, direction, force_both):
    span = _sample_span(rng, regime)
    if force_both and span == 2:
        span = 3
    lane_count = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.23, 0.16, 0.22, 0.17, 0.13, 0.09]))
    if regime == "target_test" and direction == "encode" and lane_count == 2:
        lane_count = 3
    left, right = _bank_margins(rng, span)
    axial_size = left + span + right + 1
    cross_size = int(rng.integers(max(6, lane_count + 1), 23))
    height, width = (cross_size, axial_size) if orientation == "H" else (axial_size, cross_size)
    lanes = _lanes(rng, cross_size, lane_count)
    bank0, bank1 = left, left + span
    active = np.zeros((height, width), dtype=bool)
    markers = np.zeros((height, width), dtype=bool)
    if orientation == "H":
        active[lanes, bank0 + 1:bank1] = True
        markers[lanes, bank0] = True
        markers[lanes, bank1] = True
    else:
        active[bank0 + 1:bank1, lanes] = True
        markers[bank0, lanes] = True
        markers[bank1, lanes] = True
    mode = ["iid", "runs", "patches", "stripes", "checker"][int(rng.choice(5, p=[0.42, 0.17, 0.16, 0.13, 0.12]))]
    bits = _binary_field(rng, height, width, orientation, mode)
    if not force_both and rng.random() < 0.19:
        lane = lanes[int(rng.integers(lane_count))]
        if orientation == "H":
            bits[lane, bank0 + 1:bank1] = int(rng.integers(2))
        else:
            bits[bank0 + 1:bank1, lane] = int(rng.integers(2))
    if not force_both and rng.random() < 0.08:
        bits[active] = int(rng.integers(2))
    positions = np.argwhere(active)
    if force_both:
        bits[tuple(positions[0])] = 0
        bits[tuple(positions[-1])] = 1
    outside = np.argwhere(~(active | markers))
    bits[tuple(outside[0])] = 0
    bits[tuple(outside[-1])] = 1
    base0, base1, high0, high1, delimiter = colors
    grid = np.where(bits == 0, base0, base1).astype(np.int8)
    grid[markers] = delimiter
    changed = grid.copy()
    changed[active & (grid == base0)] = high0
    changed[active & (grid == base1)] = high1
    if direction == "encode":
        return {"input": grid, "output": changed}
    return {"input": changed, "output": grid}


def _array_key(array):
    return (array.shape, array.tobytes())


def _source_key(pair):
    return tuple(tuple("".join(str(int(v)) for v in row) for row in pair[side]) for side in ("input", "output"))


def _geometry_candidates(grid, delimiter):
    marks = grid == delimiter
    candidates = []
    columns = np.flatnonzero(marks.any(axis=0))
    if len(columns) == 2:
        a, b = map(int, columns)
        lanes = np.flatnonzero(marks[:, a])
        if b - a >= 2 and len(lanes) and np.array_equal(lanes, np.flatnonzero(marks[:, b])) and int(marks.sum()) == 2 * len(lanes):
            active = np.zeros_like(marks)
            active[lanes, a + 1:b] = True
            candidates.append(("H", a, b, tuple(map(int, lanes)), active))
    rows = np.flatnonzero(marks.any(axis=1))
    if len(rows) == 2:
        a, b = map(int, rows)
        lanes = np.flatnonzero(marks[a])
        if b - a >= 2 and len(lanes) and np.array_equal(lanes, np.flatnonzero(marks[b])) and int(marks.sum()) == 2 * len(lanes):
            active = np.zeros_like(marks)
            active[a + 1:b, lanes] = True
            candidates.append(("V", a, b, tuple(map(int, lanes)), active))
    return candidates


def _learn(train):
    changes = set()
    unchanged = set()
    palette = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        difference = inp != out
        changes.update((int(a), int(b)) for a, b in zip(inp[difference], out[difference]))
        unchanged.update(int(x) for x in np.unique(inp[~difference]))
        palette.update(int(x) for x in np.unique(inp))
        palette.update(int(x) for x in np.unique(out))
    changed_colors = {color for edge in changes for color in edge}
    assert len(changed_colors) == 4
    base = changed_colors & unchanged
    high = changed_colors - base
    assert len(base) == len(high) == 2
    marker = palette - changed_colors
    assert len(marker) == 1 and len(palette) == 5
    marker = next(iter(marker))
    mapping = {}
    for b in base:
        options = [h for h in high if (b, h) in changes and (h, b) in changes]
        assert len(options) == 1
        mapping[b] = options[0]
    assert set(mapping.values()) == high
    assert changes == {(b, h) for b, h in mapping.items()} | {(h, b) for b, h in mapping.items()}
    return marker, mapping


def _interpret_input(inp, marker, mapping, output=None):
    base = set(mapping)
    high = set(mapping.values())
    candidates = _geometry_candidates(inp, marker)
    assert candidates
    highlighted = np.isin(inp, list(high))
    direction = "decode" if np.any(highlighted) else "encode"
    if output is not None:
        difference = inp != output
        candidates = [c for c in candidates if np.array_equal(c[4], difference)]
    elif direction == "decode":
        candidates = [c for c in candidates if np.array_equal(highlighted & c[4], highlighted)]
    assert len(candidates) == 1
    candidate = candidates[0]
    active = candidate[4]
    assert set(int(x) for x in np.unique(inp[active])) <= (high if direction == "decode" else base)
    assert np.any(active)
    assert set(int(x) for x in np.unique(inp[~active])) <= base | {marker}
    assert not np.any(highlighted & ~active)
    expected = inp.copy()
    if direction == "encode":
        for b, h in mapping.items():
            expected[active & (inp == b)] = h
    else:
        for b, h in mapping.items():
            expected[active & (inp == h)] = b
    return expected, candidate, direction


def _predict_from_training(train, test_input):
    marker, mapping = _learn(train)
    prediction, _, _ = _interpret_input(test_input, marker, mapping)
    return prediction


def _analyze_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        key = _array_key(inp)
        assert key not in seen
        seen.add(key)
        assert _source_key(pair) not in _SOURCE_PAIRS
    marker, mapping = _learn(puzzle["train"])
    metadata = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            predicted, geometry, direction = _interpret_input(inp, marker, mapping, out if split == "train" else None)
            assert np.array_equal(predicted, out)
            if split == "test":
                assert np.array_equal(_predict_from_training(puzzle["train"], inp.copy()), out)
            orientation, bank0, bank1, lanes, active = geometry
            assert np.array_equal(inp == marker, out == marker)
            assert np.array_equal(inp != out, active)
            assert set(int(x) for x in np.unique(inp[~(active | (inp == marker))])) == set(mapping)
            axial_size = inp.shape[1] if orientation == "H" else inp.shape[0]
            placement = "both_edges" if bank0 == 0 and bank1 == axial_size - 1 else "one_edge" if bank0 == 0 or bank1 == axial_size - 1 else "interior"
            metadata.append({"split": split, "orientation": orientation, "direction": direction,
                             "span": bank1 - bank0, "lanes": len(lanes), "lane_positions": lanes,
                             "placement": placement, "height": inp.shape[0], "width": inp.shape[1],
                             "uniform_active": len(np.unique(inp[active])) == 1})
    train_meta = [item for item in metadata if item["split"] == "train"]
    assert {item["orientation"] for item in train_meta} == {"H", "V"}
    assert {item["direction"] for item in train_meta} == {"encode", "decode"}
    return {"marker": marker, "mapping": mapping, "metadata": metadata}


def validate_generated_puzzle(puzzle: dict) -> None:
    _analyze_generated_puzzle(puzzle)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    train_specs, test_specs = _schedule(rng, n_train, n_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for index, (orientation, direction) in enumerate(specs):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                pair = _make_pair(pair_rng, colors, "source_train" if split == "train" else "target_test",
                                  orientation, direction, split == "train" and index < 2)
                key = _array_key(pair["input"])
                if key not in seen and _source_key(pair) not in _SOURCE_PAIRS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(np.array_equal(a[side], b[side]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for side in ("input", "output")) and all(
                   len(first[split]) == len(second[split]) for split in ("train", "test"))


def _observed_texture_motifs(pair, marker, mapping):
    from numpy.lib.stride_tricks import sliding_window_view

    inp, out = pair["input"], pair["output"]
    base_grid = out if np.isin(inp, list(mapping.values())).any() else inp
    binary = base_grid == sorted(mapping)[1]
    known = base_grid != marker
    motifs = set()
    if min(base_grid.shape) >= 4:
        windows = sliding_window_view(binary, (4, 4))
        valid = sliding_window_view(known, (4, 4)).all(axis=(-1, -2))
        checker = np.indices((4, 4)).sum(axis=0) % 2 == 0
        stripe = np.tile([False, True, False, True], (4, 1))
        for name, pattern in (("checker", checker), ("stripes", stripe), ("stripes", stripe.T)):
            matching = (windows == pattern).all(axis=(-1, -2)) | (windows != pattern).all(axis=(-1, -2))
            if np.any(valid & matching):
                motifs.add(name)
    if min(base_grid.shape) >= 3:
        windows = sliding_window_view(binary, (3, 3))
        valid = sliding_window_view(known, (3, 3)).all(axis=(-1, -2))
        if np.any(valid & (windows.all(axis=(-1, -2)) | (~windows).all(axis=(-1, -2)))):
            motifs.add("patch")
    for shape in ((1, 5), (5, 1)):
        if base_grid.shape[0] < shape[0] or base_grid.shape[1] < shape[1]:
            continue
        windows = sliding_window_view(binary, shape)
        valid = sliding_window_view(known, shape).all(axis=(-1, -2))
        if np.any(valid & (windows.all(axis=(-1, -2)) | (~windows).all(axis=(-1, -2)))):
            motifs.add("run")
    return motifs


def self_test() -> None:
    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    combos, lanes, placements, span_bands = set(), set(), set(), set()
    short_tests = long_tests = uniform_pairs = mixed_test_sets = vertical_decodes = 0
    train_combos, test_combos_seen, spacing = set(), set(), set()
    varied_lanes = varied_spans = varied_placements = 0
    two_lane_decoding_tests = 0
    texture_motifs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        analysis = _analyze_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            texture_motifs.update(_observed_texture_motifs(pair, analysis["marker"], analysis["mapping"]))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palettes.add((analysis["marker"], tuple(sorted(analysis["mapping"].items()))))
        test_combos = set()
        varied_lanes += len({m["lanes"] for m in analysis["metadata"]}) >= 2
        varied_spans += len({m["span"] for m in analysis["metadata"]}) >= 2
        varied_placements += len({m["placement"] for m in analysis["metadata"]}) >= 2
        for item in analysis["metadata"]:
            combo = (item["orientation"], item["direction"])
            combos.add(combo)
            (train_combos if item["split"] == "train" else test_combos_seen).add(combo)
            lanes.add(item["lanes"])
            placements.add(item["placement"])
            shapes.add((item["height"], item["width"]))
            layouts.add((combo, item["lanes"], item["span"], item["placement"], item["lane_positions"]))
            span_bands.add(0 if item["span"] <= 5 else 1 if item["span"] <= 10 else 2)
            uniform_pairs += item["uniform_active"]
            if item["lanes"] >= 2:
                gaps = np.diff(item["lane_positions"])
                if max(gaps) <= 2:
                    spacing.add("close")
                if min(gaps) >= 3:
                    spacing.add("wide")
                if len(gaps) >= 2 and max(gaps) - min(gaps) >= 2:
                    spacing.add("irregular")
            if item["split"] == "test":
                test_combos.add(combo)
                short_tests += item["span"] <= 10
                long_tests += item["span"] >= 11
                vertical_decodes += combo == ("V", "decode")
                two_lane_decoding_tests += item["lanes"] == 2 and item["direction"] == "decode"
                assert not (item["lanes"] == 2 and item["direction"] == "encode")
        mixed_test_sets += len(test_combos) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 30 and len(shapes) >= 55 and len(layouts) >= 100
    assert combos == {("H", "encode"), ("H", "decode"), ("V", "encode"), ("V", "decode")}
    assert train_combos == combos and test_combos_seen == combos
    assert lanes == {1, 2, 3, 4, 5, 6} and placements == {"both_edges", "one_edge", "interior"}
    assert spacing == {"close", "wide", "irregular"}
    assert varied_lanes >= 35 and varied_spans >= 35 and varied_placements >= 35
    assert span_bands == {0, 1, 2} and short_tests >= 10 and long_tests >= 10
    assert uniform_pairs >= 5 and mixed_test_sets >= 5 and vertical_decodes >= 5
    assert two_lane_decoding_tests >= 1
    assert texture_motifs == {"checker", "stripes", "patch", "run"}
    for seed, nt, ns in ((700, 2, 1), (701, 10, 1), (702, 10, 4), (703, 6, 2),
                         (800, None, 1), (801, None, 4), (802, 2, None), (803, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(generate_puzzle(991, 7, 3), generate_puzzle(991, 7, 3))
    assert not _puzzles_equal(generate_puzzle(991, 7, 3), generate_puzzle(992, 7, 3))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
