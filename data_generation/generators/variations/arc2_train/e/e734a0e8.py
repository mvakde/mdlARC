import numpy as np


# Source inputs are kept as literal data so exact originals cannot be emitted.
_SOURCE_INPUT_KEYS = {
    ((23, 23), "".join((
        "77777077777077777077777", "77777077777077777077777",
        "77777077077077777077777", "77777077777077777077777",
        "77777077777077777077777", "00000000000000000000000",
        "77777077777077777077777", "77777077777077777077777",
        "77777077077077777077777", "77777077777077777077777",
        "77777077777077777077777", "00000000000000000000000",
        "77777077777077799077777", "77777077777077797077777",
        "77777077777077797077077", "77777077777079997077777",
        "77777077777079777077777", "00000000000000000000000",
        "77777077777077777077777", "77777077777077777077777",
        "77077077777077777077077", "77777077777077777077777",
        "77777077777077777077777",
    ))),
    ((11, 11), "".join((
        "72707770777", "72707770777", "72707770777",
        "00000000000", "77707770777", "77707070777",
        "77707770777", "00000000000", "77707770777",
        "77707070707", "77707770777",
    ))),
    ((17, 17), "".join((
        "77777077777077477", "77777077777044444",
        "77777077777077477", "77777077777074747",
        "77777077777047774", "00000000000000000",
        "77777077777077777", "77777077777077777",
        "77777077077077777", "77777077777077777",
        "77777077777077777", "00000000000000000",
        "77777077777077777", "77777077777077777",
        "77077077777077077", "77777077777077777",
        "77777077777077777",
    ))),
}


def _plain_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None and (not _plain_int(num_train) or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 through 10")
    if num_test is not None and (not _plain_int(num_test) or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train is None and num_test is None:
        # Several compact puzzles, as well as larger teaching sets.
        num_train, num_test = (
            (2, 1), (2, 1), (2, 1), (2, 1),
            (3, 1), (3, 2), (4, 1), (4, 3),
            (5, 2), (6, 3), (8, 4), (10, 4),
        )[seed % 12]
    elif num_train is None:
        num_train = int(num_test) + 1 + seed % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _layouts():
    return tuple(
        (rows, cols, height, width)
        for height in (3, 5, 7)
        for width in (3, 5, 7)
        for rows in range(2, 8)
        for cols in range(2, 8)
        if rows * (height + 1) - 1 <= 30
        and cols * (width + 1) - 1 <= 30
    )


_LAYOUTS = _layouts()


def sample_puzzle_parameters(rng):
    return {
        "palette": tuple(int(x) for x in rng.permutation(10)[:6]),
        "color_mode": ("stable", "accent", "swap", "free")[int(rng.integers(0, 4))],
    }


def _colors(rng, parameters, index):
    palette = parameters["palette"]
    mode = parameters["color_mode"]
    if mode == "stable":
        return palette[:3]
    if mode == "accent":
        return palette[0], palette[1], palette[2 + index % 3]
    if mode == "swap":
        offset = index % 3
        return tuple(palette[(offset + i) % 3] for i in range(3))
    selected = rng.choice(6, size=3, replace=False)
    return tuple(palette[int(i)] for i in selected)


def _motif_mask(rng, height, width, center_accent, forbidden):
    cy, cx = height // 2, width // 2
    for _ in range(180):
        kind = int(rng.integers(0, 9))
        mask = np.zeros((height, width), dtype=bool)
        if kind == 0:  # bar, sometimes centered and sometimes offset
            if rng.integers(0, 2):
                y = cy if rng.integers(0, 2) else int(rng.integers(0, height))
                left = int(rng.integers(0, width - 1))
                right = int(rng.integers(left + 2, width + 1))
                mask[y, left:right] = True
            else:
                x = cx if rng.integers(0, 2) else int(rng.integers(0, width))
                top = int(rng.integers(0, height - 1))
                bottom = int(rng.integers(top + 2, height + 1))
                mask[top:bottom, x] = True
        elif kind == 1:  # hook
            y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
            mask[y, :] = True
            mask[:, x] = True
            if rng.integers(0, 2):
                mask[:y, x] = False
            else:
                mask[y + 1:, x] = False
        elif kind == 2:  # cross
            y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
            mask[y, :] = True
            mask[:, x] = True
        elif kind == 3:  # diagonal or X
            for y in range(height):
                x = round(y * (width - 1) / (height - 1))
                mask[y, x] = True
                if rng.integers(0, 2):
                    mask[y, width - 1 - x] = True
            if rng.integers(0, 2):
                mask = np.fliplr(mask)
        elif kind in (4, 5):  # outlined or filled rectangle
            y0 = int(rng.integers(0, height - 1))
            x0 = int(rng.integers(0, width - 1))
            y1 = int(rng.integers(y0 + 2, height + 1))
            x1 = int(rng.integers(x0 + 2, width + 1))
            mask[y0:y1, x0:x1] = True
            if kind == 4:
                mask[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = False
        elif kind == 6:  # parallel bars or a structured combination
            if rng.integers(0, 2):
                mask[0, :] = True
                mask[-1, :] = True
            else:
                mask[:, 0] = True
                mask[:, -1] = True
            if rng.integers(0, 2):
                mask[cy, cx] = True
        elif kind == 7:  # sparse, possibly disconnected points
            count = int(rng.integers(2, min(height * width - 1, 7) + 1))
            chosen = rng.choice(height * width, size=count, replace=False)
            mask.flat[chosen] = True
        else:  # irregular or nearly filled tiles
            density = float(rng.choice((0.16, 0.35, 0.55, 0.80)))
            mask = rng.random((height, width)) < density

        symmetry = int(rng.choice((0, 0, 0, 1, 2, 3, 4)))
        if symmetry == 1:
            mask |= np.fliplr(mask)
        elif symmetry == 2:
            mask |= np.flipud(mask)
        elif symmetry == 3:
            mask |= np.fliplr(mask) | np.flipud(mask)
        elif symmetry == 4:
            mask |= np.flipud(np.fliplr(mask))
        if symmetry and rng.random() < 0.12:
            y = int(rng.integers(0, height))
            x = int(rng.integers(0, width))
            if (y, x) != (cy, cx):
                mask[y, x] = not mask[y, x]
        mask[cy, cx] = center_accent
        count = int(np.count_nonzero(mask))
        key = mask.tobytes()
        if 2 <= count <= height * width - 2 and key not in forbidden:
            return mask
    raise RuntimeError("could not find a distinct motif")


def _destination_positions(rng, rows, cols, source, index):
    available = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != source]
    count_available = len(available)
    mode = int(rng.integers(0, 5))
    if index == 0:
        mode = 1 + mode % 4  # the first demonstration shows multiple copies
    if mode == 0:
        chosen = [available[int(rng.integers(0, count_available))]]
    elif mode == 4:
        chosen = available
    else:
        if mode == 1:
            count = int(rng.integers(2, min(4, count_available) + 1))
        elif mode == 2:
            count = int(rng.integers(2, min(6, count_available) + 1))
        else:
            count = int(rng.integers(2, count_available + 1))
        if mode == 2:
            anchor = available[int(rng.integers(0, count_available))]
            order = sorted(available, key=lambda p: (abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]), float(rng.random())))
            chosen = order[:count]
        else:
            indices = rng.choice(count_available, size=count, replace=False)
            chosen = [available[int(i)] for i in indices]
    return tuple(sorted(chosen))


def _choose_layout(rng, used_train, regime, index):
    candidates = list(_LAYOUTS)
    if regime == "source_train" and index == 1:
        candidates = [x for x in candidates if x != used_train[0]]
    if regime == "target_test":
        layout_relation = float(rng.random())
        if layout_relation < 0.25:
            return used_train[int(rng.integers(0, len(used_train)))]
        if layout_relation < 0.75:
            candidates = [x for x in candidates if x not in used_train]
    # Balanced shape groups let both small grids and long or tall grids occur.
    group = int(rng.integers(0, 4))
    filtered = [x for x in candidates if (
        (group == 0 and x[0] <= 3 and x[1] <= 3)
        or (group == 1 and x[0] >= 5)
        or (group == 2 and x[1] >= 5)
        or (group == 3 and x[2] != x[3])
    )]
    if filtered:
        candidates = filtered
    return candidates[int(rng.integers(0, len(candidates)))]


def generate_pair(rng, regime, puzzle_parameters, pair_index=0, layout=None, forbidden_masks=()):
    if regime not in ("source_train", "target_test"):
        raise ValueError("invalid regime")
    rows, cols, height, width = layout if layout is not None else _LAYOUTS[int(rng.integers(0, len(_LAYOUTS)))]
    structural, carrier, accent = _colors(rng, puzzle_parameters, pair_index)
    source = (int(rng.integers(0, rows)), int(rng.integers(0, cols)))
    destinations = _destination_positions(rng, rows, cols, source, pair_index)
    center_accent = (pair_index == 1) if regime == "source_train" and pair_index < 2 else bool(rng.integers(0, 2))
    mask = _motif_mask(rng, height, width, center_accent, set(forbidden_masks))
    motif = np.where(mask, accent, carrier).astype(np.int8)
    grid_height = rows * (height + 1) - 1
    grid_width = cols * (width + 1) - 1
    inp = np.full((grid_height, grid_width), carrier, dtype=np.int8)
    inp[height::height + 1, :] = structural
    inp[:, width::width + 1] = structural
    sy, sx = source
    inp[sy * (height + 1):sy * (height + 1) + height,
        sx * (width + 1):sx * (width + 1) + width] = motif
    for y, x in destinations:
        inp[y * (height + 1) + height // 2, x * (width + 1) + width // 2] = structural
    out = inp.copy()
    for y, x in destinations:
        out[y * (height + 1):y * (height + 1) + height,
            x * (width + 1):x * (width + 1) + width] = motif
    return inp, out


def _array_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_key(grid):
    return grid.shape, "".join(str(int(x)) for x in grid.flat)


def _analyze_pair(inp, out):
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == out.ndim == 2 and inp.shape == out.shape
    assert all(1 <= dim <= 30 for dim in inp.shape)
    assert np.issubdtype(inp.dtype, np.integer) and np.issubdtype(out.dtype, np.integer)
    assert 0 <= int(inp.min()) <= int(inp.max()) <= 9
    assert 0 <= int(out.min()) <= int(out.max()) <= 9
    colors = [int(x) for x in np.unique(inp)]
    assert len(colors) == 3 and set(colors) == set(int(x) for x in np.unique(out))
    candidates = []
    for color in colors:
        full_rows = np.flatnonzero(np.all(inp == color, axis=1))
        full_cols = np.flatnonzero(np.all(inp == color, axis=0))
        if len(full_rows) and len(full_cols):
            candidates.append((color, full_rows, full_cols))
    assert len(candidates) == 1
    structural, separator_rows, separator_cols = candidates[0]
    height, width = int(separator_rows[0]), int(separator_cols[0])
    assert height in (3, 5, 7) and width in (3, 5, 7)
    assert np.array_equal(separator_rows, np.arange(height, inp.shape[0], height + 1))
    assert np.array_equal(separator_cols, np.arange(width, inp.shape[1], width + 1))
    rows, cols = len(separator_rows) + 1, len(separator_cols) + 1
    assert 2 <= rows <= 7 and 2 <= cols <= 7
    assert inp.shape == (rows * (height + 1) - 1, cols * (width + 1) - 1)
    assert np.array_equal(inp[separator_rows, :], out[separator_rows, :])
    assert np.array_equal(inp[:, separator_cols], out[:, separator_cols])
    other = [c for c in colors if c != structural]
    carrier, accent = sorted(other, key=lambda c: int(np.count_nonzero(inp == c)), reverse=True)
    assert np.count_nonzero(inp == carrier) > np.count_nonzero(inp == accent)
    source = None
    destinations = []
    blank_count = 0
    motif = None
    for y in range(rows):
        for x in range(cols):
            block = inp[y * (height + 1):y * (height + 1) + height,
                        x * (width + 1):x * (width + 1) + width]
            if np.any(block == accent):
                assert source is None and np.all((block == carrier) | (block == accent))
                source, motif = (y, x), block.copy()
            elif np.all(block == carrier):
                blank_count += 1
            else:
                assert np.count_nonzero(block == structural) == 1
                assert block[height // 2, width // 2] == structural
                trial = block.copy()
                trial[height // 2, width // 2] = carrier
                assert np.all(trial == carrier)
                destinations.append((y, x))
    assert source is not None and len(destinations) >= 1
    mask = motif == accent
    accent_count = int(np.count_nonzero(mask))
    assert 2 <= accent_count <= height * width - 2
    expected = inp.copy()
    for y, x in destinations:
        expected[y * (height + 1):y * (height + 1) + height,
                 x * (width + 1):x * (width + 1) + width] = motif
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    symmetric = any(np.array_equal(mask, transformed) for transformed in
                    (np.fliplr(mask), np.flipud(mask), np.flipud(np.fliplr(mask))))
    return {
        "layout": (rows, cols, height, width), "structural": structural,
        "carrier": carrier, "accent": accent, "source": source,
        "destinations": tuple(destinations), "blank_count": blank_count,
        "motif": mask.tobytes(), "accent_count": accent_count,
        "center_accent": bool(mask[height // 2, width // 2]),
        "symmetric": symmetric,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, seen_motifs = set(), set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            analysis = _analyze_pair(inp, out)
            inp_key = _array_key(inp)
            pair_key = inp_key, _array_key(out)
            motif_key = analysis["layout"][2:], analysis["motif"]
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            assert motif_key not in seen_motifs
            assert _source_key(inp) not in _SOURCE_INPUT_KEYS
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            seen_motifs.add(motif_key)
            analyses[split].append(analysis)
    assert len({a["layout"] for a in analyses["train"]}) >= 2
    assert {a["center_accent"] for a in analyses["train"]} == {False, True}
    assert len(analyses["train"][0]["destinations"]) >= 2
    # These are the only conventions needed at test time: visible source and
    # lattice-colored center marks cause exact copying in every training pair.
    for a in analyses["test"]:
        assert a["source"] not in a["destinations"]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _plain_int(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    used_train = []
    used_inputs, used_motifs = set(), set()
    for split, count, regime in (("train", num_train, "source_train"),
                                 ("test", num_test, "target_test")):
        for index in range(count):
            for _ in range(120):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                layout = _choose_layout(pair_rng, used_train, regime, index)
                inp, out = generate_pair(pair_rng, regime, parameters,
                                         index if split == "train" else num_train + index,
                                         layout, ())
                analysis = _analyze_pair(inp, out)
                motif_key = layout[2:], analysis["motif"]
                if (_array_key(inp) in used_inputs or motif_key in used_motifs
                        or _source_key(inp) in _SOURCE_INPUT_KEYS):
                    continue
                used_inputs.add(_array_key(inp))
                used_motifs.add(motif_key)
                puzzle[split].append({"input": inp, "output": out})
                if split == "train":
                    used_train.append(layout)
                break
            else:
                raise RuntimeError("could not produce a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[role], b[role])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for role in ("input", "output")) and all(
                   len(left[split]) == len(right[split]) for split in ("train", "test"))


def _predict_visible_copy(inp):
    # A second reconstruction path for self-test, using only the input grid.
    structural = next(int(color) for color in np.unique(inp)
                      if np.any(np.all(inp == color, axis=1))
                      and np.any(np.all(inp == color, axis=0)))
    cuts_y = np.flatnonzero(np.all(inp == structural, axis=1))
    cuts_x = np.flatnonzero(np.all(inp == structural, axis=0))
    carrier = max((int(color) for color in np.unique(inp) if color != structural),
                  key=lambda color: int(np.count_nonzero(inp == color)))
    accent = next(int(color) for color in np.unique(inp)
                  if color not in (structural, carrier))
    row_ranges = list(zip([0] + [int(y) + 1 for y in cuts_y],
                          [int(y) for y in cuts_y] + [inp.shape[0]]))
    col_ranges = list(zip([0] + [int(x) + 1 for x in cuts_x],
                          [int(x) for x in cuts_x] + [inp.shape[1]]))
    motif = None
    marked = []
    for y0, y1 in row_ranges:
        for x0, x1 in col_ranges:
            tile = inp[y0:y1, x0:x1]
            if np.any(tile == accent):
                assert motif is None
                motif = tile.copy()
            elif np.any(tile == structural):
                marked.append((y0, y1, x0, x1))
    assert motif is not None and marked
    predicted = inp.copy()
    for y0, y1, x0, x1 in marked:
        predicted[y0:y1, x0:x1] = motif
    return predicted


def self_test() -> None:
    analyses = []
    puzzle_analyses = []
    counts = set()
    within_layouts = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        local = []
        local_by_split = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair["input"], pair["output"])
                assert np.array_equal(_predict_visible_copy(pair["input"]), pair["output"])
                analyses.append((split, a))
                local.append(a["layout"])
                local_by_split[split].append(a)
        within_layouts.append(len(set(local)))
        puzzle_analyses.append(local_by_split)
    layouts = {a["layout"] for _, a in analyses}
    sources = {a["source"] for _, a in analyses}
    role_triples = {(a["structural"], a["carrier"], a["accent"]) for _, a in analyses}
    marker_counts = {len(a["destinations"]) for _, a in analyses}
    assert len({x for x, _ in counts}) >= 4 and len({y for _, y in counts}) >= 3
    assert (2, 1) in counts and len(layouts) >= 35 and len(role_triples) >= 25
    assert len(sources) >= 12 and len(marker_counts) >= 10
    assert sum(a["symmetric"] for _, a in analyses) >= 50
    assert sum(not a["symmetric"] for _, a in analyses) >= 30
    assert sum(a["layout"][2] != a["layout"][3] for _, a in analyses) >= 50
    assert sum(7 in a["layout"][2:] for _, a in analyses) >= 20
    assert sum(len(a["destinations"]) == 1 for _, a in analyses) >= 10
    assert sum(len(a["destinations"]) == a["layout"][0] * a["layout"][1] - 1
               for _, a in analyses) >= 10
    assert sum(a["blank_count"] == 0 for _, a in analyses) >= 10
    assert sum(a["accent_count"] / (a["layout"][2] * a["layout"][3]) <= 0.20
               for _, a in analyses) >= 10
    assert sum(a["accent_count"] / (a["layout"][2] * a["layout"][3]) >= 0.70
               for _, a in analyses) >= 10
    assert any(a["accent_count"] / (a["layout"][2] * a["layout"][3]) >= 0.85
               for _, a in analyses)
    assert sum(a["layout"][0] >= 5 and a["layout"][1] <= 3 for _, a in analyses) >= 10
    assert sum(a["layout"][1] >= 5 and a["layout"][0] <= 3 for _, a in analyses) >= 10
    cluster_count = 0
    scatter_count = 0
    near_symmetry_count = 0
    for _, a in analyses:
        marks = a["destinations"]
        diameter = max(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p in marks for q in marks)
        cluster_count += 2 <= len(marks) <= 5 and diameter <= 2
        scatter_count += len(marks) >= 2 and diameter >= 5
        height, width = a["layout"][2:]
        mask = np.frombuffer(a["motif"], dtype=bool).reshape(height, width)
        distances = [int(np.count_nonzero(mask != transformed)) for transformed in
                     (np.fliplr(mask), np.flipud(mask), np.flipud(np.fliplr(mask)))]
        near_symmetry_count += 0 < min(distances) <= 2
    assert cluster_count >= 10 and scatter_count >= 10 and near_symmetry_count >= 10
    assert any(a["source"] == (0, a["layout"][1] - 1) for split, a in analyses if split == "train")
    assert any(0 < a["source"][0] < a["layout"][0] - 1 and
               0 < a["source"][1] < a["layout"][1] - 1
               for split, a in analyses if split == "test")
    assert min(within_layouts) >= 2
    assert sum(train == 2 and test == 1 for train, test in
               ((len(p["train"]), len(p["test"])) for p in puzzle_analyses)) >= 10
    familiar, novel, stable, exchanged, reused_accent = 0, 0, 0, 0, 0
    for p in puzzle_analyses:
        train_layouts = {a["layout"] for a in p["train"]}
        familiar += any(a["layout"] in train_layouts for a in p["test"])
        novel += any(a["layout"] not in train_layouts for a in p["test"])
        all_pairs = p["train"] + p["test"]
        stable += len({(a["structural"], a["carrier"], a["accent"])
                       for a in all_pairs}) == 1
        exchanged += any(a["structural"] == b["carrier"]
                         for a in all_pairs for b in all_pairs)
        reused_accent += bool({a["accent"] for a in p["train"]} &
                              {a["accent"] for a in p["test"]})
    assert familiar >= 5 and novel >= 10
    assert stable >= 5 and exchanged >= 5 and reused_accent >= 10
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(20000 + train * 10 + test, train, test)
        validate_generated_puzzle(p)
        assert _puzzles_equal(p, generate_puzzle(20000 + train * 10 + test, train, test))
    assert not _puzzles_equal(generate_puzzle(123), generate_puzzle(124))
    for seed, train, test in ((30001, None, 1), (30002, None, 4),
                              (30003, 2, None), (30004, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        assert (train is None or len(p["train"]) == train)
        assert (test is None or len(p["test"]) == test)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
