"""Scattered length-coded lines become ranked concentric square layers."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "38e8d34b134230aa68d62514ad90417e13b6840456a5fb20bd4d3d601d82ea19",
    "436037abccb5e55f38b37c450d0845f77abd96cb861e533630b5ffdd8f0c509d",
    "e6d7dd96025dc0285f02b1829b2686ed4c30c842fc84f1bb3d958a20e131ab1c",
    "7df5dcf1caf3ca4658745e5fd5f8952ebcf68d4fd0d7a8eb15debc6d2b4546f3",
})


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _child_rng(rng):
    return np.random.default_rng(int(rng.integers(0, 1 << 63)))


def _output(side, rank_colors):
    axis = np.arange(side)
    depth = np.minimum.reduce([
        np.broadcast_to(axis[:, None], (side, side)),
        np.broadcast_to(axis[None, :], (side, side)),
        np.broadcast_to((side - 1 - axis)[:, None], (side, side)),
        np.broadcast_to((side - 1 - axis)[None, :], (side, side)),
    ])
    return np.asarray(rank_colors, dtype=np.int8)[depth]


def _lengths(rng, m, side, witness):
    if not witness and rng.random() < 0.22:
        return tuple(range(side, 0, -2))
    if witness:
        canonical_shortest = 1 if side % 2 else 2
        possible = [v for v in range(1, side - m + 2) if v != canonical_shortest]
        shortest = int(rng.choice(possible))
        middle = rng.choice(np.arange(shortest + 1, side), size=m - 2, replace=False)
        return (side,) + tuple(sorted((int(v) for v in middle), reverse=True)) + (shortest,)
    shorter = rng.choice(np.arange(1, side), size=m - 1, replace=False)
    return (side,) + tuple(sorted((int(v) for v in shorter), reverse=True))


def _orientations(rng, lengths, style):
    if style == "M":
        result = ["H" if rng.integers(0, 2) == 0 else "V" for _ in lengths]
        movable = [i for i, length in enumerate(lengths) if length > 1]
        if len(movable) >= 2:
            result[movable[0]], result[movable[1]] = "H", "V"
    else:
        result = [style] * len(lengths)
    return tuple("S" if length == 1 else result[i] for i, length in enumerate(lengths))


def _shape(rng, lengths, orientations):
    m = len(lengths)
    h_need = max((length for length, o in zip(lengths, orientations) if o == "V"), default=1)
    w_need = max((length for length, o in zip(lengths, orientations) if o == "H"), default=1)
    mode = int(rng.integers(0, 4))
    if mode == 0:  # compact, including canvases smaller than the output on one axis
        h_low = max(h_need, 2 * m - 1 if all(o != "V" for o in orientations) else m + 2)
        w_low = max(w_need, 2 * m - 1 if all(o != "H" for o in orientations) else m + 2)
        h = min(30, h_low + int(rng.integers(0, 4)))
        w = min(30, w_low + int(rng.integers(0, 4)))
    elif mode == 1:  # medium
        h_low, w_low = max(h_need, m + 5), max(w_need, m + 5)
        h = int(rng.integers(h_low, min(30, h_low + 13) + 1))
        w = int(rng.integers(w_low, min(30, w_low + 13) + 1))
    elif mode == 2:  # elongated
        if rng.integers(0, 2) == 0:
            h = int(rng.integers(max(h_need, m + 3), max(h_need, m + 3, 21) + 1))
            w = int(rng.integers(max(w_need, 23), 31))
        else:
            h = int(rng.integers(max(h_need, 23), 31))
            w = int(rng.integers(max(w_need, m + 3), max(w_need, m + 3, 21) + 1))
    else:  # spacious
        h = int(rng.integers(max(h_need, 22), 31))
        w = int(rng.integers(max(w_need, 22), 31))
    return h, w


def _place(rng, shape, lengths, orientations, colors, background):
    h, w = shape
    if h * w <= 2 * sum(lengths) + 2:
        return None
    candidates = []
    for length, orientation in zip(lengths, orientations):
        if orientation == "H":
            candidates.append([(r, c) for r in range(h) for c in range(w - length + 1)])
        elif orientation == "V":
            candidates.append([(r, c) for r in range(h - length + 1) for c in range(w)])
        else:
            candidates.append([(r, c) for r in range(h) for c in range(w)])
    if any(not positions for positions in candidates):
        return None
    for _ in range(24):
        blocked = np.zeros(shape, dtype=bool)
        grid = np.full(shape, background, dtype=np.int8)
        good = True
        for length, orientation, color, positions in zip(lengths, orientations, colors, candidates):
            found = False
            for k in rng.permutation(len(positions)):
                row, col = positions[int(k)]
                cells = [(row, col + d) if orientation == "H" else
                         (row + d, col) if orientation == "V" else (row, col)
                         for d in range(length)]
                if any(blocked[r, c] for r, c in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                    blocked[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
                found = True
                break
            if not found:
                good = False
                break
        if good:
            return grid
    return None


def _colors(rng, m, background, anchor=None, anchor_rank=None, novel=None,
            excluded=None):
    available = [c for c in range(10) if c != background and c != anchor
                 and c != excluded]
    chosen = [int(c) for c in rng.choice(available, size=m - (anchor is not None), replace=False)]
    if anchor is not None:
        chosen.insert(anchor_rank, anchor)
    if novel is not None and novel != background and novel not in chosen:
        replaceable = [i for i, c in enumerate(chosen) if c != anchor]
        if replaceable:
            chosen[int(rng.choice(replaceable))] = novel
    return tuple(chosen)


def _make_pair(rng, m, parity, background, colors, orientation_style, witness,
               preferred_shape=None):
    side = 2 * m - 1 if parity == 1 else 2 * m
    lengths = _lengths(rng, m, side, witness)
    orientations = _orientations(rng, lengths, orientation_style)
    if preferred_shape is not None:
        grid = _place(rng, preferred_shape, lengths, orientations, colors, background)
        if grid is not None:
            return {"input": grid, "output": _output(side, colors)}
    for _ in range(28):
        shape = _shape(rng, lengths, orientations)
        grid = _place(rng, shape, lengths, orientations, colors, background)
        if grid is not None:
            return {"input": grid, "output": _output(side, colors)}
    grid = _place(rng, (30, 30), lengths, orientations, colors, background)
    _require(grid is not None, "spacious fallback could not place segments")
    return {"input": grid, "output": _output(side, colors)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(_seed(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    base_background = 0 if rng.random() < 0.5 else int(rng.integers(0, 10))
    vary_background = bool(rng.integers(0, 2))
    orientation_plan = str(rng.choice(("all_H", "all_V", "H_to_V", "V_to_H", "varied")))
    reserved_test_color = int(rng.integers(0, 10)) if rng.random() < 0.35 else None

    def background_for(split, index, anchor=None):
        if not vary_background or (split == "train" and index == 0):
            candidate = base_background
        elif split == "train" and index == 1:
            candidate = int(rng.choice([c for c in range(10) if c != base_background and c != anchor]))
        else:
            candidate = base_background if rng.random() < 0.5 else int(rng.integers(0, 10))
        if candidate == anchor:
            candidate = int(rng.choice([c for c in range(10) if c != anchor]))
        return candidate

    def style_for(split):
        if orientation_plan == "all_H":
            return "H"
        if orientation_plan == "all_V":
            return "V"
        if orientation_plan == "H_to_V":
            return "H" if split == "train" else "V"
        if orientation_plan == "V_to_H":
            return "V" if split == "train" else "H"
        return str(rng.choice(("H", "V", "M")))

    train = []
    infos = []
    previous_shapes = []
    max_train_m = 8 if reserved_test_color is not None else 9
    first_m = int(rng.integers(2, max_train_m + 1))
    second_m = int(rng.choice([m for m in range(2, max_train_m + 1) if m != first_m]))
    first_parity = int(rng.integers(0, 2))
    specifications = [(first_m, first_parity), (second_m, 1 - first_parity)]
    specifications += [(int(rng.integers(2, max_train_m + 1)), int(rng.integers(0, 2)))
                       for _ in range(train_count - 2)]
    anchor = None
    for i, (m, parity) in enumerate(specifications):
        pair_rng = _child_rng(rng)
        bg = background_for("train", i, anchor if i == 1 else None)
        colors = (_colors(pair_rng, m, bg, anchor, 1, excluded=reserved_test_color)
                  if i == 1 else _colors(pair_rng, m, bg,
                                         excluded=reserved_test_color))
        if i == 0:
            anchor = colors[0]
        preferred = previous_shapes[int(rng.integers(0, len(previous_shapes)))] \
            if previous_shapes and rng.random() < 0.35 else None
        pair = _make_pair(pair_rng, m, parity, bg, colors, style_for("train"),
                          i == 0, preferred)
        train.append(pair)
        previous_shapes.append(pair["input"].shape)
        infos.append((m, parity))

    test = []
    available_novel = [c for c in range(10) if c not in set(int(v) for p in train
                                                             for v in np.unique(p["output"]))]
    for i in range(test_count):
        pair_rng = _child_rng(rng)
        if (test_count > 1 and i == 0) or (test_count == 1 and rng.random() < 0.4):
            m, parity = infos[int(rng.integers(0, len(infos)))]
        elif i == 1 and test_count > 1:
            missing = [(m, parity) for m in range(2, 10) for parity in (0, 1)
                       if (m, parity) not in infos]
            m, parity = missing[int(rng.integers(0, len(missing)))] if missing else infos[0]
        else:
            m, parity = int(rng.integers(2, 10)), int(rng.integers(0, 2))
        bg = background_for("test", i, reserved_test_color if i == 0 else None)
        novel = (reserved_test_color if i == 0 and reserved_test_color is not None else
                 int(pair_rng.choice(available_novel)) if available_novel and pair_rng.random() < 0.5 else None)
        colors = _colors(pair_rng, m, bg, novel=novel)
        preferred = previous_shapes[int(rng.integers(0, len(previous_shapes)))] \
            if rng.random() < 0.35 else None
        pair = _make_pair(pair_rng, m, parity, bg, colors, style_for("test"),
                          False, preferred)
        test.append(pair)
        previous_shapes.append(pair["input"].shape)

    train = [train[int(i)] for i in rng.permutation(len(train))]
    test = [test[int(i)] for i in rng.permutation(len(test))]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_digest(grid):
    h, w = grid.shape
    return hashlib.sha256(f"{h}x{w}:".encode() + np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def _validate_grid(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
    _require(all(1 <= size <= 30 for size in grid.shape), "grid size outside ARC limits")
    _require(np.issubdtype(grid.dtype, np.integer), "grid needs integer dtype")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    values, counts = np.unique(grid, return_counts=True)
    majority = int(np.argmax(counts))
    bg = int(values[majority])
    _require(int(counts[majority]) > grid.size // 2, "background must be a clear majority")
    colors = [int(c) for c in values if c != bg]
    _require(2 <= len(colors) <= 9, "segment count outside supported range")
    h, w = grid.shape
    for r, c in np.argwhere(grid != bg):
        r, c = int(r), int(c)
        neighborhood = grid[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)]
        _require(bool(np.all((neighborhood == bg) | (neighborhood == grid[r, c]))),
                 "different colors touch")
    segments = []
    for color in colors:
        positions = np.argwhere(grid == color)
        rows, cols = positions[:, 0], positions[:, 1]
        length = len(positions)
        if length == 1:
            orientation = "S"
        elif np.all(rows == rows[0]):
            _require(len(np.unique(cols)) == int(cols.max() - cols.min() + 1), "broken horizontal line")
            orientation = "H"
        elif np.all(cols == cols[0]):
            _require(len(np.unique(rows)) == int(rows.max() - rows.min() + 1), "broken vertical line")
            orientation = "V"
        else:
            raise AssertionError("colored object is not a straight line")
        segments.append((length, color, orientation))
    segments.sort(reverse=True)
    lengths = tuple(x[0] for x in segments)
    _require(len(set(lengths)) == len(lengths), "length order is tied")
    side = lengths[0]
    _require(side in (2 * len(segments) - 1, 2 * len(segments)), "wrong depth count")
    rank_colors = tuple(x[1] for x in segments)
    _require(output.shape == (side, side), "output size must equal longest length")
    _require(bool(np.array_equal(output, _output(side, rank_colors))), "incorrect ranked output")
    _require(bg not in np.unique(output), "background appears in output")
    _require(_source_digest(grid) not in _SOURCE_INPUT_DIGESTS, "exact source input copied")
    return {"m": len(segments), "side": side, "parity": side % 2, "bg": bg,
            "colors": rank_colors, "lengths": lengths,
            "orientations": tuple(x[2] for x in segments),
            "signature": (len(segments), side % 2),
            "witness": lengths[-1] + 2 * (len(segments) - 1) != side}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "invalid counts")
    seen = set()
    info = {"train": [], "test": []}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            _validate_grid(pair["input"])
            _validate_grid(pair["output"])
            key = _grid_key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info[split].append(_inspect_pair(pair))
    training = info["train"]
    _require({i["parity"] for i in training} == {0, 1}, "training must show both center parities")
    _require(any(i["witness"] for i in training), "training does not establish longest-side sizing")
    reassignment = any(
        color in b["colors"] and a["colors"].index(color) != b["colors"].index(color)
        for i, a in enumerate(training) for b in training[i + 1:] for color in a["colors"]
    )
    _require(reassignment, "training does not disprove fixed color roles")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    def visible_prediction(grid, side_rule, color_rule):
        values, counts = np.unique(grid, return_counts=True)
        background = int(values[np.argmax(counts)])
        lines = sorted(((int(np.count_nonzero(grid == color)), int(color))
                        for color in values if int(color) != background), reverse=True)
        side = lines[0][0] if side_rule == "longest" else lines[-1][0] + 2 * (len(lines) - 1)
        rank = [color for _, color in (lines if color_rule == "descending" else lines[::-1])]
        answer = np.empty((side, side), dtype=np.int8)
        for row in range(side):
            for col in range(side):
                depth = min(row, col, side - 1 - row, side - 1 - col)
                answer[row, col] = rank[min(depth, len(rank) - 1)]
        return answer

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    signatures, shapes, lengths, palettes, orientations, backgrounds = set(), set(), set(), set(), set(), set()
    within_count, within_shape, within_palette = 0, 0, 0
    coverage = {name: 0 for name in (
        "canonical", "irregular", "compact", "elongated", "spacious",
        "zero_foreground", "nonzero_background", "novel_test_color",
        "background_varies", "shape_repeats", "orientation_varies",
        "uniform_h", "uniform_v", "h_to_v", "v_to_h",
        "test_overlap", "test_heldout", "original_shape_reused",
    )}
    all_keys = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        models = [(s, o) for s in ("longest", "shortest")
                  for o in ("descending", "ascending")]
        matching = [model for model in models if all(
            np.array_equal(visible_prediction(p["input"], *model), p["output"])
            for p in puzzle["train"])]
        _require(matching == [("longest", "descending")],
                 "training does not identify the intended rule")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        all_keys.add(_puzzle_key(puzzle))
        infos = [_inspect_pair(p) for split in ("train", "test") for p in puzzle[split]]
        train_infos = infos[:len(puzzle["train"])]
        test_infos = infos[len(puzzle["train"]):]
        train_signatures = {i["signature"] for i in train_infos}
        train_colors = set(c for i in train_infos for c in i["colors"])
        coverage["test_overlap"] += any(i["signature"] in train_signatures for i in test_infos)
        coverage["test_heldout"] += any(i["signature"] not in train_signatures for i in test_infos)
        coverage["novel_test_color"] += any(set(i["colors"]) - train_colors for i in test_infos)
        coverage["background_varies"] += len({i["bg"] for i in infos}) > 1
        puzzle_shapes = [p["input"].shape for split in ("train", "test") for p in puzzle[split]]
        coverage["shape_repeats"] += len(set(puzzle_shapes)) < len(puzzle_shapes)
        orientation_kinds = [tuple(sorted(set(i["orientations"]) - {"S"})) for i in infos]
        coverage["orientation_varies"] += len(set(orientation_kinds)) > 1
        coverage["uniform_h"] += all(o == ("H",) for o in orientation_kinds)
        coverage["uniform_v"] += all(o == ("V",) for o in orientation_kinds)
        coverage["h_to_v"] += (all(o == ("H",) for o in orientation_kinds[:len(train_infos)])
                                 and all(o == ("V",) for o in orientation_kinds[len(train_infos):]))
        coverage["v_to_h"] += (all(o == ("V",) for o in orientation_kinds[:len(train_infos)])
                                 and all(o == ("H",) for o in orientation_kinds[len(train_infos):]))
        within_count += len({i["m"] for i in infos}) > 1
        within_shape += len({p["input"].shape for split in ("train", "test") for p in puzzle[split]}) > 1
        within_palette += len({frozenset(i["colors"]) for i in infos}) > 1
        for split in ("train", "test"):
            for p in puzzle[split]:
                item = _inspect_pair(p)
                _require(np.array_equal(visible_prediction(p["input"], *matching[0]),
                                        p["output"]), "visible prediction mismatch")
                coverage["canonical"] += item["lengths"] == tuple(range(item["side"], 0, -2))
                coverage["irregular"] += item["lengths"] != tuple(range(item["side"], 0, -2))
                coverage["compact"] += min(p["input"].shape) < item["side"]
                coverage["elongated"] += max(p["input"].shape) >= 2 * min(p["input"].shape)
                coverage["spacious"] += min(p["input"].shape) >= 22
                coverage["zero_foreground"] += item["bg"] != 0 and 0 in item["colors"]
                coverage["nonzero_background"] += item["bg"] != 0
                coverage["original_shape_reused"] += p["input"].shape in (
                    (21, 22), (23, 29), (24, 20), (21, 23))
                signatures.add(item["signature"])
                shapes.add(p["input"].shape)
                lengths.add(item["lengths"])
                palettes.add(frozenset(item["colors"]))
                orientations.add(tuple(sorted(set(item["orientations"]) - {"S"})))
                backgrounds.add(item["bg"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity too low")
    _require(len(all_keys) >= 40, "cross-seed puzzle diversity too low")
    _require(len(signatures) == 16, "count/parity coverage incomplete")
    _require(len(shapes) >= 65 and len(lengths) >= 60, "geometry diversity too low")
    _require(len(palettes) >= 40 and len(backgrounds) >= 7, "color diversity too low")
    _require(len(orientations) == 3, "orientation regimes incomplete")
    _require(within_count >= 38 and within_shape >= 38 and within_palette >= 38,
             "within-puzzle diversity too low")
    minimums = {"canonical": 20, "irregular": 50, "compact": 10,
                "elongated": 10, "spacious": 20, "zero_foreground": 10,
                "nonzero_background": 20, "novel_test_color": 8,
                "background_varies": 5, "shape_repeats": 5,
                "orientation_varies": 10, "uniform_h": 2, "uniform_v": 2,
                "h_to_v": 1, "v_to_h": 1, "test_overlap": 15,
                "test_heldout": 15, "original_shape_reused": 1}
    for name, minimum in minimums.items():
        _require(coverage[name] >= minimum, f"insufficient {name} coverage")
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1234 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(p)
        if train is not None:
            _require(len(p["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(p["test"]) == test, "explicit test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(10, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for args in ((4567,), (4567, 7, 3), (4567, None, 4)):
        _require(_puzzle_key(generate_puzzle(*args)) == _puzzle_key(generate_puzzle(*args)),
                 "nondeterministic generation")
