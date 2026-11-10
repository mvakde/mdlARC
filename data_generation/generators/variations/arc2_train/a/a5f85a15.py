"""Generate diagonal-run recoloring puzzles in the source ARC family."""

import numpy as np


# Exact source pairs are excluded; their dimensions and silhouettes are valid.
_SOURCE = (
    (("00900000", "00090000", "00009000", "00000900",
      "90000090", "09000009", "00900000", "00090000"),
     ("00900000", "00040000", "00009000", "00000400",
      "90000090", "04000004", "00900000", "00040000")),
    (("200", "020", "002"), ("200", "040", "002")),
    (("003000", "000300", "000030", "300003", "030000", "003000"),
     ("003000", "000400", "000030", "300004", "040000", "003000")),
    (("000060000000", "600006000000", "060000600000",
      "006000060000", "000600006000", "000060000600",
      "000006000060", "000000600006", "600000060000",
      "060000006000", "006000000600", "000600000060"),
     ("000060000000", "600004000000", "040000600000",
      "006000040000", "000400006000", "000060000400",
      "000004000060", "000000600004", "600000040000",
      "040000006000", "006000000400", "000400000060")),
)


def _signature(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _pair_signature(pair):
    return _signature(pair["input"]), _signature(pair["output"])


def _raw_signature(rows):
    return (len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row)


_SOURCE_PAIRS = frozenset((_raw_signature(a), _raw_signature(b)) for a, b in _SOURCE)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        # The count schedule remains diverse even for arithmetic seed suites.
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _segment(row, col, length):
    return tuple((row + k, col + k) for k in range(length))


def _compatible(first, second):
    a = first[0][1] - first[0][0]
    b = second[0][1] - second[0][0]
    if a != b:
        # Offset one gives side-touching runs and uncertain membership.
        return abs(a - b) >= 2
    a0, a1 = first[0][0], first[-1][0]
    b0, b1 = second[0][0], second[-1][0]
    return a1 + 1 < b0 or b1 + 1 < a0


def _candidate_segments(height, width, length):
    return [_segment(r, c, length)
            for r in range(height - length + 1)
            for c in range(width - length + 1)]


def _phase_contrast(segments):
    return any(
        a[0][0] % 2 != b[0][0] % 2
        and a[0][1] % 2 != b[0][1] % 2
        for i, a in enumerate(segments) for b in segments[i + 1:]
    )


def _arrangement_ok(first, second, arrangement):
    if not _compatible(first, second):
        return False
    a = first[0][1] - first[0][0]
    b = second[0][1] - second[0][0]
    if arrangement == "same":
        return a == b
    if arrangement == "close":
        return abs(a - b) == 2
    if arrangement == "separate":
        return (
            first[-1][0] < second[0][0] and first[-1][1] < second[0][1]
        ) or (
            second[-1][0] < first[0][0] and second[-1][1] < first[0][1]
        )
    return True


def _sample_layout(rng, height, width, lengths, arrangement="free",
                   require_phase=False):
    if sum(lengths) >= height * width / 2:
        return None
    candidates = [_candidate_segments(height, width, length) for length in lengths]
    if any(not group for group in candidates):
        return None
    for _ in range(36):
        first = candidates[0][int(rng.integers(len(candidates[0])))]
        chosen = [first]
        possible = True
        for index in range(1, len(lengths)):
            order = rng.permutation(len(candidates[index]))
            next_segment = None
            for position in order:
                candidate = candidates[index][int(position)]
                if not all(_compatible(candidate, prior) for prior in chosen):
                    continue
                if index == 1 and not _arrangement_ok(first, candidate, arrangement):
                    continue
                if index == 1 and require_phase and not _phase_contrast((first, candidate)):
                    continue
                next_segment = candidate
                break
            if next_segment is None:
                possible = False
                break
            chosen.append(next_segment)
        if possible:
            return tuple(sorted(chosen, key=lambda segment: segment[0]))
    return None


def _construct(height, width, segments, background, colors, highlight):
    inp = np.full((height, width), background, dtype=np.int64)
    out = inp.copy()
    for segment, color in zip(segments, colors):
        for index, (row, col) in enumerate(segment):
            inp[row, col] = color
            out[row, col] = color if index % 2 == 0 else highlight
    return {"input": inp, "output": out}


def _dimensions(rng, regime, special):
    if special == "anchor":
        return int(rng.integers(9, 13)), int(rng.integers(9, 13))
    if special == "minimal":
        return 3, 3
    if special == "source_like":
        return int(rng.integers(12, 20)), int(rng.integers(12, 20))
    bucket = int(rng.choice(3, p=(0.27, 0.46, 0.27) if regime == "train"
                            else (0.22, 0.37, 0.41)))
    ranges = ((3, 7), (7, 15), (14, 25)) if regime == "train" else (
        (3, 9), (8, 17), (13, 27))
    low, high = ranges[bucket]
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _length(rng, limit):
    category = int(rng.integers(3))
    if category == 0:
        low, high = 2, min(limit, 5)
    elif category == 1:
        low, high = min(4, limit), min(limit, 9)
    else:
        low, high = min(7, limit), limit
    return int(rng.integers(low, high + 1))


def _design(rng, regime, special, same_shape=None):
    if same_shape is not None and special == "ordinary" and rng.random() < 0.24:
        height, width = same_shape
    else:
        height, width = _dimensions(rng, regime, special)
    limit = min(height, width)
    if special == "anchor":
        return height, width, (6, int(rng.integers(3, 6))), "free", True
    if special == "minimal":
        return 3, 3, (3,), "free", False
    if special == "single":
        return height, width, (_length(rng, limit),), "free", False
    if special == "source_like":
        return height, width, (
            int(rng.integers(8, min(12, limit - 2) + 1)),
            int(rng.integers(7, min(11, limit - 2) + 1)),
            int(rng.integers(2, 6)),
        ), "free", False

    count = int(rng.integers(1, min(4 if regime == "test" else 3,
                                   max(1, limit // 3)) + 1))
    arrangement = "free"
    if count >= 2:
        arrangement = str(rng.choice(("free", "same", "close", "separate"),
                                     p=(0.40, 0.20, 0.20, 0.20)))
    if arrangement in ("same", "separate"):
        cap = max(2, (limit - 1) // 2)
        lengths = [int(rng.integers(2, cap + 1)) for _ in range(2)]
        if count == 3:
            lengths.append(_length(rng, limit))
    else:
        lengths = [_length(rng, limit) for _ in range(count)]
    return height, width, tuple(lengths), arrangement, False


def _sample_pair(rng, regime, special, background, highlight, palette,
                 multi_color=False, same_shape=None, required_source=None):
    for _ in range(90):
        height, width, lengths, arrangement, require_phase = _design(
            rng, regime, special, same_shape)
        segments = _sample_layout(rng, height, width, lengths,
                                  arrangement, require_phase)
        if segments is None:
            continue
        available = [color for color in palette if color != background]
        if required_source is not None and required_source != background:
            first_color = required_source
        else:
            first_color = available[int(rng.integers(len(available)))]
        colors = [first_color]
        for _segment in segments[1:]:
            if multi_color:
                alternatives = [color for color in available if color != colors[0]]
                colors.append(alternatives[int(rng.integers(len(alternatives)))])
            else:
                colors.append(first_color if rng.random() < 0.72 else
                              available[int(rng.integers(len(available)))])
        return _construct(height, width, segments, background, colors, highlight)
    raise RuntimeError(f"could not place {special} diagonal layout")


def _validate_grid(grid, label):
    _need(isinstance(grid, np.ndarray), f"{label} must be an array")
    _need(grid.ndim == 2, f"{label} must be two-dimensional")
    _need(all(1 <= side <= 30 for side in grid.shape), f"{label} has illegal shape")
    _need(np.issubdtype(grid.dtype, np.integer), f"{label} has noninteger cells")
    _need(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} has illegal color")


def _extract_segments(mask):
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    _need(bool(cells), "input has no diagonal cells")
    starts = sorted((r, c) for r, c in cells if (r - 1, c - 1) not in cells)
    segments = []
    seen = set()
    for row, col in starts:
        segment = []
        while (row, col) in cells:
            segment.append((row, col))
            seen.add((row, col))
            row += 1
            col += 1
        segments.append(tuple(segment))
    _need(seen == cells, "not every foreground cell lies on a diagonal run")
    for i, first in enumerate(segments):
        for second in segments[i + 1:]:
            _need(_compatible(first, second), "ambiguous or touching runs")
    return tuple(segments)


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    _need(inp.shape == out.shape, "output shape differs from input shape")
    colors, counts = np.unique(inp, return_counts=True)
    background_index = int(np.argmax(counts))
    _need(int(counts[background_index]) > int(sum(counts) - counts[background_index]),
          "background is not dominant")
    background = int(colors[background_index])
    segments = _extract_segments(inp != background)
    changed = inp != out
    _need(bool(np.any(changed)), "pair does not demonstrate recoloring")
    highlights = np.unique(out[changed])
    _need(len(highlights) == 1, "pair has inconsistent highlight colors")
    highlight = int(highlights[0])
    _need(highlight not in colors, "highlight must be new in input")
    expected = inp.copy()
    segment_colors = []
    for segment in segments:
        own_colors = {int(inp[row, col]) for row, col in segment}
        _need(len(own_colors) == 1, "a diagonal changes source color along its run")
        color = next(iter(own_colors))
        segment_colors.append(color)
        for index, (row, col) in enumerate(segment):
            if index % 2:
                expected[row, col] = highlight
    _need(np.array_equal(out, expected), "independent endpoint alternation failed")
    return {"shape": inp.shape, "background": background, "highlight": highlight,
            "segments": segments, "sources": tuple(segment_colors)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
          "puzzle must contain exactly train and test")
    _need(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
          "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"},
                  f"{split}[{index}] has invalid fields")
            _validate_grid(pair["input"], f"{split}[{index}].input")
            _validate_grid(pair["output"], f"{split}[{index}].output")
            signature = _pair_signature(pair)
            _need(signature[0] not in seen_inputs, "duplicate input")
            _need(signature not in seen_pairs, "duplicate pair")
            _need(signature not in _SOURCE_PAIRS, "exact original source pair")
            seen_inputs.add(signature[0])
            seen_pairs.add(signature)
            analysis = _analyze(pair)
            _need(len(analysis["segments"]) <= (3 if split == "train" else 4),
                  "too many diagonal runs")
            analyses[split].append(analysis)
    all_pairs = analyses["train"] + analyses["test"]
    _need(len({a["highlight"] for a in all_pairs}) == 1,
          "highlight convention changes within puzzle")
    train = analyses["train"]
    _need(any(_phase_contrast(a["segments"]) for a in train),
          "training lacks independent phase contrast")
    _need(any(len(segment) % 2 == 0 for a in train for segment in a["segments"]),
          "training lacks even-length endpoint example")
    _need(any(len(segment) >= 5 for a in train for segment in a["segments"]),
          "training lacks extended alternation")
    _need(any(len(set(a["sources"])) >= 2 for a in train),
          "training lacks individually colored lines")
    _need(len({a["background"] for a in train}) >= 2,
          "training lacks background color variation")
    _need(len({source for a in train for source in a["sources"]}) >= 2,
          "training lacks source color variation")
    _need(any(len(a["segments"]) == 1 for a in train),
          "training lacks a single-line demonstration")
    _need(any(len(a["segments"]) >= 2 for a in train),
          "training lacks a multi-line demonstration")


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
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    order = [int(value) for value in rng.permutation(10)]
    highlight = order[0]
    palette = order[1:]
    backgrounds = [palette[0], palette[1]]
    puzzle = {"train": [], "test": []}
    used_inputs, used_layouts = set(), set()

    def add(split, special, background, multi=False, same_shape=None,
            required_source=None):
        for _ in range(100):
            pair = _sample_pair(rng, split, special, background, highlight,
                                palette, multi, same_shape, required_source)
            signature = _signature(pair["input"])
            mask = pair["input"] != background
            layout = (tuple(mask.shape), tuple(bool(x) for x in mask.flat))
            if signature in used_inputs or _pair_signature(pair) in _SOURCE_PAIRS:
                continue
            if layout in used_layouts:
                continue
            puzzle[split].append(pair)
            used_inputs.add(signature)
            used_layouts.add(layout)
            return
        raise RuntimeError("could not generate a unique pair")

    add("train", "anchor", backgrounds[0], multi=True)
    second_special = "minimal" if seed % 7 == 0 else "single"
    add("train", second_special, backgrounds[1])
    for index in range(2, train_count):
        background = palette[int(rng.integers(len(palette)))]
        repeat = seed % 9 == 1 and index == 2
        if repeat:
            original = _analyze(puzzle["train"][1])
            source = next(color for color in palette
                          if color not in (background, original["sources"][0]))
            pair = _construct(*original["shape"], original["segments"],
                              background, [source], highlight)
            if (_signature(pair["input"]) not in used_inputs
                    and _pair_signature(pair) not in _SOURCE_PAIRS):
                puzzle["train"].append(pair)
                used_inputs.add(_signature(pair["input"]))
                continue
        add("train", "ordinary", background, multi=rng.random() < 0.35)
    first_train_source = int(next(
        value for value in puzzle["train"][0]["input"].flat
        if value != backgrounds[0]))
    for index in range(test_count):
        special = "source_like" if index == 0 and seed % 5 == 0 else "ordinary"
        background = palette[int(rng.integers(len(palette)))]
        same_shape = puzzle["train"][int(rng.integers(train_count))]["input"].shape
        add("test", special, background, multi=rng.random() < 0.45,
            same_shape=same_shape,
            required_source=first_train_source if seed % 3 == 0 else None)
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x[key], y[key])
            for x, y in zip(a[split], b[split]) for key in ("input", "output"))
        for split in ("train", "test"))


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(99173, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def _predict_from_examples(train, test_input):
    """Reconstruct a test output using only demonstrations and the input."""
    introduced = {
        int(color)
        for pair in train
        for color in np.unique(pair["output"][
            pair["input"] != pair["output"]])
    }
    _need(len(introduced) == 1, "training does not identify one highlight")
    highlight = next(iter(introduced))
    colors, counts = np.unique(test_input, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    runs = _extract_segments(test_input != background)
    prediction = test_input.copy()
    for run in runs:
        for index, (row, col) in enumerate(run):
            if index % 2:
                prediction[row, col] = highlight
    return prediction


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            _need(np.array_equal(
                _predict_from_examples(puzzle["train"], pair["input"]),
                pair["output"]),
                "training-only prediction differs from generated answer")
    _need(_equal(generate_puzzle(17), puzzles[17]), "nondeterministic automatic result")
    _need(not _equal(puzzles[0], puzzles[1]), "distinct seeds generated one puzzle")
    for seed, train, test in ((1201, 2, 1), (1202, 10, 1), (1203, 10, 4),
                              (1301, None, 1), (1302, None, 4),
                              (1303, 2, None), (1304, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _need(len(puzzle["train"]) == train, "train count not honored")
        if test is not None:
            _need(len(puzzle["test"]) == test, "test count not honored")
    _need(_equal(generate_puzzle(731, 7, 3), generate_puzzle(731, 7, 3)),
          "nondeterministic explicit result")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        _expect_value_error(**kwargs)
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, layouts, color_roles, segment_counts = set(), set(), set(), set()
    rectangular = small = long_train = three_test = same_diag = close = 0
    three_by_three = same_size_test = source_like = repeated_layout = 0
    new_source = reused_source = mixed_line_colors = 0
    for puzzle in puzzles:
        training = [_analyze(pair) for pair in puzzle["train"]]
        train_shapes = {a["shape"] for a in training}
        train_sources = {color for a in training for color in a["sources"]}
        train_layouts = [
            (a["shape"], tuple(bool(x) for x in
             (pair["input"] != a["background"]).flat))
            for pair, a in zip(puzzle["train"], training)]
        repeated_layout += len(train_layouts) != len(set(train_layouts))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                shapes.add(a["shape"])
                color_roles.add((a["background"], a["highlight"]))
                mask = pair["input"] != a["background"]
                layouts.add((a["shape"], tuple(bool(x) for x in mask.flat)))
                segment_counts.add(len(a["segments"]))
                rectangular += a["shape"][0] != a["shape"][1]
                small += min(a["shape"]) <= 5
                long_train += split == "train" and any(
                    len(s) >= 8 for s in a["segments"])
                three_test += split == "test" and len(a["segments"]) >= 3
                offsets = [s[0][1] - s[0][0] for s in a["segments"]]
                same_diag += len(offsets) != len(set(offsets))
                close += any(abs(x - y) == 2 for i, x in enumerate(offsets)
                             for y in offsets[i + 1:])
                mixed_line_colors += len(set(a["sources"])) >= 2
                if split == "train":
                    three_by_three += a["shape"] == (3, 3)
                else:
                    same_size_test += a["shape"] in train_shapes
                    source_like += (
                        len(a["segments"]) >= 3
                        and sorted(len(s) for s in a["segments"])[-2] >= 7
                        and max(len(s) for s in a["segments"]) >= 8
                        and min(len(s) for s in a["segments"]) <= 5)
                    new_source += any(color not in train_sources
                                      for color in a["sources"])
                    reused_source += any(color in train_sources
                                         for color in a["sources"])
    _need(len(train_counts) >= 4 and len(test_counts) >= 3,
          "automatic counts lack diversity")
    _need(len(shapes) >= 35 and len(layouts) >= 100 and len(color_roles) >= 20,
          "structural or color diversity is too narrow")
    _need(rectangular >= 30 and small >= 8 and long_train >= 5,
          "size and aspect-ratio diversity is too narrow")
    _need(three_test >= 5 and same_diag >= 3 and close >= 3,
          "test composition or spacing diversity is too narrow")
    _need(three_by_three >= 3 and same_size_test >= 3 and source_like >= 5,
          "source-sized or three-run diversity is too narrow")
    _need(repeated_layout >= 2 and mixed_line_colors >= 30,
          "color independence is not adequately demonstrated")
    _need(new_source >= 5 and reused_source >= 5,
          "test source colors lack reuse or novelty")
