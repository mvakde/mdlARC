"""Generate aligned horizontal copies of keyed border sequences."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    ("000000000000000", "002000000000000", "000000000008000",
     "000000000000000", "000000000000000", "000010000000000",
     "000000000000000", "000000000000000", "006000004000000",
     "000000000000000", "000000000000000", "000000000000000",
     "124300000005786"),
    ("00000000000000", "00000000000000", "00000002000000",
     "00000000000000", "00000000000000", "00100000000000",
     "00000000000000", "00000000000400", "00000000000000",
     "00000000000000", "00000000000000", "23516400000000"),
    ("0000000004378", "0000000000000", "0010000000000",
     "0000000000000", "0000008000000", "0000000000000",
     "0000000000020", "0400000000000", "0000000000000",
     "0000000000000", "1620000000000"),
    ("953400000000000000", "000000000008000000",
     "000000000000000000", "000900000000000000",
     "000000000000000000", "000000000000060000",
     "000000000000000000", "000000000000000000",
     "000000000000000000", "000000010000000000",
     "000000000000000000", "000000000000020000",
     "000000000000000000", "000070000000000000",
     "000000000000000000", "000000000000000000",
     "213800000000004567"),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _counts(seed, rng, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sequences(rng, palette, count, core):
    """Allocate private queried colors and independently varied shared colors."""
    if count == 1:
        length = int(rng.integers(3 if core else 2, 10))
        return [[int(x) for x in rng.choice(palette, size=length, replace=False)]]
    mode = str(rng.choice(("disjoint", "pairwise", "threeway")
                          if count >= 3 else ("disjoint", "pairwise")))
    for _ in range(100):
        colors = [int(x) for x in rng.permutation(palette)]
        low = 2 * count if mode == "disjoint" else count + 2
        used = int(rng.integers(low, 10))
        members = [{i} for i in range(count)]
        if mode == "pairwise":
            pair = rng.choice(count, size=2, replace=False)
            members.append({int(pair[0]), int(pair[1])})
        elif mode == "threeway":
            members.append(set(range(count)))
        while len(members) < used:
            if mode == "disjoint" or rng.random() < 0.55:
                members.append({int(rng.integers(count))})
            elif mode == "threeway" and rng.random() < 0.13:
                members.append(set(range(count)))
            else:
                pair = rng.choice(count, size=2, replace=False)
                members.append({int(pair[0]), int(pair[1])})
        sequences = []
        for ref in range(count):
            values = [colors[i] for i, group in enumerate(members) if ref in group]
            sequences.append([int(x) for x in rng.permutation(values)])
        if all(2 <= len(seq) <= 8 for seq in sequences):
            return sequences
    raise ValueError("Could not allocate reference colors")


def _borders(rng, count, multi_demonstration):
    if count == 1:
        return [str(rng.choice(("top", "bottom")))]
    if multi_demonstration:
        shared = str(rng.choice(("top", "bottom")))
        borders = [shared, shared]
        borders.extend(str(rng.choice(("top", "bottom")))
                       for _ in range(count - 2))
        return [borders[int(i)] for i in rng.permutation(count)]
    return [str(rng.choice(("top", "bottom"))) for _ in range(count)]


def _place_references(grid, sequences, borders, background, rng):
    height, width = grid.shape
    for border in ("top", "bottom"):
        ids = [i for i, value in enumerate(borders) if value == border]
        if not ids:
            continue
        ids = [ids[int(i)] for i in rng.permutation(len(ids))]
        required = sum(len(sequences[i]) for i in ids) + len(ids) - 1
        free = width - required
        if free < 0:
            raise ValueError("References do not fit")
        # First and last slots are margins; interior slots extend the gaps.
        slots = rng.multinomial(free, np.ones(len(ids) + 1) / (len(ids) + 1))
        column = int(slots[0])
        row = 0 if border == "top" else height - 1
        for ordinal, ref in enumerate(ids):
            sequence = sequences[ref]
            grid[row, column:column + len(sequence)] = sequence
            column += len(sequence)
            if ordinal + 1 < len(ids):
                column += 1 + int(slots[ordinal + 1])


def _private_positions(sequences):
    memberships = {}
    for ref, sequence in enumerate(sequences):
        for color in sequence:
            memberships.setdefault(color, []).append(ref)
    return [[i for i, color in enumerate(sequence)
             if len(memberships[color]) == 1]
            for sequence in sequences]


def _marker_specs(rng, sequences, role, regime):
    private = _private_positions(sequences)
    count = len(sequences)
    unique_options = [(ref, index) for ref, positions in enumerate(private)
                      for index in positions]
    if role == "core":
        specs = [(0, 1), (0, len(sequences[0]) - 2), (0, 0)]
        marker_count = int(rng.integers(3, 7))
    elif role == "multi":
        marker_count = int(rng.integers(4, 8))
        specs = [(0, int(rng.choice(private[0]))),
                 (1, int(rng.choice(private[1])))]
        specs.append(specs[0])
    else:
        if rng.random() < 0.38:
            marker_count = int(rng.integers(2, min(8, len(unique_options)) + 1))
            chosen = rng.choice(len(unique_options), size=marker_count,
                                replace=False)
            specs = [unique_options[int(i)] for i in chosen]
        else:
            marker_count = (int(rng.integers(3, 8)) if regime == "train"
                            else int(rng.integers(2, 9)))
            specs = []
    weights = rng.dirichlet(np.full(count, 0.7))
    while len(specs) < marker_count:
        ref = int(rng.choice(count, p=weights))
        specs.append((ref, int(rng.choice(private[ref]))))
    if role == "core":
        kinds = ["left", "right", "full"] + ["full"] * (len(specs) - 3)
    else:
        profile = str(rng.choice(("complete", "left", "right", "both", "multiple")))
        kinds = ["full"] * len(specs)
        requests = {"complete": [], "left": ["left"], "right": ["right"],
                    "both": ["left", "right"],
                    "multiple": ["left", "right", str(rng.choice(("left", "right")))]}[profile]
        for kind in requests:
            eligible = [j for j, (ref, index) in enumerate(specs)
                        if kinds[j] == "full" and
                        (index > 0 if kind == "left" else index < len(sequences[ref]) - 1)]
            if eligible:
                kinds[int(rng.choice(eligible))] = kind
    return [{"ref": ref, "index": index, "kind": kind}
            for (ref, index), kind in zip(specs, kinds)]


def _columns(width, length, index, kind):
    result = []
    for column in range(width):
        start = column - index
        visible = min(width, start + length) - max(0, start)
        if visible < 2:
            continue
        actual = "left" if start < 0 else "right" if start + length > width else "full"
        if actual == kind:
            result.append(column)
    return result


def _place_markers(grid, sequences, specs, background, rng):
    height, width = grid.shape
    pattern = str(rng.choice(("free", "shared", "adjacent", "mixed")))
    for _ in range(120):
        placed = []
        valid = True
        for ordinal, spec in enumerate(specs):
            ref, index, kind = spec["ref"], spec["index"], spec["kind"]
            sequence = sequences[ref]
            columns = _columns(width, len(sequence), index, kind)
            if not columns:
                valid = False
                break
            if ordinal == 1 and pattern in ("shared", "mixed"):
                rows = [placed[0][0]]
            elif ordinal == 1 and pattern == "adjacent":
                rows = [row for row in (placed[0][0] - 1, placed[0][0] + 1)
                        if 1 <= row < height - 1]
            elif ordinal == 2 and pattern == "mixed":
                rows = [row for row in (placed[0][0] - 1, placed[0][0] + 1)
                        if 1 <= row < height - 1]
            else:
                rows = list(range(1, height - 1))
            candidates = [(row, column) for row in rows for column in columns]
            candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
            found = False
            for row, column in candidates:
                r0, r1 = max(0, row - 1), min(height, row + 2)
                c0, c1 = max(0, column - 1), min(width, column + 2)
                if np.any(grid[r0:r1, c0:c1] != background):
                    continue
                if any(abs(row - old_row) <= 1 and abs(column - old_col) <= 1
                       for old_row, old_col, _, _ in placed):
                    continue
                start = column - index
                end = start + len(sequence) - 1
                if any(row == old_row and not (end + 1 < old_start or
                                               old_end + 1 < start)
                       for old_row, _, old_start, old_end in placed):
                    continue
                placed.append((row, column, start, end))
                found = True
                break
            if not found:
                valid = False
                break
        if valid:
            for spec, (row, column, _, _) in zip(specs, placed):
                spec["row"], spec["column"] = row, column
                grid[row, column] = sequences[spec["ref"]][spec["index"]]
            return
    raise ValueError("Could not place isolated, nonconflicting markers")


def _size(rng, regime, parameters):
    scale_mode = parameters["scale_mode"]
    if scale_mode == "source_like":
        hr, wr = ((10, 18), (11, 20)) if regime == "train" else ((17, 26), (18, 29))
    elif scale_mode == "reverse":
        hr, wr = ((17, 26), (18, 29)) if regime == "train" else ((10, 19), (11, 21))
    else:
        center_h, center_w = parameters["size_center"]
        hr = (max(9, center_h - 3), min(27, center_h + 4))
        wr = (max(10, center_w - 3), min(30, center_w + 4))
    return int(rng.integers(*hr)), int(rng.integers(*wr))


def _generate_pair(rng, regime, role, parameters, ordinal):
    background = parameters["background"]
    if not parameters["constant_background"] and ordinal > 0:
        background = (background + int(rng.integers(1, 10))) % 10
    palette = np.asarray([color for color in range(10) if color != background])
    if role == "core":
        count = 1
    elif role == "multi":
        count = int(rng.integers(2, 4))
    else:
        count = int(rng.choice((1, 2, 3, 4), p=(.24, .30, .32, .14)))
    sequences = _sequences(rng, palette, count, role == "core")
    borders = _borders(rng, count, role == "multi")
    height, width = _size(rng, regime, parameters)
    required = max(sum(len(sequences[i]) for i, border in enumerate(borders)
                       if border == side) + max(0, borders.count(side) - 1)
                   for side in ("top", "bottom"))
    width = max(width, required + int(rng.integers(0, 4)))
    if width > 30:
        raise ValueError("Reference layout exceeds ARC bounds")
    grid = np.full((height, width), background, dtype=np.int8)
    _place_references(grid, sequences, borders, background, rng)
    specs = _marker_specs(rng, sequences, role, regime)
    _place_markers(grid, sequences, specs, background, rng)
    output = grid.copy()
    for spec in specs:
        sequence = sequences[spec["ref"]]
        start = spec["column"] - spec["index"]
        for offset, color in enumerate(sequence):
            column = start + offset
            if 0 <= column < width:
                output[spec["row"], column] = color
    return grid, output


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, "Output shape differs")
    height, width = input_grid.shape
    counts = np.bincount(input_grid.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    _require(int(counts[background]) > input_grid.size // 2,
             "Background is not dominant")
    refs = []
    for row in (0, height - 1):
        column = 0
        while column < width:
            if int(input_grid[row, column]) == background:
                column += 1
                continue
            start = column
            while column < width and int(input_grid[row, column]) != background:
                column += 1
            sequence = tuple(int(x) for x in input_grid[row, start:column])
            _require(2 <= len(sequence) <= 9 and len(set(sequence)) == len(sequence),
                     "A border run must contain 2-9 distinct colors")
            refs.append((row, start, sequence))
    _require(1 <= len(refs) <= 4, "Illegal reference count")
    coords = [(int(r) + 1, int(c)) for r, c in
              np.argwhere(input_grid[1:-1] != background)]
    _require(2 <= len(coords) <= 8, "Illegal marker count")
    expected = input_grid.copy()
    written = {}
    markers = []
    per_ref = [0] * len(refs)
    for row, column in coords:
        neighborhood = input_grid[max(0, row-1):min(height, row+2),
                                  max(0, column-1):min(width, column+2)]
        _require(np.count_nonzero(neighborhood != background) == 1,
                 "Input markers must be isolated")
        color = int(input_grid[row, column])
        matches = [(ref, seq.index(color)) for ref, (_, _, seq) in enumerate(refs)
                   if color in seq]
        _require(len(matches) == 1, "Marker must have one reference and offset")
        ref, index = matches[0]
        sequence = refs[ref][2]
        start = column - index
        per_ref[ref] += 1
        for offset, value in enumerate(sequence):
            copied_column = start + offset
            if 0 <= copied_column < width:
                location = (row, copied_column)
                _require(location == (row, column) or
                         int(input_grid[location]) == background,
                         "Expansion overwrites an input object")
                _require(location not in written or written[location] == value,
                         "Expansions conflict")
                written[location] = value
                expected[location] = value
        markers.append({"row": row, "column": column, "color": color,
                        "ref": ref, "index": index,
                        "left": start < 0,
                        "right": start + len(sequence) > width})
    _require(np.array_equal(expected, output_grid), "Incorrect aligned copy")
    _require(not np.array_equal(input_grid, output_grid), "Degenerate pair")
    return {"background": background, "height": height, "width": width,
            "refs": refs, "markers": markers, "per_ref": tuple(per_ref)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle needs train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "Illegal pair counts")
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pair needs input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "Grid must be a two-dimensional NumPy array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape)
                         and np.issubdtype(grid.dtype, np.integer),
                         "Illegal shape or dtype")
                _require(np.all((0 <= grid) & (grid <= 9)), "Illegal ARC color")
            key = _key(pair["input"])
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            _require(_rows(pair["input"]) not in _SOURCE_INPUT_ROWS,
                     "Source input copied")
            infos[split].append(_analyze_pair(pair["input"], pair["output"]))
    train = infos["train"]
    markers = [marker for info in train for marker in info["markers"]]
    _require(any(len(info["refs"]) == 1 for info in train),
             "Training needs a single-reference demonstration")
    _require(any(len(info["refs"]) >= 2 and
                 any(sum(ref[0] == row for ref in info["refs"]) >= 2
                     for row in (0, info["height"] - 1)) for info in train),
             "Training needs separated runs on a shared border")
    _require(any(marker["left"] for marker in markers) and
             any(marker["right"] for marker in markers) and
             any(not marker["left"] and not marker["right"] for marker in markers),
             "Training must demonstrate clipping and complete copies")
    _require(any(marker["index"] == 0 for marker in markers) and
             any(marker["index"] > 0 for marker in markers),
             "Training must demonstrate marker offsets")
    _require(any(len({m["color"] for m in info["markers"]}) < len(info["markers"])
                 for info in train), "Training must demonstrate repeated marker colors")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, rng, num_train, num_test)
    parameters = {"background": int(rng.integers(10)),
                  "constant_background": bool(rng.integers(2)),
                  "scale_mode": str(rng.choice(("source_like", "similar", "reverse"))),
                  "size_center": (int(rng.integers(14, 22)),
                                  int(rng.integers(15, 26)))}
    train_roles = ["core", "multi"] + ["free"] * (ntrain - 2)
    train_roles = [train_roles[int(i)] for i in rng.permutation(ntrain)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, roles in (("train", train_roles), ("test", ["free"] * ntest)):
        for role in roles:
            for _ in range(250):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    input_grid, output_grid = _generate_pair(
                        pair_rng, split, role, parameters,
                        len(puzzle["train"]) + len(puzzle["test"]))
                    if _key(input_grid) in seen or _rows(input_grid) in _SOURCE_INPUT_ROWS:
                        continue
                    _analyze_pair(input_grid, output_grid)
                except (ValueError, AssertionError):
                    continue
                pair = {"input": input_grid, "output": output_grid}
                puzzle[split].append(pair)
                seen.add(_key(input_grid))
                break
            else:
                raise ValueError("Could not build a valid unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    auto = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in auto:
        validate_generated_puzzle(puzzle)
        # Reconstruct each answer afresh from its visible input, without using
        # the placements or reference identities held during construction.
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                height, width = grid.shape
                background = int(np.argmax(np.bincount(grid.ravel(), minlength=10)))
                runs = []
                for row in (0, height - 1):
                    start = 0
                    while start < width:
                        if int(grid[row, start]) == background:
                            start += 1
                            continue
                        end = start + 1
                        while end < width and int(grid[row, end]) != background:
                            end += 1
                        runs.append(tuple(int(color) for color in grid[row, start:end]))
                        start = end
                predicted = grid.copy()
                for local_row, column in np.argwhere(grid[1:-1] != background):
                    row, column = int(local_row) + 1, int(column)
                    color = int(grid[row, column])
                    matches = [(sequence, sequence.index(color)) for sequence in runs
                               if color in sequence]
                    _require(len(matches) == 1, "Ambiguous prediction key")
                    sequence, index = matches[0]
                    for position, value in enumerate(sequence):
                        target = column - index + position
                        if 0 <= target < width:
                            predicted[row, target] = value
                _require(np.array_equal(predicted, pair["output"]),
                         "Independent reconstruction failed")
    _require(len({len(p["train"]) for p in auto}) >= 4 and
             len({len(p["test"]) for p in auto}) >= 3,
             "Automatic counts lack diversity")
    _require(len({tuple(tuple(_key(pair["input"]) for pair in p[split])
                        for split in ("train", "test")) for p in auto}) >= 40,
             "Cross-seed puzzles repeat")
    infos = [_analyze_pair(pair["input"], pair["output"])
             for p in auto for split in ("train", "test") for pair in p[split]]
    _require(len({(info["height"], info["width"]) for info in infos}) >= 30,
             "Shapes lack diversity")
    _require({len(info["refs"]) for info in infos} >= {1, 2, 3},
             "Reference counts lack diversity")
    _require({len(info["markers"]) for info in infos} >= {2, 3, 4, 5, 6},
             "Marker counts lack diversity")
    _require(any(len({m["row"] for m in info["markers"]}) < len(info["markers"])
                 for info in infos), "Same-row markers missing")
    _require(any(any(abs(a["row"] - b["row"]) == 1 for a in info["markers"]
                     for b in info["markers"] if a is not b) for info in infos),
             "Adjacent-row markers missing")
    _require(any(all(not m["left"] and not m["right"] for m in info["markers"])
                 for info in infos), "All-complete pairs missing")
    _require(any(sum(m["left"] or m["right"] for m in info["markers"]) >= 2
                 for info in infos), "Multiple-clipped pairs missing")
    _require(any(0 in info["per_ref"] for info in infos if len(info["refs"]) > 1),
             "Unqueried references missing")
    _require(len({info["background"] for info in infos}) >= 8,
             "Background color diversity missing")
    clip_profiles = set()
    overlap_profiles = set()
    same_row_different_refs = False
    repeated_profiles = set()
    top_only = top_multi = bottom_multi = two_cell = long_run = False
    allocations = set()
    for info in infos:
        references = info["refs"]
        markers = info["markers"]
        left = any(marker["left"] for marker in markers)
        right = any(marker["right"] for marker in markers)
        clip_profiles.add((left, right))
        reference_sets = [set(ref[2]) for ref in references]
        overlap = "disjoint"
        if any(first & second for i, first in enumerate(reference_sets)
               for second in reference_sets[i+1:]):
            overlap = "pairwise"
        if any(reference_sets[i] & reference_sets[j] & reference_sets[k]
               for i in range(len(references)) for j in range(i+1, len(references))
               for k in range(j+1, len(references))):
            overlap = "threeway"
        overlap_profiles.add(overlap)
        same_row_different_refs |= any(
            first["row"] == second["row"] and first["ref"] != second["ref"]
            for i, first in enumerate(markers) for second in markers[i+1:])
        repeated_profiles.add(len({m["color"] for m in markers}) < len(markers))
        top_count = sum(ref[0] == 0 for ref in references)
        bottom_count = len(references) - top_count
        top_only |= bottom_count == 0
        top_multi |= top_count >= 2
        bottom_multi |= bottom_count >= 2
        two_cell |= any(len(ref[2]) == 2 for ref in references)
        long_run |= any(len(ref[2]) >= 7 for ref in references)
        allocations.add(tuple(sorted(info["per_ref"])))
    _require(len(clip_profiles) == 4 and len(overlap_profiles) == 3,
             "Clipping or overlap profiles lack diversity")
    _require(same_row_different_refs and repeated_profiles == {False, True},
             "Marker arrangements or queried colors lack diversity")
    _require(top_only and top_multi and bottom_multi and two_cell and long_run,
             "Border placement or run lengths lack diversity")
    _require(any(0 in allocation for allocation in allocations) and
             (1, 2, 3) in allocations,
             "Reference query allocations lack diversity")
    background_counts = set()
    scale_relations = set()
    for puzzle in auto:
        train_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        background_counts.add(len({x["background"] for x in train_info + test_info}))
        train_area = np.mean([x["height"] * x["width"] for x in train_info])
        test_area = np.mean([x["height"] * x["width"] for x in test_info])
        scale_relations.add("test_larger" if test_area > train_area * 1.1 else
                            "train_larger" if train_area > test_area * 1.1 else
                            "similar")
    _require(1 in background_counts and any(value > 1 for value in background_counts),
             "Background must be constant in some puzzles and varied in others")
    _require(scale_relations == {"test_larger", "train_larger", "similar"},
             "Train/test scale relationships lack diversity")
    explicit = generate_puzzle(137, 7, 3)
    _require(_key(explicit["train"][0]["input"]) ==
             _key(generate_puzzle(137, 7, 3)["train"][0]["input"]),
             "Nondeterministic explicit generation")
    for offset, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(700 + offset, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b),
                 "Boundary count request ignored")
    for offset, (a, b) in enumerate(((None, 1), (None, 4),
                                     (2, None), (10, None))):
        p = generate_puzzle(800 + offset, a, b)
        _require((a is None or len(p["train"]) == a) and
                 (b is None or len(p["test"]) == b),
                 "Partial count request ignored")
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(900, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")
