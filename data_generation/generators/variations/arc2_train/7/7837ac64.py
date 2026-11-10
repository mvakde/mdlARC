"""Lattice intersections encode the corners of a 3 by 3 mosaic."""

import numpy as np


_SOURCE_SIGNATURES = {
    ((29, 29), 0, 1, 3, 3, 9, 9, 2, 2, (3, 4), ((6, 6, 0), (0, 0, 0), (3, 3, 3))),
    ((27, 27), 0, 3, 4, 4, 6, 6, 3, 3, (1, 0), ((0, 2, 0), (2, 0, 0), (0, 0, 8))),
    ((27, 27), 0, 8, 4, 4, 6, 6, 3, 3, (1, 1), ((1, 0, 2), (0, 0, 2), (2, 2, 2))),
    ((29, 29), 0, 4, 3, 3, 9, 9, 2, 2, (1, 1), ((1, 0, 3), (1, 0, 0), (1, 0, 0))),
    ((29, 29), 0, 2, 5, 5, 5, 5, 4, 4, (0, 0), ((1, 0, 4), (0, 0, 0), (8, 8, 8))),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _canonical_output(output, background):
    mapping = {background: 0}
    result = []
    for value in output.flat:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        result.append(mapping[value])
    return tuple(result)


def _components(mask):
    remaining = {(r, c) for r in range(3) for c in range(3)
                 if mask & (1 << (3 * r + c))}
    result = []
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            r, c = frontier.pop()
            neighbors = {p for p in remaining
                         if max(abs(p[0] - r), abs(p[1] - c)) == 1}
            remaining -= neighbors
            component |= neighbors
            frontier.extend(neighbors)
        result.append(tuple(sorted(component)))
    return tuple(sorted(result))


def _make_catalogue():
    catalogue = {1: {}, 2: {}, 3: {}}
    for mask in range(1, 512):
        cells = {(r, c) for r in range(3) for c in range(3)
                 if mask & (1 << (3 * r + c))}
        if not (any(r == 0 for r, c in cells)
                and any(r == 2 for r, c in cells)
                and any(c == 0 for r, c in cells)
                and any(c == 2 for r, c in cells)):
            continue
        components = _components(mask)
        for count in (1, 2, 3):
            if len(components) >= count:
                catalogue[count].setdefault(len(cells), []).append(components)
    return catalogue


_CATALOGUE = _make_catalogue()


def _encode_vertices(output, background, structure):
    vertices = np.full((4, 4), structure, dtype=np.int8)
    for r in range(3):
        for c in range(3):
            color = int(output[r, c])
            if color == background:
                continue
            _require(color != structure, "payload is structural color")
            for vr in (r, r + 1):
                for vc in (c, c + 1):
                    old = int(vertices[vr, vc])
                    _require(old in (structure, color), "corner color conflict")
                    vertices[vr, vc] = color
    return vertices


def _decode_vertices(vertices, background, structure):
    output = np.full((3, 3), background, dtype=np.int8)
    for r in range(3):
        for c in range(3):
            square = vertices[r:r + 2, c:c + 2]
            color = int(square[0, 0])
            if color != structure and np.all(square == color):
                output[r, c] = color
    return output


def _has_shared_side(output, background):
    for r in range(3):
        for c in range(3):
            color = int(output[r, c])
            if color != background and ((r < 2 and output[r + 1, c] == color)
                                        or (c < 2 and output[r, c + 1] == color)):
                return True
    return False


def _sample_mosaic(rng, background, colors, illustrate_sharing=False):
    count = len(colors)
    sizes = sorted(_CATALOGUE[count])
    for _ in range(256):
        size = int(rng.choice(sizes))
        options = _CATALOGUE[count][size]
        components = options[int(rng.integers(len(options)))]
        order = list(rng.permutation(len(components)))
        labels = [0] * len(components)
        for i, index in enumerate(order):
            labels[index] = i if i < count else int(rng.integers(count))
        output = np.full((3, 3), background, dtype=np.int8)
        for component, label in zip(components, labels):
            for r, c in component:
                output[r, c] = colors[label]
        vertices = _encode_vertices(output, background, -1)
        if np.array_equal(_decode_vertices(vertices, background, -1), output) and (
            not illustrate_sharing or _has_shared_side(output, background)
        ):
            return output
    raise RuntimeError("could not find an exactly decodable mosaic")


def _pick_axis(rng, regime):
    if regime == "source_train":
        pitch = int(rng.choice((2, 3, 4, 5), p=(.12, .34, .38, .16)))
    else:
        pitch = int(rng.choice((2, 3, 4, 5, 6), p=(.10, .19, .25, .32, .14)))
    low = 6 if pitch == 2 else 5
    high = min(12, 1 + 29 // pitch)
    count = int(rng.integers(low, high + 1))
    return pitch, count, (count - 1) * pitch + 1


def _margins(rng, extra):
    if rng.random() < .36:
        first = extra // 2 + int(rng.integers(2)) * (extra % 2)
    else:
        first = int(rng.integers(extra + 1))
    return first, extra - first


def _sample_geometry(rng, regime):
    row_pitch, row_count, row_span = _pick_axis(rng, regime)
    col_pitch, col_count, col_span = _pick_axis(rng, regime)
    if rng.random() < .35:
        side = int(rng.integers(max(row_span, col_span), 31))
        height = width = side
    else:
        height = int(rng.integers(row_span, 31))
        width = int(rng.integers(col_span, 31))
    top, bottom = _margins(rng, height - row_span)
    left, right = _margins(rng, width - col_span)
    block_row = int(rng.integers(row_count - 3))
    block_col = int(rng.integers(col_count - 3))
    return (row_pitch, col_pitch, row_count, col_count, top, bottom,
            left, right, block_row, block_col)


def _choose_colors(rng, preferred_background, previous, count):
    background = preferred_background if rng.random() < .60 else int(rng.integers(10))
    candidates = [v for v in range(10) if v != background]
    reuse = [v for v in previous["payload"] if v != background] if previous else []
    structure = int(rng.choice(reuse if reuse and rng.random() < .35 else candidates))
    candidates.remove(structure)
    colors = []
    if previous and previous["structure"] in candidates and rng.random() < .35:
        colors.append(previous["structure"])
        candidates.remove(colors[-1])
    if len(colors) < count:
        colors.extend(int(v) for v in rng.choice(candidates, size=count-len(colors), replace=False))
    rng.shuffle(colors)
    return background, structure, tuple(colors)


def _construct_pair(rng, regime, preferred_background, previous, index):
    if regime == "source_train" and index == 0:
        color_count = 2
    else:
        probs = (.22, .53, .25) if regime == "source_train" else (.16, .43, .41)
        color_count = int(rng.choice((1, 2, 3), p=probs))
    background, structure, colors = _choose_colors(rng, preferred_background,
                                                    previous, color_count)
    output = _sample_mosaic(rng, background, colors,
                            illustrate_sharing=(regime == "source_train" and index == 0))
    (row_pitch, col_pitch, row_count, col_count, top, bottom,
     left, right, block_row, block_col) = _sample_geometry(rng, regime)
    height = top + (row_count - 1) * row_pitch + 1 + bottom
    width = left + (col_count - 1) * col_pitch + 1 + right
    rows = top + np.arange(row_count) * row_pitch
    cols = left + np.arange(col_count) * col_pitch
    grid = np.full((height, width), background, dtype=np.int8)
    grid[rows, :] = structure
    grid[:, cols] = structure
    vertices = _encode_vertices(output, background, structure)
    grid[np.ix_(rows[block_row:block_row + 4],
                cols[block_col:block_col + 4])] = vertices
    return {"input": grid, "output": output}


def _analyze_candidate(grid, output, structure):
    rows = np.array([r for r in range(grid.shape[0])
                     if np.count_nonzero(grid[r] != structure) <= 4], dtype=int)
    cols = np.array([c for c in range(grid.shape[1])
                     if np.count_nonzero(grid[:, c] != structure) <= 4], dtype=int)
    _require(len(rows) >= 5 and len(cols) >= 5, "too few lattice lines")
    row_steps, col_steps = np.diff(rows), np.diff(cols)
    _require(np.all(row_steps == row_steps[0]) and 2 <= row_steps[0] <= 6,
             "irregular row lattice")
    _require(np.all(col_steps == col_steps[0]) and 2 <= col_steps[0] <= 6,
             "irregular column lattice")
    row_pitch, col_pitch = int(row_steps[0]), int(col_steps[0])
    _require(len(rows) <= 12 and len(cols) <= 12, "too many lines")
    _require((row_pitch != 2 or len(rows) >= 6)
             and (col_pitch != 2 or len(cols) >= 6), "dense lattice ambiguity")
    line = np.zeros(grid.shape, dtype=bool)
    line[rows, :] = True
    line[:, cols] = True
    crossings_mask = np.zeros(grid.shape, dtype=bool)
    crossings_mask[np.ix_(rows, cols)] = True
    _require(np.all(grid[line & ~crossings_mask] == structure), "broken line")
    off = np.unique(grid[~line])
    _require(len(off) == 1 and int(off[0]) != structure, "unclear background")
    background = int(off[0])
    crossings = grid[np.ix_(rows, cols)]
    _require(np.all(crossings != background), "background on crossing")
    changed = np.argwhere(crossings != structure)
    _require(len(changed) > 0, "no altered crossings")
    start = tuple(int(v) for v in changed.min(axis=0))
    stop = tuple(int(v) for v in changed.max(axis=0))
    _require(stop == (start[0] + 3, start[1] + 3), "altered extent is not 4 by 4")
    _require(output.shape == (3, 3), "output is not 3 by 3")
    _require(not np.any(output == structure), "structure in output")
    vertices = _encode_vertices(output, background, structure)
    _require(np.array_equal(_decode_vertices(vertices, background, structure), output),
             "four-corner rule does not recover output")
    expected = np.full(crossings.shape, structure, dtype=np.int8)
    expected[start[0]:start[0] + 4, start[1]:start[1] + 4] = vertices
    _require(np.array_equal(crossings, expected), "altered crossings mismatch output")
    payload = frozenset(int(v) for v in np.unique(output) if v != background)
    _require(1 <= len(payload) <= 3, "invalid payload count")
    signature = (grid.shape, background, structure, row_pitch, col_pitch,
                 len(rows), len(cols), int(rows[0]), int(cols[0]), start,
                 tuple(tuple(int(v) for v in row) for row in output))
    _require(signature not in _SOURCE_SIGNATURES, "copied source pair")
    return {"background": background, "structure": structure,
            "payload": payload, "row_pitch": row_pitch, "col_pitch": col_pitch,
            "row_count": len(rows), "col_count": len(cols), "block": start,
            "margins": (int(rows[0]), grid.shape[0] - 1 - int(rows[-1]),
                        int(cols[0]), grid.shape[1] - 1 - int(cols[-1])),
            "square": grid.shape[0] == grid.shape[1],
            "pattern": _canonical_output(output, background)}


def _analyze_pair(pair):
    candidates = []
    for color in np.unique(pair["input"]):
        try:
            candidates.append(_analyze_candidate(pair["input"], pair["output"], int(color)))
        except AssertionError:
            pass
    _require(len(candidates) == 1, "lattice interpretation is ambiguous")
    return candidates[0]


def _predict_from_input(grid):
    predictions = []
    for value in np.unique(grid):
        structure = int(value)
        rows = [r for r in range(grid.shape[0])
                if np.count_nonzero(grid[r] != structure) <= 4]
        cols = [c for c in range(grid.shape[1])
                if np.count_nonzero(grid[:, c] != structure) <= 4]
        if len(rows) < 5 or len(cols) < 5:
            continue
        if len(set(np.diff(rows))) != 1 or len(set(np.diff(cols))) != 1:
            continue
        line = np.zeros(grid.shape, dtype=bool)
        line[rows, :] = True
        line[:, cols] = True
        off = np.unique(grid[~line])
        if len(off) != 1 or int(off[0]) == structure:
            continue
        background = int(off[0])
        crossings = grid[np.ix_(rows, cols)]
        changed = np.argwhere(crossings != structure)
        if len(changed) == 0:
            continue
        start, stop = changed.min(axis=0), changed.max(axis=0)
        if not np.array_equal(stop - start, (3, 3)):
            continue
        vertices = crossings[start[0]:start[0] + 4, start[1]:start[1] + 4]
        predictions.append(_decode_vertices(vertices, background, structure))
    _require(len(predictions) == 1, "input alone does not determine the lattice")
    return predictions[0]


def _output_components(output, background):
    unseen = {(r, c) for r in range(3) for c in range(3)
              if output[r, c] != background}
    components = []
    while unseen:
        start = unseen.pop()
        color = int(output[start])
        part = {start}
        frontier = [start]
        while frontier:
            r, c = frontier.pop()
            neighbors = {(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c),
                                                  (r, c - 1), (r, c + 1))
                         if (rr, cc) in unseen and output[rr, cc] == color}
            unseen -= neighbors
            part |= neighbors
            frontier.extend(neighbors)
        components.append((color, part))
    return components


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen_inputs, seen_pairs, seen_patterns = set(), set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid type or rank")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(np.all((grid >= 0) & (grid <= 9)), "grid color")
            key = _grid_key(pair["input"])
            pair_key = key, _grid_key(pair["output"])
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analysis = _analyze_pair(pair)
            _require(np.array_equal(_predict_from_input(pair["input"]), pair["output"]),
                     "input-only corner decoding disagrees with output")
            _require(analysis["pattern"] not in seen_patterns,
                     "duplicate decoded mosaic pattern")
            seen_patterns.add(analysis["pattern"])
            analyses[split].append(analysis)
    _require(len(analyses["train"][0]["payload"]) == 2,
             "first train example must expose two payload colors")
    first_train = puzzle["train"][0]["output"]
    first_background = analyses["train"][0]["background"]
    _require(_has_shared_side(first_train, first_background),
             "first train example must illustrate shared corners")
    _require(all(a["row_pitch"] <= 5 and a["col_pitch"] <= 5
                 for a in analyses["train"]), "training pitch out of regime")
    _require(any(a["row_pitch"] >= 5 or a["col_pitch"] >= 5
                 or len(a["payload"]) == 3
                 or a["block"][0] in (0, a["row_count"] - 4)
                 or a["block"][1] in (0, a["col_count"] - 4)
                 for a in analyses["test"]), "test lacks supported shift")


def _resolve_counts(seed, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer or None")
            if not low <= int(value) <= high:
                raise ValueError(f"{name} is outside its legal range")
    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        low_train = max(2, test_count + 1)
        train_count = low_train + (seed // 4) % (11 - low_train)
    elif num_train is None:
        test_count = int(num_test)
        low_train = max(2, test_count + 1)
        train_count = low_train + seed % (11 - low_train)
    elif num_test is None:
        train_count = int(num_train)
        test_count = 1 + seed % min(4, train_count - 1)
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    preferred_background = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_patterns = set(), set()
    previous = None
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            for _ in range(1024):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _construct_pair(pair_rng, regime, preferred_background, previous, index)
                key = _grid_key(pair["input"])
                try:
                    analysis = _analyze_pair(pair)
                except AssertionError:
                    continue
                pattern = analysis["pattern"]
                if key in seen_inputs or pattern in seen_patterns:
                    continue
                if split == "test" and index == count - 1:
                    test_analyses = [_analyze_pair(p) for p in puzzle["test"]] + [analysis]
                    if not any(a["row_pitch"] >= 5 or a["col_pitch"] >= 5
                               or len(a["payload"]) == 3
                               or a["block"][0] in (0, a["row_count"] - 4)
                               or a["block"][1] in (0, a["col_count"] - 4)
                               for a in test_analyses):
                        continue
                puzzle[split].append(pair)
                seen_inputs.add(key)
                seen_patterns.add(pattern)
                previous = analysis
                break
            else:
                raise RuntimeError("could not construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k])
        for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    counts, features = set(), []
    review = {name: 0 for name in (
        "background_changes", "structure_changes", "train_structure_changes",
        "train_background_changes", "structure_to_payload", "payload_to_structure",
        "geometry_changes", "pitch_changes", "count_changes", "block_changes",
        "payload_count_changes", "solid", "unequal_components",
        "disconnected_repeat", "diagonal_repeat", "bent", "branch",
        "combined", "square", "rectangular", "asymmetric_pitch",
        "asymmetric_count", "balanced_margins", "offcenter_margins",
        "top", "bottom", "left", "right", "interior", "train_left",
        "test_top_left", "test_interior", "test_three_color_no_large_pitch",
        "test_large_pitch_no_three_color", "test_combined_shift",
    )}
    empty_counts, row_counts, col_counts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzles_equal(puzzle, generate_puzzle(seed)), "nondeterministic puzzle")
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        analyses = {s: [_analyze_pair(p) for p in puzzle[s]]
                    for s in ("train", "test")}
        all_features = analyses["train"] + analyses["test"]
        features.extend(all_features)
        review["background_changes"] += len({a["background"] for a in all_features}) > 1
        review["structure_changes"] += len({a["structure"] for a in all_features}) > 1
        review["train_structure_changes"] += len({a["structure"] for a in analyses["train"]}) > 1
        review["train_background_changes"] += len({a["background"] for a in analyses["train"]}) > 1
        review["geometry_changes"] += len({(a["row_pitch"], a["col_pitch"],
                                            a["row_count"], a["col_count"])
                                           for a in all_features}) > 1
        review["pitch_changes"] += len({(a["row_pitch"], a["col_pitch"])
                                         for a in all_features}) > 1
        review["count_changes"] += len({(a["row_count"], a["col_count"])
                                         for a in all_features}) > 1
        review["block_changes"] += len({a["block"] for a in all_features}) > 1
        review["payload_count_changes"] += len({len(a["payload"]) for a in all_features}) > 1
        train_structures = {a["structure"] for a in analyses["train"]}
        test_structures = {a["structure"] for a in analyses["test"]}
        train_payloads = set().union(*(a["payload"] for a in analyses["train"]))
        test_payloads = set().union(*(a["payload"] for a in analyses["test"]))
        review["structure_to_payload"] += bool(train_structures & test_payloads)
        review["payload_to_structure"] += bool(train_payloads & test_structures)
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split], analyses[split]):
                output = pair["output"]
                background = a["background"]
                _require(np.array_equal(_predict_from_input(pair["input"]), output),
                         "independent input-only decoding failed")
                components = _output_components(output, background)
                sizes = [len(cells) for _, cells in components]
                color_counts = {color: sum(x == color for x, _ in components)
                                for color, _ in components}
                review["solid"] += bool(np.all(output == output[0, 0]))
                review["unequal_components"] += len(set(sizes)) > 1
                review["disconnected_repeat"] += any(n > 1 for n in color_counts.values())
                review["diagonal_repeat"] += any(
                    color1 == color2 and any(
                        max(abs(r1 - r2), abs(c1 - c2)) == 1
                        for r1, c1 in cells1 for r2, c2 in cells2)
                    for i, (color1, cells1) in enumerate(components)
                    for color2, cells2 in components[i + 1:])
                review["bent"] += any(len(cells) >= 3
                                      and len({r for r, c in cells}) > 1
                                      and len({c for r, c in cells}) > 1
                                      for _, cells in components)
                review["branch"] += any(any(
                    sum((rr, cc) in cells for rr, cc in ((r - 1, c), (r + 1, c),
                                                        (r, c - 1), (r, c + 1))) >= 3
                    for r, c in cells) for _, cells in components)
                review["square"] += a["square"]
                review["rectangular"] += not a["square"]
                review["asymmetric_pitch"] += a["row_pitch"] != a["col_pitch"]
                review["asymmetric_count"] += a["row_count"] != a["col_count"]
                top, bottom, left, right = a["margins"]
                review["balanced_margins"] += abs(top - bottom) <= 1 and abs(left - right) <= 1
                review["offcenter_margins"] += abs(top - bottom) >= 2 or abs(left - right) >= 2
                r, c = a["block"]
                review["top"] += r == 0
                review["bottom"] += r == a["row_count"] - 4
                review["left"] += c == 0
                review["right"] += c == a["col_count"] - 4
                interior = 0 < r < a["row_count"] - 4 and 0 < c < a["col_count"] - 4
                review["interior"] += interior
                review["train_left"] += split == "train" and c == 0
                review["test_top_left"] += split == "test" and (r, c) == (0, 0)
                review["test_interior"] += split == "test" and interior
                review["combined"] += len(a["payload"]) == 3 and a["square"] and a["row_pitch"] != a["col_pitch"]
                large_pitch = a["row_pitch"] >= 5 or a["col_pitch"] >= 5
                if split == "test":
                    review["test_three_color_no_large_pitch"] += len(a["payload"]) == 3 and not large_pitch
                    review["test_large_pitch_no_three_color"] += large_pitch and len(a["payload"]) < 3
                    review["test_combined_shift"] += len(a["payload"]) == 3 and large_pitch and (
                        r in (0, a["row_count"] - 4) or c in (0, a["col_count"] - 4))
                empty_counts.add(int(np.count_nonzero(output == background)))
                row_counts.add(a["row_count"])
                col_counts.add(a["col_count"])
    _require(len({a for a, b in counts}) >= 4 and len({b for a, b in counts}) >= 3,
             "counts lack diversity")
    _require(len({f["pattern"] for f in features}) >= 40, "mosaics lack diversity")
    _require({f["row_pitch"] for f in features} >= {2, 3, 4, 5, 6},
             "pitch coverage")
    _require(any(f["square"] for f in features) and any(not f["square"] for f in features),
             "canvas aspect coverage")
    _require({len(f["payload"]) for f in features} == {1, 2, 3},
             "payload count coverage")
    _require(len({f["background"] for f in features}) >= 6
             and len({f["structure"] for f in features}) >= 6, "palette coverage")
    _require(len(row_counts) >= 5 and len(col_counts) >= 5,
             "crossing count coverage")
    _require(len(empty_counts) >= 5, "output density coverage")
    for name in ("background_changes", "structure_changes", "train_structure_changes",
                 "train_background_changes", "structure_to_payload", "payload_to_structure",
                 "geometry_changes", "pitch_changes", "count_changes",
                 "block_changes", "payload_count_changes"):
        _require(review[name] >= 10, f"within-puzzle review coverage: {name}")
    for name in ("solid", "unequal_components", "disconnected_repeat",
                 "diagonal_repeat", "bent",
                 "branch", "combined", "square", "rectangular", "asymmetric_pitch",
                 "asymmetric_count", "balanced_margins", "offcenter_margins",
                 "top", "bottom", "left", "right", "interior", "train_left",
                 "test_top_left", "test_interior", "test_three_color_no_large_pitch",
                 "test_large_pitch_no_three_color", "test_combined_shift"):
        _require(review[name] >= 2, f"cross-seed review coverage: {name}")
    _require(not _puzzles_equal(generate_puzzle(0), generate_puzzle(1)),
             "different seeds repeat")
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(137, nt, ns)
        if nt is not None:
            _require(len(puzzle["train"]) == nt, "explicit train count")
        if ns is not None:
            _require(len(puzzle["test"]) == ns, "explicit test count")
    for nt, ns in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count was accepted")
