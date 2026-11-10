"""Solved synthetic puzzles for the divider-and-docking ARC family."""

import numpy as np


_GUIDE_OFFSETS = frozenset({(-2, 2), (-1, 2), (0, 2), (0, 0), (1, 1)})
_DIAGONAL = frozenset({(0, 1), (1, 0)})
_SQUARE = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
_ZIGZAG = frozenset({(0, 1), (1, 0), (2, 1)})
_BOXES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))

# Original task pairs are used only as a regression check and to reject copies.
_SOURCE_ROWS = (
    (
        ("77777777777", "77777777777", "77777777777", "77777777777", "77777777777", "44444444444", "77772777776", "77772277776", "77777777676", "77777777767", "77777777777"),
        ("77777777776", "77777772776", "77777772276", "77777777767", "77777777777", "44444444444", "77777777777", "77777777777", "77777777777", "77777777777", "77777777777"),
    ),
    (
        ("77777777777", "77777777777", "77777777777", "77777777777", "77777777777", "44444444444", "77777777677", "77777777677", "78877767677", "78877776777", "77777777777"),
        ("77777777677", "77777887677", "77777887677", "77777776777", "77777777777", "44444444444", "77777777777", "77777777777", "77777777777", "77777777777", "77777777777"),
    ),
    (
        ("77777777777", "77777777777", "77777777777", "77777777777", "77777777777", "44444444444", "77777677717", "77777677177", "77767677777", "77776777777", "77777777777"),
        ("77777677777", "77717677777", "77167677777", "77776777777", "77777777777", "44444444444", "77777777777", "77777777777", "77777777777", "77777777777", "77777777777"),
    ),
    (
        ("77777777777", "77777777777", "77777777777", "77777777777", "77777777777", "44444444444", "77777777677", "77787777677", "77877767677", "77787776777", "77777777777"),
        ("77777787677", "77777877677", "77777787677", "77777776777", "77777777777", "44444444444", "77777777777", "77777777777", "77777777777", "77777777777", "77777777777"),
    ),
)
_SOURCE_INPUTS = {rows[0] for rows in _SOURCE_ROWS}


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _text_rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _read_rows(rows):
    return np.array([[int(x) for x in row] for row in rows], dtype=np.int8)


def _count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(num_test + 1, 3 + (seed // 4) % 8)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _orient(grid, orientation):
    result = np.rot90(grid, orientation % 4)
    return np.fliplr(result) if orientation >= 4 else result


def _unorient(grid, orientation):
    result = np.fliplr(grid) if orientation >= 4 else grid
    return np.rot90(result, -(orientation % 4))


def _guide_silhouettes():
    mask = np.zeros((4, 3), dtype=np.int8)
    for dr, dc in _GUIDE_OFFSETS:
        mask[dr + 2, dc] = 1
    return {frozenset(map(tuple, np.argwhere(_orient(mask, code)))) for code in range(8)}


_GUIDE_SILHOUETTES = _guide_silhouettes()


def _guide_at(row, col):
    return frozenset((row + dr, col + dc) for dr, dc in _GUIDE_OFFSETS)


def _connected(cells):
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        row, col = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (row + dr, col + dc)
                if neighbor in cells and neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
    return seen == set(cells)


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    return r0, r1, c0, c1, frozenset((r - r0, c - c0) for r, c in cells)


def _make_patterns():
    pools = {"all": [], "empty": [], "occupied": [], "tall": [], "wide": [], "large": []}
    for height, width in _BOXES:
        area = height * width
        for bits in range(1, 1 << area):
            size = bits.bit_count()
            if size < 3 or size > min(8, area):
                continue
            cells = frozenset((i // width, i % width) for i in range(area) if bits & (1 << i))
            if not _connected(cells):
                continue
            r0, r1, c0, c1, normalized = _bbox(cells)
            if (r0, r1, c0, c1) != (0, height - 1, 0, width - 1):
                continue
            if cells == _SQUARE or cells == _ZIGZAG:
                continue
            # A payload must not masquerade as the five-cell guide.
            if cells in _GUIDE_SILHOUETTES:
                continue
            pools["all"].append(cells)
            pools["occupied" if (height - 1, width - 1) in cells else "empty"].append(cells)
            if height > width:
                pools["tall"].append(cells)
            if width > height:
                pools["wide"].append(cells)
            if area >= 9:
                pools["large"].append(cells)
    return pools


_PATTERNS = _make_patterns()


def _group_patterns():
    groups = {}
    for kind, patterns in _PATTERNS.items():
        by_box = {}
        for cells in patterns:
            box = (max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1)
            by_box.setdefault(box, {}).setdefault(len(cells), []).append(cells)
        groups[kind] = by_box
    return groups


_PATTERN_GROUPS = _group_patterns()


def _pattern(rng, kind):
    if kind == "diagonal":
        return _DIAGONAL
    if kind == "square":
        return _SQUARE
    if kind == "zigzag":
        return _ZIGZAG
    if kind == "l":
        missing = ((0, 0), (0, 1), (1, 0))[int(rng.integers(0, 3))]
        return _SQUARE - {missing}
    # Choose geometry before a particular mask. Otherwise the many masks
    # in a large box overwhelm the smaller, equally useful silhouettes.
    by_box = _PATTERN_GROUPS[kind]
    boxes = tuple(sorted(by_box))
    box = boxes[int(rng.integers(0, len(boxes)))]
    sizes = tuple(sorted(by_box[box]))
    size = sizes[int(rng.integers(0, len(sizes)))]
    pool = by_box[box][size]
    return pool[int(rng.integers(0, len(pool)))]


def _source_placements(depth, width, height, pwidth, guide):
    placements = []
    for top in range(depth - height + 1):
        for left in range(width - pwidth + 1):
            if all(
                max(abs(row - gr), abs(col - gc)) > 1
                for row in range(top, top + height)
                for col in range(left, left + pwidth)
                for gr, gc in guide
            ):
                placements.append((top, left))
    return placements


def _draw_canonical(rng, depth, width, colors, pattern):
    height = max(r for r, _ in pattern) + 1
    pwidth = max(c for _, c in pattern) + 1
    candidates = []
    for row in range(max(2, height - 1), depth - 1):
        for col in range(max(2, pwidth - 1), width - 2):
            guide = _guide_at(row, col)
            placements = _source_placements(depth, width, height, pwidth, guide)
            if placements:
                candidates.append((row, col, guide, placements))
    _check(bool(candidates), "no separated initial placement")
    row, col, guide, placements = candidates[int(rng.integers(0, len(candidates)))]
    top, left = placements[int(rng.integers(0, len(placements)))]
    background, separator, guide_color, payload_color = colors
    input_grid = np.full((2 * depth + 1, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    input_grid[depth, :] = separator
    output_grid[depth, :] = separator
    for gr, gc in guide:
        input_grid[depth + 1 + gr, gc] = guide_color
        output_grid[gr, gc] = guide_color
    for pr, pc in pattern:
        input_grid[depth + 1 + top + pr, left + pc] = payload_color
        output_grid[row - height + 1 + pr, col - pwidth + 1 + pc] = payload_color
    return input_grid, output_grid


def _analyze_canonical(grid):
    height, width = grid.shape
    _check(height % 2 == 1 and height >= 11 and width >= 9, "canonical dimensions")
    depth = height // 2
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    separator = int(grid[depth, 0])
    _check(separator != background and np.all(grid[depth] == separator), "center divider")
    _check(np.all(grid[:depth] == background), "destination is not blank")
    _check(all(not np.all(grid[r] == grid[r, 0]) or int(grid[r, 0]) == background for r in range(height) if r != depth), "extra divider")
    source = grid[depth + 1:]
    object_colors = [int(x) for x in np.unique(source) if int(x) != background]
    _check(len(object_colors) == 2 and separator not in object_colors, "one guide and one payload")
    guide_candidates = []
    for color in object_colors:
        cells = frozenset(map(tuple, np.argwhere(source == color)))
        if len(cells) == 5:
            matches = [(r, c) for r, c in cells if _guide_at(r, c) == cells]
            if len(matches) == 1:
                guide_candidates.append((color, cells, matches[0]))
    _check(len(guide_candidates) == 1, "unique open guide")
    guide_color, guide, endpoint = guide_candidates[0]
    payload_color = next(color for color in object_colors if color != guide_color)
    payload = frozenset(map(tuple, np.argwhere(source == payload_color)))
    _check(2 <= len(payload) <= 8 and _connected(payload), "one connected payload")
    r0, r1, c0, c1, pattern = _bbox(payload)
    _check((r1 - r0 + 1, c1 - c0 + 1) in _BOXES, "supported bounding box")
    _check(_source_placements(depth, width, r1-r0+1, c1-c0+1, guide) != [], "payload placement")
    _check(all(max(abs(r-gr), abs(c-gc)) > 1 for r in range(r0, r1+1) for c in range(c0, c1+1) for gr, gc in guide), "payload is not separate")
    er, ec = endpoint
    _check(er >= r1-r0 and ec >= c1-c0, "docked payload cannot fit")
    return dict(depth=depth, width=width, background=background, separator=separator,
                guide_color=guide_color, payload_color=payload_color, guide=guide,
                endpoint=endpoint, payload=payload, bbox=(r0, r1, c0, c1), pattern=pattern)


def _analyze_input(grid):
    matches = []
    for orientation in range(8):
        canonical = _unorient(grid, orientation)
        try:
            info = _analyze_canonical(canonical)
        except (AssertionError, ValueError):
            continue
        info["orientation"] = orientation
        matches.append(info)
    _check(len(matches) == 1, "guide and divider must identify one orientation")
    return matches[0]


def _predict_canonical(input_grid, info, corner=(1, 1)):
    canonical = _unorient(input_grid, info["orientation"])
    depth, width = info["depth"], info["width"]
    output = np.full(canonical.shape, info["background"], dtype=input_grid.dtype)
    output[depth, :] = info["separator"]
    for row, col in info["guide"]:
        output[row, col] = info["guide_color"]
    r0, r1, c0, c1 = info["bbox"]
    anchor_row = r1 if corner[0] else r0
    anchor_col = c1 if corner[1] else c0
    shift_row = info["endpoint"][0] - anchor_row
    shift_col = info["endpoint"][1] - anchor_col
    for row, col in info["payload"]:
        target_row, target_col = row + shift_row, col + shift_col
        if not (0 <= target_row < depth and 0 <= target_col < width):
            return None
        output[target_row, target_col] = info["payload_color"]
    return output


def _predict(input_grid, info, corner=(1, 1)):
    canonical = _predict_canonical(input_grid, info, corner)
    return None if canonical is None else _orient(canonical, info["orientation"])


def _pair_features(pair):
    info = _analyze_input(pair["input"])
    return (info["orientation"], (info["depth"], info["width"]),
            (info["background"], info["separator"], info["guide_color"]),
            info["pattern"], (info["bbox"][1]-info["bbox"][0]+1, info["bbox"][3]-info["bbox"][2]+1))


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed puzzle") from exc


def _validate(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train, list) and isinstance(test, list), "pair lists")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_infos = []
    test_has_square = False
    possible_corners = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _check(all(1 <= n <= 30 for n in grid.shape), "grid size")
                _check(np.issubdtype(grid.dtype, np.integer), "integer grid")
                _check(bool(np.all((grid >= 0) & (grid <= 9))), "ARC colors")
            _check(inp.shape == out.shape and inp.dtype == out.dtype, "matching dimensions and dtype")
            _check(not np.array_equal(inp, out), "unchanged pair")
            insig, outsig = _signature(inp), _signature(out)
            _check(insig not in seen_inputs and (insig, outsig) not in seen_pairs, "duplicate pair")
            seen_inputs.add(insig)
            seen_pairs.add((insig, outsig))
            _check(_text_rows(inp) not in _SOURCE_INPUTS, "original input copy")
            info = _analyze_input(inp)
            _check(np.array_equal(out, _predict(inp, info)), "unchanged guide, corner docking, or panel transfer violated")
            if section == "train":
                train_infos.append(info)
                possible_corners = {corner for corner in possible_corners
                    if (prediction := _predict(inp, info, corner)) is not None and np.array_equal(out, prediction)}
            elif info["pattern"] == _SQUARE:
                test_has_square = True
    _check(possible_corners == {(1, 1)}, "training does not identify the bounding-box corner")
    _check(any((i["bbox"][1], i["bbox"][3]) not in i["payload"] for i in train_infos), "training lacks an empty docking corner")
    _check(any((i["bbox"][1], i["bbox"][3]) in i["payload"] for i in train_infos), "training lacks an occupied docking corner")
    _check(not test_has_square or any(i["pattern"] == _SQUARE for i in train_infos),
           "square test lacks a square training demonstration")


def _role_colors(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _variation_mode(seed):
    # The count schedules use divisions by four. A three-seed cycle
    # keeps orientation and palette variation from tracking those counts.
    return (seed // 3) % 4


def _orientation_plan(rng, seed, count_train, count_test):
    mode = _variation_mode(seed)
    first = int(rng.integers(0, 8))
    if mode == 0:
        return [first] * (count_train + count_test)
    second = int(rng.choice([x for x in range(8) if x != first]))
    train = [first, second]
    while len(train) < count_train:
        train.append(int(rng.integers(0, 8 if mode == 3 else 4)) if mode == 3 else int(rng.choice((first, second))))
    if mode == 2:
        held_out = [x for x in range(8) if x not in set(train)]
        tests = [int(rng.choice(held_out)) for _ in range(count_test)]
    elif mode == 1:
        tests = [int(rng.choice(train)) for _ in range(count_test)]
    else:
        tests = [int(rng.integers(0, 8)) for _ in range(count_test)]
    return train + tests


def _pattern_plan(rng, seed, count_train, count_test):
    first_test = ("square", "zigzag", "tall", "wide", "large")[seed % 5]
    second_train = "square" if first_test == "square" else ("l", "square", "zigzag")[seed % 3]
    train = ["diagonal", second_train]
    choices = ("l", "square", "zigzag", "empty", "occupied", "tall", "wide", "large", "all")
    while len(train) < count_train:
        train.append(choices[int(rng.integers(0, len(choices)))])
    test = [first_test]
    test_choices = ("square", "zigzag", "empty", "occupied", "tall", "wide", "large", "all")
    while len(test) < count_test:
        test.append(test_choices[int(rng.integers(0, len(test_choices)))])
    if "square" in test and "square" not in train:
        train[1] = "square"
    return train + test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    count_train, count_test = _resolve_counts(seed, rng, num_train, num_test)
    count = count_train + count_test
    mode = _variation_mode(seed)
    base_depth = int(rng.integers(6, 13))
    base_width = int(rng.integers(10, 19))
    base_roles = _role_colors(rng)
    orientations = _orientation_plan(rng, seed, count_train, count_test)
    kinds = _pattern_plan(rng, seed, count_train, count_test)
    puzzle = {"train": [], "test": []}
    used = set()
    for index in range(count):
        accepted = None
        for attempt in range(48):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            if mode == 0:
                depth, width, roles = base_depth, base_width, base_roles
            else:
                depth = base_depth if index == 0 else int(pair_rng.integers(6, 13))
                width = base_width if index == 0 else int(pair_rng.integers(10, 19))
                roles = base_roles if mode == 1 or index == 0 else _role_colors(pair_rng)
                if index == 1 and (depth, width) == (base_depth, base_width):
                    width = 10 + (width - 9) % 9
                if index == 1 and mode >= 2 and roles == base_roles:
                    roles = (roles[1], roles[2], roles[0])
            free_colors = [c for c in range(10) if c not in roles]
            payload_color = int(pair_rng.choice(free_colors))
            pattern = _pattern(pair_rng, kinds[index])
            canonical_input, canonical_output = _draw_canonical(
                pair_rng, depth, width, (*roles, payload_color), pattern)
            inp = np.ascontiguousarray(_orient(canonical_input, orientations[index]))
            out = np.ascontiguousarray(_orient(canonical_output, orientations[index]))
            sig = _signature(inp)
            if sig not in used and _text_rows(inp) not in _SOURCE_INPUTS:
                used.add(sig)
                accepted = {"input": inp, "output": out}
                break
        _check(accepted is not None, "could not construct unique pair")
        puzzle["train" if index < count_train else "test"].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for input_rows, output_rows in _SOURCE_ROWS:
        source_input, source_output = _read_rows(input_rows), _read_rows(output_rows)
        info = _analyze_input(source_input)
        _check(np.array_equal(_predict(source_input, info), source_output), "original task regression")
    train_counts, test_counts, shapes, orientations, role_triples = set(), set(), set(), set(), set()
    patterns, feature_combos, different_orientations, varied_dims, varied_roles = set(), set(), 0, 0, 0
    square_tests, zigzag_trains, empty_tests = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_equal_puzzles(puzzle, generate_puzzle(seed)), "nondeterminism")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_features = [_pair_features(pair) for section in ("train", "test") for pair in puzzle[section]]
        os = {f[0] for f in pair_features}
        ds = {f[1] for f in pair_features}
        rs = {f[2] for f in pair_features}
        different_orientations += len(os) > 1
        varied_dims += len(ds) > 1
        varied_roles += len(rs) > 1
        for o, d, r, p, box in pair_features:
            orientations.add(o)
            shapes.add(d)
            role_triples.add(r)
            patterns.add((box, p))
            feature_combos.add((o, d, r, box, p))
        zigzag_trains += any(f[3] == _ZIGZAG for f in pair_features[:len(puzzle["train"])])
        square_tests += any(f[3] == _SQUARE for f in pair_features[len(puzzle["train"]):])
        empty_tests += any((f[4][0]-1, f[4][1]-1) not in f[3] for f in pair_features[len(puzzle["train"]):])
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _check(len(shapes) >= 20 and len(orientations) == 8 and len(role_triples) >= 20, "geometry and palette diversity")
    _check(len(patterns) >= 35 and len(feature_combos) >= 150, "structural diversity")
    _check(different_orientations >= 20 and varied_dims >= 20 and varied_roles >= 10, "within-puzzle variation")
    _check(square_tests >= 3 and zigzag_trains >= 3 and empty_tests >= 3, "train/test shape mixing")
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            _check(len(puzzle["train"]) == counts[0], "requested train count")
        if counts[1] is not None:
            _check(len(puzzle["test"]) == counts[1], "requested test count")
    for counts in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2)):
        try:
            generate_puzzle(5, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
