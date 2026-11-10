"""Generate falling-strip puzzles with sparse, visible scaffold cues."""

import numpy as np


# The source inputs are used only to exclude exact copies.
_SOURCE_INPUTS = (
    ("7777777777777999", "7777777777577999", "7777777777757111", "7777777777777777", "7777755777777777", "7777557777777777", "7777777777555777", "7777777777555777", "7777777777755577", "7755777777775577", "7577777777777777", "7777777755777777", "7777777555777777", "7777775555557777", "7777777777777777", "7777777777777777"),
    ("77777999777777", "77777999777777", "77777919777777", "75577777777777", "77777775777777", "77755775777777", "77775777777577", "77777777777777", "77777777777777", "77777755555777", "77577777557777", "75757777777777", "77777777777777", "77777775555577"),
    ("7777777779997777", "7777777779997777", "7777777771197777", "7777777777777777", "7777775577777777", "7775775557777777", "7777777555777777", "7777777777777777", "7777777777777777", "7777777777777777", "7777777577557775", "7757777777557557", "7757775777577577", "5555775777775777", "7777777777755777", "7777777777777777"),
    ("7779997777777777", "7779997777777777", "7771997777777777", "7777777777777777", "7777757777775777", "7777777777777777", "7777777775777777", "7777577777777777", "7757777777777777", "7777777557777777", "7777775777777755", "7775575777777755", "7777555777777777", "7777755777777777", "7777777777775777", "7777777777777777"),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
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


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_copy(grid):
    for rows in _SOURCE_INPUTS:
        if grid.shape == (len(rows), len(rows[0])):
            if all("".join(str(int(x)) for x in grid[r]) == rows[r] for r in range(len(rows))):
                return True
    return False


def _roles_and_payload(grid):
    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 4, "input must have four distinct role colors")
    background = int(colors[int(np.argmax(counts))])
    starts = np.flatnonzero(grid[0] != background)
    _require(len(starts) == 3 and np.array_equal(starts, np.arange(starts[0], starts[0] + 3)), "top payload must be a 3-cell block")
    start = int(starts[0])
    primary = int(grid[0, start])
    _require(primary != background, "primary equals background")
    _require(bool(np.all(grid[:2, start:start + 3] == primary)), "first two payload rows differ")
    _require(bool(np.all(grid[2, start:start + 3] != background)), "payload bottom has a gap")
    _require(bool(np.all(grid[0, :start] == background) and np.all(grid[0, start + 3:] == background)), "row zero has extra content")
    accent_colors = set(int(x) for x in grid[2, start:start + 3]) - {primary}
    _require(len(accent_colors) == 1, "payload must visibly contain an accent")
    accent = next(iter(accent_colors))
    scaffold_colors = set(int(x) for x in colors) - {background, primary, accent}
    _require(len(scaffold_colors) == 1, "scaffold role is missing")
    scaffold = next(iter(scaffold_colors))
    _require(bool(np.all(np.isin(grid[1:3, :start], [background, scaffold]))) and bool(np.all(np.isin(grid[1:3, start + 3:], [background, scaffold]))), "extra top colors")
    _require(bool(np.all(np.isin(grid[3:], [background, scaffold]))), "payload colors appear in the scaffold")
    return (background, scaffold, primary, accent), start, grid[:3, start:start + 3].copy()


def _predict(input_grid):
    """Read the visible ledges and move the three ordered strips downward.

    A horizontal gray structure catches the leftmost strip it meets. A strip
    at that structure's right edge is pushed right when the ledge continues
    across the next row. Other strips pass it and continue toward the bottom.
    """
    (background, scaffold, primary, accent), start, payload = _roles_and_payload(input_grid)
    height, width = input_grid.shape
    output = input_grid.copy()
    output[:3, start:start + 3] = background
    columns = [start + i for i in range(3)]
    tops = [None, None, None]
    shifted = [0, 0, 0]
    previous_ledges = []
    for row in range(6, height - 4):
        ledges = []
        col = 0
        while col < width:
            if int(input_grid[row, col]) != scaffold:
                col += 1
                continue
            left = col
            while col < width and int(input_grid[row, col]) == scaffold:
                col += 1
            if col - left >= 2:
                ledges.append((left, col - 1))
        current_ledges = []
        for left, right in ledges:
            continuation = any(a <= right and left <= b for a, b in previous_ledges)
            if continuation:
                current_ledges.append((left, right))
                continue
            touching = [i for i in range(3) if tops[i] is None and left <= columns[i] <= right]
            if not touching:
                continue
            chosen = min(touching)
            tops[chosen] = row - 3
            current_ledges.append((left, right))
            if row + 1 < height and bool(np.all(input_grid[row + 1, left:right + 1] == scaffold)):
                for i in touching:
                    if i != chosen and columns[i] == right and right + 1 < width:
                        if not any(tops[j] is None and j != i and columns[j] == right + 1 for j in range(3)):
                            columns[i] += 1
                            shifted[i] += 1
        previous_ledges = current_ledges
    terminals = tuple((height - 3 if tops[i] is None else tops[i], columns[i]) for i in range(3))
    cells = [(r, c) for top, col in terminals for r in range(top, top + 3) for c in (col,)]
    _require(len(set(cells)) == 9, "terminal strips overlap")
    for i, (top, col) in enumerate(terminals):
        output[top:top + 3, col] = payload[:, i]
    return output, {"roles": (background, scaffold, primary, accent), "source_start": start,
                    "pattern": tuple(int(x) == accent for x in payload[2]), "terminals": tuple(terminals),
                    "turns": tuple(shifted), "shape": input_grid.shape,
                    "gray_overlaps": sum(int(input_grid[r, c]) == scaffold for r, c in cells),
                    "gray_row3": int(np.count_nonzero(input_grid[3] == scaffold)),
                    "empty_gray_rows": sum(not np.any(input_grid[r] == scaffold) for r in range(3, height)),
                    "max_gray_row": max(int(np.count_nonzero(input_grid[r] == scaffold)) for r in range(height)),
                    "top_side_gray": int(np.count_nonzero(input_grid[1:3] == scaffold))}


def _palette(rng):
    return tuple(int(x) for x in rng.choice(np.arange(10), size=4, replace=False))


def _motif(rng, height, width):
    kind = int(rng.integers(0, 6))
    row = int(rng.integers(3, height))
    col = int(rng.integers(0, width))
    if kind == 0:
        return [(row, col)]
    if kind in (1, 2):
        length = int(rng.integers(2, 7))
        return [(row, col + j) for j in range(length)]
    if kind == 3:
        return [(row + j, col) for j in range(int(rng.integers(2, 4)))]
    if kind == 4:
        return [(row + dr, col + dc) for dr in range(2) for dc in range(2)]
    return [(row, col), (row + 1, col + 1), (row + 2, col + 1)]


def _make_pair(rng, regime, palette, pattern, require_shift, require_straight, placement_mode):
    for _ in range(160):
        height = int(rng.integers(12, 21))
        width = int(rng.integers(12, 21))
        if placement_mode == "shifted" and regime == "source_train":
            source_start = int(rng.integers(max(6, width // 2), width - 2))
        elif placement_mode == "shifted" and regime == "target_test":
            source_start = int(rng.integers(0, min(4, width - 2)))
        elif regime == "target_test" and rng.random() < 0.7:
            source_start = int(rng.integers(0, max(2, width // 2)))
        else:
            source_start = int(rng.integers(0, width - 2))
        if source_start > width - 3:
            continue
        background, scaffold, primary, accent = palette
        grid = np.full((height, width), background, dtype=np.int8)
        payload = np.full((3, 3), primary, dtype=np.int8)
        for i in range(3):
            if pattern[i]:
                payload[2, i] = accent
        grid[:3, source_start:source_start + 3] = payload

        if require_straight:
            profiles = (1, 3)
        elif require_shift:
            profiles = (2, 4, 5, 6)
        else:
            profiles = (0, 1, 2, 3, 4, 5, 6)
        profile = int(profiles[int(rng.integers(0, len(profiles)))])
        if profile in (4, 5, 6) and height < 15:
            continue
        rows = list(range(6, height - 4))
        gap = 3 if profile in (4, 5, 6) else 1
        pairs = [(a, b) for a in rows for b in rows if b - a >= gap]
        if profile in (3, 4, 5, 6):
            if not pairs:
                continue
            row_a, row_b = pairs[int(rng.integers(0, len(pairs)))]
        elif profile in (1, 2):
            row_a = rows[int(rng.integers(0, len(rows)))]
        events = []
        if profile == 1:
            events = [(row_a, int(rng.integers(0, 3)), None)]
        elif profile == 2:
            events = [(row_a, int(rng.integers(0, 2)), 2)]
        elif profile == 3:
            first, second = [int(x) for x in rng.permutation(3)[:2]]
            events = [(row_a, first, None), (row_b, second, None)]
        elif profile == 4:
            if rng.random() < 0.5:
                events = [(row_a, 0, None), (row_b, 1, 2)]
            else:
                events = [(row_a, 1, 2), (row_b, 0, None)]
        elif profile == 5:
            events = [(row_a, 0, 2), (row_b, 1, 2)]
        elif profile == 6:
            events = [(row_a, 2, None), (row_b, 0, 1)]
        positions = [source_start + i for i in range(3)]
        captured = {}
        shift_events = []
        valid = True
        for row, target, right_edge_strip in events:
            left = positions[target]
            if right_edge_strip is None:
                right = left + int(rng.integers(1, 5))
            else:
                right = positions[right_edge_strip]
            if right >= width or right <= left or (right_edge_strip is not None and right + 1 >= width):
                valid = False
                break
            grid[row, left:right + 1] = scaffold
            if right_edge_strip is not None:
                grid[row + 1, left:right + 1] = scaffold
                positions[right_edge_strip] += 1
                shift_events.append((row, right_edge_strip))
            captured[target] = row - 3
        if not valid:
            continue
        wanted_terminals = tuple((captured.get(i, height - 3), positions[i]) for i in range(3))
        wanted_turns = tuple(sum(j == i for _, j in shift_events) for i in range(3))
        blocked = set()
        for i in range(3):
            end = captured[i] + 3 if i in captured else height - 1
            for row in range(3, end + 1):
                col = source_start + i + sum(j == i and turn_row < row for turn_row, j in shift_events)
                blocked.add((row, col))
        if events:
            try:
                _, base_info = _predict(grid)
            except AssertionError:
                continue
            if base_info["terminals"] != wanted_terminals or base_info["turns"] != wanted_turns:
                continue

        # Sparse detached marks, blocks, bars, and steps supply the broad gray
        # field seen in the source without painting complete routes.
        for _ in range(int(rng.integers(7, 16))):
            cells = _motif(rng, height, width)
            if all(3 <= r < height and 0 <= c < width and (r, c) not in blocked and grid[r, c] == background for r, c in cells):
                for r, c in cells:
                    grid[r, c] = scaffold
        if rng.random() < 0.55:
            for _ in range(int(rng.integers(1, 4))):
                r = int(rng.integers(1, 3))
                c = int(rng.integers(0, width))
                if not source_start <= c < source_start + 3:
                    grid[r, c] = scaffold
        # A gray cell in a terminal footprint is occasionally overwritten, as
        # in three of the four source pairs.
        for i in rng.permutation(3)[:int(rng.integers(0, 3))]:
            top, col = wanted_terminals[int(i)]
            grid[top + 2, col] = scaffold
        try:
            output, info = _predict(grid)
        except AssertionError:
            continue
        if info["terminals"] != wanted_terminals:
            continue
        if info["turns"] != wanted_turns:
            continue
        if info["gray_overlaps"] > 2:
            continue
        if info["empty_gray_rows"] < 1 or info["max_gray_row"] < 3:
            continue
        if _source_copy(grid) or np.array_equal(grid, output):
            continue
        return {"input": grid, "output": output}
    raise ValueError("could not construct a clear sparse-scaffold pair")


def _analyze_pair(input_grid, output_grid):
    predicted, info = _predict(input_grid)
    _require(np.array_equal(predicted, output_grid), "output does not follow the visible strip cues")
    _require(not _source_copy(input_grid), "source input was copied")
    _require(info["empty_gray_rows"] >= 1, "scaffold has no open row")
    _require(info["max_gray_row"] >= 3, "scaffold lacks source-like clusters")
    _require(info["gray_overlaps"] <= 2, "too much scaffold was covered")
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts")
    inputs = set()
    pairs = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid grid type")
                _require(all(1 <= x <= 30 for x in grid.shape), "invalid grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "invalid grid dtype")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid grid color")
            a, b = pair["input"], pair["output"]
            _require(a.shape == b.shape and not np.array_equal(a, b), "invalid output shape or unchanged pair")
            ka, kb = _grid_key(a), _grid_key(b)
            _require(ka not in inputs and (ka, kb) not in pairs, "duplicate pair")
            inputs.add(ka)
            pairs.add((ka, kb))
            facts[split].append(_analyze_pair(a, b))
    train = facts["train"]
    _require(any(all(t == 0 for t in f["turns"]) for f in train), "training needs a straight-drop example")
    _require(any(any(t > 0 for t in f["turns"]) for f in train), "training needs a lateral cue example")
    _require(any(any(top < f["shape"][0] - 3 for top, _ in f["terminals"]) for f in train), "training needs an intermediate catch")
    _require(any(any(top == f["shape"][0] - 3 for top, _ in f["terminals"]) for f in train), "training needs a bottom landing")
    _require(len({f["pattern"] for f in train}) >= 2, "training needs different packet signatures")
    _require(all({f["pattern"][i] for f in train} == {False, True} for i in range(3)), "training must show both strip signatures in each position")
    train_palettes = {f["roles"] for f in train}
    if any(f["roles"] not in train_palettes for f in facts["test"]):
        _require(len(train_palettes) >= 2, "new test palette without train palette variation")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    palette_mode = "fixed" if rng.random() < 0.5 else "pair"
    placement_mode = "shifted" if rng.random() < 0.4 else "free"
    common_palette = _palette(rng)
    first_mask = int(rng.integers(1, 7))
    patterns = [first_mask, 7 ^ first_mask]
    patterns += [int(rng.integers(1, 8)) for _ in range(train_count + test_count - 2)]
    puzzle = {"train": [], "test": []}
    used = set()
    for split, regime, count in (("train", "source_train", train_count), ("test", "target_test", test_count)):
        for index in range(count):
            pattern_mask = patterns[index if split == "train" else train_count + index]
            pattern = tuple(bool(pattern_mask & (1 << i)) for i in range(3))
            accepted = None
            for _ in range(20):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                palette = common_palette if palette_mode == "fixed" else _palette(child)
                try:
                    pair = _make_pair(child, regime, palette, pattern,
                                      require_shift=(split == "train" and index == 1),
                                      require_straight=(split == "train" and index == 0),
                                      placement_mode=placement_mode)
                except ValueError:
                    continue
                key = _grid_key(pair["input"])
                if key not in used:
                    used.add(key)
                    accepted = pair
                    break
            if accepted is None:
                raise ValueError("unable to generate a distinct pair")
            puzzle[split].append(accepted)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    for seed, tr, te in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4),
                         (9004, None, 1), (9005, None, 4), (9006, 2, None),
                         (9007, 10, None), (314159, None, None)):
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, tr, te)), "nondeterministic puzzle")
        _require(tr is None or len(puzzle["train"]) == tr, "train count ignored")
        _require(te is None or len(puzzle["test"]) == te, "test count ignored")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    counts = set()
    tests = set()
    structures = set()
    shapes = set()
    positions = set()
    roles = set()
    patterns = set()
    shift_cases = set()
    overlaps = set()
    captures = set()
    max_gray = set()
    side_gray = set()
    shared_columns = set()
    row3_empty = set()
    early_catches = set()
    edge_sources = set()
    rectangles = set()
    palette_modes = set()
    left_shift_regimes = set()
    for seed in range(42000, 42042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_facts = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_facts = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        palette_modes.add(len({info["roles"] for info in train_facts}) > 1)
        left_shift_regimes.add(max(info["source_start"] for info in test_facts) < min(info["source_start"] for info in train_facts))
        counts.add(len(puzzle["train"]))
        tests.add(len(puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                info = _analyze_pair(a, b)
                _require(a.dtype == np.int8 and b.dtype == np.int8, "expected compact dtype")
                structures.add(_grid_key(a))
                shapes.add(info["shape"])
                positions.add(info["source_start"])
                roles.add(info["roles"])
                patterns.add(info["pattern"])
                shift_cases.add(sum(info["turns"]))
                overlaps.add(info["gray_overlaps"])
                captures.add(sum(top < a.shape[0] - 3 for top, _ in info["terminals"]))
                max_gray.add(info["max_gray_row"])
                side_gray.add(info["top_side_gray"] > 0)
                shared_columns.add(len({col for _, col in info["terminals"]}) < 3)
                row3_empty.add(info["gray_row3"] == 0)
                early_catches.add(any(top <= 5 for top, _ in info["terminals"]))
                edge_sources.add(info["source_start"] in (0, a.shape[1] - 3))
                rectangles.add(a.shape[0] != a.shape[1])
    _require(len(counts) >= 4 and len(tests) >= 3, "automatic counts lack diversity")
    _require(len(structures) >= 110 and len(shapes) >= 12 and len(positions) >= 8, "geometry lacks diversity")
    _require(len(roles) >= 30 and len(patterns) >= 5, "palette or signature lacks diversity")
    _require({0, 1, 2} <= shift_cases and {0, 1, 2} <= overlaps, "shift or overlay diversity missing")
    _require({0, 1, 2} <= captures and len(max_gray) >= 4 and side_gray == {False, True}, "scaffold diversity missing")
    _require(shared_columns == {False, True}, "shared-column landing variation missing")
    _require(all(values == {False, True} for values in (row3_empty, early_catches, edge_sources, rectangles, palette_modes, left_shift_regimes)), "review regime diversity missing")
