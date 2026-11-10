"""Generate corrupted periodic lattices and their clean intersection grids."""

import hashlib
import numpy as np


PITCHES = (3, 4, 5, 6, 7)
SOURCE_INPUT_HASHES = frozenset({
    ((21, 17), "d91f5456254baa5100ca470262ce1626729726875ae29c01ae353fd57dcfc936"),
    ((19, 13), "6fb5da8907f4beba529a5c28973eca759e3e92dcdc99890c06b91e45764300ef"),
    ((21, 24), "029c0aa15b1840b98494f7f85ce7faacf886a47cac0ca2cf40a0739ed41c9e7f"),
    ((22, 27), "6e646118dffd8c7b4ede5c49be852bb8be50f28d3dd106c88eae3bef098ca6ed"),
})


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    else:
        num_test = 1 + seed % min(4, num_train - 1)
    return num_train, num_test


def _axis_options(pitch, shifted):
    """(dimension, first-line coordinate, line count, trailing cells)."""
    phases = range(pitch) if shifted else (pitch - 1,)
    return [
        (dimension, phase, count, tail)
        for phase in phases
        for count in range(3, 11)
        for tail in range(pitch)
        for dimension in (phase + (count - 1) * pitch + 1 + tail,)
        if 10 <= dimension <= 30
    ]


def _masks(height, width, pitch, row_phase, col_phase):
    rows = np.arange(row_phase, height, pitch)
    cols = np.arange(col_phase, width, pitch)
    row_mask = np.zeros((height, width), dtype=bool)
    col_mask = np.zeros((height, width), dtype=bool)
    row_mask[rows, :] = True
    col_mask[:, cols] = True
    return rows, cols, row_mask, col_mask


def _clean(height, width, pitch, row_phase, col_phase, background, structure, accent):
    _, _, row_mask, col_mask = _masks(height, width, pitch, row_phase, col_phase)
    result = np.full((height, width), background, dtype=np.int8)
    result[row_mask | col_mask] = structure
    result[row_mask & col_mask] = accent
    return result


def _scaffold(height, width, pitch, row_phase, col_phase, background, structure):
    _, _, row_mask, col_mask = _masks(height, width, pitch, row_phase, col_phase)
    result = np.full((height, width), background, dtype=np.int8)
    result[row_mask | col_mask] = structure
    return result


def _input_candidates(grid, limit=2):
    """Find all role assignments and periodic scaffolds visible in an input."""
    colors = [int(x) for x in np.unique(grid)]
    if len(colors) != 3:
        return []
    height, width = grid.shape
    found = []
    for accent in colors:
        other = [x for x in colors if x != accent]
        for background, structure in (other, other[::-1]):
            sr, sc = np.where(grid == structure)
            br, bc = np.where(grid == background)
            for pitch in PITCHES:
                sr_mod, sc_mod = sr % pitch, sc % pitch
                br_mod, bc_mod = br % pitch, bc % pitch
                for row_phase in range(pitch):
                    if len(range(row_phase, height, pitch)) < 3:
                        continue
                    remaining = sr_mod != row_phase
                    for col_phase in range(pitch):
                        if len(range(col_phase, width, pitch)) < 3:
                            continue
                        if np.any(sc_mod[remaining] != col_phase):
                            continue
                        if np.any((br_mod == row_phase) | (bc_mod == col_phase)):
                            continue
                        found.append((pitch, row_phase, col_phase, background, structure, accent))
                        if len(found) >= limit:
                            return found
    return found


def _output_interpretations(grid):
    """Infer geometry and colors from the clean grid, without generator metadata."""
    height, width = grid.shape
    colors = [int(x) for x in np.unique(grid)]
    found = []
    if len(colors) != 3:
        return found
    for accent in colors:
        rr, cc = np.where(grid == accent)
        rows, cols = np.unique(rr), np.unique(cc)
        if len(rows) < 3 or len(cols) < 3 or len(rr) != len(rows) * len(cols):
            continue
        row_steps, col_steps = np.diff(rows), np.diff(cols)
        if not (np.all(row_steps == row_steps[0]) and np.all(col_steps == row_steps[0])):
            continue
        pitch = int(row_steps[0])
        if pitch not in PITCHES or rows[0] >= pitch or cols[0] >= pitch:
            continue
        if not (np.array_equal(rows, np.arange(rows[0], height, pitch))
                and np.array_equal(cols, np.arange(cols[0], width, pitch))):
            continue
        _, _, row_mask, col_mask = _masks(height, width, pitch, int(rows[0]), int(cols[0]))
        if not np.all(grid[row_mask & col_mask] == accent):
            continue
        background_sites = grid[~(row_mask | col_mask)]
        structure_sites = grid[row_mask ^ col_mask]
        if len(background_sites) == 0 or len(structure_sites) == 0:
            continue
        background, structure = int(background_sites[0]), int(structure_sites[0])
        if len({background, structure, accent}) != 3:
            continue
        if not (np.all(background_sites == background)
                and np.all(structure_sites == structure)):
            continue
        found.append((pitch, int(rows[0]), int(cols[0]), background, structure, accent))
    return found


def _evidence_is_strong(grid, pitch, row_phase, col_phase, structure):
    rows, cols, _, _ = _masks(*grid.shape, pitch, row_phase, col_phase)
    row_min = int(np.ceil(0.50 * grid.shape[1]))
    col_min = int(np.ceil(0.50 * grid.shape[0]))
    return (all(np.count_nonzero(grid[row, :] == structure) >= row_min for row in rows)
            and all(np.count_nonzero(grid[:, col] == structure) >= col_min for col in cols))


def _pick_axis(rng, options, desired_count=None):
    if desired_count is None:
        counts = sorted({option[2] for option in options})
        weights = np.array([1 / (1 + abs(count - 4)) for count in counts], dtype=float)
        weights /= weights.sum()
        desired_count = int(rng.choice(counts, p=weights))
    eligible = [option for option in options if option[2] == desired_count]
    return eligible[int(rng.integers(len(eligible)))]


def _choose_train_layout(rng, pitch, shifted, row_count=None, col_count=None, force_shift=False):
    options = _axis_options(pitch, shifted)
    row = _pick_axis(rng, options, row_count)
    col = _pick_axis(rng, options, col_count)
    if force_shift and row[1] == pitch - 1 and col[1] == pitch - 1:
        shifted_options = [option for option in options if option[1] != pitch - 1]
        if rng.random() < 0.5:
            row = _pick_axis(rng, shifted_options, row_count)
        else:
            col = _pick_axis(rng, shifted_options, col_count)
    return (pitch, row, col)


def _choose_test_layout(rng, training, mode, shifted):
    order = [int(x) for x in rng.permutation(len(training))]
    for index in order:
        pitch, old_row, old_col = training[index]
        options = _axis_options(pitch, shifted)
        if mode == 0:  # demonstrated line counts and comparable canvas
            row_options = [x for x in options if x[2] == old_row[2] and abs(x[0] - old_row[0]) <= 3]
            col_options = [x for x in options if x[2] == old_col[2] and abs(x[0] - old_col[0]) <= 3]
        elif mode == 1:  # extend rows alone
            row_options = [x for x in options if x[2] > old_row[2] and x[0] > old_row[0]]
            col_options = [x for x in options if x[2] == old_col[2] and abs(x[0] - old_col[0]) <= 3]
        elif mode == 2:  # extend columns alone
            row_options = [x for x in options if x[2] == old_row[2] and abs(x[0] - old_row[0]) <= 3]
            col_options = [x for x in options if x[2] > old_col[2] and x[0] > old_col[0]]
        elif mode == 3:  # extend both
            row_options = [x for x in options if x[2] > old_row[2] and x[0] > old_row[0]]
            col_options = [x for x in options if x[2] > old_col[2] and x[0] > old_col[0]]
        else:  # a larger one-axis or two-axis extension
            row_options = [x for x in options if x[0] >= old_row[0] and x[2] >= old_row[2]]
            col_options = [x for x in options if x[0] >= old_col[0] and x[2] >= old_col[2]]
            large_rows = [x for x in row_options if x[0] >= old_row[0] + 6]
            large_cols = [x for x in col_options if x[0] >= old_col[0] + 6]
            if not row_options or not col_options or (not large_rows and not large_cols):
                continue
            if large_rows and (not large_cols or rng.random() < 0.5):
                row_options = large_rows
            else:
                col_options = large_cols
        if row_options and col_options:
            row = row_options[int(rng.integers(len(row_options)))]
            col = col_options[int(rng.integers(len(col_options)))]
            return (pitch, row, col)
    # Legal count requests can leave no extension room; use a fresh
    # arrangement at a pitch the training set already demonstrates.
    pitch = training[order[0]][0]
    return _choose_train_layout(rng, pitch, shifted)


def _rate(rng, bands):
    low, high = bands[int(rng.integers(len(bands)))]
    return float(rng.uniform(low, high))


def _select_corruption(rng, category_mask, count, clustered):
    candidates = np.flatnonzero(category_mask)
    count = min(count, len(candidates))
    selected = set()
    height, width = category_mask.shape
    if clustered and count >= 3:
        starts = candidates[
            np.array([
                (index // width in (0, 1, height - 2, height - 1)
                 or index % width in (0, 1, width - 2, width - 1))
                for index in candidates
            ])
        ]
        pool = starts if len(starts) and rng.random() < 0.5 else candidates
        start = int(pool[int(rng.integers(len(pool)))])
        selected.add(start)
        frontier = [start]
        while frontier and len(selected) < min(count, int(rng.integers(3, 6))):
            current = frontier.pop(0)
            row, col = divmod(current, width)
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
            for choice in rng.permutation(4):
                nr, nc = neighbors[int(choice)]
                if 0 <= nr < height and 0 <= nc < width and category_mask[nr, nc]:
                    neighbor = nr * width + nc
                    if neighbor not in selected:
                        selected.add(neighbor)
                        frontier.append(neighbor)
                        if len(selected) >= count:
                            break
    remaining = np.array([x for x in candidates if int(x) not in selected], dtype=np.int64)
    need = count - len(selected)
    if need:
        selected.update(int(x) for x in rng.choice(remaining, size=need, replace=False))
    return np.array(sorted(selected), dtype=np.int64)


def _generate_pair(rng, layout, colors):
    pitch, row, col = layout
    height, row_phase = row[:2]
    width, col_phase = col[:2]
    background, structure, accent = colors
    output = _clean(height, width, pitch, row_phase, col_phase, *colors)
    scaffold = _scaffold(height, width, pitch, row_phase, col_phase, background, structure)
    _, _, row_mask, col_mask = _masks(height, width, pitch, row_phase, col_phase)
    masks = (~(row_mask | col_mask), row_mask ^ col_mask, row_mask & col_mask)
    rates = (
        _rate(rng, ((0.015, 0.05), (0.06, 0.11), (0.12, 0.20))),
        _rate(rng, ((0.02, 0.08), (0.09, 0.16), (0.17, 0.24))),
        _rate(rng, ((0.0, 0.12), (0.13, 0.34), (0.35, 0.60))),
    )
    counts = [
        min(len(np.flatnonzero(mask)), max(1 if index < 2 else 0,
            int(round(rate * np.count_nonzero(mask)))))
        for index, (rate, mask) in enumerate(zip(rates, masks))
    ]
    clustered = rng.random(2) < 0.52
    for _ in range(24):
        grid = scaffold.copy()
        for index, mask in enumerate(masks):
            cells = _select_corruption(rng, mask, counts[index],
                                       bool(clustered[index]) if index < 2 else False)
            grid.flat[cells] = accent
        if not _evidence_is_strong(grid, pitch, row_phase, col_phase, structure):
            continue
        if _input_candidates(grid) == [(pitch, row_phase, col_phase, *colors)]:
            return grid, output
    raise ValueError("could not place uniquely recoverable corruption")


def _choose_colors(rng, base, fixed_roles, training_accents, distinct):
    if distinct:
        fresh = [int(x) for x in base if x not in training_accents
                 and (not fixed_roles or x not in base[:2])]
        if fresh:
            accent = fresh[int(rng.integers(len(fresh)))]
            if fixed_roles:
                return (base[0], base[1], accent)
            roles = [int(x) for x in rng.choice(
                [x for x in base if x != accent], size=2, replace=False)]
            return (*roles, accent)
    if fixed_roles:
        background, structure = base[:2]
    else:
        background, structure = [int(x) for x in rng.choice(base, size=2, replace=False)]
    eligible = [int(x) for x in base if x not in (background, structure)]
    if distinct:
        fresh = [x for x in eligible if x not in training_accents]
        if fresh:
            eligible = fresh
    accent = eligible[int(rng.integers(len(eligible)))]
    return (background, structure, accent)


def _grid_key(grid):
    return (grid.shape, grid.dtype.str, grid.tobytes())


def _has_adjacent(mask):
    return bool(np.any(mask[:-1, :] & mask[1:, :])
                or np.any(mask[:, :-1] & mask[:, 1:]))


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input and output dimensions differ")
    interpretations = _output_interpretations(out)
    _require(len(interpretations) == 1, "output is not a unique clean lattice")
    expected = interpretations[0]
    _require(_input_candidates(inp) == [expected], "input has a competing lattice or role assignment")
    pitch, row_phase, col_phase, background, structure, accent = expected
    _require(_evidence_is_strong(inp, pitch, row_phase, col_phase, structure),
             "too much structural evidence was erased")
    _, _, row_mask, col_mask = _masks(*inp.shape, pitch, row_phase, col_phase)
    noise = inp == accent
    _require(np.any(noise & ~(row_mask | col_mask)), "no exterior accent corruption")
    _require(np.any(noise & (row_mask ^ col_mask)), "no structural accent corruption")
    _require(not np.array_equal(inp, out), "input equals output")
    source_key = (inp.shape, hashlib.sha256(inp.astype(np.int8).tobytes()).hexdigest())
    _require(source_key not in SOURCE_INPUT_HASHES, "exact source input was copied")
    background_noise = noise & ~(row_mask | col_mask)
    line_noise = noise & (row_mask ^ col_mask)
    return {
        "pitch": pitch, "row_phase": row_phase, "col_phase": col_phase,
        "background": background, "structure": structure, "accent": accent,
        "height": inp.shape[0], "width": inp.shape[1],
        "row_count": len(range(row_phase, inp.shape[0], pitch)),
        "col_count": len(range(col_phase, inp.shape[1], pitch)),
        "row_tail": (inp.shape[0] - 1 - row_phase) % pitch,
        "col_tail": (inp.shape[1] - 1 - col_phase) % pitch,
        "background_patch": _has_adjacent(background_noise),
        "line_run": _has_adjacent(line_noise),
        "background_rate": float(np.count_nonzero(background_noise)
                                 / np.count_nonzero(~(row_mask | col_mask))),
        "line_rate": float(np.count_nonzero(line_noise)
                           / np.count_nonzero(row_mask ^ col_mask)),
        "crossing_rate": float(np.count_nonzero(noise & (row_mask & col_mask))
                               / np.count_nonzero(row_mask & col_mask)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         "grid dimensions are outside 1..30")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and np.all((0 <= grid) & (grid <= 9)), "invalid ARC cells")
            key = (_grid_key(pair["input"]), _grid_key(pair["output"]))
            _require(key[0] not in seen_inputs and key not in seen_pairs, "duplicate input or pair")
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            info[split].append(_inspect_pair(pair))
    train_pitches = {x["pitch"] for x in info["train"]}
    _require(len(train_pitches) >= 2, "training must demonstrate multiple pitches")
    _require(len({(x["row_count"], x["col_count"]) for x in info["train"]}) >= 2,
             "training must demonstrate multiple line arrangements")
    _require(all(x["pitch"] in train_pitches for x in info["test"]),
             "test pitch is not demonstrated in training")
    train_shifted = any(x["row_phase"] != x["pitch"] - 1
                        or x["col_phase"] != x["pitch"] - 1 for x in info["train"])
    test_shifted = any(x["row_phase"] != x["pitch"] - 1
                       or x["col_phase"] != x["pitch"] - 1 for x in info["test"])
    _require(not test_shifted or train_shifted, "shifted test phase lacks training demonstration")
    _require(len({x["accent"] for x in info["train"]}) >= 2,
             "training needs at least two local accent examples")
    # Colors may change between pairs. Every local input still reveals all
    # three roles through a single uniquely compatible scaffold.


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    base = [int(x) for x in rng.permutation(10)]
    fixed_roles = seed % 2 == 0
    shifted = seed % 4 != 0
    distinct = seed % 3 == 0
    all_distinct = seed % 6 == 1 and num_train + num_test <= 10
    pair_seeds = [int(x) for x in rng.integers(0, 2**63 - 1, size=num_train + num_test)]
    pitches = [int(rng.choice(PITCHES)) for _ in range(num_train)]
    if pitches[1] == pitches[0]:
        pitches[1] = PITCHES[(PITCHES.index(pitches[0]) + 1 + int(rng.integers(4))) % len(PITCHES)]
    if num_train >= 3 and seed % 5 == 0:
        pitches[2] = int(rng.choice((4, 5)))
    layouts = []
    puzzle = {"train": [], "test": []}
    training_accents = []
    used_accents = []
    for index, pitch in enumerate(pitches):
        local_rng = np.random.default_rng(pair_seeds[index])
        rows = cols = None
        if num_train >= 3 and index == 2 and seed % 5 == 0:
            rows, cols = 5, 6
        elif num_train >= 2 and seed % 5 == 1 and index < 2:
            rows, cols = ((3, 4) if index == 0 else (4, 3))
        layout = _choose_train_layout(local_rng, pitch, shifted, rows, cols,
                                      force_shift=shifted and index == 0)
        if index == 1 and (layout[1][2], layout[2][2]) == (layouts[0][1][2], layouts[0][2][2]):
            other_counts = sorted({option[2] for option in _axis_options(pitch, shifted)}
                                  - {layout[2][2]})
            if other_counts:
                new_count = int(local_rng.choice(other_counts))
                layout = _choose_train_layout(local_rng, pitch, shifted,
                                              layout[1][2], new_count)
        colors = _choose_colors(local_rng, base, fixed_roles, used_accents,
                                all_distinct or distinct or index == 1)
        training_accents.append(colors[2])
        used_accents.append(colors[2])
        inp, out = _generate_pair(local_rng, layout, colors)
        layouts.append(layout)
        puzzle["train"].append({"input": inp, "output": out})
    for index in range(num_test):
        local_rng = np.random.default_rng(pair_seeds[num_train + index])
        mode = (seed + index) % 5
        layout = _choose_test_layout(local_rng, layouts, mode, shifted)
        reuse_accent = seed % 6 != 1 and (seed + index) % 2 == 0
        colors = _choose_colors(local_rng, base, fixed_roles,
                                used_accents if all_distinct else training_accents,
                                all_distinct or distinct or not reuse_accent)
        if reuse_accent:
            reused = [x for x in training_accents if x not in colors[:2]]
            if reused:
                colors = (colors[0], colors[1], reused[int(local_rng.integers(len(reused)))])
        inp, out = _generate_pair(local_rng, layout, colors)
        used_accents.append(colors[2])
        puzzle["test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts, arrangements, pitches = set(), set(), set(), set()
    phases, background_rates, line_rates, crossing_rates = set(), [], [], []
    role_schedules, test_novelty, accent_schedules, aspect_ratios = set(), set(), set(), set()
    test_relations, tails = set(), set()
    source_shapes_reused = patch_pairs = run_pairs = combined_cases = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_info = [_inspect_pair(pair) for pair in puzzle["test"]]
        roles = {(x["background"], x["structure"]) for x in train_info + test_info}
        role_schedules.add("fixed" if len(roles) == 1 else "varying")
        test_novelty.add(any(x["accent"] in {y["accent"] for y in train_info}
                             for x in test_info))
        accent_schedules.add(len({x["accent"] for x in train_info + test_info})
                             == len(train_info + test_info))
        for test in test_info:
            for train in train_info:
                if test["pitch"] != train["pitch"]:
                    continue
                row_change = test["row_count"] - train["row_count"]
                col_change = test["col_count"] - train["col_count"]
                if row_change == col_change == 0:
                    test_relations.add("retained")
                elif row_change > 0 and col_change == 0:
                    test_relations.add("rows")
                elif row_change == 0 and col_change > 0:
                    test_relations.add("columns")
                elif row_change > 0 and col_change > 0:
                    test_relations.add("both")
        for x in train_info + test_info:
            arrangements.add((x["row_count"], x["col_count"]))
            pitches.add(x["pitch"])
            phases.add((x["row_phase"], x["col_phase"]))
            background_rates.append(x["background_rate"])
            line_rates.append(x["line_rate"])
            crossing_rates.append(x["crossing_rate"])
            aspect_ratios.add("wide" if x["width"] > x["height"] else
                              "tall" if x["height"] > x["width"] else "square")
            tails.add((x["row_tail"], x["col_tail"]))
            source_shapes_reused += int((x["height"], x["width"]) in
                                        {(21, 17), (19, 13), (21, 24), (22, 27)})
            patch_pairs += int(x["background_patch"])
            run_pairs += int(x["line_run"])
            combined_cases += int(x["background_rate"] >= 0.15
                                  and (x["row_phase"] != x["pitch"] - 1
                                       or x["col_phase"] != x["pitch"] - 1)
                                  and len(roles) > 1)
    _require(len(signatures) == 42, "automatic seeds repeated puzzles")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(pitches == set(PITCHES), "pitches lack coverage")
    _require({(3, 4), (4, 3), (5, 6)} <= arrangements, "line arrangements lack coverage")
    _require(role_schedules == {"fixed", "varying"}, "role schedules lack coverage")
    _require(test_novelty == {False, True}, "test accents lack reuse or novelty")
    _require(accent_schedules == {False, True}, "accent schedules lack repeats or all-distinct cases")
    _require({"wide", "tall"} <= aspect_ratios, "aspect ratios lack coverage")
    _require(len(phases) >= 12, "phase locations lack coverage")
    _require(test_relations == {"retained", "rows", "columns", "both"},
             "test line-count regimes lack coverage")
    _require((0, 0) in tails and len(tails) >= 12, "trailing strips lack coverage")
    _require(source_shapes_reused > 0, "source canvas dimensions were unnecessarily banned")
    _require(patch_pairs >= 20 and run_pairs >= 20, "clustered corruption lacks coverage")
    _require(combined_cases > 0, "compatible variation axes do not combine")
    _require(max(background_rates) >= 0.15 and min(background_rates) <= 0.05,
             "background corruption lacks dense or sparse cases")
    _require(max(line_rates) >= 0.16 and min(line_rates) <= 0.08,
             "line corruption lacks diversity")
    _require(max(crossing_rates) >= 0.30 and min(crossing_rates) <= 0.10,
             "crossing corruption lacks diversity")
    for seed, nt, nv in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 2),
                         (201, None, 1), (202, None, 4), (203, 2, None),
                         (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, nv)
        validate_generated_puzzle(puzzle)
        _require(nt is None or len(puzzle["train"]) == nt, "explicit training count ignored")
        _require(nv is None or len(puzzle["test"]) == nv, "explicit test count ignored")
    _require(_puzzle_key(generate_puzzle(98765, 7, 3))
             == _puzzle_key(generate_puzzle(98765, 7, 3)), "generation is not deterministic")
    for nt, nv in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (True, 1)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count combination did not raise ValueError")
