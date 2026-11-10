"""Generate bottom-up strands that step right around head-on blockers."""

import numpy as np


_ORIGINAL_INPUT_ROWS = {
    ("0000000000", "0000000000", "0000000000", "0000050000", "0000000000", "0500000000", "0000000050", "0000000000", "0000000000", "0200200200"),
    ("0000000000", "0000000000", "0000000000", "0000005000", "0000000000", "0050000000", "0000000000", "0000000000", "0000000000", "0200202000"),
    ("0000000000", "0000000000", "0000000500", "0000000000", "0500000000", "0000000000", "0000500000", "0000000000", "0000000000", "0200200020"),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        if rng.random() < 0.27:
            return 2, 1
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _palette(rng):
    return tuple(int(v) for v in rng.choice(10, size=3, replace=False))


def _infer_roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 3, "input must visibly identify three color roles")
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be dominant")
    background = int(values[int(np.argmax(counts))])
    edge_colors = {int(v) for v in grid[-1] if int(v) != background}
    _require(len(edge_colors) == 1, "bottom edge must identify the strand")
    strand = edge_colors.pop()
    blocker = ({int(v) for v in values} - {background, strand}).pop()
    _require(np.all((grid[-1] == background) | (grid[-1] == strand)), "bottom edge contains a blocker")
    _require(np.all(grid[:-1] != strand), "input strand extends above its source edge")
    sources = np.flatnonzero(grid[-1] == strand)
    _require(1 <= len(sources) <= 5, "invalid source count")
    _require(len(sources) < 2 or np.all(np.diff(sources) >= 2), "sources must remain distinct")
    blockers = [(int(r), int(c)) for r, c in np.argwhere(grid == blocker)]
    _require(bool(blockers), "input has no blocker")
    for i, (r, c) in enumerate(blockers):
        for rr, cc in blockers[i + 1:]:
            _require(max(abs(r - rr), abs(c - cc)) > 1, "blockers must be isolated")
    return background, strand, blocker


def _trace(grid, roles):
    """Trace from visible input roles, rejecting blocked steps and path overlap."""
    background, strand, blocker = roles
    height, width = grid.shape
    sources = tuple(int(c) for c in np.flatnonzero(grid[-1] == strand))
    output = grid.copy()
    owner = np.full(grid.shape, -1, dtype=np.int16)
    for index, col in enumerate(sources):
        owner[-1, col] = index
    hits = []
    for index, source in enumerate(sources):
        col = source
        strand_hits = []
        for row in range(height - 2, -1, -1):
            value = int(grid[row, col])
            if value == blocker:
                new_col = col + 1
                _require(new_col < width, "right step leaves grid")
                _require(int(grid[row, new_col]) == background, "cell beside blocker is not clear")
                _require(int(grid[row + 1, new_col]) == background, "horizontal step is not clear")
                _require(owner[row, new_col] == -1 and owner[row + 1, new_col] == -1, "paths overlap")
                output[row + 1, new_col] = strand
                output[row, new_col] = strand
                owner[row + 1, new_col] = index
                owner[row, new_col] = index
                strand_hits.append((row, col))
                col = new_col
            else:
                _require(value == background, "path meets an unexpected input cell")
                _require(owner[row, col] == -1, "paths overlap")
                output[row, col] = strand
                owner[row, col] = index
        hits.append(tuple(strand_hits))
    return output, {"sources": sources, "hits": tuple(hits), "owner": owner}


def _far_enough(point, blockers):
    return all(max(abs(point[0] - r), abs(point[1] - c)) > 1 for r, c in blockers)


def _inert_groups(grid, output, roles, info, blockers):
    background, strand, _ = roles
    height, width = grid.shape
    groups = {"straight": [], "pre": [], "post": [], "distant": [], "other": []}
    path_cells = np.argwhere(info["owner"] >= 0)
    for row in range(height - 1):
        for col in range(width):
            point = (row, col)
            if int(grid[row, col]) != background or int(output[row, col]) != background or not _far_enough(point, blockers):
                continue
            adjacent = []
            for neighbor in (col - 1, col + 1):
                if not 0 <= neighbor < width or int(output[row, neighbor]) != strand:
                    continue
                index = int(info["owner"][row, neighbor])
                if index < 0:
                    continue
                strand_hits = info["hits"][index]
                if not strand_hits:
                    adjacent.append("straight")
                elif row > strand_hits[0][0]:
                    adjacent.append("pre")
                elif row < strand_hits[-1][0]:
                    adjacent.append("post")
            if adjacent:
                for name in set(adjacent):
                    groups[name].append(point)
            elif all(max(abs(row - int(r)), abs(col - int(c))) >= 2 for r, c in path_cells):
                groups["distant"].append(point)
            else:
                groups["other"].append(point)
    return groups


def _shape(rng):
    height = 10 if rng.random() < 0.22 else int(rng.integers(7, 25))
    if rng.random() < 0.22:
        width = 10
    elif rng.random() < 0.55:
        width = int(rng.integers(7, 15))
    else:
        width = int(rng.integers(15, 31))
    return height, width


def _mode(rng, regime):
    if regime == "source_train":
        return str(rng.choice(("none", "single", "multi", "repeat", "multi_repeat"), p=(0.18, 0.38, 0.24, 0.14, 0.06)))
    return str(rng.choice(("none", "single", "multi", "repeat", "multi_repeat"), p=(0.12, 0.28, 0.34, 0.16, 0.10)))


def _plan_hits(rng, mode, sources, height, width):
    eligible = [i for i, col in enumerate(sources) if col < width - 1]
    repeat_eligible = [i for i, col in enumerate(sources) if col < width - 2]
    if mode == "none":
        return [[] for _ in sources]
    if mode == "single" and not eligible:
        return None
    if mode in ("repeat", "multi_repeat") and not repeat_eligible:
        return None
    if mode in ("multi", "multi_repeat") and len(eligible) < 2:
        return None
    rows = [[] for _ in sources]
    if mode in ("repeat", "multi_repeat"):
        repeat_index = int(rng.choice(repeat_eligible))
        lower = int(rng.integers(2, height - 1))
        upper = lower - 2 if rng.random() < 0.38 else int(rng.integers(0, lower - 1))
        rows[repeat_index] = [lower, upper]
    if mode == "single":
        index = int(rng.choice(eligible))
        rows[index] = [int(rng.integers(0, height - 1))]
    elif mode == "multi":
        count = int(rng.integers(2, min(len(eligible), 4) + 1))
        for index in rng.choice(eligible, size=count, replace=False):
            rows[int(index)] = [int(rng.integers(0, height - 1))]
    elif mode == "multi_repeat":
        others = [i for i in eligible if i != repeat_index]
        count = int(rng.integers(1, min(len(others), 2) + 1))
        for index in rng.choice(others, size=count, replace=False):
            rows[int(index)] = [int(rng.integers(0, height - 1))]
    return rows


def _generate_pair(rng, regime, mode, roles, required_inert=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    background, strand, blocker = roles
    for _ in range(800):
        height, width = _shape(rng)
        max_sources = min(5, (width + 1) // 2)
        minimum = 2 if required_inert == "straight" else 1
        if mode in ("multi", "multi_repeat"):
            minimum = max(minimum, 2)
        count = int(rng.integers(minimum, max_sources + 1))
        bases = np.sort(rng.choice(width - count + 1, size=count, replace=False))
        sources = tuple(int(bases[i]) + i for i in range(count))
        planned_rows = _plan_hits(rng, mode, sources, height, width)
        if planned_rows is None:
            continue
        blockers = set()
        valid = True
        for i, rows in enumerate(planned_rows):
            for shift, row in enumerate(rows):
                point = (row, sources[i] + shift)
                if point[1] + 1 >= width or not _far_enough(point, blockers):
                    valid = False
                    break
                blockers.add(point)
            if not valid:
                break
        if not valid:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        grid[-1, list(sources)] = strand
        for row, col in blockers:
            grid[row, col] = blocker
        try:
            output, info = _trace(grid, roles)
        except AssertionError:
            continue
        if {point for strand_hits in info["hits"] for point in strand_hits} != blockers:
            continue
        if required_inert == "straight" and not any(not hits for hits in info["hits"]):
            continue
        desired_inert = int(rng.integers(0, 7))
        if not blockers:
            desired_inert = max(1, desired_inert)
        if required_inert is not None:
            desired_inert = max(1, desired_inert)
        for inert_index in range(desired_inert):
            groups = _inert_groups(grid, output, roles, info, blockers)
            if inert_index == 0 and required_inert is not None:
                name = required_inert
            else:
                name = str(rng.choice(("straight", "pre", "post", "distant", "other")))
            candidates = groups[name]
            if not candidates and required_inert == name and inert_index == 0:
                valid = False
                break
            if not candidates:
                candidates = [p for group in groups.values() for p in group]
            if not candidates:
                break
            point = candidates[int(rng.integers(0, len(candidates)))]
            blockers.add(point)
            grid[point] = blocker
        if not valid or not blockers:
            continue
        try:
            actual, final_info = _trace(grid, roles)
            _infer_roles(grid)
        except AssertionError:
            continue
        if final_info["hits"] != info["hits"] or not np.array_equal(final_info["owner"], info["owner"]):
            continue
        if _rows(grid) in _ORIGINAL_INPUT_ROWS:
            continue
        return grid, actual
    raise RuntimeError("could not construct a valid strand pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Orientation and local turning rule are the shared convention. Colors are
    # visible semantic roles in each input and can therefore vary by pair.
    repeat_witness = bool(rng.random() < 0.35)
    test_modes = [_mode(rng, "target_test") for _ in range(test_count)]
    if repeat_witness:
        test_modes[int(rng.integers(0, test_count))] = str(rng.choice(("repeat", "multi_repeat")))
    if any(mode in ("repeat", "multi_repeat") for mode in test_modes):
        repeat_witness = True
    train_modes = ["single", "repeat" if repeat_witness else _mode(rng, "source_train")]
    train_modes += [_mode(rng, "source_train") for _ in range(train_count - 2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    training_palettes = set()
    for split, modes in (("train", train_modes), ("test", test_modes)):
        for index, mode in enumerate(modes):
            for _ in range(80):
                roles = _palette(rng)
                if split == "train" and index == 1 and roles in training_palettes:
                    continue
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                required = "straight" if split == "train" and index == 0 else None
                inp, out = _generate_pair(pair_rng, "source_train" if split == "train" else "target_test", mode, roles, required)
                key = _signature(inp)
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                if split == "train":
                    training_palettes.add(roles)
                break
            else:
                raise RuntimeError("could not construct a distinct puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _adjacent_inert(grid, output, roles, info):
    blocker = roles[2]
    hits = {p for strand_hits in info["hits"] for p in strand_hits}
    for row, col in np.argwhere(grid == blocker):
        row, col = int(row), int(col)
        if (row, col) in hits:
            continue
        for neighbor in (col - 1, col + 1):
            if 0 <= neighbor < grid.shape[1] and int(output[row, neighbor]) == roles[1]:
                index = int(info["owner"][row, neighbor])
                if index >= 0 and not info["hits"][index]:
                    return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys are invalid")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs, train_roles, train_info, test_info = set(), set(), set(), [], []
    adjacency_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys are invalid")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a two-dimensional array")
                _require(all(1 <= size <= 30 for size in grid.shape), "grid dimensions out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid grid transformation")
            ikey, okey = _signature(inp), _signature(out)
            _require(ikey not in seen_inputs and (ikey, okey) not in seen_pairs, "duplicate pair")
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            _require(_rows(inp) not in _ORIGINAL_INPUT_ROWS, "source pair was copied")
            roles = _infer_roles(inp)
            expected, info = _trace(inp, roles)
            _require(np.array_equal(expected, out), "output violates strand rule")
            _require(all(np.array_equal(inp == color, out == color) for color in (roles[2],)), "blocker was changed")
            if split == "train":
                train_roles.add(roles)
                train_info.append(info)
                adjacency_witness |= _adjacent_inert(inp, out, roles, info)
            else:
                test_info.append(info)
    _require(len(train_roles) >= 2, "training does not demonstrate pair-level palette variation")
    _require(any(sum(len(h) for h in info["hits"]) > 0 for info in train_info), "training has no real detour")
    _require(adjacency_witness, "training has no inert adjacent blocker witness")
    if any(any(len(h) >= 2 for h in info["hits"]) for info in test_info):
        _require(any(any(len(h) >= 2 for h in info["hits"]) for info in train_info), "repeated test turn lacks training witness")


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(202, 6, 3)
    _require(_equal(a, generate_puzzle(202, 6, 3)), "explicit generation is nondeterministic")
    _require(not _equal(a, generate_puzzle(203, 6, 3)), "seeds have no effect")
    for seed, train, test in ((300, 2, 1), (301, 10, 1), (302, 10, 4), (400, None, 1), (401, None, 4), (402, 2, None), (403, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require(train is None or len(puzzle["train"]) == train, "train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "test count ignored")
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(500, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    train_counts, test_counts, palettes, shapes, sources, gaps, collision_totals, layouts = set(), set(), set(), set(), set(), set(), set(), set()
    within_palette_changes = lean = repeats_train = repeats_test = zero_hits = multi_hits = edge_sources = compact = top_hits = low_hits = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_equal(puzzle, generate_puzzle(seed)), "automatic generation is nondeterministic")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        lean += (len(puzzle["train"]), len(puzzle["test"])) == (2, 1)
        within_palette_changes += len({_infer_roles(p["input"]) for p in puzzle["train"]}) > 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                roles = _infer_roles(inp)
                expected, info = _trace(inp, roles)
                _require(np.array_equal(expected, pair["output"]), "trace reconstruction failed")
                palettes.add(roles)
                shapes.add(inp.shape)
                sources.add(len(info["sources"]))
                gaps.update(int(v) for v in np.diff(info["sources"]))
                total = sum(len(h) for h in info["hits"])
                collision_totals.add(total)
                zero_hits += total == 0
                multi_hits += sum(bool(h) for h in info["hits"]) >= 2
                edge_sources += 0 in info["sources"] or inp.shape[1] - 1 in info["sources"]
                compact += inp.shape[1] == 10
                top_hits += any(r == 0 for h in info["hits"] for r, _ in h)
                low_hits += any(r == inp.shape[0] - 2 for h in info["hits"] for r, _ in h)
                layouts.add((inp.shape, info["sources"], tuple((int(r), int(c)) for r, c in np.argwhere(inp == roles[2]))))
                if any(len(h) >= 2 for h in info["hits"]):
                    if split == "train":
                        repeats_train += 1
                    else:
                        repeats_test += 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3 and lean >= 3, "count diversity is insufficient")
    _require(len(palettes) >= 30 and within_palette_changes == 42, "palette diversity is insufficient")
    _require(len(shapes) >= 40 and compact >= 8, "shape diversity is insufficient")
    _require(sources == {1, 2, 3, 4, 5} and 2 in gaps and len(gaps) >= 5, "source layout diversity is insufficient")
    _require(len(layouts) >= 100 and edge_sources >= 8, "layout diversity is insufficient")
    _require(0 in collision_totals and 1 in collision_totals and max(collision_totals) >= 2, "collision regimes are insufficient")
    _require(zero_hits >= 5 and multi_hits >= 5 and repeats_train >= 1 and repeats_test >= 1, "turn regimes are insufficient")
    _require(top_hits >= 1 and low_hits >= 1, "boundary turn heights are absent")
