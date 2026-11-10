import numpy as np


_COLORS = tuple(range(10))
_SOURCE_INPUTS = frozenset("".join(rows) for rows in (
    ("250060000", "255700601", "030001940", "070600000", "090001008", "000000000", "004000000", "000100004", "050000000"),
    ("900000060", "040705081", "020071445", "060040000", "830420097", "002302067", "404034707", "710000300", "320040000"),
    ("050080004", "000000300", "000021003", "010000300", "100100000", "000000080", "000000000", "009400000", "307002006"),
    ("300700970", "840660484", "170000400", "110910700", "000077000", "800170840", "070992100", "000000500", "000240800"),
    ("690010589", "290608090", "000009920", "926008068", "774070900", "007001574", "410075009", "990000100", "492000840"),
))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _palette(grid: np.ndarray) -> frozenset[int]:
    return frozenset(int(x) for x in np.unique(grid))


def _asymmetric(grid: np.ndarray) -> bool:
    rotations = [np.rot90(grid, k) for k in range(4)]
    return len({_key(x) for x in rotations + [np.fliplr(x) for x in rotations]}) == 8


def _crop(grid: np.ndarray) -> np.ndarray:
    return grid[:3, -3:]


def _exact_copies(grid: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    return [(r, c) for r in range(grid.shape[0] - 2)
            for c in range(grid.shape[1] - 2)
            if np.array_equal(grid[r:r + 3, c:c + 3], patch)]


def _d4_copies(grid: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    rotations = [np.rot90(patch, k) for k in range(4)]
    variants = rotations + [np.fliplr(x) for x in rotations]
    return [(r, c) for r in range(grid.shape[0] - 2)
            for c in range(grid.shape[1] - 2)
            if any(np.array_equal(grid[r:r + 3, c:c + 3], x) for x in variants)]


def _count(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    if num_train is not None:
        if not _count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _size(rng: np.random.Generator, variable: bool, split: str, index: int,
          train_sizes: set[tuple[int, int]]) -> tuple[int, int]:
    if not variable or (split == "train" and index == 0):
        return 9, 9
    if split == "train" and index > 1 and rng.random() < 0.38:
        return 9, 9
    if split == "test" and rng.random() < 0.16:
        return 9, 9
    for _ in range(32):
        shape = (int(rng.integers(6, 17)), int(rng.integers(6, 18)))
        if split == "train" and index == 1 and shape == (9, 9):
            continue
        if split == "test" and shape in train_sizes:
            continue
        return shape
    return (16, 17) if (16, 17) not in train_sizes else (15, 17)


def _groups(kind: str) -> list[list[tuple[int, int]]]:
    groups = []
    seen = set()
    for r in range(3):
        for c in range(3):
            if (r, c) in seen:
                continue
            if kind == "mirror_lr":
                group = {(r, c), (r, 2 - c)}
            elif kind == "mirror_tb":
                group = {(r, c), (2 - r, c)}
            elif kind == "rotate":
                group = {(r, c), (2 - r, 2 - c)}
            elif kind == "repeat_rows":
                group = {(rr, c) for rr in range(3)}
            elif kind == "repeat_cols":
                group = {(r, cc) for cc in range(3)}
            else:
                group = {(r, c)}
            seen.update(group)
            groups.append(sorted(group))
    return groups


def _make_payload(rng: np.random.Generator, allowed: tuple[int, ...],
                  required: int | None = None,
                  reuse: frozenset[int] | None = None,
                  teaching: bool = False) -> np.ndarray:
    nonzero_allowed = [c for c in allowed if c != 0]
    for _ in range(96):
        if reuse is not None:
            selected = list(reuse)
            kind = "free"
            zero_count = (9 if selected == [0] else
                          int(rng.integers(1, 10 - len(selected))) if 0 in selected else 0)
        else:
            kind = "free" if teaching else str(rng.choice(
                ("free", "mirror_lr", "mirror_tb", "rotate", "repeat_rows", "repeat_cols"),
                p=(0.54, 0.11, 0.11, 0.08, 0.08, 0.08)))
            blank_mode = str(rng.choice(("sparse", "middle", "dense", "zero_free"),
                                        p=(0.28, 0.36, 0.20, 0.16)))
            if teaching and rng.random() < 0.48:
                blank_mode = "sparse"
            if blank_mode == "sparse":
                zero_count = int(rng.integers(5, 7 if teaching else 9))
            elif blank_mode == "middle":
                zero_count = int(rng.integers(2, 6))
            elif blank_mode == "dense":
                zero_count = int(rng.integers(0, 3))
            else:
                zero_count = 0
            if teaching:
                zero_count = max(1, zero_count)
            maximum = min(7, len(nonzero_allowed), 9 - zero_count)
            minimum = 2 if teaching else 1
            if maximum < minimum:
                continue
            n_colors = int(rng.integers(minimum, maximum + 1))
            chosen = [required] if required is not None else []
            extras = [c for c in nonzero_allowed if c not in chosen]
            chosen += [int(x) for x in rng.choice(extras, size=n_colors - len(chosen), replace=False)]
            selected = chosen + ([0] if zero_count else [])

        patch = np.zeros((3, 3), dtype=np.int8)
        groups = _groups(kind)
        if kind == "free":
            positions = [int(x) for x in rng.permutation(9)]
            n_nonzero = 9 - zero_count
            colors = [c for c in selected if c != 0]
            if n_nonzero < len(colors):
                continue
            values = colors + [int(x) for x in rng.choice(colors, size=n_nonzero - len(colors))]
            values = [int(x) for x in rng.permutation(values)]
            for pos, color in zip(positions[:n_nonzero], values):
                patch[pos // 3, pos % 3] = color
        else:
            color_pool = [c for c in selected if c != 0]
            p_zero = zero_count / 9
            for group in groups:
                color = 0 if zero_count and rng.random() < p_zero else int(rng.choice(color_pool))
                for r, c in group:
                    patch[r, c] = color
            if required is not None and required not in patch:
                for r, c in groups[int(rng.integers(len(groups)))]:
                    patch[r, c] = required
        if required is not None and required not in patch:
            continue
        if reuse is not None and _palette(patch) != reuse:
            continue
        if teaching and not _asymmetric(patch):
            continue
        return patch
    fallback = np.zeros((3, 3), dtype=np.int8)
    base = [c for c in allowed if c != 0]
    fallback[0, 0] = required if required is not None else base[0]
    fallback[1, 2] = base[-1]
    fallback[2, 1] = base[0]
    if teaching and not _asymmetric(fallback):
        fallback[0, 2] = base[-2]
    return fallback


def _decoy_position(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    options = [(0, 0), (h - 3, 0), (h - 3, w - 3), (0, w - 6), (3, w - 3)]
    return options[int(rng.integers(len(options)))]


def _make_input(rng: np.random.Generator, shape: tuple[int, int],
                payload: np.ndarray, teaching: bool, test: bool,
                force_outside: int | None) -> np.ndarray:
    h, w = shape
    full_palette = shape == (9, 9) and rng.random() < (0.24 if not test else 0.09)
    candidates = [c for c in _palette(payload) if c != 0 and c != force_outside]
    exclusive = int(rng.choice(candidates)) if candidates and not full_palette and rng.random() < 0.34 else None
    pool = [c for c in _COLORS if c != exclusive]
    if full_palette:
        outside_colors = list(_COLORS)
    else:
        n = int(rng.integers(2, len(pool) + 1))
        chosen = [force_outside] if force_outside is not None else []
        if rng.random() < 0.83 and 0 not in chosen:
            chosen.append(0)
        n = max(n, len(chosen))
        remaining = [c for c in pool if c not in chosen]
        chosen += [int(x) for x in rng.choice(remaining, size=n - len(chosen), replace=False)]
        outside_colors = chosen
    nonzero = [c for c in outside_colors if c != 0]
    zero_probability = float(rng.uniform(0.08, 0.89)) if 0 in outside_colors else 0.0
    values = np.asarray([0] + nonzero if 0 in outside_colors else nonzero, dtype=np.int8)
    probs = ([zero_probability] + [(1 - zero_probability) / len(nonzero)] * len(nonzero)
             if 0 in outside_colors else None)
    grid = rng.choice(values, size=shape, p=probs).astype(np.int8)

    style = str(rng.choice(("iid", "runs", "blocks", "mixed")))
    if style in ("runs", "mixed"):
        for _ in range(int(rng.integers(2, 7))):
            row, col = int(rng.integers(h)), int(rng.integers(w))
            length, color = int(rng.integers(2, 7)), int(rng.choice(outside_colors))
            if rng.random() < 0.5:
                grid[row, col:min(w, col + length)] = color
            else:
                grid[row:min(h, row + length), col] = color
    if style in ("blocks", "mixed"):
        for _ in range(int(rng.integers(1, 5))):
            row, col = int(rng.integers(h)), int(rng.integers(w))
            color = int(rng.choice(outside_colors))
            grid[row:min(h, row + int(rng.integers(2, 5))),
                 col:min(w, col + int(rng.integers(2, 5)))] = color
    if rng.random() < 0.68:
        color = int(rng.choice(outside_colors))
        if rng.random() < 0.5:
            grid[:3, w - 4] = color
        else:
            grid[3, w - 3:] = color
    grid[:3, -3:] = payload

    protected = set()
    if not teaching and rng.random() < (0.62 if test else 0.39):
        r, c = _decoy_position(rng, shape)
        kind = str(rng.choice(("exact", "rotate", "near"), p=(0.55, 0.23, 0.22)))
        decoy = payload.copy()
        if kind == "rotate":
            decoy = np.rot90(decoy, int(rng.integers(1, 4))).copy()
        elif kind == "near":
            rr, cc = int(rng.integers(3)), int(rng.integers(3))
            decoy[rr, cc] = int(rng.choice([x for x in _COLORS if x != decoy[rr, cc]]))
        grid[r:r + 3, c:c + 3] = decoy
        protected = {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)}
        grid[:3, -3:] = payload

    needed = set(_COLORS) - set(_palette(grid)) if full_palette else set()
    if force_outside is not None and not np.any(grid[3:, :] == force_outside) and not np.any(grid[:3, :w - 3] == force_outside):
        needed.add(force_outside)
    positions = [(r, c) for r in range(h) for c in range(w)
                 if not (r < 3 and c >= w - 3) and (r, c) not in protected]
    rng.shuffle(positions)
    for color, (r, c) in zip(sorted(needed), positions):
        grid[r, c] = color
    if teaching:
        for _ in range(h * w):
            duplicates = [p for p in _d4_copies(grid, payload) if p != (0, w - 3)]
            if not duplicates:
                break
            r, c = duplicates[0]
            cells = [(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)
                     if not (rr < 3 and cc >= w - 3)]
            rr, cc = cells[-1]
            grid[rr, cc] = (int(grid[rr, cc]) + 1) % 10
        if force_outside is not None and force_outside not in grid:
            # This color is absent from the target, so restoring it cannot
            # create a transformed copy of the target elsewhere.
            grid[h - 1, 0] = force_outside
    return grid


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    variable_size = bool(rng.random() < 0.63)
    heldout = int(rng.integers(1, 10)) if rng.random() < 0.57 else None
    puzzle = {"train": [], "test": []}
    train_sizes = set()
    prior_train_palettes = []
    for index in range(num_train):
        shape = _size(rng, variable_size, "train", index, train_sizes)
        allowed = tuple(c for c in _COLORS if c != heldout)
        reuse = (prior_train_palettes[int(rng.integers(len(prior_train_palettes)))]
                 if index and rng.random() < 0.30 else None)
        payload = _make_payload(rng, allowed, reuse=reuse, teaching=index == 0)
        input_grid = _make_input(rng, shape, payload, index == 0, False,
                                 heldout if index == 0 else None)
        puzzle["train"].append({"input": input_grid, "output": _crop(input_grid).copy()})
        train_sizes.add(shape)
        prior_train_palettes.append(_palette(payload))
    demonstrated = set().union(*prior_train_palettes)
    for index in range(num_test):
        shape = _size(rng, variable_size, "test", index, train_sizes)
        required = heldout if index == 0 else None
        allowed = tuple(sorted(demonstrated | ({heldout} if heldout is not None else set())))
        reuse = (prior_train_palettes[int(rng.integers(len(prior_train_palettes)))]
                 if required is None and rng.random() < 0.33 else None)
        payload = _make_payload(rng, allowed, required=required, reuse=reuse)
        input_grid = _make_input(rng, shape, payload, False, True, None)
        puzzle["test"].append({"input": input_grid, "output": _crop(input_grid).copy()})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    inputs, pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
            _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
            _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
            _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
        grid, output = pair["input"], pair["output"]
        _require(grid.shape[0] >= 6 and grid.shape[1] >= 6, "input too small")
        _require(output.shape == (3, 3) and np.array_equal(output, _crop(grid)), "upper-right crop")
        if grid.shape == (9, 9):
            _require("".join(str(int(x)) for x in grid.flat) not in _SOURCE_INPUTS, "source copy")
        input_key = _key(grid)
        pair_key = (input_key, _key(output))
        _require(input_key not in inputs and pair_key not in pairs, "duplicate pair or input")
        inputs.add(input_key)
        pairs.add(pair_key)
    first = puzzle["train"][0]
    _require(_asymmetric(first["output"]), "training must teach orientation")
    _require(_d4_copies(first["input"], first["output"]) ==
             [(0, first["input"].shape[1] - 3)], "training must teach unique corner")
    train_sizes = {pair["input"].shape for pair in puzzle["train"]}
    if any(pair["input"].shape != (9, 9) for pair in puzzle["test"]):
        _require(len(train_sizes) >= 2, "training must show size variation")
    train_input_colors = set().union(*(_palette(p["input"]) for p in puzzle["train"]))
    train_output_colors = set().union(*(_palette(p["output"]) for p in puzzle["train"]))
    unseen = set().union(*(_palette(p["output"]) for p in puzzle["test"])) - train_output_colors
    _require(len(unseen) <= 1 and unseen <= train_input_colors,
             "unseen output colors need input exposure")


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((split, _key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    _require(_fingerprint(generate_puzzle(1701)) == _fingerprint(generate_puzzle(1701)),
             "automatic determinism")
    _require(_fingerprint(generate_puzzle(2701, 7, 3)) ==
             _fingerprint(generate_puzzle(2701, 7, 3)), "explicit determinism")
    _require(_fingerprint(generate_puzzle(2701, 7, 3)) !=
             _fingerprint(generate_puzzle(2702, 7, 3)), "seed diversity")
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                        (2, None), (10, None)):
        puzzle = generate_puzzle(3100 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (True, 1)):
        _expect_value_error(lambda a=train, b=test: generate_puzzle(1, a, b))
    puzzles = [generate_puzzle(seed) for seed in range(4200, 4242)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3, "count diversity")
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "puzzle diversity")
    all_pairs = [pair for p in puzzles for split in ("train", "test") for pair in p[split]]
    _require(len({_key(p["input"]) for p in all_pairs}) == len(all_pairs), "input diversity")
    _require(len({p["input"].shape for p in all_pairs}) >= 25, "size diversity")
    _require(len({tuple(int(x != 0) for x in p["input"].flat) for p in all_pairs})
             >= int(0.95 * len(all_pairs)), "distractor layout diversity")
    _require(len({_key(p["output"]) for p in all_pairs}) >= int(0.80 * len(all_pairs)),
             "payload arrangement diversity")
    holdout = ordinary = symmetric = sparse = zero_free = test_decoys = 0
    full_inputs = palette_reuse = variable_puzzles = 0
    fixed_puzzles = source_sparse = few_colors = repeated_rows = repeated_cols = 0
    exclusive_payloads = train_decoys = varied_test_palettes = nine_color_tests = 0
    varied_adjacent = varied_density = holdout_sparse = holdout_decoy = non9_decoy = 0
    input_palette_counts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_outputs = set().union(*(_palette(p["output"]) for p in puzzle["train"]))
        test_outputs = set().union(*(_palette(p["output"]) for p in puzzle["test"]))
        unseen = test_outputs - train_outputs
        holdout += bool(unseen)
        ordinary += not unseen
        variable_puzzles += len({p["input"].shape for p in puzzle["train"]}) > 1
        fixed_puzzles += all(p["input"].shape == (9, 9)
                             for p in puzzle["train"] + puzzle["test"])
        test_palettes = [_palette(p["input"]) for p in puzzle["test"]]
        varied_test_palettes += len(test_palettes) > 1 and len(set(test_palettes)) > 1
        nine_color_tests += sum(len(p) == 9 for p in test_palettes)
        signatures = [_palette(p["output"]) for p in puzzle["train"] + puzzle["test"]]
        palette_reuse += len(signatures) != len(set(signatures))
        adjacent = set()
        densities = []
        for pair in puzzle["train"] + puzzle["test"]:
            grid, output = pair["input"], pair["output"]
            _require(np.array_equal(output, grid[:3, -3:]), "independent crop check")
            zeros = int(np.count_nonzero(output == 0))
            symmetric += not _asymmetric(output)
            sparse += zeros >= 6
            source_sparse += zeros >= 6 and len(_palette(output)) <= 3
            few_colors += len(_palette(output)) in (2, 3)
            zero_free += 0 not in output
            repeated_rows += any(np.array_equal(output[i], output[j])
                                 for i in range(3) for j in range(i + 1, 3))
            repeated_cols += any(np.array_equal(output[:, i], output[:, j])
                                 for i in range(3) for j in range(i + 1, 3))
            full_inputs += len(_palette(grid)) == 10
            input_palette_counts.add(len(_palette(grid)))
            outside = grid.copy()
            outside[:3, -3:] = 0
            exclusive_payloads += any(c != 0 and c not in _palette(outside)
                                      for c in _palette(output))
            densities.append(float(np.count_nonzero(outside)) / (grid.size - 9))
            adjacent.add((tuple(int(x) for x in grid[:3, -4]),
                          tuple(int(x) for x in grid[3, -3:])))
        varied_adjacent += len(adjacent) > 1
        varied_density += max(densities) - min(densities) >= 0.2
        for pair in puzzle["train"]:
            train_decoys += len(_exact_copies(pair["input"], pair["output"])) > 1
        for pair in puzzle["test"]:
            decoy = len(_exact_copies(pair["input"], pair["output"])) > 1
            test_decoys += decoy
            holdout_sparse += bool(unseen and np.count_nonzero(pair["output"] == 0) >= 6)
            holdout_decoy += bool(unseen and decoy)
            non9_decoy += bool(pair["input"].shape != (9, 9) and decoy)
    _require(holdout >= 10 and ordinary >= 10, "holdout regimes")
    _require(symmetric >= 15 and sparse >= 15 and zero_free >= 12, "payload diversity")
    _require(test_decoys >= 12, "exact decoy diversity")
    _require(full_inputs >= 12 and palette_reuse >= 5, "palette diversity")
    _require(variable_puzzles >= 15, "variable-size training diversity")
    _require(fixed_puzzles >= 8 and len(input_palette_counts) >= 6,
             "source-size and input-palette diversity")
    _require(5 <= nine_color_tests < sum(len(p["test"]) for p in puzzles) - 5 and
             varied_test_palettes >= 8, "test palette variation")
    _require(source_sparse >= 15 and few_colors >= 30 and
             repeated_rows >= 10 and repeated_cols >= 10, "payload repetition")
    _require(exclusive_payloads >= 15 and train_decoys >= 8,
             "distractor and palette variation")
    _require(varied_adjacent >= 35 and varied_density >= 25,
             "within-puzzle distractor variation")
    _require(holdout_sparse >= 2 and holdout_decoy >= 2 and non9_decoy >= 2,
             "combined variation")
