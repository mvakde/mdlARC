import numpy as np


_SOURCE_CONFIGURATIONS = {(3, 5), (15, 6), (7, 8), (10, 3)}
_ALL_SIZES = tuple(range(3, 31))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _coerce_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _optional_count(num_train, "num_train")
    test = _optional_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = test + 1
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = test + 1
        train = minimum + seed % (11 - minimum)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _sample_sizes(rng: np.random.Generator, train_count: int, test_count: int) -> tuple[list[int], list[int]]:
    # Modes: source-like odd/even, reversed, mixed, odd/odd, and even/even.
    mode = int(rng.integers(0, 5))
    if mode == 0:
        train_parities = [1] * train_count
        test_parities = [0] * test_count
    elif mode == 1:
        train_parities = [0] * train_count
        test_parities = [1] * test_count
    elif mode == 2:
        first = int(rng.integers(0, 2))
        train_parities = [first, 1 - first] + [int(rng.integers(0, 2)) for _ in range(train_count - 2)]
        test_parities = [int(rng.integers(0, 2)) for _ in range(test_count)]
    else:
        parity = mode - 3
        train_parities = [parity] * train_count
        test_parities = [parity] * test_count

    train_sizes = []
    for index, parity in enumerate(train_parities):
        if index == 0:
            pool = [s for s in range(3, 11) if s % 2 == parity]
        elif index == 1:
            pool = [s for s in range(20, 31) if s % 2 == parity]
        else:
            pool = [s for s in _ALL_SIZES if s % 2 == parity and s not in train_sizes]
        train_sizes.append(int(rng.choice(pool)))
    train_sizes = [train_sizes[int(i)] for i in rng.permutation(train_count)]

    test_sizes = []
    for parity in test_parities:
        reusable = [s for s in train_sizes if s % 2 == parity and s not in test_sizes]
        fresh = [s for s in _ALL_SIZES if s % 2 == parity and s not in train_sizes and s not in test_sizes]
        if reusable and (not fresh or rng.random() < 0.35):
            pool = reusable
        else:
            pool = fresh
        test_sizes.append(int(rng.choice(pool)))
    return train_sizes, test_sizes


def _sample_lines(rng: np.random.Generator, train_count: int, test_count: int,
                  diagonal: int, base: int) -> tuple[list[int], list[int]]:
    available = [int(v) for v in rng.permutation([c for c in range(10) if c not in (diagonal, base)])]
    palette = available[:min(train_count, 4)]
    train = palette[:] + [int(rng.choice(palette)) for _ in range(train_count - len(palette))]
    train = [train[int(i)] for i in rng.permutation(train_count)]
    novel = [c for c in available if c not in palette]
    mode = int(rng.integers(0, 4))
    test = []
    for index in range(test_count):
        if mode == 0:
            choices = novel
        elif mode == 1:
            choices = palette
        elif mode == 2 and test_count >= 2:
            choices = novel if index % 2 == 0 else palette
        else:
            choices = novel if rng.random() < 0.5 else palette
        test.append(int(rng.choice(choices)))
    return train, test


def _sample_orientations(rng: np.random.Generator, train_count: int,
                         test_count: int) -> tuple[list[int], list[int]]:
    if int(rng.integers(0, 3)) == 0:
        edge = int(rng.integers(0, 4))
        return [edge] * train_count, [edge] * test_count
    train = [int(rng.integers(0, 4))]
    train.append(int(rng.choice([edge for edge in range(4) if edge != train[0]])))
    train += [int(rng.integers(0, 4)) for _ in range(train_count - 2)]
    train = [train[int(i)] for i in rng.permutation(train_count)]
    test = []
    for _ in range(test_count):
        unseen = [edge for edge in range(4) if edge not in train]
        pool = unseen if unseen and rng.random() < 0.5 else range(4)
        test.append(int(rng.choice(pool)))
    return train, test


def _left_pair(side: int, background: int, line: int, diagonal: int,
               base: int, reflected: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((side, side), background, dtype=np.int8)
    inp[:, 0] = line
    out = inp.copy()
    for row in range(side - 1):
        out[row, side - 1 - row] = diagonal
    out[-1, 1:] = base
    if reflected:
        inp = np.flipud(inp)
        out = np.flipud(out)
    return inp, out


def _rotate_pair(inp: np.ndarray, out: np.ndarray, edge: int) -> tuple[np.ndarray, np.ndarray]:
    return np.rot90(inp, -edge).copy(), np.rot90(out, -edge).copy()


def _input_signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _would_copy_source(side: int, background: int, line: int, diagonal: int,
                       base: int, reflected: int, edge: int) -> bool:
    return (edge == 0 and reflected == 0 and background == 0 and
            diagonal == 2 and base == 4 and (side, line) in _SOURCE_CONFIGURATIONS)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _coerce_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    roles = [int(v) for v in rng.permutation(10)]
    diagonal, base = roles[:2]
    reflected = int(rng.integers(0, 2))

    train_sizes, test_sizes = _sample_sizes(rng, train_count, test_count)
    train_lines, test_lines = _sample_lines(rng, train_count, test_count, diagonal, base)
    train_edges, test_edges = _sample_orientations(rng, train_count, test_count)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_train_background = None
    for regime, sizes, lines, edges in (("train", train_sizes, train_lines, train_edges),
                                        ("test", test_sizes, test_lines, test_edges)):
        for index, (side, line, edge) in enumerate(zip(sizes, lines, edges)):
            candidates = [int(v) for v in rng.permutation(
                [c for c in range(10) if c not in (line, diagonal, base)])]
            if regime == "train" and index == 1:
                candidates = [c for c in candidates if c != first_train_background]
            for background in candidates:
                if _would_copy_source(side, background, line, diagonal, base, reflected, edge):
                    continue
                left_in, left_out = _left_pair(side, background, line, diagonal, base, reflected)
                inp, out = _rotate_pair(left_in, left_out, edge)
                signature = _input_signature(inp)
                if signature not in seen_inputs:
                    break
            else:
                raise AssertionError("No distinct legal background remains")
            seen_inputs.add(signature)
            if regime == "train" and index == 0:
                first_train_background = background
            puzzle[regime].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be an ndarray")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= value <= 30 for value in grid.shape), f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has an invalid ARC color")


def _read_input(grid: np.ndarray) -> dict:
    _check_grid(grid, "input")
    side = grid.shape[0]
    _require(grid.shape == (side, side) and side >= 3, "Input must be a square of side 3-30")
    candidates = []
    for edge in range(4):
        left = np.rot90(grid, edge)
        line = int(left[0, 0])
        background = int(left[0, 1])
        if line != background and bool(np.all(left[:, 0] == line)) and bool(np.all(left[:, 1:] == background)):
            candidates.append((edge, line, background))
    _require(len(candidates) == 1, "Input must have exactly one full edge line")
    edge, line, background = candidates[0]
    return {"size": side, "edge": edge, "line": line, "background": background}


def _analyze_pair(inp: np.ndarray, out: np.ndarray) -> dict:
    facts = _read_input(inp)
    _check_grid(out, "output")
    _require(out.shape == inp.shape, "Input and output shapes differ")
    normalized = np.rot90(out, facts["edge"])
    for reflected in (0, 1):
        if reflected:
            diagonal = int(normalized[-1, -1])
            base = int(normalized[0, 1])
        else:
            diagonal = int(normalized[0, -1])
            base = int(normalized[-1, 1])
        if len({facts["background"], facts["line"], diagonal, base}) != 4:
            continue
        _, expected = _left_pair(facts["size"], facts["background"], facts["line"],
                                 diagonal, base, reflected)
        if np.array_equal(normalized, expected):
            facts.update({"diagonal": diagonal, "base": base, "reflected": reflected})
            return facts
    raise AssertionError("Output does not preserve the line and draw the diagonal and base")


def _predict_from_training(train: list, test_input: np.ndarray) -> np.ndarray:
    learned = [_analyze_pair(pair["input"], pair["output"]) for pair in train]
    diagonal = learned[0]["diagonal"]
    base = learned[0]["base"]
    reflected = learned[0]["reflected"]
    _require(all((f["diagonal"], f["base"], f["reflected"]) ==
                 (diagonal, base, reflected) for f in learned), "Training conventions conflict")
    facts = _read_input(test_input)
    _, out = _left_pair(facts["size"], facts["background"], facts["line"],
                        diagonal, base, reflected)
    return np.rot90(out, -facts["edge"]).copy()


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[key], list) for key in ("train", "test")),
             "Train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "Illegal pair counts")
    facts = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pair must contain exactly input and output")
            info = _analyze_pair(pair["input"], pair["output"])
            input_key = _input_signature(pair["input"])
            pair_key = (input_key, _input_signature(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "Duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _would_copy_source(info["size"], info["background"], info["line"],
                                             info["diagonal"], info["base"],
                                             info["reflected"], info["edge"]),
                     "Exact source pair copied")
            facts[regime].append(info)

    train = facts["train"]
    test = facts["test"]
    all_facts = train + test
    conventions = {(f["diagonal"], f["base"], f["reflected"]) for f in all_facts}
    _require(len(conventions) == 1, "Puzzle-level drawing convention changes")
    _require(len({f["background"] for f in train}) >= 2,
             "Training must show varying backgrounds")
    _require(len({f["line"] for f in train}) >= 2,
             "Training must show varying line colors")
    _require(any(f["size"] <= 10 for f in train) and any(f["size"] >= 20 for f in train),
             "Training must show compact and large scales")
    _require(len({f["size"] for f in train}) == train_count,
             "Training sizes must vary")
    _require(len({f["size"] for f in test}) == test_count,
             "Test sizes must vary")
    train_edges = {f["edge"] for f in train}
    _require(len(train_edges) >= 2 or all(f["edge"] in train_edges for f in test),
             "A new test edge needs demonstrations on multiple training edges")
    for pair in puzzle["test"]:
        prediction = _predict_from_training(puzzle["train"], pair["input"].copy())
        _require(np.array_equal(prediction, pair["output"]),
                 "Training and test input do not determine the test output")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_input_signature(p["input"]), _input_signature(p["output"]))
                       for p in puzzle[regime]) for regime in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "Automatic pair counts lack diversity")
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40,
             "Different seeds repeat puzzles")

    edges, conventions, backgrounds = set(), set(), set()
    train_sizes, test_sizes = set(), set()
    zeros = {"line": 0, "diagonal": 0, "base": 0}
    parity_regimes = set()
    reuse_size = reuse_line = novel_line = 0
    mixed_edges = fixed_edges = combined = 0
    patterns = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        # Recheck the construction cell by cell without the pair builder.
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                details = _analyze_pair(pair["input"], pair["output"])
                side = details["size"]
                before = np.rot90(pair["input"], details["edge"])
                after = np.rot90(pair["output"], details["edge"])
                expected = before.copy()
                for row in range(side - 1):
                    target_row = side - 1 - row if details["reflected"] else row
                    expected[target_row, side - 1 - row] = details["diagonal"]
                base_row = 0 if details["reflected"] else side - 1
                expected[base_row, 1:] = details["base"]
                _require(np.array_equal(after, expected), "Independent cell check failed")
                _require(np.array_equal(after[:, 0], before[:, 0]),
                         "Input edge line was not preserved")
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_facts = train + test
        train_size_set = {f["size"] for f in train}
        train_line_set = {f["line"] for f in train}
        train_edge_set = {f["edge"] for f in train}
        if len(train_edge_set) >= 2:
            mixed_edges += 1
        else:
            fixed_edges += 1
        parity_regimes.add((tuple(sorted({f["size"] % 2 for f in train})),
                            tuple(sorted({f["size"] % 2 for f in test}))))
        backgrounds.update(f["background"] for f in all_facts)
        edges.update(f["edge"] for f in all_facts)
        conventions.add((train[0]["diagonal"], train[0]["base"], train[0]["reflected"]))
        train_sizes.update(train_size_set)
        test_sizes.update(f["size"] for f in test)
        patterns.add((tuple(sorted(train_size_set)), tuple(sorted(f["size"] for f in test))))
        for role in zeros:
            zeros[role] += sum(f[role] == 0 for f in all_facts)
        reuse_size += sum(f["size"] in train_size_set for f in test)
        reuse_line += sum(f["line"] in train_line_set for f in test)
        novel_line += sum(f["line"] not in train_line_set for f in test)
        if len(train_edge_set) >= 2 and len({f["background"] for f in all_facts}) >= 2 and \
                any(f["size"] in train_size_set for f in test):
            combined += 1
        for pair in puzzle["test"]:
            predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]), "Held-out prediction failed")

    _require(edges == {0, 1, 2, 3}, "Not all edges appeared")
    _require({v[2] for v in conventions} == {0, 1}, "Both endpoint conventions are needed")
    _require(len(conventions) >= 15 and len(backgrounds) == 10,
             "Colors lack cross-seed coverage")
    _require(all(value > 0 for value in zeros.values()), "Zero did not appear in every requested role")
    _require({3, 7, 15}.issubset(train_sizes) and 10 in test_sizes,
             "Source dimensions were not restored")
    _require(len(train_sizes) >= 20 and len(test_sizes) >= 20 and len(patterns) >= 35,
             "Dimension diversity is too narrow")
    _require(mixed_edges >= 8 and fixed_edges >= 5,
             "Mixed and fixed orientation modes must both appear")
    _require(reuse_size >= 3 and reuse_line >= 3 and novel_line >= 3 and combined >= 2,
             "Test shifts or combined variation are missing")
    _require(len(parity_regimes) >= 4, "Parity regimes lack diversity")

    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                                          (2001, None, 1), (2002, None, 4),
                                          (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "Explicit training count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "Explicit test count ignored")
    for seed, counts in ((314159, (None, None)), (271828, (7, 3))):
        first = generate_puzzle(seed, *counts)
        second = generate_puzzle(seed, *counts)
        _require(_puzzle_signature(first) == _puzzle_signature(second), "Generation is nondeterministic")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(3001, train_count, test_count)
        except ValueError:
            continue
        raise AssertionError("Illegal counts did not raise ValueError")
