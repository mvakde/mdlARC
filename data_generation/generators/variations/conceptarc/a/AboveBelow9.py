import numpy as np


_SOURCE_INPUTS = {
    (
        "000030000000", "000030000000", "444444444444",
        "000033300000", "000000300000", "000000300000",
        "003003300000", "003003000000", "003333000000",
        "000000000000",
    ),
    (
        "000077777700000", "007770000000000", "007000000000000",
        "007000000000000", "007777000000000", "000007000000000",
        "444444444444444", "000777000000000", "000700000000000",
        "000700000000000", "000777777700000", "000000000700000",
        "000000777700000",
    ),
    (
        "008000000000000", "008888800000000", "000000800000000",
        "000000800000000", "000088800000000", "000080000000000",
        "000080000000000", "000088880000000", "999999999999999",
        "000000080000000", "000008880000000", "000008000000000",
    ),
    (
        "000100000000", "333333333333", "000110000000",
        "000010000000", "000011100000", "000000100000",
        "000000111000", "000000001000", "000001111000",
        "000001000000", "000111000000", "000100000000",
    ),
    (
        "22222222222", "00005000000", "00005000000",
        "00055000000", "00050000000", "55550000000",
        "50000000000", "55555500000", "00000500000",
    ),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _choose_counts(rng, num_train, num_test):
    for value, name, low, high in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(
                f"{name} must be an integer from {low} through {high}"
            )

    legal = [
        (train_count, test_count)
        for train_count in range(2, 11)
        for test_count in range(1, 5)
        if train_count > test_count
        and (num_train is None or train_count == num_train)
        and (num_test is None or test_count == num_test)
    ]
    if not legal:
        raise ValueError("num_train must exceed num_test")

    return legal[int(rng.integers(len(legal)))]



def _sample_puzzle_parameters(rng):
    palette = [int(c) for c in rng.choice(10, int(rng.integers(5, 9)), replace=False)]
    mode = str(rng.choice(("stable", "background_stable", "varied"),
                          p=(0.23, 0.27, 0.50)))
    triple = tuple(int(c) for c in rng.choice(palette, 3, replace=False))
    return palette, mode, triple


def _colors(rng, palette, mode, triple):
    if mode == "stable":
        return triple
    if mode == "background_stable":
        others = [c for c in palette if c != triple[0]]
        chosen = rng.choice(others, 2, replace=False)
        return triple[0], int(chosen[0]), int(chosen[1])
    chosen = rng.choice(palette, 3, replace=False)
    return tuple(int(c) for c in chosen)


def _divider_row(rng, height, regime, anchor):
    if anchor:
        return int(rng.integers(2, height - 2))
    weight = 0.27 if regime == "source_train" else 0.42
    if rng.random() < weight:
        return int(rng.choice((0, 1, height - 4, height - 3, height - 2)))
    return int(rng.integers(0, height - 1))


def _draw_path(grid, rng, color, start, end, divider):
    # One path advances down the canvas. Horizontal moves on adjacent rows
    # cannot reverse, which prevents overlap, branches, and 2x2 blocks.
    width = grid.shape[1]
    x = int(rng.integers(width))
    if rng.random() < 0.18:
        x = int(rng.choice((0, width - 1)))
    bend_rate = float(rng.uniform(0.18, 0.82))
    max_run = int(rng.integers(2, min(width - 2, 9) + 1))
    previous = 0
    for y in range(start, end + 1):
        grid[y, x] = color
        if y == divider:
            previous = 0
            continue
        direction = 0
        if rng.random() < bend_rate:
            choices = [d for d in (-1, 1) if 0 <= x + d < width]
            if previous:
                choices = [d for d in choices if d == previous]
            if choices:
                direction = int(rng.choice(choices))
        if direction:
            free = x if direction < 0 else width - 1 - x
            limit = min(free, max_run, width - 2)
            if rng.random() < 0.23 and free <= width - 2:
                step = free
            else:
                step = int(rng.integers(1, limit + 1))
            new_x = x + direction * step
            grid[y, min(x, new_x):max(x, new_x) + 1] = color
            x = new_x
        previous = direction


def _generate_pair(rng, regime, parameters, anchor=False, unchanged=False):
    height = int(rng.integers(7, 29))
    width = int(rng.integers(7, 29))
    divider = _divider_row(rng, height, regime, anchor)
    if unchanged:
        divider = int(rng.integers(2, height - 1))
    background, divider_color, foreground = _colors(rng, *parameters)
    grid = np.full((height, width), background, dtype=np.int8)
    if unchanged:
        end = int(rng.integers(1, divider))
        start = int(rng.integers(0, end))
    else:
        if divider == 0:
            start = 1
        elif anchor:
            start = int(rng.integers(0, divider - 1))
        else:
            start = int(rng.integers(0, divider))
        if anchor:
            end = int(rng.integers(divider + 2, height))
        else:
            end = int(rng.integers(divider + 1, height))
    for _ in range(12):
        grid.fill(background)
        _draw_path(grid, rng, foreground, start, end, divider)
        if np.count_nonzero(grid == foreground) <= 0.42 * (height * width - width):
            break
    else:
        grid.fill(background)
        grid[start:end + 1, int(rng.integers(width))] = foreground
    grid[divider, :] = divider_color
    output = grid.copy()
    output[divider + 1:, :] = background
    return grid, output


def _grid_signature(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _source_rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, part in (
        ("source_train", train_count, "train"),
        ("target_test", test_count, "test"),
    ):
        for index in range(count):
            anchor = regime == "source_train" and index == 0
            unchanged = (regime == "target_test" and index == test_count - 1
                         and rng.random() < 0.085)
            for _ in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = _generate_pair(pair_rng, regime, parameters,
                                          anchor, unchanged)
                sig = _grid_signature(inp)
                if sig not in seen and _source_rows(inp) not in _SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError("could not construct distinct pair")
            seen.add(sig)
            puzzle[part].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _components_and_degrees(mask):
    cells = {tuple(int(z) for z in p) for p in np.argwhere(mask)}
    if not cells:
        return 0, 0
    unseen = set(cells)
    components = 0
    maximum = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            y, x = stack.pop()
            neighbors = {(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)} & cells
            maximum = max(maximum, len(neighbors))
            new = neighbors & unseen
            unseen -= new
            stack.extend(new)
    return components, maximum


def _pair_facts(pair, part):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "bad pair keys")
    for name in ("input", "output"):
        grid = pair[name]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 "grid must be a 2D NumPy array")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                 "grid dimensions outside ARC bounds")
        _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
        _require(np.all((grid >= 0) & (grid <= 9)),
                 "grid color outside ARC palette")
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "grid size changed")
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 3, "input must have three local color roles")
    background = int(values[int(np.argmax(counts))])
    full_rows = [r for r in range(inp.shape[0])
                 if len(set(int(c) for c in inp[r])) == 1
                 and int(inp[r, 0]) != background]
    _require(len(full_rows) == 1, "divider must be unique non-background full row")
    row = full_rows[0]
    divider = int(inp[row, 0])
    foreground = next(int(c) for c in values if c not in (background, divider))
    _require(np.count_nonzero(inp == divider) == inp.shape[1],
             "divider color appears outside its row")
    _require(np.array_equal(out[:row + 1], inp[:row + 1]),
             "upper region or divider changed")
    _require(np.all(out[row + 1:] == background), "lower region not erased")
    lower = inp[row + 1:] == foreground
    upper = inp[:row] == foreground
    _require(np.any(upper) or np.any(lower), "foreground missing")
    if part == "train":
        _require(np.any(lower), "training must demonstrate erasure")
    if not np.any(lower):
        _require(part == "test" and np.array_equal(inp, out) and np.any(upper),
                 "invalid unchanged example")
    else:
        _require(not np.array_equal(inp, out), "lower foreground was not erased")
    _require(not np.any(np.all(inp == foreground, axis=1)),
             "foreground makes a competing full row")
    mask = inp == foreground
    _require(not np.any(mask[:-1, :-1] & mask[1:, :-1]
                        & mask[:-1, 1:] & mask[1:, 1:]), "thick 2x2 linework")
    for region in (upper, lower):
        components, degree = _components_and_degrees(region)
        _require(components <= 1 and degree <= 2,
                 "foreground branched or disconnected on one side")
    if np.any(upper) and np.any(lower):
        _require(np.any((inp[row - 1] == foreground)
                        & (inp[row + 1] == foreground)),
                 "path does not cross divider in shared column")
    if row == 0:
        _require(np.any(inp[1] == foreground),
                 "top-edge path must start immediately below divider")
    _require(_source_rows(inp) not in _SOURCE_INPUTS, "original source input copied")
    return background, divider, foreground, row, int(upper.sum()), int(lower.sum())


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "bad puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "puzzle parts must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    train_facts = []
    for part in ("train", "test"):
        for pair in puzzle[part]:
            facts = _pair_facts(pair, part)
            if part == "train":
                train_facts.append(facts)
            insig = _grid_signature(pair["input"])
            psig = insig, _grid_signature(pair["output"])
            _require(insig not in seen_inputs and psig not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(insig)
            seen_pairs.add(psig)
    _require(any(u >= 2 and l >= 2 and r >= 2 for _, _, _, r, u, l in train_facts),
             "training lacks a clear preserve-and-erase demonstration")


def _predict_from_visible_input(train_pairs, test_input):
    _require(any(np.any(p["input"] != p["output"]) for p in train_pairs),
             "training does not show an erasure")
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    rows = [r for r in range(test_input.shape[0])
            if np.all(test_input[r] == test_input[r, 0])
            and int(test_input[r, 0]) != background]
    _require(len(rows) == 1, "cannot locate test divider")
    predicted = test_input.copy()
    predicted[rows[0] + 1:] = background
    return predicted


def _same_puzzle(a, b):
    return all(len(a[part]) == len(b[part]) and all(
        np.array_equal(x[key], y[key])
        for x, y in zip(a[part], b[part]) for key in ("input", "output")
    ) for part in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, structures = set(), set(), set(), set()
    train_positions, test_positions = set(), set()
    retained, erased, edges, unchanged, stable, varied = set(), set(), 0, 0, 0, 0
    puzzles = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_same_puzzle(puzzle, generate_puzzle(seed)), "not deterministic")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        triples = []
        signature = []
        for part in ("train", "test"):
            for pair in puzzle[part]:
                background, divider, foreground, row, u, l = _pair_facts(pair, part)
                inp, out = pair["input"], pair["output"]
                triples.append((background, divider, foreground))
                shapes.add(inp.shape)
                retained.add(u)
                erased.add(l)
                (train_positions if part == "train" else test_positions).add(
                    "top" if row == 0 else "shallow" if row <= 2
                    else "deep" if row >= inp.shape[0] - 4 else "interior"
                )
                mask = inp == foreground
                edges += int(np.any(mask[:, 0] | mask[:, -1]))
                unchanged += int(np.array_equal(inp, out))
                structures.add((inp.shape, row, tuple(
                    tuple(int(v) for v in line) for line in mask
                )))
                signature.append((_grid_signature(inp), _grid_signature(out)))
                if part == "test":
                    _require(np.array_equal(_predict_from_visible_input(
                        puzzle["train"], inp.copy()), out),
                        "test not inferable from training and input")
        stable += int(len(set(triples)) == 1)
        varied += int(len(set(t[0] for t in triples)) > 1
                      and len(set(t[1] for t in triples)) > 1
                      and len(set(t[2] for t in triples)) > 1)
        puzzles.add(tuple(signature))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts lack diversity")
    _require(len(puzzles) == 42 and len(shapes) >= 80 and len(structures) >= 150,
             "structural diversity too narrow")
    _require({"top", "shallow", "interior", "deep"} <= train_positions
             and {"top", "shallow", "interior", "deep"} <= test_positions,
             "divider placement diversity too narrow")
    _require(len(retained) >= 15 and len(erased) >= 15 and edges >= 12,
             "path length or edge diversity too narrow")
    _require(unchanged >= 2 and stable >= 3 and varied >= 8,
             "boundary case or palette diversity too narrow")
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        p = generate_puzzle(8917, nt, ns)
        validate_generated_puzzle(p)
        _require((nt is None or len(p["train"]) == nt)
                 and (ns is None or len(p["test"]) == ns)
                 and _same_puzzle(p, generate_puzzle(8917, nt, ns)),
                 "explicit or partial counts failed")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (1, 1), (10, 10)):
        try:
            generate_puzzle(12, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
