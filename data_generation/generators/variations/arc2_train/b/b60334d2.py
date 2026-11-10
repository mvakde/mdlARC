import numpy as np


SOURCE = {
    (9, 9, ((1, 2), (1, 7), (5, 2), (7, 6))),
    (9, 9, ((2, 3), (5, 6), (7, 2))),
    (9, 9, ((1, 1), (3, 4), (5, 7), (7, 2))),
}
CORNERS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SIDES = ((-1, 0), (0, -1), (0, 1), (1, 0))
MODES = ("single", "pair", "row", "column", "array", "symmetric",
         "chain", "cluster", "cluster_isolated", "scatter")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def count_arg(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def counts(rng, train, test):
    train = count_arg(train, "num_train", 2, 10)
    test = count_arg(test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def assignments(centers, height, width):
    if not centers or len(set(centers)) != len(centers):
        return None
    painted = {}
    for row, col in centers:
        if not (1 <= row < height - 1 and 1 <= col < width - 1):
            return None
        if any((row, col) != (r, c) and max(abs(row - r), abs(col - c)) <= 1
               for r, c in centers):
            return None
        for offsets, role in ((CORNERS, 1), (SIDES, 2)):
            for dr, dc in offsets:
                cell = (row + dr, col + dc)
                if cell in painted and painted[cell] != role:
                    return None
                painted[cell] = role
    return painted


def disjoint(centers):
    return all(abs(r - s) >= 3 or abs(c - d) >= 3
               for i, (r, c) in enumerate(centers) for s, d in centers[i + 1:])


def separated(centers):
    return all(abs(r - s) >= 4 or abs(c - d) >= 4
               for i, (r, c) in enumerate(centers) for s, d in centers[i + 1:])


def addable(centers, candidate, height, width, spacing):
    proposed = centers + [candidate]
    return (assignments(proposed, height, width) is not None
            and (spacing != "disjoint" or disjoint(proposed))
            and (spacing != "separated" or separated(proposed)))


def fill(rng, centers, count, height, width, spacing="disjoint"):
    centers = list(centers)
    candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for index in rng.permutation(len(candidates)):
        if len(centers) >= count:
            break
        candidate = candidates[int(index)]
        if addable(centers, candidate, height, width, spacing):
            centers.append(candidate)
    return tuple(sorted(centers)) if len(centers) == count else None


def shape(rng):
    band = int(rng.choice(5, p=(0.20, 0.34, 0.24, 0.11, 0.11)))
    if band == 0:
        return int(rng.integers(3, 9)), int(rng.integers(3, 9))
    if band == 1:
        return int(rng.integers(8, 17)), int(rng.integers(8, 17))
    if band == 2:
        return int(rng.integers(17, 31)), int(rng.integers(17, 31))
    if band == 3:
        return int(rng.integers(3, 8)), int(rng.integers(12, 31))
    return int(rng.integers(12, 31)), int(rng.integers(3, 8))


def line(rng, height, width, vertical=False, overlap=False):
    length = height if vertical else width
    feasible = [(step, n) for step in ((2,) if overlap else (3, 4, 5, 6))
                for n in range(3, 9) if 3 + (n - 1) * step <= length]
    if not feasible:
        return None
    step, n = feasible[int(rng.integers(len(feasible)))]
    start = int(rng.integers(1, length - (n - 1) * step - 1))
    other = int(rng.integers(1, (width if vertical else height) - 1))
    if vertical:
        return tuple((start + k * step, other) for k in range(n))
    return tuple((other, start + k * step) for k in range(n))


def array(rng, height, width, overlap=False):
    steps = (2, 3, 4, 5) if overlap else (3, 4, 5, 6)
    feasible = [(nr, nc, dr, dc) for nr in range(2, 5) for nc in range(2, 6)
                for dr in steps for dc in steps
                if 3 + (nr - 1) * dr <= height and 3 + (nc - 1) * dc <= width
                and (not overlap or dr == 2 or dc == 2)]
    if not feasible:
        return None
    nr, nc, dr, dc = feasible[int(rng.integers(len(feasible)))]
    row = int(rng.integers(1, height - (nr - 1) * dr - 1))
    col = int(rng.integers(1, width - (nc - 1) * dc - 1))
    return tuple((row + i * dr, col + j * dc) for i in range(nr) for j in range(nc))


def chain(rng, height, width):
    dr, dc = (2, 3) if rng.integers(2) else (3, 2)
    dr *= 1 if rng.integers(2) else -1
    dc *= 1 if rng.integers(2) else -1
    n = int(rng.integers(3, 5))
    span_r, span_c = (n - 1) * abs(dr), (n - 1) * abs(dc)
    if height < span_r + 3 or width < span_c + 3:
        return None
    row = int(rng.integers(1, height - span_r - 1))
    col = int(rng.integers(1, width - span_c - 1))
    if dr < 0:
        row += span_r
    if dc < 0:
        col += span_c
    return tuple((row + k * dr, col + k * dc) for k in range(n))


def cluster(rng, height, width, isolated=False):
    target = int(rng.integers(3, 8))
    centers = [(int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1)))]
    deltas = [(dr * sr, dc * sc) for dr, dc in ((3, 0), (0, 3), (3, 1), (1, 3), (3, 3))
              for sr in (-1, 1) for sc in (-1, 1)]
    for _ in range(target * 18):
        if len(centers) == target:
            break
        parent = centers[int(rng.integers(len(centers)))]
        dr, dc = deltas[int(rng.integers(len(deltas)))]
        candidate = (parent[0] + dr, parent[1] + dc)
        if addable(centers, candidate, height, width, "disjoint"):
            centers.append(candidate)
    if len(centers) != target:
        return None
    if isolated:
        candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)
                      if all(max(abs(r - pr), abs(c - pc)) >= 6 for pr, pc in centers)
                      and addable(centers, (r, c), height, width, "disjoint")]
        if not candidates:
            return None
        centers.append(candidates[int(rng.integers(len(candidates)))])
    return tuple(sorted(centers))


def construct(rng, mode, height, width, allow_overlap):
    if mode == "single":
        return ((int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))),)
    if mode == "pair":
        return fill(rng, [], 2, height, width, "separated")
    if mode in ("row", "column", "overlap"):
        vertical = mode == "column" or (mode == "overlap" and bool(rng.integers(2)))
        return line(rng, height, width, vertical, mode == "overlap")
    if mode == "array":
        return array(rng, height, width, allow_overlap and bool(rng.integers(3) == 0))
    if mode == "symmetric":
        if height < 9 or width < 9:
            return None
        row = int(rng.integers(4, height - 4))
        col = int(rng.integers(4, width - 4))
        dr = int(rng.integers(3, min(row, height - 1 - row, 6) + 1))
        dc = int(rng.integers(3, min(col, width - 1 - col, 6) + 1))
        return ((row, col), (row - dr, col), (row + dr, col),
                (row, col - dc), (row, col + dc))
    if mode == "chain":
        centers = chain(rng, height, width)
        if centers is None:
            return None
        return fill(rng, list(centers), len(centers) + int(rng.integers(0, 4)), height, width)
    if mode in ("cluster", "cluster_isolated"):
        return cluster(rng, height, width, mode == "cluster_isolated")
    return fill(rng, [], int(rng.integers(2, 19)), height, width,
                "separated" if rng.integers(2) else "disjoint")


def make_pair(rng, mode, accent, fixed_colors, allow_overlap):
    for _ in range(160):
        height, width = shape(rng)
        centers = construct(rng, mode, height, width, allow_overlap)
        if centers is None:
            continue
        centers = tuple(sorted(centers))
        painted = assignments(centers, height, width)
        if painted is None or (height, width, centers) in SOURCE:
            continue
        if fixed_colors is None:
            available = [c for c in range(10) if c != accent]
            colors = rng.choice(available, size=2, replace=False)
            background, seed = int(colors[0]), int(colors[1])
        else:
            background, seed = fixed_colors
        inp = np.full((height, width), background, dtype=np.int8)
        out = inp.copy()
        for row, col in centers:
            inp[row, col] = seed
        for cell, role in painted.items():
            out[cell] = seed if role == 1 else accent
        return {"input": inp, "output": out}
    raise ValueError("could not construct a pair")


def grid_key(grid):
    return grid.shape, grid.tobytes()


def analyze(pair):
    inp, out = pair["input"], pair["output"]
    check(inp.shape == out.shape, "output shape changed")
    values, counts = np.unique(inp, return_counts=True)
    check(len(values) == 2 and max(counts) > min(counts), "unclear input roles")
    background = int(values[int(np.argmax(counts))])
    seed = int(values[int(np.argmin(counts))])
    height, width = inp.shape
    centers = tuple(map(tuple, np.argwhere(inp == seed).tolist()))
    painted = assignments(centers, height, width)
    check(painted is not None, "incompatible rings")
    check((height, width, centers) not in SOURCE, "source layout copied")
    extras = set(map(int, np.unique(out))) - {background, seed}
    check(len(extras) == 1, "exactly one accent required")
    accent = extras.pop()
    expected = np.full((height, width), background, dtype=np.int8)
    for cell, role in painted.items():
        expected[cell] = seed if role == 1 else accent
    check(np.array_equal(out, expected), "incorrect ring replacement")
    check(not np.array_equal(inp, out), "unchanged pair")
    return background, seed, accent, centers


def validate_generated_puzzle(puzzle: dict) -> None:
    check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists required")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    inputs, pairs, train_accents, test_accents = set(), set(), set(), set()
    train_overlap = test_overlap = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            for grid in pair.values():
                check(isinstance(grid, np.ndarray) and grid.ndim == 2, "2D array required")
                check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "illegal size")
                check(np.issubdtype(grid.dtype, np.integer), "integer grid required")
                check(bool(np.all((0 <= grid) & (grid <= 9))), "illegal color")
            key = grid_key(pair["input"])
            pair_key = key, grid_key(pair["output"])
            check(key not in inputs and pair_key not in pairs, "duplicate input or pair")
            inputs.add(key)
            pairs.add(pair_key)
            _, _, accent, centers = analyze(pair)
            if regime == "train":
                train_accents.add(accent)
                train_overlap |= not disjoint(centers)
            else:
                test_accents.add(accent)
                test_overlap |= not disjoint(centers)
    check(len(train_accents) == 1 and test_accents == train_accents, "accent must be learned")
    check(not test_overlap or train_overlap, "test overlap must be demonstrated")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = counts(rng, num_train, num_test)
    accent = int(rng.integers(10))
    fixed_colors = None
    if rng.integers(3) == 0:
        colors = rng.choice([c for c in range(10) if c != accent], size=2, replace=False)
        fixed_colors = int(colors[0]), int(colors[1])
    overlap_puzzle = bool(rng.integers(4) == 0)
    puzzle = {"train": [], "test": []}
    used, previous_mode = set(), None
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if regime == "train" and index == 0 and overlap_puzzle:
                mode = "overlap"
            else:
                choices = [m for m in MODES if m != previous_mode]
                if overlap_puzzle:
                    choices += ["overlap"] * (6 if regime == "test" else 2)
                if regime == "test":
                    choices += ["array", "chain", "cluster", "scatter"]
                mode = choices[int(rng.integers(len(choices)))]
            previous_mode = mode
            for _ in range(80):
                child = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                pair = make_pair(child, mode, accent, fixed_colors, overlap_puzzle)
                key = grid_key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise ValueError("could not create a unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def fingerprint(puzzle):
    return tuple((regime, tuple((grid_key(p["input"]), grid_key(p["output"]))
                               for p in puzzle[regime])) for regime in ("train", "test"))


def has_chain(centers):
    points = set(centers)
    return any((r + dr, c + dc) in points and (r + 2 * dr, c + 2 * dc) in points
               for r, c in points for dr in (-3, -2, 2, 3) for dc in (-3, -2, 2, 3)
               if sorted((abs(dr), abs(dc))) == [2, 3])


def has_array(centers):
    points = set(centers)
    return any((r, d) in points and (s, c) in points and (s, d) in points
               for r, c in points for s, d in points if r != s and c != d)


def has_symmetric_cross(centers):
    points = set(centers)
    return len(points) == 5 and any(
        {(r, c), (r - dr, c), (r + dr, c), (r, c - dc), (r, c + dc)} == points
        for r, c in points for dr in range(2, 9) for dc in range(2, 9))


def has_cluster_with_isolate(centers):
    if len(centers) < 4:
        return False
    for isolate in centers:
        remaining = set(centers) - {isolate}
        if any(max(abs(isolate[0] - r), abs(isolate[1] - c)) < 6
               for r, c in remaining):
            continue
        found = {next(iter(remaining))}
        while True:
            expanded = found | {point for point in remaining
                                if any(max(abs(point[0] - q[0]), abs(point[1] - q[1])) <= 3
                                       for q in found)}
            if expanded == found:
                break
            found = expanded
        if found == remaining:
            return True
    return False


def has_irregular_cluster(centers):
    if (len(centers) < 4 or has_symmetric_cross(centers) or has_array(centers)
            or has_chain(centers) or len({r for r, c in centers}) == 1
            or len({c for r, c in centers}) == 1):
        return False
    remaining = set(centers)
    largest = 0
    while remaining:
        found = {remaining.pop()}
        while True:
            expanded = found | {point for point in remaining
                                if any(max(abs(point[0] - q[0]), abs(point[1] - q[1])) <= 3
                                       for q in found)}
            if expanded == found:
                break
            found = expanded
            remaining -= found
        largest = max(largest, len(found))
    return largest >= 4


def predict_from_examples(train, test_input):
    train_input, train_output = train[0]["input"], train[0]["output"]
    accent_values = set(map(int, np.unique(train_output))) - set(map(int, np.unique(train_input)))
    check(len(accent_values) == 1, "training must identify the accent")
    accent = accent_values.pop()
    values, frequencies = np.unique(test_input, return_counts=True)
    background = int(values[int(np.argmax(frequencies))])
    seed = int(values[int(np.argmin(frequencies))])
    answer = np.full(test_input.shape, background, dtype=np.int8)
    for row, col in np.argwhere(test_input == seed):
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            answer[row + dr, col + dc] = seed
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            answer[row + dr, col + dc] = accent
    return answer


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            check(np.array_equal(predict_from_examples(puzzle["train"], pair["input"]),
                                 pair["output"]), "test cannot be inferred from training")
    check(fingerprint(generate_puzzle(1234567, 7, 3)) ==
          fingerprint(generate_puzzle(1234567, 7, 3)), "nondeterministic")
    check(fingerprint(generate_puzzle(111, 6, 2)) !=
          fingerprint(generate_puzzle(112, 6, 2)), "seed has no effect")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                        (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + i, train, test)
        check((train is None or len(puzzle["train"]) == train) and
              (test is None or len(puzzle["test"]) == test), "count not honored")
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (2, 3)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    check(len({len(p["train"]) for p in puzzles}) >= 4 and
          len({len(p["test"]) for p in puzzles}) >= 3, "pair-count diversity")
    check(len({fingerprint(p) for p in puzzles}) >= 40, "puzzle diversity")
    shapes, counts_seen, layouts, colors = set(), set(), set(), set()
    properties = {name: 0 for name in ("single", "two", "many", "small", "large", "thin",
                                      "overlap", "separated", "abut", "array", "chain",
                                      "row", "column", "corner", "one_edge", "several_edges",
                                      "symmetric", "cluster_isolate", "irregular_cluster",
                                      "interior", "boundary", "train_chain", "test_no_chain",
                                      "test_overlap")}
    varying_colors = varying_shapes = varying_counts = denser_test = sparser_test = 0
    for puzzle in puzzles:
        roles = set()
        pair_shapes, pair_counts, train_sizes, test_sizes = set(), set(), [], []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                background, seed, accent, centers = analyze(pair)
                height, width = pair["input"].shape
                shapes.add((height, width))
                counts_seen.add(len(centers))
                layouts.add((height, width, centers))
                colors.add((background, seed, accent))
                roles.add((background, seed))
                pair_shapes.add((height, width))
                pair_counts.add(len(centers))
                (train_sizes if regime == "train" else test_sizes).append(len(centers))
                touched = {edge for r, c in centers for edge, on in
                           (("top", r == 1), ("bottom", r == height - 2),
                            ("left", c == 1), ("right", c == width - 2)) if on}
                touch = bool(touched)
                overlap = not disjoint(centers)
                chain = has_chain(centers)
                for name, truth in (
                    ("single", len(centers) == 1), ("two", len(centers) == 2),
                    ("many", len(centers) >= 8), ("small", min(height, width) <= 7),
                    ("large", max(height, width) >= 24),
                    ("thin", max(height, width) / min(height, width) >= 2),
                    ("overlap", overlap), ("separated", len(centers) >= 2 and separated(centers)),
                    ("abut", disjoint(centers) and not separated(centers)),
                    ("array", has_array(centers)), ("chain", chain),
                    ("symmetric", has_symmetric_cross(centers)),
                    ("cluster_isolate", has_cluster_with_isolate(centers)),
                    ("irregular_cluster", has_irregular_cluster(centers)),
                    ("row", any(sum(r == rr for rr, cc in centers) >= 3 for r, c in centers)),
                    ("column", any(sum(c == cc for rr, cc in centers) >= 3 for r, c in centers)),
                    ("corner", any(r in (1, height - 2) and c in (1, width - 2)
                                   for r, c in centers)),
                    ("one_edge", len(touched) == 1), ("several_edges", len(touched) >= 2),
                    ("interior", not touch), ("boundary", touch),
                    ("train_chain", regime == "train" and chain),
                    ("test_no_chain", regime == "test" and not chain),
                    ("test_overlap", regime == "test" and overlap),
                ):
                    properties[name] += int(truth)
        varying_colors += len(roles) > 1
        varying_shapes += len(pair_shapes) > 1
        varying_counts += len(pair_counts) > 1
        denser_test += max(test_sizes) > max(train_sizes)
        sparser_test += max(test_sizes) < max(train_sizes)
    check(len(shapes) >= 35 and len(counts_seen) >= 8 and len(layouts) >= 110,
          "insufficient structural diversity")
    check(len(colors) >= 30 and varying_colors >= 20, "insufficient color variation")
    check(varying_shapes >= 35 and varying_counts >= 30, "insufficient within-puzzle variation")
    check(denser_test >= 4 and sparser_test >= 4, "test density always shifts one way")
    check(all(value >= 2 for value in properties.values()), "review axis absent")
