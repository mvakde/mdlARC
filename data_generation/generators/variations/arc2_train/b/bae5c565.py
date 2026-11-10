"""ARC top-row templates revealed around a downward vertical spine."""

import numpy as np


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


# Exact source inputs as visible features (height, width, spine start,
# background, spine color, top row). The output is deterministic from these.
_SOURCE = {
    (13, 13, 5, 5, 8, (2, 2, 7, 1, 9, 1, 5, 8, 6, 0, 3, 2, 2)),
    (9, 9, 3, 5, 8, (0, 1, 6, 9, 5, 9, 6, 1, 0)),
    (11, 11, 5, 5, 8, (4, 6, 7, 2, 9, 5, 3, 3, 4, 3, 3)),
}


def _output(grid, background, spine, start):
    height, width = grid.shape
    center = width // 2
    out = np.full(grid.shape, background, dtype=np.int8)
    for row in range(start, height):
        depth = row - start
        lo, hi = max(0, center - depth), min(width, center + depth + 1)
        out[row, lo:hi] = grid[0, lo:hi]
        out[row, center] = spine
    return out


def _counts(seed, train, test):
    for value, name, lo, hi in ((train, "num_train", 2, 10),
                                 (test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _other(rng, choices, value):
    return int(rng.choice([v for v in choices if v != value]))


def _template(rng, width, background, spine, style, spine_count):
    radius, n = width // 2, width - 1
    ordinary = [c for c in range(10) if c not in (background, spine)]
    if style == "distinct":
        payload = [int(v) for v in rng.choice(ordinary, size=n, replace=False)]
        spine_count = min(spine_count, 1)
    elif style in ("palindrome", "near_palindrome"):
        palette = [int(v) for v in rng.choice(
            ordinary, size=int(rng.integers(1, min(6, radius) + 1)),
            replace=False)]
        left = [int(v) for v in rng.choice(palette, size=radius)]
        for j in rng.choice(radius, size=min(radius, spine_count // 2),
                            replace=False):
            left[int(j)] = spine
        payload = left + left[::-1]
        if style == "near_palindrome":
            if spine_count % 2:
                candidates = [j for j in range(radius, n) if payload[j] != spine]
                j = int(rng.choice(candidates))
                payload[j] = spine
            else:
                j = int(rng.integers(radius, n))
                payload[j] = _other(rng, ordinary, payload[j])
    elif style == "monochrome":
        payload = [int(rng.choice(ordinary))] * n
        spine_count = 0
    else:
        size = int(rng.integers(3 if style == "anchor" else 2,
                                min(8, n) + 1))
        palette = [int(v) for v in rng.choice(ordinary, size=size,
                                              replace=False)]
        if style == "runs":
            cuts = sorted(int(v) for v in rng.choice(
                np.arange(1, n), size=int(rng.integers(1, min(4, n - 1) + 1)),
                replace=False))
            bounds = [0] + cuts + [n]
            payload = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                payload += [int(rng.choice(palette))] * (hi - lo)
        elif style == "alternating":
            a, b = [int(v) for v in rng.choice(palette, size=2, replace=False)]
            payload = [a if j % 2 == 0 else b for j in range(n)]
            for j in rng.choice(n, size=int(rng.integers(0, n // 4 + 1)),
                                replace=False):
                payload[int(j)] = int(rng.choice(palette))
        elif style == "word":
            length = int(rng.integers(2, min(5, n // 2) + 1))
            word = [int(v) for v in rng.choice(palette, size=length)]
            payload = [word[j % length] for j in range(n)]
        elif style == "unequal":
            simple = [int(rng.choice(palette))] * radius
            varied = [int(v) for v in rng.choice(palette, size=radius)]
            payload = simple + varied if rng.integers(2) else varied + simple
        else:
            payload = [int(v) for v in rng.choice(palette, size=n)]
            if style == "anchor":
                for j, color in enumerate(palette[:min(4, n)]):
                    payload[j] = color
                payload[radius] = _other(rng, palette, payload[radius - 1])
        for j in rng.choice(n, size=min(spine_count, n), replace=False):
            payload[int(j)] = spine
    if style == "distinct" and spine_count:
        payload[int(rng.integers(n))] = spine
    if style == "anchor":
        # A plainly asymmetric saturated example establishes column order.
        if payload[:radius] == payload[radius:][::-1]:
            payload[-1] = _other(rng, ordinary, payload[0])
        if len(set(payload)) < 3:
            payload[0], payload[1], payload[-1] = palette[:3]
    template = np.full(width, background, dtype=np.int8)
    template[:radius], template[radius + 1:] = payload[:radius], payload[radius:]
    return template


def _shape(rng, width, depth_kind):
    radius = width // 2
    if depth_kind == "short":
        depth = int(rng.integers(2, radius))
    elif depth_kind == "edge":
        depth = radius
    else:
        depth = radius + int(rng.integers(1, min(6, 29 - radius)))
    start = int(rng.integers(2, min(15, 30 - depth)))
    return start, depth


def _pair(rng, width, roles, style, depth_kind, spine_count):
    background, spine = roles
    start, depth = _shape(rng, width, depth_kind)
    grid = np.full((start + depth + 1, width), background, dtype=np.int8)
    grid[0] = _template(rng, width, background, spine, style, spine_count)
    grid[start:, width // 2] = spine
    return {"input": grid, "output": _output(grid, background, spine, start)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    total = train_count + test_count
    choices = list(range(5, 30, 2))
    if rng.random() < 0.23:
        widths = [int(rng.choice(choices[1:]))] * total
    else:
        widths = [int(v) for v in rng.choice(choices, size=total)]
        widths[0] = int(rng.choice(choices[1:]))
        width_route = rng.random()
        if width_route < 0.35:
            widths[train_count] = int(rng.choice(widths[:train_count]))
        elif width_route < 0.60:
            smaller = [w for w in choices if w < min(widths[:train_count])]
            if smaller:
                widths[train_count] = int(rng.choice(smaller))
        elif width_route < 0.85:
            larger = [w for w in choices if w > max(widths[:train_count])]
            if larger:
                widths[train_count] = int(rng.choice(larger))
    first_role = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    if rng.random() < 0.10:
        first_role = (5, 8)
    roles = [first_role] * total
    if rng.random() < 0.48:
        roles = [tuple(int(v) for v in rng.choice(10, size=2, replace=False))
                 for _ in range(total)]
        roles[0] = first_role
        while roles[1] == first_role:
            roles[1] = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    styles = ["palindrome", "near_palindrome", "runs", "alternating",
              "word", "unequal", "shuffled", "monochrome", "distinct"]
    style_plan = [str(rng.choice(styles)) for _ in range(total)]
    style_plan[0] = "anchor"
    for i in range(total):
        if widths[i] > 9 and style_plan[i] == "distinct":
            style_plan[i] = "shuffled"
    depths = [str(rng.choice(["short", "edge", "plateau"],
                              p=[0.27, 0.34, 0.39])) for _ in range(total)]
    depths[0] = "plateau"
    for i in range(total):
        if widths[i] == 5 and depths[i] == "short":
            depths[i] = "edge"
    if "short" in depths[train_count:]:
        if widths[1] == 5:
            widths[1] = 7
        depths[1] = "short"
    spine_counts = [int(rng.choice([0, 1, 2, 3, 4],
                                    p=[0.32, 0.29, 0.20, 0.12, 0.07]))
                    for _ in range(total)]
    for i, style in enumerate(style_plan):
        if style == "palindrome" and spine_counts[i] % 2:
            spine_counts[i] += 1
        if style == "monochrome":
            spine_counts[i] = 0
        if style == "distinct":
            spine_counts[i] = min(spine_counts[i], 1)
    if any(spine_counts[train_count:]) and not any(spine_counts[:train_count]):
        eligible = [i for i in range(train_count) if style_plan[i] != "monochrome"]
        i = int(rng.choice(eligible))
        spine_counts[i] = 2 if style_plan[i] == "palindrome" else 1
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(total):
        section = "train" if i < train_count else "test"
        for _ in range(128):
            pair = _pair(rng, widths[i], roles[i], style_plan[i],
                         depths[i], spine_counts[i])
            inp = pair["input"]
            info = _analyze_input(inp)
            if _key(inp) not in seen and _signature(inp, info) not in _SOURCE:
                seen.add(_key(inp))
                puzzle[section].append(pair)
                break
        else:
            raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid, info):
    return (grid.shape[0], grid.shape[1], info["start"], info["background"],
            info["spine"], tuple(int(v) for v in grid[0]))


def _analyze_input(grid):
    height, width = grid.shape
    _check(width % 2 == 1 and 5 <= width <= 29 and height >= 5,
           "invalid dimensions")
    center = width // 2
    background = int(grid[1, 0])
    _check(grid[0, center] == background and np.all(grid[1] == background),
           "center placeholder or gap missing")
    payload = np.concatenate((grid[0, :center], grid[0, center + 1:]))
    _check(np.all(payload != background), "background in template payload")
    starts = np.flatnonzero(grid[2:, center] != background)
    _check(len(starts) > 0, "spine missing")
    start = int(starts[0] + 2)
    spine = int(grid[start, center])
    _check(np.all(grid[1:start] == background), "gap not empty")
    _check(np.all(grid[start:, center] == spine), "spine discontinuous")
    mask = np.ones(width, dtype=bool)
    mask[center] = False
    _check(np.all(grid[start:, mask] == background), "unrelated input cells")
    _check(height - start >= 3, "insufficient growth depth")
    return {"background": background, "spine": spine, "start": start,
            "radius": center, "depth": height - start - 1,
            "payload": payload, "template": grid[0]}


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
               "invalid puzzle keys")
        _check(all(isinstance(puzzle[s], list) for s in ("train", "test")),
               "sections must be lists")
        nt, ne = len(puzzle["train"]), len(puzzle["test"])
        _check(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
        seen_inputs, seen_pairs = set(), set()
        records = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                       "invalid pair keys")
                for grid in pair.values():
                    _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                           "grid must be a two-dimensional array")
                    _check(all(1 <= d <= 30 for d in grid.shape), "grid size")
                    _check(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                    _check(np.all((0 <= grid) & (grid <= 9)), "grid color")
                inp, out = pair["input"], pair["output"]
                _check(inp.shape == out.shape, "dimensions changed")
                input_key, pair_key = _key(inp), (_key(inp), _key(out))
                _check(input_key not in seen_inputs and pair_key not in seen_pairs,
                       "duplicate input or pair")
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                info = _analyze_input(inp)
                _check(_signature(inp, info) not in _SOURCE, "exact source copy")
                _check(np.array_equal(out, _output(inp, info["background"],
                                                   info["spine"], info["start"])),
                       "output violates same-column expansion")
                _check(not np.array_equal(inp, out), "unchanged pair")
                records[section].append(info)
        train, test = records["train"], records["test"]
        anchor = train[0]
        _check(anchor["depth"] > anchor["radius"], "missing plateau demonstration")
        _check(len(set(int(v) for v in anchor["payload"])) >= 3,
               "training anchor lacks color order")
        _check(not np.array_equal(anchor["template"], anchor["template"][::-1]),
               "training anchor is symmetric")
        if any(r["depth"] < r["radius"] for r in test):
            _check(any(r["depth"] < r["radius"] for r in train),
                   "short-spine test lacks a training example")
        if any(np.any(r["payload"] == r["spine"]) for r in test):
            _check(any(np.any(r["payload"] == r["spine"]) for r in train),
                   "spine-colored template data not demonstrated")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed puzzle: {exc}") from exc


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_puzzle_key(p) for p in puzzles}) == 42, "repeated puzzles")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    shapes, regimes, roles, starts, templates = set(), set(), set(), set(), set()
    same_width = recolored = spine_data = palindromes = distinct = one_color = 0
    review = {name: 0 for name in (
        "source_roles", "test_same_width", "test_smaller", "test_larger",
        "near_palindrome", "rich_palette", "long_run", "alternating",
        "repeated_word", "unequal_halves", "start_outside_source_range",
        "spine_once", "spine_several", "train_short", "test_short",
        "train_edge", "test_edge", "test_plateau")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pairs = puzzle["train"] + puzzle["test"]
        records = [_analyze_input(p["input"]) for p in pairs]
        same_width += len({p["input"].shape[1] for p in pairs}) == 1
        recolored += len({(r["background"], r["spine"]) for r in records}) > 1
        train_widths = {p["input"].shape[1] for p in puzzle["train"]}
        test_widths = [p["input"].shape[1] for p in puzzle["test"]]
        review["test_same_width"] += any(w in train_widths for w in test_widths)
        review["test_smaller"] += any(w < min(train_widths) for w in test_widths)
        review["test_larger"] += any(w > max(train_widths) for w in test_widths)
        review["source_roles"] += any((r["background"], r["spine"]) == (5, 8)
                                      for r in records)
        for index, (pair, r) in enumerate(zip(pairs, records)):
            shapes.add(pair["input"].shape)
            regimes.add(int(np.sign(r["depth"] - r["radius"])))
            roles.add((r["background"], r["spine"]))
            starts.add(r["start"])
            templates.add(tuple(int(v) for v in r["payload"]))
            spine_data += bool(np.any(r["payload"] == r["spine"]))
            palindromes += bool(np.array_equal(r["template"], r["template"][::-1]))
            distinct += len(set(int(v) for v in r["payload"])) == len(r["payload"])
            one_color += len(set(int(v) for v in r["payload"])) == 1
            payload = [int(v) for v in r["payload"]]
            radius = r["radius"]
            left, right = payload[:radius], payload[radius:]
            errors = sum(a != b for a, b in zip(left, right[::-1]))
            review["near_palindrome"] += errors == 1
            review["rich_palette"] += len(set(payload)) >= 8
            review["long_run"] += any(payload[j] == payload[j + 1] == payload[j + 2]
                                      for j in range(len(payload) - 2))
            review["alternating"] += any(all(payload[j] == payload[j + 2]
                                              and payload[j] != payload[j + 1]
                                              for j in range(lo, lo + 4))
                                          for lo in range(len(payload) - 5))
            review["repeated_word"] += any(all(payload[j] == payload[j % period]
                                                for j in range(len(payload)))
                                           for period in range(2, min(5, radius) + 1))
            review["unequal_halves"] += ((len(set(left)) <= 2 and len(set(right)) >= 3)
                                         or (len(set(right)) <= 2 and len(set(left)) >= 3))
            review["start_outside_source_range"] += r["start"] < 3 or r["start"] > 7
            review["spine_once"] += payload.count(r["spine"]) == 1
            review["spine_several"] += payload.count(r["spine"]) >= 2
            split = "train" if index < len(puzzle["train"]) else "test"
            if r["depth"] < radius:
                review[split + "_short"] += 1
            elif r["depth"] == radius:
                review[split + "_edge"] += 1
            elif split == "test":
                review["test_plateau"] += 1
    _check(len(shapes) >= 50 and len(templates) >= 150, "low structural diversity")
    _check(len(roles) >= 20 and len(starts) >= 9, "low role or gap diversity")
    _check(regimes == {-1, 0, 1}, "missing depth regime")
    _check(all(v > 0 for v in (same_width, recolored, spine_data, palindromes,
                               distinct, one_color)), "reviewed variation missing")
    _check(all(v > 0 for v in review.values()), "review axis missing in fixed seeds")
    for counts in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(500 + sum(counts), *counts)
        _check((len(p["train"]), len(p["test"])) == counts, "explicit count ignored")
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(600 + (args[0] or 0) + (args[1] or 0), *args)
        _check(args[0] is None or len(p["train"]) == args[0], "partial train count")
        _check(args[1] is None or len(p["test"]) == args[1], "partial test count")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(777, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _check(_puzzle_key(generate_puzzle(314159, 7, 3)) ==
           _puzzle_key(generate_puzzle(314159, 7, 3)), "not deterministic")
