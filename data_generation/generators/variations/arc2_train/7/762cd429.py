"""Generate ARC puzzles by doubling a 2 by 2 template to the right."""

import numpy as np


# (background, number of copies, symmetric padding, row-major template)
_ORIGINAL_INSTANCES = {
    (0, 3, 1, (2, 5, 5, 3)),
    (0, 3, 1, (2, 3, 1, 1)),
    (0, 4, 0, (1, 2, 3, 4)),
    (0, 4, 0, (1, 1, 8, 8)),
}
_PATTERNS = ("distinct", "three", "row", "column", "diagonal", "three_one", "single")


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _template(rng, background, pattern):
    count = {"distinct": 4, "three": 3, "single": 1}.get(pattern, 2)
    colors = rng.choice(np.array([c for c in range(10) if c != background]),
                        size=count, replace=False).astype(np.int8)
    if pattern == "distinct":
        labels = [0, 1, 2, 3]
    elif pattern == "three":
        labels = [0, 1, 2, 2]
        rng.shuffle(labels)
    elif pattern == "row":
        labels = [0, 0, 1, 1]
    elif pattern == "column":
        labels = [0, 1, 0, 1]
    elif pattern == "diagonal":
        labels = [0, 1, 1, 0]
    elif pattern == "three_one":
        labels = [0, 0, 0, 1]
        rng.shuffle(labels)
    elif pattern == "single":
        labels = [0, 0, 0, 0]
    else:
        raise ValueError("unknown template pattern")
    if pattern in ("row", "column", "diagonal") and rng.integers(2):
        colors = colors[::-1]
    return colors[np.array(labels)].reshape(2, 2)


def _layout(rng, copies, vertical, horizontal):
    scale = 1 << (copies - 1)
    max_margin = 30 - 2 * ((1 << copies) - 1)
    left = 0 if horizontal == "left" or max_margin == 0 else int(rng.integers(1, max_margin + 1))
    max_extra = 30 - 2 * scale
    if vertical == "center":
        extra = 2 * int(rng.integers(0, max_extra // 2 + 1))
        upper = extra // 2
    else:
        extra = int(rng.integers(1, max_extra + 1))
        possible = [p for p in range(extra + 1) if p != extra - p]
        upper = int(possible[int(rng.integers(len(possible)))])
    height = 2 * scale + extra
    top = upper + scale - 1
    return height, left, top


def _make_pair(background, template, copies, height, left, top):
    width = left + 2 * ((1 << copies) - 1)
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[top:top + 2, left:left + 2] = template
    boundary = top + 1
    x = left
    for exponent in range(copies):
        scale = 1 << exponent
        block = np.repeat(np.repeat(template, scale, axis=0), scale, axis=1)
        out[boundary - scale:boundary + scale, x:x + 2 * scale] = block
        x += 2 * scale
    return {"input": inp, "output": out}


def _source_copy(background, copies, height, left, top, template):
    if left or height % 2 or top != height // 2 - 1:
        return False
    padding = (height - 2 * (1 << (copies - 1))) // 2
    return (background, copies, padding, tuple(int(c) for c in template.flat)) in _ORIGINAL_INSTANCES


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _sample_pair(rng, background, copies, pattern, vertical, horizontal, used):
    for _ in range(80):
        template = _template(rng, background, pattern)
        height, left, top = _layout(rng, copies, vertical, horizontal)
        if _source_copy(background, copies, height, left, top, template):
            continue
        pair = _make_pair(background, template, copies, height, left, top)
        key = _signature(pair["input"])
        if key not in used:
            used.add(key)
            return pair
    # Systematic fallback avoids a seed-dependent rejection failure.
    labels = {
        "distinct": (0, 1, 2, 3), "three": (0, 1, 2, 2),
        "row": (0, 0, 1, 1), "column": (0, 1, 0, 1),
        "diagonal": (0, 1, 1, 0), "three_one": (0, 0, 0, 1),
        "single": (0, 0, 0, 0),
    }[pattern]
    available = [c for c in range(10) if c != background]
    scale = 1 << (copies - 1)
    max_margin = 30 - 2 * ((1 << copies) - 1)
    left_options = [0] if horizontal == "left" or max_margin == 0 else range(1, max_margin + 1)
    for offset in range(9):
        template = np.array([available[(label + offset) % 9] for label in labels],
                            dtype=np.int8).reshape(2, 2)
        for height in range(2 * scale, 31):
            if vertical == "center":
                top_options = [height // 2 - 1] if height % 2 == 0 else []
            else:
                top_options = [t for t in range(scale - 1, height - scale)
                               if 2 * (t + 1) != height]
            for left in left_options:
                for top in top_options:
                    if _source_copy(background, copies, height, left, top, template):
                        continue
                    pair = _make_pair(background, template, copies, height, left, top)
                    key = _signature(pair["input"])
                    if key not in used:
                        used.add(key)
                        return pair
    raise RuntimeError("no unique pair exists for this specification")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    # These are sampling regimes; no hidden parameter changes the puzzle rule.
    extrapolate = int(seed) % 4 == 0
    mixed_background = bool(rng.integers(2))
    base_background = int(rng.integers(10))
    other_background = int(rng.choice([c for c in range(10) if c != base_background]))
    used = set()

    def background(index):
        if not mixed_background:
            return base_background
        if index == 0:
            return base_background
        if index == 1:
            return other_background
        return int(rng.integers(10))

    train_specs = [
        (3 if extrapolate else 4, "distinct", "center", "left"),
        (3, _PATTERNS[1 + int(rng.integers(0, 6))], "shifted", "margin"),
    ]
    for _ in range(train_count - 2):
        copies = int(rng.choice([2, 3] if extrapolate else [2, 3, 4]))
        pattern = _PATTERNS[int(rng.integers(len(_PATTERNS)))]
        vertical = "center" if rng.integers(2) == 0 else "shifted"
        horizontal = "left" if rng.integers(2) == 0 else "margin"
        train_specs.append((copies, pattern, vertical, horizontal))
    rng.shuffle(train_specs)
    train = [_sample_pair(rng, background(i), copies, pattern, vertical, horizontal, used)
             for i, (copies, pattern, vertical, horizontal) in enumerate(train_specs)]

    test = []
    for index in range(test_count):
        if index == 0:
            copies = 4 if extrapolate else (2, 3, 4)[(int(seed) // 4) % 3]
            pattern = _PATTERNS[int(seed) % len(_PATTERNS)]
        else:
            copies = int(rng.integers(2, 5))
            pattern = _PATTERNS[int(rng.integers(len(_PATTERNS)))]
        vertical = "center" if rng.integers(2) == 0 else "shifted"
        horizontal = "left" if rng.integers(2) == 0 else "margin"
        test.append(_sample_pair(rng, background(train_count + index),
                                 copies, pattern, vertical, horizontal, used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    height, width = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    mask = inp != background
    assert int(mask.sum()) == 4
    rows, cols = np.where(mask)
    top, left = int(rows.min()), int(cols.min())
    assert top + 1 < height and left + 1 < width
    expected_mask = np.zeros(inp.shape, dtype=bool)
    expected_mask[top:top + 2, left:left + 2] = True
    assert np.array_equal(mask, expected_mask)
    template = inp[top:top + 2, left:left + 2]
    assert np.all(template != background)
    span = width - left
    copies = {6: 2, 14: 3, 30: 4}.get(span)
    assert copies is not None
    largest = 1 << (copies - 1)
    boundary = top + 1
    assert boundary >= largest and height - boundary >= largest
    expected = np.full(inp.shape, background, dtype=np.int8)
    x = left
    for exponent in range(copies):
        scale = 1 << exponent
        for tr in range(2):
            for tc in range(2):
                expected[boundary - scale + tr * scale:boundary - scale + (tr + 1) * scale,
                         x + tc * scale:x + (tc + 1) * scale] = template[tr, tc]
        x += 2 * scale
    assert x == width
    assert np.array_equal(out, expected)
    assert not _source_copy(background, copies, height, left, top, template)
    return {"background": background, "copies": copies, "height": height,
            "width": width, "top": top, "left": left,
            "distinct": len(np.unique(template)), "template": template.copy(),
            "shifted": 2 * boundary != height}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.min(grid) >= 0 and np.max(grid) <= 9
            inp_sig = _signature(pair["input"])
            pair_sig = (inp_sig, _signature(pair["output"]))
            assert inp_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(inp_sig)
            seen_pairs.add(pair_sig)
            info[split].append(_analyze_pair(pair["input"], pair["output"]))
    train = info["train"]
    assert any(item["distinct"] == 4 for item in train)
    assert any(item["distinct"] < 4 for item in train)
    assert any(item["copies"] == 3 for item in train)
    assert any(item["left"] == 0 and not item["shifted"] for item in train)
    assert any(item["left"] > 0 and item["shifted"] for item in train)
    assert max(item["copies"] for item in info["test"]) <= max(item["copies"] for item in train) + 1


def _fingerprint(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _pattern_name(template):
    a, b, c, d = (int(v) for v in template.flat)
    distinct = len({a, b, c, d})
    if distinct == 4:
        return "distinct"
    if distinct == 3:
        return "three"
    if distinct == 1:
        return "single"
    if a == b and c == d:
        return "row"
    if a == c and b == d:
        return "column"
    if a == d and b == c:
        return "diagonal"
    return "three_one"


def self_test() -> None:
    seen = set()
    train_counts, test_counts = set(), set()
    test_lengths, test_patterns, shapes = set(), set(), set()
    extrapolated = mixed = odd = shifted_test = margined_test = 0
    combinations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        seen.add(_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        mixed += len({p["background"] for p in train_info + test_info}) > 1
        extrapolated += max(p["copies"] for p in train_info) == 3 and any(p["copies"] == 4 for p in test_info)
        for p in train_info + test_info:
            shapes.add((p["height"], p["width"]))
            odd += p["height"] % 2
        for p in test_info:
            test_lengths.add(p["copies"])
            pattern = _pattern_name(p["template"])
            test_patterns.add(pattern)
            shifted_test += p["shifted"]
            margined_test += p["left"] > 0
            combinations.add((p["copies"], pattern, p["left"] > 0, p["shifted"]))
    assert len(seen) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert test_lengths == {2, 3, 4} and test_patterns == set(_PATTERNS)
    assert extrapolated >= 5 and mixed >= 8
    assert odd >= 20 and shifted_test >= 15 and margined_test >= 10
    assert len(shapes) >= 40 and len(combinations) >= 25
    for seed, ntrain, ntest in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                                (1101, None, 1), (1102, None, 4),
                                (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    assert _fingerprint(generate_puzzle(987654, 7, 3)) == _fingerprint(generate_puzzle(987654, 7, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_test": False}):
        try:
            generate_puzzle(2000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
