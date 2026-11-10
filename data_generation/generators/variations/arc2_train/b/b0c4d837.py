"""One upright vessel; encode its empty headspace in a 3 by 3 spiral."""

import numpy as np


_SPIRAL = ((0, 0), (0, 1), (0, 2), (1, 2))

# Exact original pairs are protected, including the solved test pair.
_SOURCE = (
    (("000000000", "000000000", "005000500", "005000500", "005888500",
      "005888500", "005888500", "005888500", "005555500"), ("880", "000", "000")),
    (("000000000", "005000500", "005000500", "005000500", "005000500",
      "005888500", "005888500", "005888500", "005555500"), ("888", "008", "000")),
    (("000000000", "000000000", "050000050", "050000050", "050000050",
      "058888850", "058888850", "058888850", "055555550"), ("888", "000", "000")),
    (("000000", "050050", "058850", "058850", "055550"), ("800", "000", "000")),
    (("000000", "050050", "050050", "050050", "058850", "055550"),
     ("888", "000", "000")),
    (("0000000", "0000000", "0500050", "0500050", "0588850", "0588850",
      "0555550"), ("880", "000", "000")),
    (("000000000", "005000500", "005888500", "005888500", "005888500",
      "005888500", "005888500", "005888500", "005555500"), ("800", "000", "000")),
)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_key(rows):
    return (len(rows), len(rows[0]), tuple(int(x) for row in rows for x in row))


_SOURCE_KEYS = {(_source_key(a), _source_key(b)) for a, b in _SOURCE}


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)
    ):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(name + " is outside its legal integer range")
    if num_train is None and num_test is None:
        # A complete count cycle occurs within 36 consecutive seeds.
        test = 1 + int(seed) % 4
        choices = list(range(max(2, test + 1), 11))
        train = choices[(int(seed) // 4) % len(choices)]
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, 3, replace=False))


def _output(background, payload, headspace):
    assert 1 <= headspace <= 4
    result = np.full((3, 3), background, dtype=np.int8)
    for r, c in _SPIRAL[:headspace]:
        result[r, c] = payload
    return result


def _construct(geometry, colors):
    top, left, right_margin, bottom_margin, width, depth, headspace = geometry
    background, vessel, payload = colors
    height = top + depth + 1 + bottom_margin
    canvas_width = left + width + 2 + right_margin
    assert 1 <= headspace < depth and height <= 30 and canvas_width <= 30
    base, right = top + depth, left + width + 1
    grid = np.full((height, canvas_width), background, dtype=np.int8)
    grid[top:base + 1, left] = vessel
    grid[top:base + 1, right] = vessel
    grid[base, left:right + 1] = vessel
    grid[top + headspace:base, left + 1:right] = payload
    return {"input": grid, "output": _output(background, payload, headspace)}


def _geometry(rng, headspace, regime):
    assert regime in ("source_train", "target_test")
    if regime == "source_train":
        width = int(rng.integers(1, 12))
        payload_height = int(rng.integers(1, 10))
        top = int(rng.integers(0, 7))
        bottom = int(rng.integers(0, 6))
        left = int(rng.integers(1, 7))
        right = int(rng.integers(1, 7))
    else:
        # Tests sample the same family with more extreme widths, fill heights,
        # and placement. None of these properties determines the answer.
        width = int(rng.integers(1, 15))
        payload_height = int(rng.integers(1, 11))
        top = int(rng.integers(0, 7))
        bottom = int(rng.integers(0, 7))
        left = int(rng.integers(1, 8))
        right = int(rng.integers(1, 8))
    return (top, left, right, bottom, width, headspace + payload_height, headspace)


def _analyze(grid):
    """Reconstruct the unique exact U and payload, without generation metadata."""
    colors = [int(x) for x in np.unique(grid)]
    assert len(colors) == 3
    candidates = []
    for background in colors:
        for vessel in colors:
            if vessel == background:
                continue
            payload = next(x for x in colors if x not in (background, vessel))
            vr, vc = np.nonzero(grid == vessel)
            pr, pc = np.nonzero(grid == payload)
            top, base = int(vr.min()), int(vr.max())
            left, right = int(vc.min()), int(vc.max())
            if right - left < 2 or base - top < 2:
                continue
            payload_top = int(pr.min())
            if not (int(pr.max()) == base - 1 and int(pc.min()) == left + 1
                    and int(pc.max()) == right - 1 and top < payload_top < base):
                continue
            expected = np.full(grid.shape, background, dtype=grid.dtype)
            expected[top:base + 1, left] = vessel
            expected[top:base + 1, right] = vessel
            expected[base, left:right + 1] = vessel
            expected[payload_top:base, left + 1:right] = payload
            if np.array_equal(grid, expected):
                candidates.append({
                    "colors": (background, vessel, payload),
                    "top": top, "base": base, "left": left, "right": right,
                    "bottom": grid.shape[0] - base - 1,
                    "right_margin": grid.shape[1] - right - 1,
                    "width": right - left - 1,
                    "headspace": payload_top - top,
                    "payload_height": base - payload_top,
                    "shape": grid.shape,
                })
    assert len(candidates) == 1
    return candidates[0]


def _shell(a):
    return (a["shape"], a["colors"], a["top"], a["base"], a["left"], a["right"])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analysis = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            key = (_key(inp), _key(out))
            assert key[0] not in seen and key not in _SOURCE_KEYS
            seen.add(key[0])
            a = _analyze(inp)
            analysis[split].append(a)
            assert 1 <= a["headspace"] <= 4
            assert out.shape == (3, 3)
            assert np.array_equal(out, _output(a["colors"][0], a["colors"][2],
                                               a["headspace"]))
            assert a["left"] >= 1 and a["right_margin"] >= 1
    train, test = analysis["train"], analysis["test"]
    train_heads = {a["headspace"] for a in train}
    assert 4 in train_heads and len(train_heads) >= 2
    assert all(a["headspace"] in train_heads for a in test)
    assert len({a["headspace"] for a in test}) == len(test)
    # A controlled shell contrast rules out counting vessel height or width.
    assert any(_shell(a) == _shell(b) and a["headspace"] != b["headspace"]
               for i, a in enumerate(train) for b in train[i + 1:])
    if len(train) >= 3:
        assert any(a["headspace"] == b["headspace"]
                   and a["payload_height"] != b["payload_height"]
                   and a["width"] != b["width"]
                   and (a["top"], a["left"], a["right_margin"], a["bottom"])
                       != (b["top"], b["left"], b["right_margin"], b["bottom"])
                   for i, a in enumerate(train) for b in train[i + 1:])
    train_palettes = {a["colors"] for a in train}
    if len(train_palettes) == 1:
        assert all(a["colors"] in train_palettes for a in test)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(seed, rng, num_train, num_test)
    base_colors = _palette(rng)
    recolor = n_train >= 3 and bool(rng.integers(0, 2))
    second_head = int(rng.integers(1, 4))

    # The first two examples share every shell property and all colors; only
    # the payload level changes. The four-cell example teaches the full
    # source-evidenced prefix of the clockwise output spiral.
    depth = int(rng.integers(6, 12))
    shell = (int(rng.integers(0, 7)), int(rng.integers(1, 7)),
             int(rng.integers(1, 7)), int(rng.integers(0, 6)),
             int(rng.integers(1, 12)), depth)
    anchor = [_construct(shell + (4,), base_colors),
              _construct(shell + (second_head,), base_colors)]
    if any((_key(pair["input"]), _key(pair["output"])) in _SOURCE_KEYS
           for pair in anchor):
        # Every source vessel has its base on the canvas edge.
        shell = shell[:3] + (1,) + shell[4:]
        anchor = [_construct(shell + (4,), base_colors),
                  _construct(shell + (second_head,), base_colors)]
    train_heads = [4, second_head]
    if n_train >= 3:
        train_heads.append(second_head)
    remaining = [x for x in (1, 2, 3) if x != second_head]
    rng.shuffle(remaining)
    train_heads.extend(remaining)
    while len(train_heads) < n_train:
        train_heads.append(int(rng.integers(1, 5)))
    train_heads = train_heads[:n_train]

    puzzle = {"train": anchor, "test": []}
    seen = {_key(pair["input"]) for pair in anchor}

    def add(split, headspace, regime, colors, contrast=False):
        # A bounded random phase is followed by a deterministic enumeration.
        # This makes even dense explicit-count requests total.
        geometries = []
        for _ in range(60):
            geometries.append(_geometry(rng, headspace, regime))
        for width in range(1, 15):
            for payload_height in range(1, 11):
                geometries.append((0, 1, 1, 0, width,
                                   headspace + payload_height, headspace))
        for geometry in geometries:
            if contrast and (geometry[4] == shell[4]
                             or geometry[5] - headspace == depth - second_head
                             or geometry[:4] == shell[:4]):
                continue
            pair = _construct(geometry, colors)
            key = _key(pair["input"])
            if key in seen or (key, _key(pair["output"])) in _SOURCE_KEYS:
                continue
            seen.add(key)
            puzzle[split].append(pair)
            return
        raise RuntimeError("unique vessel construction failed")

    for index in range(2, n_train):
        colors = base_colors
        if recolor:
            colors = _palette(rng)
            while colors == base_colors:
                colors = _palette(rng)
        add("train", train_heads[index], "source_train", colors, index == 2)

    rng.shuffle(puzzle["train"])
    demonstrated = sorted(set(train_heads))
    if n_test == 1:
        # A one-test puzzle never reveals its answer by copying train[0].
        eligible = [h for h in demonstrated
                    if h != _analyze(puzzle["train"][0]["input"])["headspace"]]
        test_heads = [int(rng.choice(eligible))]
    else:
        test_heads = [int(x) for x in rng.choice(demonstrated, n_test, replace=False)]
    rng.shuffle(test_heads)
    for headspace in test_heads:
        colors = _palette(rng) if recolor else base_colors
        add("test", headspace, "target_test", colors)

    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _signature(generate_puzzle(123, 7, 3)) == _signature(generate_puzzle(123, 7, 3))
    assert _signature(generate_puzzle(123)) != _signature(generate_puzzle(124))
    for n_train, n_test in ((2, 1), (10, 1), (10, 4), (7, 3),
                            (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(800 + (n_train or 0) * 10 + (n_test or 0),
                            n_train, n_test)
        assert n_train is None or len(p["train"]) == n_train
        assert n_test is None or len(p["test"]) == n_test
        validate_generated_puzzle(p)
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4)):
        try:
            generate_puzzle(999, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")

    heads, test_heads, widths, payloads, tops, bottoms, palettes, layouts = (
        set(), set(), set(), set(), set(), set(), set(), set())
    one_cell_tests = multivalued_tests = recolored = same_head_contrast = 0
    tall_train = short_train = tall_test = short_test = 0
    margin_coincidence = margin_noncoincidence = 0
    first_output_copies = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        train = [_analyze(x["input"]) for x in p["train"]]
        test = [_analyze(x["input"]) for x in p["test"]]
        one_cell_tests += any(a["headspace"] == 1 for a in test)
        multivalued_tests += len({a["headspace"] for a in test}) > 1
        recolored += len({a["colors"] for a in train}) > 1
        if len(p["test"]) == 1:
            first_output_copies += np.array_equal(
                p["test"][0]["output"], p["train"][0]["output"])
        same_head_contrast += any(
            a["headspace"] == b["headspace"] and a["width"] != b["width"]
            and a["payload_height"] != b["payload_height"]
            for i, a in enumerate(train) for b in train[i + 1:])
        for split, group in (("train", train), ("test", test)):
            for a in group:
                heads.add(a["headspace"])
                if split == "test":
                    test_heads.add(a["headspace"])
                widths.add(a["width"])
                payloads.add(a["payload_height"])
                tops.add(a["top"])
                bottoms.add(a["bottom"])
                palettes.add(a["colors"])
                layouts.add((a["shape"], a["top"], a["left"], a["width"],
                             a["headspace"], a["payload_height"]))
                margin_coincidence += a["top"] == a["headspace"]
                margin_noncoincidence += a["top"] != a["headspace"]
                if split == "train":
                    tall_train += a["payload_height"] >= 5
                    short_train += a["payload_height"] <= 4
                else:
                    tall_test += a["payload_height"] >= 5
                    short_test += a["payload_height"] <= 4
    assert heads == test_heads == {1, 2, 3, 4}
    assert 1 in widths and len(widths) >= 10
    assert len(payloads) >= 9 and 0 in tops and 0 in bottoms and max(bottoms) >= 3
    assert len(palettes) >= 30 and len(layouts) >= 150
    assert one_cell_tests >= 2 and multivalued_tests >= 5
    assert recolored >= 5 and same_head_contrast >= 10
    assert tall_train and short_train and tall_test and short_test
    assert margin_coincidence and margin_noncoincidence
    assert first_output_copies == 0
