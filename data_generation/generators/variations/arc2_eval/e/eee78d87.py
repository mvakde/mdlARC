"""Extract a 3x3 pattern, tile it, and recolor its central four copies.

The summary's fixed hash-shaped overlay was incorrect: carrier cells survive
EVERYWHERE. Only accent cells acquire the common ground/highlight colors.
Output size and replacement colors are puzzle conventions; input container,
placement, carrier, accent and motif are observable pair-level data.
"""

import numpy as np


_OUTPUT_SIZES = (10, 16, 22, 28)
_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0))
_ORDERS = tuple((transpose, rows, cols) for transpose in (False, True)
                for rows in _PERMUTATIONS for cols in _PERMUTATIONS)
_DIRECT = _ORDERS.index((False, (1, 2, 0), (1, 2, 0)))


def _full_box(mask):
    return (mask.shape == (3, 3) and 1 < int(mask.sum()) < 9
            and all((mask[0].any(), mask[2].any(),
                     mask[:, 0].any(), mask[:, 2].any())))


# Exhaust the binary space, not a hand-written vocabulary. Empty internal rows
# and columns are permitted: only the four bounding extrema must be occupied.
_MASKS = np.array([m for code in range(1, 511)
                   if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                              dtype=bool).reshape(3, 3))])
_DENSITY = _MASKS.sum(axis=(1, 2))
_CENTERS = _MASKS[:, 1, 1]
_MASK_IDS = {m.tobytes(): i for i, m in enumerate(_MASKS)}
_CLASSICS = ("010111010", "101111101", "101010101", "101000101")
_CLASSIC_IDS = tuple(_MASK_IDS[np.array([x == '1' for x in s]).reshape(3, 3).tobytes()]
                     for s in _CLASSICS)


def _ordered(mask, order):
    transpose, rows, cols = _ORDERS[order]
    return (mask.T if transpose else mask)[np.ix_(rows, cols)]


# The training pair selector uses actual mask symmetries to rule out all 71
# alternatives, including independent row/column permutations and transpose.
_AGREEMENT = np.array([[np.array_equal(_ordered(m, k), _ordered(m, _DIRECT))
                        for k in range(len(_ORDERS))] for m in _MASKS])


# Literal source inputs and output rows, transcribed from the staged evidence.
# Used both for exact-label regressions and content-based source-copy exclusion.
_SOURCE_INPUTS = (
    "777777 777777 777777 777737 777333 777737",
    "777777 717177 711177 717177 777777 777777",
    "777777 767677 776777 767677 777777 777777",
    "772727 777777 772727 777777 777777 777777",
)
_SOURCE_ROWS = (
    ("0000000000000000", "0770770770770770", "0770779779770770",
     "0000099999900000"),
    ("0000000000000000", "7007007007007007", "7007097997907007",
     "0000099999900000"),
    ("0770770770770770", "7007007007007007", "7007097997907007",
     "0770779779770770"),
    ("7777777777777777", "7007007007007007", "7007097997907007",
     "7777777777777777"),
)
_SOURCE_ROW_SEQUENCE = (0, 1, 1, 0, 1, 2, 3, 2, 2, 3, 2, 1, 0, 1, 1, 0)


def _source_pairs():
    return [{"input": np.array([[int(c) for c in row] for row in text.split()], dtype=np.int8),
             "output": np.array([[int(c) for c in rows[k]]
                                  for k in _SOURCE_ROW_SEQUENCE], dtype=np.int8)}
            for text, rows in zip(_SOURCE_INPUTS, _SOURCE_ROWS)]


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


_SOURCE_KEYS = {_pair_key(pair) for pair in _source_pairs()}


def _normalize_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _normalize_count(num_train, "num_train", 2, 10)
    test = _normalize_count(num_test, "num_test", 1, 4)
    code = abs(int(seed))
    if train is None and test is None:
        test = 1 + code % 4
        train = test + 1 + (code // 4) % (10 - test)
    elif train is None:
        train = test + 1 + code % (10 - test)
    elif test is None:
        test = 1 + code % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _analyze_input(grid):
    assert min(grid.shape) >= 4 and grid.size >= 20
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    carrier = int(colors[np.argmax(counts)])
    accent = int(colors[np.argmin(counts)])
    assert int(counts.max()) > grid.size // 2
    coords = np.argwhere(grid == accent)
    top, left = (int(x) for x in coords.min(axis=0))
    bottom, right = (int(x) for x in coords.max(axis=0))
    assert (bottom - top, right - left) == (2, 2)
    mask = grid[top:bottom + 1, left:right + 1] == accent
    assert _full_box(mask)
    return {"carrier": carrier, "accent": accent, "top": top, "left": left,
            "mask": mask, "mask_id": _MASK_IDS[mask.tobytes()]}


def _central(shape):
    result = np.zeros(shape, dtype=bool)
    top, left = ((n - 6) // 2 for n in shape)
    result[top:top + 6, left:left + 6] = True
    return result


def _render_output(mask, carrier, ground, highlight, shape):
    rows = (np.arange(shape[0]) + 1) % 3
    cols = (np.arange(shape[1]) + 1) % 3
    occupancy = mask[rows[:, None], cols[None, :]]
    result = np.full(shape, carrier, dtype=np.int8)
    result[occupancy] = ground
    result[occupancy & _central(shape)] = highlight
    return result


def _reference_output(grid, shape, ground, highlight):
    """Separate implementation: stamp complete patches at physical origins."""
    info = _analyze_input(grid)
    result = np.full(shape, info["carrier"], dtype=np.int8)
    top, left = ((n - 6) // 2 for n in shape)
    for r in range(-1, shape[0], 3):
        for c in range(-1, shape[1], 3):
            for dr, dc in np.argwhere(info["mask"]):
                y, x = r + int(dr), c + int(dc)
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    result[y, x] = (highlight if top <= y < top + 6
                                    and left <= x < left + 6 else ground)
    return result


def _learn_conventions(train):
    """Recover roles, dimensions and axis order from demonstrations only."""
    conventions = None
    orders = set(range(len(_ORDERS)))
    for pair in train:
        info = _analyze_input(pair["input"])
        out = pair["output"]
        shape = out.shape
        assert all(n in _OUTPUT_SIZES for n in shape)
        central = _central(shape)
        outer = set(int(x) for x in out[~central]) - {info["carrier"]}
        inner = set(int(x) for x in out[central]) - {info["carrier"]}
        assert len(outer) == len(inner) == 1
        ground, highlight = next(iter(outer)), next(iter(inner))
        assert len({ground, highlight, info["carrier"], info["accent"]}) == 4
        current = (shape, ground, highlight)
        assert conventions is None or conventions == current
        conventions = current
        observed = out[:3, :3] != info["carrier"]
        orders &= {k for k in orders if np.array_equal(_ordered(info["mask"], k), observed)}
        assert orders
    return conventions, sorted(orders)


def _predict(train, test_input):
    """Prediction API receives no seed, test label or generation metadata."""
    (shape, ground, highlight), orders = _learn_conventions(train)
    info = _analyze_input(test_input)
    answers = []
    for order in orders:
        tile = np.where(_ordered(info["mask"], order), ground, info["carrier"])
        answer = np.tile(tile, ((shape[0] + 2) // 3, (shape[1] + 2) // 3))[:shape[0], :shape[1]].copy()
        middle = _central(shape)
        answer[middle & (answer == ground)] = highlight
        answers.append(answer.astype(np.int8))
    assert all(np.array_equal(answers[0], other) for other in answers)
    return answers[0]


def _choose_mask(rng, allowed):
    candidates = np.flatnonzero(allowed)
    assert candidates.size
    classics = [k for k in _CLASSIC_IDS if allowed[k]]
    if classics and rng.random() < 0.22:
        return int(rng.choice(classics))
    # Equalize available densities, so rare two-cell and eight-cell motifs are
    # not drowned out by the many medium-density masks.
    density = rng.choice(np.unique(_DENSITY[candidates]))
    return int(rng.choice(candidates[_DENSITY[candidates] == density]))


def _sample_shape(rng, complexity, different=None):
    maximum = (8, 16, 30)[complexity]
    choices = [(h, w) for h in range(4, maximum + 1)
               for w in range(4, maximum + 1) if h * w >= 20 and (h, w) != different]
    if different != (6, 6) and rng.random() < (0.6, 0.25, 0.1)[complexity]:
        return (6, 6)
    return choices[int(rng.integers(len(choices)))]


def _positions(shape):
    groups = {}
    for r in range(shape[0] - 2):
        for c in range(shape[1] - 2):
            row_kind = 0 if r == 0 else 2 if r == shape[0] - 3 else 1
            col_kind = 0 if c == 0 else 2 if c == shape[1] - 3 else 1
            groups.setdefault((row_kind, col_kind), []).append((r, c))
    return groups


def _make_pair(rng, mask_id, shape, carrier, accent, convention, used,
               excluded_positions=()):
    output_shape, ground, highlight = convention
    out = _render_output(_MASKS[mask_id], carrier, ground, highlight, output_shape)
    groups = list(_positions(shape).values())
    # Randomize categories first, then locations within a category. Enumerating
    # the finite candidates also provides a guaranteed, non-rejection fallback.
    for group_index in rng.permutation(len(groups)):
        positions = groups[int(group_index)]
        for index in rng.permutation(len(positions)):
            top, left = positions[int(index)]
            if (top, left) in excluded_positions:
                continue
            inp = np.full(shape, carrier, dtype=np.int8)
            inp[top:top + 3, left:left + 3][_MASKS[mask_id]] = accent
            pair = {"input": inp, "output": out.copy()}
            if _grid_key(inp) not in used and _pair_key(pair) not in _SOURCE_KEYS:
                return pair
    raise AssertionError("finite placement space exhausted")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    palette = [int(c) for c in rng.permutation(10)]
    ground, highlight = palette[:2]
    if rng.random() < 0.15:
        ground, highlight = 0, 9
        palette = [ground, highlight] + [c for c in palette if c not in (ground, highlight)]
    complexity = int(rng.integers(3))
    ncarriers = int(rng.integers(2, 5))
    carriers = palette[2:2 + ncarriers]
    accents = palette[2 + ncarriers:]
    nknown = int(rng.integers(2, min(len(accents) - 1, max(2, ntrain)) + 1))
    known = accents[:nknown]
    output_shape = tuple(int(rng.choice(_OUTPUT_SIZES, p=(0.2, 0.4, 0.2, 0.2))) for _ in range(2))
    convention = (output_shape, ground, highlight)
    # Regimes change the distribution of visible data, never the transformation.
    regime = str(rng.choice(("center_shift", "mixed", "denser", "sparser"),
                            p=(0.25, 0.35, 0.2, 0.2)))
    train_allowed = np.ones(len(_MASKS), dtype=bool)
    if regime == "center_shift":
        train_allowed &= _CENTERS
    elif regime == "denser":
        train_allowed &= _DENSITY <= 5
    elif regime == "sparser":
        train_allowed &= _DENSITY >= 5
    first = _choose_mask(rng, train_allowed)
    second_allowed = train_allowed & (_DENSITY != _DENSITY[first])
    second_allowed &= (_AGREEMENT & _AGREEMENT[first]).sum(axis=1) == 1
    if regime != "center_shift":
        second_allowed &= _CENTERS != _CENTERS[first]
    second = _choose_mask(rng, second_allowed)
    train_ids = [first, second]
    repeat_index = int(rng.integers(2, ntrain)) if ntrain >= 4 and rng.random() < 0.35 else -1
    repeat_source = int(rng.integers(2))
    for index in range(2, ntrain):
        if index == repeat_index:
            train_ids.append(train_ids[repeat_source])
        else:
            allowed = train_allowed.copy()
            allowed[train_ids] = False
            train_ids.append(_choose_mask(rng, allowed))
    test_ids = []
    first_test_center = bool(rng.integers(2))
    for index in range(ntest):
        allowed = np.ones(len(_MASKS), dtype=bool)
        if regime == "center_shift":
            allowed &= ~_CENTERS
        else:
            center = first_test_center if index % 2 == 0 else not first_test_center
            allowed &= _CENTERS == center
        if regime == "denser":
            allowed &= _DENSITY >= 6
        elif regime == "sparser":
            allowed &= _DENSITY <= 4
        familiar = allowed.copy()
        familiar[:] = False
        familiar[train_ids] = allowed[train_ids]
        familiar[test_ids] = False
        if regime == "mixed" and familiar.any() and rng.random() < 0.35:
            allowed = familiar
        else:
            allowed[train_ids] = False
        allowed[test_ids] = False
        test_ids.append(_choose_mask(rng, allowed))

    puzzle = {"train": [], "test": []}
    used = set()
    for split, ids in (("train", train_ids), ("test", test_ids)):
        for index, mask_id in enumerate(ids):
            if split == "train" and index == repeat_index:
                original = puzzle["train"][repeat_source]
                info = _analyze_input(original["input"])
                shape = original["input"].shape
                carrier, accent = info["carrier"], info["accent"]
            else:
                different = puzzle["train"][0]["input"].shape if split == "train" and index == 1 else None
                shape = _sample_shape(rng, complexity, different)
                carrier = carriers[index] if split == "train" and index < 2 else int(rng.choice(carriers))
                if split == "train":
                    accent = known[index] if index < nknown else int(rng.choice(known))
                else:
                    demonstrated = sorted({_analyze_input(p["input"])["accent"]
                                           for p in puzzle["train"]})
                    unseen = [c for c in accents if c not in demonstrated]
                    accent = int(rng.choice(demonstrated if rng.random() < 0.5 else unseen))
            previous = [_analyze_input(p["input"]) for p in puzzle["train"] + puzzle["test"]]
            excluded = [(p["top"], p["left"]) for p in previous if p["mask_id"] == mask_id]
            pair = _make_pair(rng, mask_id, shape, carrier, accent, convention, used, excluded)
            puzzle[split].append(pair)
            used.add(_grid_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            ik, pk = _grid_key(pair["input"]), _pair_key(pair)
            assert ik not in inputs and pk not in pairs and pk not in _SOURCE_KEYS
            assert not np.array_equal(pair["input"], pair["output"])
            inputs.add(ik)
            pairs.add(pk)
            infos[split].append(_analyze_input(pair["input"]))
    convention, orders = _learn_conventions(puzzle["train"])
    assert orders == [_DIRECT], "training must establish orientation"
    shape, ground, highlight = convention
    all_infos = infos["train"] + infos["test"]
    carriers = {i["carrier"] for i in all_infos}
    accents = {i["accent"] for i in all_infos}
    assert carriers.isdisjoint(accents)
    assert not (carriers | accents) & {ground, highlight}
    train_ids = [i["mask_id"] for i in infos["train"]]
    test_ids = [i["mask_id"] for i in infos["test"]]
    assert len(set(train_ids)) >= len(train_ids) - 1
    assert len(set(test_ids)) == len(test_ids)
    assert len(set(int(_DENSITY[k]) for k in train_ids)) >= 2
    assert len({i["carrier"] for i in infos["train"]}) >= 2
    assert len({i["accent"] for i in infos["train"]}) >= 2
    assert len({p["input"].shape for p in puzzle["train"]}) >= 2
    # Observable regime guarantees, without trusting the sampled regime name.
    if all(_CENTERS[k] for k in train_ids):
        assert all(not _CENTERS[k] for k in test_ids)
    else:
        assert {bool(_CENTERS[k]) for k in train_ids} == {False, True}
        if len(test_ids) > 1:
            assert {bool(_CENTERS[k]) for k in test_ids} == {False, True}
    locations = {}
    for split in ("train", "test"):
        for pair, info in zip(puzzle[split], infos[split]):
            out, mask = pair["output"], info["mask"]
            assert out.shape == shape
            assert np.array_equal(out, _reference_output(pair["input"], shape, ground, highlight))
            assert int(np.count_nonzero(out == highlight)) == 4 * int(mask.sum())
            normalized = np.tile(_ordered(mask, _DIRECT),
                                 ((shape[0] + 2) // 3, (shape[1] + 2) // 3))[:shape[0], :shape[1]]
            assert np.array_equal(out == info["carrier"], ~normalized)
            assert np.array_equal(out == highlight, normalized & _central(shape))
            pos = (info["top"], info["left"])
            assert pos not in locations.setdefault(info["mask_id"], set())
            locations[info["mask_id"]].add(pos)
    training_copies = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
    for pair in puzzle["test"]:
        predicted = _predict(training_copies, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _components(mask, diagonal=False):
    pending = {tuple(int(x) for x in p) for p in np.argwhere(mask)}
    count = 0
    while pending:
        count += 1
        stack = [pending.pop()]
        while stack:
            r, c = stack.pop()
            neighbors = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                         if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)}
            found = neighbors & pending
            pending -= found
            stack.extend(found)
    return count


def _fingerprint(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    source = _source_pairs()
    for pair in source:
        info = _analyze_input(pair["input"])
        assert np.array_equal(_render_output(info["mask"], 7, 0, 9, (16, 16)), pair["output"])
        assert np.array_equal(_reference_output(pair["input"], (16, 16), 0, 9), pair["output"])
    # All source-compatible orders give the same unseen source-test answer.
    prediction = _predict(source[:3], source[3]["input"].copy())
    assert np.array_equal(prediction, source[3]["output"])
    assert len(_MASKS) == 321
    # Exhaustive motif/size regression, including empty interior rows/columns.
    for mask in _MASKS:
        grid = np.full((6, 6), 7, dtype=np.int8)
        grid[1:4, 1:4][mask] = 2
        for h in _OUTPUT_SIZES:
            for w in _OUTPUT_SIZES:
                assert np.array_equal(_render_output(mask, 7, 0, 9, (h, w)),
                                      _reference_output(grid, (h, w), 0, 9))
    train_counts, test_counts = set(), set()
    motifs, test_motifs, outputs, shapes, sizes, palettes = set(), set(), set(), set(), set(), set()
    densities, connectivity, diagonal_connectivity, highlights = set(), set(), set(), set()
    positions = {s: set() for s in ("train", "test")}
    colors = {s: set() for s in ("carrier", "accent")}
    source_shifts = mixed_splits = familiar = novel = familiar_colors = novel_colors = repeated = 0
    density_up = density_down = holes = empty_middle = combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        convention, orders = _learn_conventions(puzzle["train"])
        assert orders == [_DIRECT]
        sizes.add(convention[0])
        palettes.add(convention[1:])
        train = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test = [_analyze_input(p["input"]) for p in puzzle["test"]]
        tids = {i["mask_id"] for i in train}
        train_accents = {i["accent"] for i in train}
        source_shifts += all(i["mask"][1, 1] for i in train) and all(not i["mask"][1, 1] for i in test)
        mixed_splits += len({bool(i["mask"][1, 1]) for i in train}) == len({bool(i["mask"][1, 1]) for i in test}) == 2
        density_up += min(int(i["mask"].sum()) for i in test) > max(int(i["mask"].sum()) for i in train)
        density_down += max(int(i["mask"].sum()) for i in test) < min(int(i["mask"].sum()) for i in train)
        repeated += len({_grid_key(p["output"]) for p in puzzle["train"]}) < len(train)
        for i in test:
            familiar += i["mask_id"] in tids
            novel += i["mask_id"] not in tids
            familiar_colors += i["accent"] in train_accents
            novel_colors += i["accent"] not in train_accents
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                mask = info["mask"]
                h, w = pair["input"].shape
                r, c = info["top"], info["left"]
                category = (0 if r == 0 else 2 if r == h - 3 else 1,
                            0 if c == 0 else 2 if c == w - 3 else 1)
                positions[split].add(category)
                motifs.add(info["mask_id"])
                if split == "test":
                    test_motifs.add(info["mask_id"])
                outputs.add(_grid_key(pair["output"]))
                shapes.add((h, w))
                densities.add(int(mask.sum()))
                connectivity.add(_components(mask))
                diagonal_connectivity.add(_components(mask, True))
                highlights.add(int((pair["output"] == convention[2]).sum()))
                holes += not mask[1, 1] and all(mask[r, c] for r, c in ((0, 1), (1, 0), (1, 2), (2, 1)))
                empty_middle += not mask[1].any() or not mask[:, 1].any()
                for role in colors:
                    colors[role].add(info[role])
                asymmetric = not np.array_equal(mask, mask[::-1]) or not np.array_equal(mask, mask[:, ::-1])
                combined += asymmetric and h != w and category != (1, 1) and info["carrier"] != 7 and convention[0] != (16, 16)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(motifs) >= 100 and len(test_motifs) >= 45 and len(outputs) >= 250
    assert set(_CLASSIC_IDS) <= motifs and densities == set(range(2, 9))
    assert len(shapes) >= 70 and len(sizes) >= 12 and len(palettes) >= 20
    assert len(connectivity) >= 4 and len(diagonal_connectivity) >= 2
    assert highlights == {4 * n for n in range(2, 9)}
    assert all(len(v) == 9 for v in positions.values())
    assert all(len(v) == 10 for v in colors.values())
    assert source_shifts >= 3 and mixed_splits >= 8
    assert density_up >= 3 and density_down >= 3
    assert familiar >= 3 and novel >= 40 and familiar_colors >= 20 and novel_colors >= 20
    assert repeated >= 3 and holes >= 3 and empty_middle >= 10 and combined >= 50
    explicit = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
    partial = [(a, None) for a in range(2, 11)] + [(None, b) for b in range(1, 5)]
    for index, (a, b) in enumerate(explicit + partial):
        puzzle = generate_puzzle(1000 + index, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(1000 + index, a, b))
    assert _fingerprint(generate_puzzle(777)) == _fingerprint(generate_puzzle(777))
    assert len({_fingerprint(generate_puzzle(2000 + k, 6, 2)) for k in range(12)}) == 12
    illegal = [(1, None), (11, None), (None, 0), (None, 5), (True, 1), (2.0, 1)]
    illegal += [(a, b) for a in range(2, 5) for b in range(a, 5)]
    for a, b in illegal:
        try:
            generate_puzzle(999, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Reject both a damaged carrier and the old fixed-stencil failure mode.
    for old_stencil in (False, True):
        damaged = generate_puzzle(2026, 3, 1)
        out = damaged["test"][0]["output"]
        _, ground, highlight = _learn_conventions(damaged["train"])[0]
        if old_stencil:
            rows, cols = np.indices(out.shape)
            central = _central(out.shape)
            out[out == highlight] = ground
            out[central & ((rows % 3 == 0) | (cols % 3 == 0))] = highlight
        else:
            out[0, 0] = (int(out[0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            continue
        raise AssertionError("corrupted output accepted")
