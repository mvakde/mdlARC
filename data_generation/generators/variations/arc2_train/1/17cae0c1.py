"""Generate puzzles that learn a tile-glyph to solid-color dictionary."""

import numpy as np


_SOURCE_INPUTS = (
    ("555555000", "000505000", "000555555"),
    ("005000000", "050000050", "500555000"),
    ("000555555", "050000505", "000000555"),
    ("555000005", "505050050", "555000500"),
    ("000005555", "000050000", "555500000"),
)
_SOURCE_OUTPUT_ROWS = ("666333111", "999111444", "444666333", "333444999", "111999666")
_SOURCE_PAIRS = frozenset(
    ((3, 9, "".join(rows)), (3, 9, output * 3))
    for rows, output in zip(_SOURCE_INPUTS, _SOURCE_OUTPUT_ROWS)
)


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    nt = _check_count("num_train", num_train, 2, 10)
    ns = _check_count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _source_masks():
    return [
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
    ]


def _mask_bank(h, w):
    bank = {}

    def add(points):
        mask = np.zeros((h, w), dtype=np.uint8)
        for r, c in points:
            mask[r, c] = 1
        count = int(mask.sum())
        if 1 <= count <= h * w // 2:
            bank[mask.tobytes()] = mask

    for r in range(h):
        add((r, c) for c in range(w))
    for c in range(w):
        add((r, c) for r in range(h))
    for r in (0, h // 2, h - 1):
        for c in (0, w // 2, w - 1):
            add(((r, c),))
    for shift in range(w):
        add((r, (r + shift) % w) for r in range(h))
        add((r, (shift - r) % w) for r in range(h))
    for r in (0, h - 1):
        for c in (0, w - 1):
            add([(r, j) for j in range(w)] + [(i, c) for i in range(h)])
    for r in (0, h - 1):
        add([(r, j) for j in range(w)] + [(i, w // 2) for i in range(h)])
    for c in (0, w - 1):
        add([(i, c) for i in range(h)] + [(h // 2, j) for j in range(w)])
    add([(h // 2, j) for j in range(w)] + [(i, w // 2) for i in range(h)])
    add([(i, i * (w - 1) // (h - 1)) for i in range(h)])
    add([(i, w - 1 - i * (w - 1) // (h - 1)) for i in range(h)])
    add([(i, i * (w - 1) // (h - 1)) for i in range(h)] +
        [(i, w - 1 - i * (w - 1) // (h - 1)) for i in range(h)])
    return list(bank.values())


def _choose_masks(rng, h, w, k, source_style):
    if source_style:
        return _source_masks()
    bank = _mask_bank(h, w)
    sparse = [m for m in bank if int(m.sum()) == 1]
    by_count = {}
    for mask in bank:
        by_count.setdefault(int(mask.sum()), []).append(mask)
    groups = [group for count, group in by_count.items() if count > 1 and len(group) >= 2]
    group = groups[int(rng.integers(len(groups)))]
    chosen = [sparse[int(rng.integers(len(sparse)))]]
    indices = rng.choice(len(group), 2, replace=False)
    chosen.extend(group[int(i)] for i in indices)
    used = {mask.tobytes() for mask in chosen}
    if len(chosen) < k:
        for index in rng.permutation(len(bank)):
            mask = bank[int(index)]
            key = mask.tobytes()
            if key in used or any(np.array_equal(mask, 1 - other) for other in chosen):
                continue
            chosen.append(mask)
            used.add(key)
            if len(chosen) == k:
                break
    assert len(chosen) == k
    if rng.integers(0, 2):
        dense = []
        for points in (
            [(h // 2, j) for j in range(w)] + [(i, w // 2) for i in range(h)],
            [(i, i * (w - 1) // (h - 1)) for i in range(h)] +
            [(i, w - 1 - i * (w - 1) // (h - 1)) for i in range(h)],
        ):
            mask = np.zeros((h, w), dtype=np.uint8)
            for r, c in points:
                mask[r, c] = 1
            if mask.tobytes() not in used and mask.sum() < h * w:
                dense.append(mask)
        if dense:
            candidate = dense[int(rng.integers(len(dense)))]
            replacement = chosen[:-1] + [candidate]
            if (2 * sum(int(mask.sum()) for mask in replacement) < k * h * w
                    and not any(np.array_equal(candidate, 1 - other)
                                for other in chosen[:-1])):
                chosen = replacement
    return chosen


def _grid_key(grid):
    return (grid.shape[0], grid.shape[1], "".join(str(int(v)) for v in grid.flat))


def _make_pair(sequence, rows, cols, masks, labels, background, inks):
    h, w = masks[0].shape
    inp = np.full((rows * h, cols * w), background, dtype=np.int8)
    out = np.empty_like(inp)
    for pos, motif in enumerate(sequence):
        r, c = divmod(pos, cols)
        tile = inp[r * h:(r + 1) * h, c * w:(c + 1) * w]
        tile[masks[motif] != 0] = inks[pos]
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = labels[motif]
    return {"input": inp, "output": out}


def _layout_options(h, w):
    return [(r, c) for r in range(1, min(5, 30 // h) + 1)
            for c in range(1, min(6, 30 // w) + 1)
            if 3 <= r * c <= 14]


def _sequence(rng, slots, k, sparse, subset=None, distinct=False):
    if distinct:
        sequence = list(int(i) for i in rng.permutation(k)[:slots])
    elif subset is not None:
        sequence = list(subset)
        while len(sequence) < slots:
            sequence.append(int(subset[int(rng.integers(len(subset)))]))
        rng.shuffle(sequence)
    else:
        sequence = [int(i) for i in rng.integers(0, k, size=slots)]
    return sequence


def _less_dense(sequence, masks, sparse, required=()):
    area = masks[0].size
    sequence = list(sequence)
    required_positions = set()
    for glyph in required:
        required_positions.add(sequence.index(glyph))
    while 2 * sum(int(masks[i].sum()) for i in sequence) >= len(sequence) * area:
        options = [p for p, i in enumerate(sequence)
                   if p not in required_positions and i != sparse]
        if not options:
            raise AssertionError("cannot make background dominant")
        sequence[options[-1]] = sparse
    return sequence


def _random_colors(rng, background, slots, first=None):
    palette = [v for v in range(10) if v != background and v != first]
    return [int(palette[int(i)]) for i in rng.integers(0, len(palette), size=slots)]


def _build(rng, n_train, n_test):
    source_style = bool(rng.integers(0, 7) == 0)
    sizes = ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (5, 5))
    h, w = (3, 3) if source_style else sizes[int(rng.integers(len(sizes)))]
    k = 5 if source_style else int(rng.integers(3, 7))
    masks = _choose_masks(rng, h, w, k, source_style)
    palette = rng.choice(np.arange(1, 10), size=k, replace=False)
    labels = [int(v) for v in palette]
    sparse = min(range(k), key=lambda i: int(masks[i].sum()))
    layouts = _layout_options(h, w)
    pairs = {"train": [], "test": []}
    signatures = {"train": set(), "test": set()}
    input_keys = set()

    def append(regime, sequence, rows, cols, background, inks, required=()):
        sequence = _less_dense(sequence, masks, sparse, required)
        pair = _make_pair(sequence, rows, cols, masks, labels, background, inks)
        signature = (rows, cols, tuple(sequence))
        key = _grid_key(pair["input"])
        if key in input_keys or signature in signatures["train"] | signatures["test"]:
            return False
        if (_grid_key(pair["input"]), _grid_key(pair["output"])) in _SOURCE_PAIRS:
            return False
        pairs[regime].append(pair)
        signatures[regime].add(signature)
        input_keys.add(key)
        return True

    bg0 = int(rng.integers(0, 10))
    ink0 = int(rng.choice([v for v in range(10) if v != bg0]))
    teaching_layout = 0 if source_style else int(rng.integers(0, 3))
    if teaching_layout == 0:
        first_rows, first_cols = 1, k
        second_rows, second_cols = 2, k - 1
    elif teaching_layout == 1:
        first_rows, first_cols = k, 1
        second_rows, second_cols = k - 1, 2
    else:
        first_rows, first_cols = 2, (k + 1) // 2
        second_rows, second_cols = 3, first_cols + 1
    first = list(int(i) for i in rng.permutation(k))
    first += [sparse] * (first_rows * first_cols - k)
    rng.shuffle(first)
    assert append("train", first, first_rows, first_cols, bg0, [ink0] * len(first))

    bg1 = int(rng.choice([v for v in range(10) if v != bg0]))
    ink1 = int(rng.choice([v for v in range(10) if v != bg1 and v != ink0]))
    second = list(int(i) for i in rng.permutation(k))
    second += [sparse] * (second_rows * second_cols - k)
    rng.shuffle(second)
    extra_ink = int(rng.choice([v for v in range(10) if v != bg1 and v != ink1]))
    # Each class is seen once with the second ink, regardless of the extra occurrences.
    second_inks = []
    seen = set()
    for glyph in second:
        if glyph not in seen:
            second_inks.append(ink1)
            seen.add(glyph)
        else:
            second_inks.append(extra_ink)
    assert append("train", second, second_rows, second_cols, bg1, second_inks)

    reserved_other = int(rng.choice([i for i in range(k) if i != sparse]))
    reserved = frozenset((sparse, reserved_other))
    for pair_index in range(2, n_train):
        for attempt in range(80):
            if source_style and rng.integers(0, 4) != 0:
                rows, cols = 1, int(rng.integers(3, 7))
            else:
                rows, cols = layouts[int(rng.integers(len(layouts)))]
            slots = rows * cols
            sequence = _sequence(rng, slots, k, sparse)
            if frozenset(sequence) == reserved:
                sequence[0] = next(i for i in range(k) if i not in reserved)
            background = int(rng.integers(0, 10))
            inks = _random_colors(rng, background, slots)
            if rng.random() < (0.8 if source_style else 0.3):
                inks = [inks[0]] * slots
            if append("train", sequence, rows, cols, background, inks):
                break
        else:
            raise AssertionError("cannot construct training pair")

    for pair_index in range(n_test):
        for attempt in range(100):
            if pair_index == 0:
                if source_style:
                    rows, cols = 1, 3
                else:
                    candidates = [(r, c) for r, c in layouts if 3 <= r * c <= 10]
                    rows, cols = candidates[int(rng.integers(len(candidates)))]
                sequence = _sequence(rng, rows * cols, k, sparse, subset=tuple(reserved))
                required = tuple(reserved)
            else:
                rows, cols = layouts[int(rng.integers(len(layouts)))]
                slots = rows * cols
                mode = int(rng.integers(3))
                if mode == 0:
                    sequence = _sequence(rng, slots, k, sparse, subset=(sparse,))
                    required = ()
                elif mode == 1:
                    if slots < k:
                        suitable = [(r, c) for r, c in layouts if r * c >= k]
                        rows, cols = suitable[int(rng.integers(len(suitable)))]
                        slots = rows * cols
                    sequence = list(int(i) for i in rng.permutation(k))
                    sequence += [int(i) for i in rng.integers(0, k, size=slots - k)]
                    rng.shuffle(sequence)
                    required = tuple(range(k))
                else:
                    sequence = _sequence(rng, slots, k, sparse)
                    required = ()
            background = 0 if source_style and pair_index == 0 else int(rng.integers(0, 10))
            inks = _random_colors(rng, background, rows * cols)
            if source_style and pair_index == 0:
                inks = [5] * (rows * cols)
            elif rng.random() < (0.8 if source_style else 0.3):
                inks = [inks[0]] * (rows * cols)
            if append("test", sequence, rows, cols, background, inks, required):
                break
        else:
            raise AssertionError("cannot construct test pair")
    return pairs


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    n_train, n_test = _counts(rng, num_train, num_test)
    for _ in range(40):
        try:
            puzzle = _build(rng, n_train, n_test)
            validate_generated_puzzle(puzzle)
            return puzzle
        except AssertionError:
            continue
    raise AssertionError("cannot construct unambiguous puzzle")


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert bool(np.all((grid >= 0) & (grid <= 9)))


def _parse(puzzle, h, w):
    mapping = {}
    labels_to_masks = {}
    occurrences = {}
    signatures = {"train": set(), "test": set()}
    parsed = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            inp, out = pair["input"], pair["output"]
            assert inp.shape[0] % h == 0 and inp.shape[1] % w == 0
            rows, cols = inp.shape[0] // h, inp.shape[1] // w
            values, counts = np.unique(inp, return_counts=True)
            bg = int(values[int(np.argmax(counts))])
            assert int(max(counts)) * 2 > inp.size
            sequence = []
            tile_inks = []
            prediction = np.empty_like(out)
            for r in range(rows):
                for c in range(cols):
                    itile = inp[r*h:(r+1)*h, c*w:(c+1)*w]
                    otile = out[r*h:(r+1)*h, c*w:(c+1)*w]
                    inks = set(int(x) for x in np.unique(itile)) - {bg}
                    assert len(inks) == 1
                    mask = (itile != bg).astype(np.uint8).tobytes()
                    assert 0 < sum(itile.flat != bg) < itile.size
                    label = int(otile[0, 0])
                    assert bool(np.all(otile == label))
                    sequence.append(mask)
                    tile_inks.append(next(iter(inks)))
                    if regime == "train":
                        if mask in mapping:
                            assert mapping[mask] == label
                        mapping[mask] = label
                        if label in labels_to_masks:
                            assert labels_to_masks[label] == mask
                        labels_to_masks[label] = mask
                        occurrences.setdefault(mask, set()).add((bg, next(iter(inks))))
                    else:
                        assert mask in mapping
                        assert mapping[mask] == label
                    prediction[r*h:(r+1)*h, c*w:(c+1)*w] = mapping.get(mask, label)
            assert np.array_equal(prediction, out)
            signature = (rows, cols, tuple(sequence))
            signatures[regime].add(signature)
            parsed[regime].append((rows, cols, sequence, tile_inks))
    return mapping, occurrences, signatures, parsed


def _candidate_predictions(puzzle, h, w):
    mapping = {}
    for pair in puzzle["train"]:
        inp, out = pair["input"], pair["output"]
        if inp.shape[0] % h or inp.shape[1] % w:
            return None
        vals, counts = np.unique(inp, return_counts=True)
        bg = int(vals[int(np.argmax(counts))])
        if max(counts) * 2 <= inp.size:
            return None
        for r in range(0, inp.shape[0], h):
            for c in range(0, inp.shape[1], w):
                tile = inp[r:r+h, c:c+w]
                if len(set(int(x) for x in np.unique(tile)) - {bg}) != 1:
                    return None
                mask = (tile != bg).astype(np.uint8).tobytes()
                label_tile = out[r:r+h, c:c+w]
                label = int(label_tile[0, 0])
                if not np.all(label_tile == label) or mask in mapping and mapping[mask] != label:
                    return None
                mapping[mask] = label
    predictions = []
    for pair in puzzle["test"]:
        inp = pair["input"]
        if inp.shape[0] % h or inp.shape[1] % w:
            return None
        vals, counts = np.unique(inp, return_counts=True)
        bg = int(vals[int(np.argmax(counts))])
        if max(counts) * 2 <= inp.size:
            return None
        result = np.empty_like(inp)
        for r in range(0, inp.shape[0], h):
            for c in range(0, inp.shape[1], w):
                tile = inp[r:r+h, c:c+w]
                if len(set(int(x) for x in np.unique(tile)) - {bg}) != 1:
                    return None
                mask = (tile != bg).astype(np.uint8).tobytes()
                if mask not in mapping:
                    return None
                result[r:r+h, c:c+w] = mapping[mask]
        predictions.append(result)
    return predictions


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in inputs
            inputs.add(key)
            assert (key, _grid_key(out)) not in _SOURCE_PAIRS
    h = int(np.gcd.reduce([p["input"].shape[0] for p in puzzle["train"]]))
    w = int(np.gcd.reduce([p["input"].shape[1] for p in puzzle["train"]]))
    assert 2 <= h <= 5 and 2 <= w <= 5
    mapping, occurrences, signatures, parsed = _parse(puzzle, h, w)
    assert 3 <= len(mapping) <= 6 and len(set(mapping.values())) == len(mapping)
    assert all(len(presentations) >= 2 for presentations in occurrences.values())
    assert all(len({ink for _, ink in presentations}) >= 2 for presentations in occurrences.values())
    assert signatures["train"].isdisjoint(signatures["test"])
    assert len(signatures["train"]) == len(puzzle["train"])
    assert len(signatures["test"]) == len(puzzle["test"])
    assert len({(r, c) for r, c, _, _ in parsed["train"]}) >= 2
    assert len({r*c for r, c, _, _ in parsed["train"]}) >= 2
    assert any(len(set(seq)) < len(seq) for _, _, seq, _ in parsed["train"])
    assert any(len(set(inks)) >= 2 for _, _, _, inks in parsed["train"])
    test_set = frozenset(parsed["test"][0][2])
    assert len(test_set) == 2
    assert all(frozenset(seq) != test_set for _, _, seq, _ in parsed["train"])
    for dh in range(1, 6):
        for dw in range(1, 6):
            if h == dh and w == dw:
                continue
            predictions = _candidate_predictions(puzzle, dh, dw)
            if predictions is not None:
                assert all(np.array_equal(prediction, pair["output"])
                           for prediction, pair in zip(predictions, puzzle["test"]))


def _same_puzzle(a, b):
    return all(np.array_equal(p[x], q[x])
               for regime in ("train", "test")
               for p, q in zip(a[regime], b[regime])
               for x in ("input", "output")) and all(
                   len(a[regime]) == len(b[regime]) for regime in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes = set()
    layouts = set()
    masks = set()
    mappings = set()
    backgrounds = set()
    colors = set()
    vocabulary_sizes = set()
    source_upper_labels = set()
    cross_puzzles = 0
    equal_population_puzzles = 0
    familiar_test_subsets = 0
    multiple_test_subsets = 0
    source_masks = {mask.tobytes() for mask in _source_masks()}
    upper_mask = _source_masks()[0].tobytes()
    for puzzle in puzzles:
        h = int(np.gcd.reduce([p["input"].shape[0] for p in puzzle["train"]]))
        w = int(np.gcd.reduce([p["input"].shape[1] for p in puzzle["train"]]))
        mapping, _, _, parsed = _parse(puzzle, h, w)
        vocabulary_sizes.add(len(mapping))
        if set(mapping) == source_masks:
            source_upper_labels.add(mapping[upper_mask])
        equal_population_puzzles += len({sum(mask) for mask in mapping}) < len(mapping)
        has_cross = False
        for mask in mapping:
            tile = np.frombuffer(mask, dtype=np.uint8).reshape(h, w)
            plus = np.zeros((h, w), dtype=np.uint8)
            plus[h // 2, :] = 1
            plus[:, w // 2] = 1
            diagonal_cross = np.zeros((h, w), dtype=np.uint8)
            for row in range(h):
                diagonal_cross[row, row * (w - 1) // (h - 1)] = 1
                diagonal_cross[row, w - 1 - row * (w - 1) // (h - 1)] = 1
            has_cross |= np.array_equal(tile, plus) or np.array_equal(tile, diagonal_cross)
        cross_puzzles += has_cross
        train_subsets = {frozenset(seq) for _, _, seq, _ in parsed["train"]}
        test_subsets = {frozenset(seq) for _, _, seq, _ in parsed["test"]}
        familiar_test_subsets += bool(train_subsets & test_subsets)
        multiple_test_subsets += len(test_subsets) > 1
        shapes.add((h, w))
        masks.update((h, w, mask) for mask in mapping)
        mappings.add(tuple(sorted(mapping.items())))
        for regime in ("train", "test"):
            for pair, (r, c, seq, _) in zip(puzzle[regime], parsed[regime]):
                layouts.add((r, c))
                backgrounds.add(int(np.unique(pair["input"], return_counts=True)[0][
                    int(np.argmax(np.unique(pair["input"], return_counts=True)[1]))]))
                colors.update(int(v) for v in np.unique(pair["output"]))
                assert len(seq) == r * c
    assert len(shapes) >= 5 and len(layouts) >= 12
    assert len(masks) >= 30 and len(mappings) >= 35
    assert len(backgrounds) >= 8 and len(colors) >= 7
    assert vocabulary_sizes == {3, 4, 5, 6}
    assert len(source_upper_labels) >= 3
    assert cross_puzzles >= 5 and equal_population_puzzles >= 20
    assert familiar_test_subsets >= 10 and multiple_test_subsets >= 10
    assert any(len(set(_parse(p, int(np.gcd.reduce([x["input"].shape[0] for x in p["train"]])),
                              int(np.gcd.reduce([x["input"].shape[1] for x in p["train"]])))[0].values())) >= 5
               for p in puzzles)
    assert _same_puzzle(generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3))
    assert not _same_puzzle(generate_puzzle(100, 6, 2), generate_puzzle(101, 6, 2))
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + nt + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (nt or 0) + (ns or 0), nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    damaged = generate_puzzle(4000, 4, 2)
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid output accepted")
