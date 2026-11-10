"""Cyclic coordinate-coloring puzzles with erased rectangular and irregular regions."""
import numpy as np

_SOURCE_ROWS = {
    "161122334455661122334", "656677112233445566771",
    "545566778811223344556", "434455667788991122334",
}


def _count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _counts(rng, num_train, num_test):
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    choices = [(t, s) for t in range(2, 11) for s in range(1, 5)
               if t > s and (nt is None or t == nt) and (ns is None or s == ns)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    if nt is None and ns is None:
        common = {(2, 1): 8, (3, 1): 30, (4, 1): 14, (4, 2): 10,
                  (5, 1): 6, (5, 2): 8, (6, 2): 6, (6, 3): 5,
                  (7, 3): 4, (8, 3): 4, (9, 4): 2, (10, 4): 3}
        weights = np.array([common.get(x, 1) for x in choices], dtype=float)
    else:
        weights = np.array([1 / (t - 1) for t, _ in choices], dtype=float)
    weights /= weights.sum()
    return choices[int(rng.choice(len(choices), p=weights))]


def _steps(n):
    coordinates = np.arange(n)
    a = np.minimum.outer(coordinates, coordinates)
    b = np.maximum.outer(coordinates, coordinates)
    return np.where(a == b, 1, (b + 2) // (a + 2) - 1)


def _complete(n, palette, anchor, k):
    cycle = np.asarray(palette[:k], dtype=np.int8)
    return cycle[(anchor - (k - 6) + _steps(n)) % k].astype(np.int8)


def _parameters(rng):
    for _ in range(16):
        palette = tuple(int(x) for x in rng.permutation(np.arange(1, 10)))
        anchor = int(rng.integers(0, 9))
        rows = {"".join(map(str, _complete(21, palette, anchor, k)[0]))
                for k in range(4, 10)}
        if rows.isdisjoint(_SOURCE_ROWS):
            return palette, anchor
    return palette[1::-1] + palette[2:], anchor


def _levels(rng, nt, ns):
    source_progression = bool(rng.random() < 0.24)
    if source_progression:
        train = [7, 8] if nt == 2 else [6, 7, 8]
        train.extend(int(rng.integers(4, 9)) for _ in range(nt - len(train)))
        test = [9] * ns
    else:
        train = [int(x) for x in rng.choice(np.arange(4, 10), size=2, replace=False)]
        train.extend(int(rng.integers(4, 10)) for _ in range(nt - 2))
        test = []
        for _ in range(ns):
            q = rng.random()
            higher = list(range(max(train) + 1, 10))
            if q < 0.30 and higher:
                test.append(int(rng.choice(higher)))
            elif q < 0.72:
                test.append(int(rng.choice(train)))
            else:
                test.append(int(rng.integers(4, 10)))
    rng.shuffle(train)
    return train, test, source_progression


def _size(rng, k, used):
    possible = [n for n in range(max(11, 2 * k - 1), 31) if (k, n) not in used]
    if not possible:
        raise RuntimeError("no distinct size remains for this color count")
    weights = np.array([2.5 if n <= 18 else 2.0 if n <= 24 else 1.5
                        for n in possible], dtype=float)
    weights /= weights.sum()
    n = int(rng.choice(possible, p=weights))
    used.add((k, n))
    return n


def _stamp(rng, mask, r, c, h, w):
    style = int(rng.integers(4))
    patch = np.ones((h, w), dtype=bool)
    if style == 1 and h > 2 and w > 2:
        patch[h // 2:, w // 2:] = False
    elif style == 2 and h > 2 and w > 2:
        patch[:h // 2, :w // 2] = False
    elif style == 3 and h > 2 and w > 2:
        patch[:] = False
        thickness = max(1, w // 2)
        for i in range(h):
            start = min(w - thickness, i * (w - thickness) // (h - 1))
            patch[i, start:start + thickness] = True
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    mask[r:r + h, c:c + w] |= patch
    return patch


def _region(rng, mask, placement, hard):
    n = len(mask)
    limit = max(3, n // 3 + 1)
    if rng.random() < 0.23:
        if rng.random() < 0.5:
            h, w = 1, int(rng.integers(3, limit + 1))
        else:
            h, w = int(rng.integers(3, limit + 1)), 1
    else:
        h = int(rng.integers(2, limit + 1))
        w = int(rng.integers(2, limit + 1))
    if placement == "corner":
        r = 0 if rng.random() < 0.5 else n - h
        c = 0 if rng.random() < 0.5 else n - w
    elif placement == "top":
        r, c = 0, int(rng.integers(n - w + 1))
    elif placement == "bottom":
        r, c = n - h, int(rng.integers(n - w + 1))
    elif placement == "left":
        r, c = int(rng.integers(n - h + 1)), 0
    elif placement == "right":
        r, c = int(rng.integers(n - h + 1)), n - w
    else:
        r = int(rng.integers(1, n - h))
        c = int(rng.integers(1, n - w))
    patch = _stamp(rng, mask, r, c, h, w)
    if hard and placement == "interior" and rng.random() < 0.23:
        mask[c:c + w, r:r + h] |= patch.T


def _mask(rng, n, regime, hard, edge_mode):
    mask = np.zeros((n, n), dtype=bool)
    if hard:
        width = int(rng.integers(2, min(5, n // 4 + 2)))
        d = int(rng.integers(1, n - width + 1))
        mask[d:d + width, d:d + width] = True
    regions = int(rng.integers(2, 6 if regime == "source_train" else 7))
    placements = [edge_mode] if edge_mode != "none" else []
    placements += ["interior"] * (regions - len(placements))
    rng.shuffle(placements)
    for placement in placements:
        _region(rng, mask, placement, hard)
    if not hard:
        both = mask & mask.T
        mask &= ~(both & np.tril(np.ones((n, n), dtype=bool)))
    return mask


def _edge_mode(rng, regime, source_progression, pair_index):
    if source_progression and regime == "target_test" and pair_index == 0:
        return "left"
    options = ("none", "left", "top", "right", "bottom", "corner")
    weights = ((0.43, 0.20, 0.12, 0.09, 0.09, 0.07)
               if regime == "target_test" else
               (0.51, 0.16, 0.12, 0.08, 0.08, 0.05))
    return str(rng.choice(options, p=weights))


def _visible_cycle(inp):
    colors = set(int(x) for x in np.unique(inp) if x != 0)
    k = len(colors)
    if not 4 <= k <= 9:
        return None
    step = _steps(len(inp)) % k
    cycle = []
    for residue in range(k):
        values = set(int(x) for x in inp[(step == residue) & (inp != 0)])
        if len(values) != 1:
            return None
        cycle.append(values.pop())
    return tuple(cycle) if set(cycle) == colors else None


def _fallback_mask(n, hard):
    mask = np.zeros((n, n), dtype=bool)
    d = n // 2
    if hard:
        mask[d:d + 2, d:d + 2] = True
    mask[2:4, n // 2:n // 2 + 3] = True
    mask[n // 2 + 2:n // 2 + 4, 2:5] = True
    return mask


def _pair(rng, regime, palette, anchor, k, n, source_progression, pair_index):
    output = _complete(n, palette, anchor, k)
    hard = bool(rng.random() < (0.55 if regime == "source_train" else 0.64))
    if regime == "source_train" and pair_index == 0:
        hard = True
    edge_mode = _edge_mode(rng, regime, source_progression, pair_index)
    for attempt in range(64):
        mask = (_mask(rng, n, regime, hard, edge_mode) if attempt < 63
                else _fallback_mask(n, hard))
        area = int(mask.sum())
        if not max(5, n // 2) <= area < n * n // 3:
            continue
        if len(np.unique(output[mask])) < 3:
            continue
        inp = output.copy()
        inp[mask] = 0
        cycle = _visible_cycle(inp)
        if cycle is not None and np.array_equal(output,
                                                np.asarray(cycle)[_steps(n) % k]):
            return {"input": inp, "output": output}
    raise RuntimeError("could not construct a recoverable mask")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    palette, anchor = _parameters(rng)
    train_levels, test_levels, source_progression = _levels(rng, nt, ns)
    puzzle = {"train": [], "test": []}
    used_sizes = set()
    for regime, levels, split in (("source_train", train_levels, "train"),
                                  ("target_test", test_levels, "test")):
        for index, k in enumerate(levels):
            n = _size(rng, k, used_sizes)
            puzzle[split].append(_pair(rng, regime, palette, anchor, k, n,
                                       source_progression, index))
    return puzzle


def _compatible_cycles(cycles):
    for anchor in range(9):
        prefixes = {}
        for k, cycle in cycles.items():
            offset = anchor - (k - 6)
            prefixes[k] = tuple(cycle[(i - offset) % k] for i in range(k))
        largest = prefixes[max(prefixes)]
        if all(largest[:k] == prefix for k, prefix in prefixes.items()):
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs, patterns, cycles = set(), set(), set(), {}
    train_hard = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= v <= 30 for v in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and inp.shape[0] == inp.shape[1]
            n = len(inp)
            inp_key = (inp.shape, inp.tobytes())
            pair_key = (inp_key, out.shape, out.tobytes())
            assert inp_key not in inputs and pair_key not in pairs
            inputs.add(inp_key)
            pairs.add(pair_key)
            assert not np.any(out == 0) and np.array_equal(out, out.T)
            mask = inp == 0
            assert max(5, n // 2) <= mask.sum() < n * n // 3
            assert np.array_equal(inp[~mask], out[~mask])
            assert len(np.unique(out[mask])) >= 3
            cycle = _visible_cycle(inp)
            assert cycle is not None
            k = len(cycle)
            assert n >= max(11, 2 * k - 1)
            assert np.array_equal(out, np.asarray(cycle)[_steps(n) % k])
            assert (k, n) not in patterns
            patterns.add((k, n))
            if n == 21:
                assert "".join(map(str, out[0])) not in _SOURCE_ROWS
            if k in cycles:
                assert cycles[k] == cycle
            else:
                cycles[k] = cycle
            if split == "train":
                paired = mask & mask.T & ~np.eye(n, dtype=bool)
                train_hard |= bool(np.any(np.diag(mask)) and np.any(paired))
    assert train_hard
    assert len({len(_visible_cycle(p["input"])) for p in puzzle["train"]}) >= 2
    assert _compatible_cycles(cycles)


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[s], b[s]) for role in ("input", "output"))
        for s in ("train", "test"))


def _predict_from_demonstrations(train, inp):
    # Receives only solved demonstrations and the queried input.
    for example in train:
        a, b = example["input"], example["output"]
        cycle = _visible_cycle(a)
        assert cycle is not None
        for r in range(len(a)):
            for c in range(len(a)):
                small, large = min(r, c), max(r, c)
                step = 1 if r == c else (large + 2) // (small + 2) - 1
                assert b[r, c] == cycle[step % len(cycle)]
                assert a[r, c] == 0 or a[r, c] == b[r, c]
    colors = set(int(x) for x in inp.flat if x != 0)
    k = len(colors)
    lookup = {}
    for r in range(len(inp)):
        for c in range(len(inp)):
            if inp[r, c] == 0:
                continue
            a, b = min(r, c), max(r, c)
            step = 1 if r == c else (b + 2) // (a + 2) - 1
            residue = step % k
            value = int(inp[r, c])
            assert residue not in lookup or lookup[residue] == value
            lookup[residue] = value
    assert len(lookup) == k and set(lookup.values()) == colors
    answer = inp.copy()
    for r in range(len(inp)):
        for c in range(len(inp)):
            if answer[r, c] == 0:
                a, b = min(r, c), max(r, c)
                step = 1 if r == c else (b + 2) // (a + 2) - 1
                answer[r, c] = lookup[step % k]
    return answer


def self_test() -> None:
    train_counts, test_counts, sizes, count_patterns = set(), set(), set(), set()
    signatures, layouts, cycles = set(), set(), set()
    names = ("train_left", "test_no_left", "test_hard", "test_easy",
             "train_top", "test_top", "train_corner", "test_corner",
             "test_seen_k", "test_unseen_k", "train_k9", "test_k9",
             "small", "large", "source_progression")
    measures = {name: 0 for name in names}
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        count_patterns.add((len(puzzle["train"]), len(puzzle["test"])))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for s in ("train", "test") for p in puzzle[s]))
        train_ks = {len(np.unique(p["output"])) for p in puzzle["train"]}
        if {6, 7, 8} <= train_ks and all(len(np.unique(p["output"])) == 9
                                       for p in puzzle["test"]):
            measures["source_progression"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                n = len(inp)
                k = len(np.unique(out))
                mask = inp == 0
                sizes.add(n)
                layouts.add((n, mask.tobytes()))
                cycles.add(_visible_cycle(inp))
                measures["small"] += int(n <= 18)
                measures["large"] += int(n >= 25)
                paired = bool(np.any(mask & mask.T & ~np.eye(n, dtype=bool)))
                hard = paired and bool(np.any(np.diag(mask)))
                corner = bool(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1])
                if split == "train":
                    measures["train_left"] += int(bool(np.any(mask[:, 0])))
                    measures["train_top"] += int(bool(np.any(mask[0])))
                    measures["train_corner"] += int(corner)
                    measures["train_k9"] += int(k == 9)
                else:
                    measures["test_no_left"] += int(not np.any(mask[:, 0]))
                    measures["test_hard"] += int(hard)
                    measures["test_easy"] += int(not paired and not np.any(np.diag(mask)))
                    measures["test_top"] += int(bool(np.any(mask[0])))
                    measures["test_corner"] += int(corner)
                    measures["test_seen_k"] += int(k in train_ks)
                    measures["test_unseen_k"] += int(k not in train_ks)
                    measures["test_k9"] += int(k == 9)
                    assert np.array_equal(_predict_from_demonstrations(
                        puzzle["train"], inp.copy()), out)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(count_patterns) >= 6 and len(signatures) == 42
    assert len(sizes) >= 12 and min(sizes) <= 13 and max(sizes) >= 28
    assert len(layouts) >= 150 and len(cycles) >= 30
    for name in names:
        assert measures[name] > 0, name
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(500 + nt + ns, nt, ns)
        validate_generated_puzzle(p)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for args in ({"num_test": 1}, {"num_test": 4},
                 {"num_train": 2}, {"num_train": 10}):
        p = generate_puzzle(607, **args)
        validate_generated_puzzle(p)
        assert all(len(p[s]) == args[f"num_{s}"] for s in ("train", "test")
                   if f"num_{s}" in args)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    assert _same_puzzle(generate_puzzle(2026, 5, 2), generate_puzzle(2026, 5, 2))
    assert _same_puzzle(generate_puzzle(2027), generate_puzzle(2027))
