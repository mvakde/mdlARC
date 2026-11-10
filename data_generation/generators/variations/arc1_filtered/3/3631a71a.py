"""Symmetry-completion puzzles with a shared erasure color and geometry."""

import numpy as np


_SOURCE_ROWS = {
    "000000006655010000105566000000",
    "300000000833108008013380000000",
    "050005008804440000444088005000",
    "000170600006330220336000060710",
    "807077110306080000806030117707",
}


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                    (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(name + " is outside its legal range")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _orbit(r, c, n, t):
    if r < t or c < t:
        return {(r, c), (c, r)}
    end = n + t - 1
    result = set()
    for a in (r, end - r):
        for b in (c, end - c):
            result.add((a, b))
            result.add((b, a))
    return result


def _witnessed(mask, t):
    n = mask.shape[0]
    return all(any(not mask[a, b] for a, b in _orbit(int(r), int(c), n, t))
               for r, c in np.argwhere(mask))


def _recover(inp, erase, t, rotation):
    x = np.rot90(inp, -rotation)
    n = x.shape[0]
    y = x.copy()
    for r, c in np.argwhere(x == erase):
        values = {int(x[a, b]) for a, b in _orbit(int(r), int(c), n, t)
                  if x[a, b] != erase}
        if len(values) != 1:
            return None
        y[r, c] = values.pop()
    return np.rot90(y, rotation)


def _symmetric(out, t, rotation):
    x = np.rot90(out, -rotation)
    if not np.array_equal(x, x.T):
        return False
    inner = x[t:, t:]
    return (np.array_equal(inner, inner[::-1, :])
            and np.array_equal(inner, inner[:, ::-1]))


def _paint(q, r, c, color):
    q[r, c] = color
    q[c, r] = color


def _segment(q, r, c, length, color, vertical=False):
    s = len(q)
    for z in range(length):
        a, b = (min(s - 1, r + z), c) if vertical else (r, min(s - 1, c + z))
        _paint(q, a, b, color)


def _output(rng, n, t, bg, colors, style):
    m = n - t
    s = (m + 1) // 2
    q = np.full((s, s), bg, dtype=np.int8)

    if style == 0:  # scattered blocks, short lines, and dots
        for _ in range(int(rng.integers(5, 13))):
            r, c = (int(v) for v in rng.integers(s, size=2))
            h, w = (int(v) for v in rng.integers(1, 4, size=2))
            color = int(rng.choice(colors))
            for a in range(r, min(s, r + h)):
                for b in range(c, min(s, c + w)):
                    _paint(q, a, b, color)
    elif style == 1:  # long arms
        for _ in range(int(rng.integers(3, 7))):
            r = int(rng.integers(s))
            c = int(rng.integers(max(1, s // 3)))
            _segment(q, r, c, int(rng.integers(max(3, s // 2), s + 1)),
                     int(rng.choice(colors)))
    elif style == 2:  # nested contours
        levels = rng.choice(s, size=int(rng.integers(2, min(6, s) + 1)), replace=False)
        for level in levels:
            a = int(level)
            _segment(q, a, a, s - a, int(rng.choice(colors)))
    elif style == 3:  # compact shapes near the center
        for _ in range(int(rng.integers(4, 9))):
            r = int(rng.integers(max(0, s // 3), s))
            c = int(rng.integers(max(0, s // 3), s))
            color = int(rng.choice(colors))
            for a in range(r, min(s, r + int(rng.integers(1, 4)))):
                for b in range(c, min(s, c + int(rng.integers(1, 4)))):
                    _paint(q, a, b, color)
    else:  # sparse line work
        for _ in range(int(rng.integers(4, 10))):
            r, c = (int(v) for v in rng.integers(s, size=2))
            _segment(q, r, c, int(rng.integers(2, min(7, s + 1))),
                     int(rng.choice(colors)), bool(rng.integers(2)))

    # Add independent details and ensure chosen colors actually occur. Their
    # coordinates are sampled anew per pair, not fixed palette markers.
    for _ in range(int(rng.integers(3, 11))):
        r, c = (int(v) for v in rng.integers(s, size=2))
        _paint(q, r, c, int(rng.choice(colors)))
    missing = set(colors) - set(int(v) for v in np.unique(q))
    for color in missing:
        r, c = (int(v) for v in rng.integers(s, size=2))
        _paint(q, r, c, int(color))

    ix = np.minimum(np.arange(m), m - 1 - np.arange(m))
    inner = q[np.ix_(ix, ix)]
    out = np.full((n, n), bg, dtype=np.int8)
    out[t:, t:] = inner

    top = np.full((t, m), bg, dtype=np.int8)
    for row in range(t):
        for _ in range(int(rng.integers(max(3, m // 5), max(5, m // 2)))):
            c = int(rng.integers(m))
            width = int(rng.integers(1, min(6, m - c + 1)))
            top[row, c:c + width] = int(rng.choice(colors))
    out[:t, t:] = top
    out[t:, :t] = top.T
    corner = np.full((t, t), bg, dtype=np.int8)
    for r in range(t):
        for c in range(r, t):
            if rng.random() < 0.45:
                corner[r, c] = corner[c, r] = int(rng.choice(colors))
    out[:t, :t] = corner
    return out


def _rectangle(mask, rng, kind, t):
    n = len(mask)
    m = n - t
    if kind == "top":
        h = int(rng.integers(1, min(t + 3, 6)))
        w = int(rng.integers(2, min(10, n // 2 + 1)))
        r = int(rng.integers(0, t))
        c = int(rng.integers(t + 1, max(t + 2, n - w + 1)))
    elif kind == "left":
        h = int(rng.integers(2, min(10, n // 2 + 1)))
        w = int(rng.integers(1, min(t + 3, 6)))
        r = int(rng.integers(t + 1, max(t + 2, n - h + 1)))
        c = int(rng.integers(0, t))
    elif kind == "axis":
        h = int(rng.integers(3, min(10, m + 1)))
        w = int(rng.integers(3, min(10, m + 1)))
        r = max(t, min(n - h, t + m // 2 - int(rng.integers(0, h))))
        c = max(t, min(n - w, t + m // 2 - int(rng.integers(0, w))))
    elif kind == "strip":
        if rng.random() < 0.5:
            h = int(rng.integers(1, 3)); w = int(rng.integers(4, min(13, n + 1)))
        else:
            h = int(rng.integers(4, min(13, n + 1))); w = int(rng.integers(1, 3))
        r = int(rng.integers(0, n - h + 1)); c = int(rng.integers(0, n - w + 1))
    else:
        h = int(rng.integers(2, min(10, n // 2 + 1)))
        w = int(rng.integers(2, min(10, n // 2 + 1)))
        r = int(rng.integers(0, n - h + 1)); c = int(rng.integers(0, n - w + 1))
    mask[r:r + h, c:c + w] = True
    return r, c, h, w


def _repair(mask, rng, t):
    n = len(mask)
    visited = set()
    for r, c in np.argwhere(mask):
        orbit = _orbit(int(r), int(c), n, t)
        if (int(r), int(c)) in visited:
            continue
        visited.update(orbit)
        if all(mask[a, b] for a, b in orbit):
            a, b = tuple(orbit)[int(rng.integers(len(orbit)))]
            mask[a, b] = False
    return mask


def _mask(rng, n, t, regime, emphasis):
    mask = np.zeros((n, n), dtype=bool)
    count = int(rng.integers(1, 6))
    if regime == "target_test" and rng.random() < 0.65:
        count = max(2, count)
    kinds = ["top", "left", "axis", "strip", "free"]
    if emphasis == "overhang":
        first = str(rng.choice(("top", "left")))
    elif emphasis == "axis":
        first = "axis"
    else:
        first = str(rng.choice(kinds))
    rect = _rectangle(mask, rng, first, t)
    for j in range(1, count):
        if j == 1 and rng.random() < 0.24:
            r, c, h, w = rect
            rr = max(0, min(n - 2, r + int(rng.integers(-2, 3))))
            cc = max(0, min(n - 2, c + int(rng.integers(-2, 3))))
            mask[rr:min(n, rr + max(2, h // 2)) , cc:min(n, cc + max(2, w + 2))] = True
        else:
            rect = _rectangle(mask, rng, str(rng.choice(kinds)), t)
    if rng.random() < (0.22 if regime == "target_test" else 0.12):
        # Hide both elementary mirror mates for a small interior region.
        a = int(rng.integers(t + 2, t + max(3, (n - t) // 2)))
        b = int(rng.integers(t + 2, t + max(3, (n - t) // 2)))
        end = n + t - 1
        for r, c in ((a, b), (end - a, b), (a, end - b)):
            mask[r:r + 2, c:c + 2] = True
    return _repair(mask, rng, t)


def _pair(rng, n, t, rotation, bg, erase, regime, emphasis):
    available = [c for c in range(10) if c not in (bg, erase)]
    for _ in range(50):
        k = int(rng.choice((3, 4, 5, 6, 7, 8), p=(.1, .13, .15, .17, .2, .25)))
        colors = tuple(int(v) for v in rng.choice(available, size=k, replace=False))
        style = int(rng.integers(5))
        out = _output(rng, n, t, bg, colors, style)
        if int(np.bincount(out.ravel(), minlength=10).argmax()) != bg:
            continue
        mask = _mask(rng, n, t, regime, emphasis)
        if mask.sum() < 6 or not _witnessed(mask, t):
            continue
        if not np.any(mask & (out != bg)) or not np.any(mask & (out == bg)):
            continue
        inp = out.copy()
        inp[mask] = erase
        out = np.rot90(out, rotation).copy()
        inp = np.rot90(inp, rotation).copy()
        if n == 30 and ''.join(str(int(v)) for v in out[0]) in _SOURCE_ROWS:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a witnessed erasure")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    n = int(rng.integers(20, 31))
    t = int(rng.integers(1, min(4, n - 15)))
    rotation = int(rng.integers(4))
    bg = int(rng.integers(10))
    erase = int(rng.choice([c for c in range(10) if c != bg]))
    emphases = ["overhang", "axis"] + ["free"] * (ntrain - 2)
    rng.shuffle(emphases)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, key in (("source_train", ntrain, "train"),
                               ("target_test", ntest, "test")):
        for i in range(count):
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _pair(pair_rng, n, t, rotation, bg, erase, regime,
                             emphases[i] if regime == "source_train" else "free")
                signature = pair["input"].tobytes()
                if signature not in seen:
                    seen.add(signature)
                    puzzle[key].append(pair)
                    break
            else:
                raise RuntimeError("duplicate pair sampling failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _candidate_geometry(train):
    n = train[0]["output"].shape[0]
    return [(t, rot) for t in range(1, 4) for rot in range(4)
            if all(_symmetric(p["output"], t, rot) for p in train)]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    all_pairs = puzzle["train"] + puzzle["test"]
    seen_inputs, seen_pairs = set(), set()
    erasures, backgrounds, sizes = set(), set(), set()
    for pair in all_pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        assert inp.shape == out.shape and inp.shape[0] == inp.shape[1]
        assert inp.shape[0] >= 20
        changed = inp != out
        assert changed.any()
        assert len(np.unique(inp[changed])) == 1
        erase = int(inp[changed][0])
        assert not np.any(out == erase)
        assert np.array_equal(changed, inp == erase)
        assert np.any(changed & (out != np.bincount(out.ravel(), minlength=10).argmax()))
        assert np.any(changed & (out == np.bincount(out.ravel(), minlength=10).argmax()))
        key = (inp.shape, inp.tobytes())
        pk = (key, out.tobytes())
        assert key not in seen_inputs and pk not in seen_pairs
        seen_inputs.add(key); seen_pairs.add(pk)
        assert ''.join(str(int(v)) for v in out[0]) not in _SOURCE_ROWS
        erasures.add(erase)
        backgrounds.add(int(np.bincount(out.ravel(), minlength=10).argmax()))
        sizes.add(inp.shape)
    assert len(erasures) == len(backgrounds) == len(sizes) == 1
    erase = next(iter(erasures))
    candidates = _candidate_geometry(puzzle["train"])
    assert candidates
    good = []
    for t, rot in candidates:
        if not all(_symmetric(p["output"], t, rot) for p in all_pairs):
            continue
        if all(np.array_equal(_recover(p["input"], erase, t, rot), p["output"])
               for p in all_pairs):
            good.append((t, rot))
    assert good
    # The demonstrations include an erased overhang and an erased interior
    # axis region, so both parts of the geometry and the erasure role appear.
    for t, rot in good:
        train_masks = [np.rot90(p["input"] != p["output"], -rot)
                       for p in puzzle["train"]]
        n = train_masks[0].shape[0]
        middle = t + (n - t) // 2
        assert any(m[:t, :].any() or m[:, :t].any() for m in train_masks)
        assert any(m[max(t, middle - 1):middle + 2, t:].any()
                   or m[t:, max(t, middle - 1):middle + 2].any()
                   for m in train_masks)
    # Every geometry compatible with the demonstrations must give the same
    # answer on every test input. This checks inference without using outputs.
    for pair in puzzle["test"]:
        predictions = []
        for t, rot in candidates:
            prediction = _recover(pair["input"], erase, t, rot)
            if prediction is not None:
                predictions.append(prediction)
        assert predictions
        assert all(np.array_equal(predictions[0], p) for p in predictions)
        assert np.array_equal(predictions[0], pair["output"])
    assert any((p["input"] != p["output"]).sum() >= 6 for p in puzzle["train"])


def _same(a, b):
    return all(len(a[k]) == len(b[k]) and
               all(np.array_equal(x[g], y[g]) for x, y in zip(a[k], b[k])
                   for g in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    erasures, backgrounds, sizes, rotations = set(), set(), set(), set()
    edge_masks, interior_masks, compact, multi = 0, 0, 0, 0
    shapes, masks, palettes = set(), set(), set()
    full_palettes, zero_erasures, three_component_masks = 0, 0, 0
    top_masks, left_masks, nonrect_masks, hidden_simple_mates = 0, 0, 0, 0
    within_mask_sizes, within_component_counts = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"])); test_counts.add(len(puzzle["test"]))
        candidates = _candidate_geometry(puzzle["train"])
        rotations.update(rot for _, rot in candidates)
        sample = puzzle["train"][0]
        erasures.add(int(sample["input"][sample["input"] != sample["output"]][0]))
        backgrounds.add(int(np.bincount(sample["output"].ravel(), minlength=10).argmax()))
        sizes.add(sample["input"].shape)
        t, rot = candidates[0]
        pair_mask_sizes, pair_component_counts = set(), set()
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            mask = inp != out
            canonical_mask = np.rot90(mask, -rot)
            pair_mask_sizes.add(int(mask.sum()))
            full_palettes += len(np.unique(out)) == 9
            zero_erasures += int(inp[mask][0]) == 0
            top_masks += bool(canonical_mask[:t, :].any())
            left_masks += bool(canonical_mask[:, :t].any())
            edge_masks += bool(mask[:3].any() or mask[:, :3].any())
            interior_masks += bool(mask[3:-3, 3:-3].any())
            compact += bool(mask.sum() <= 30)
            multi += bool(mask.sum() >= 50)
            shapes.add((out != np.bincount(out.ravel(), minlength=10).argmax()).tobytes())
            masks.add(mask.tobytes())
            palettes.add(tuple(int(c) for c in np.unique(out)))
            remaining = canonical_mask.copy()
            components = []
            while remaining.any():
                start = tuple(np.argwhere(remaining)[0])
                stack = [start]
                remaining[start] = False
                cells = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= a < len(remaining) and 0 <= b < len(remaining) and remaining[a, b]:
                            remaining[a, b] = False
                            stack.append((a, b))
                components.append(cells)
            pair_component_counts.add(len(components))
            three_component_masks += len(components) >= 3
            nonrect_masks += any(len(cells) < (max(a for a, b in cells) - min(a for a, b in cells) + 1)
                                 * (max(b for a, b in cells) - min(b for a, b in cells) + 1)
                                 for cells in components)
            edge = len(mask) + t - 1
            hidden_simple_mates += any(canonical_mask[edge - r, c]
                                       and canonical_mask[r, edge - c]
                                       and not canonical_mask[c, r]
                                       for r, c in np.argwhere(canonical_mask)
                                       if r >= t and c >= t)
        within_mask_sizes += len(pair_mask_sizes) > 1
        within_component_counts += len(pair_component_counts) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(erasures) >= 5 and len(backgrounds) >= 4 and len(sizes) >= 6
    assert len(rotations) >= 4 and len(palettes) >= 25
    assert len(shapes) >= 100 and len(masks) >= 100
    assert edge_masks >= 15 and interior_masks >= 100
    assert compact >= 10 and multi >= 10
    assert full_palettes >= 15 and zero_erasures >= 5
    assert top_masks >= 30 and left_masks >= 15
    assert three_component_masks >= 20 and nonrect_masks >= 20
    assert hidden_simple_mates >= 10
    assert within_mask_sizes >= 35 and within_component_counts >= 30
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(180 + nt * 10 + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(302, nt, ns))
    for seed in (0, 7, 41):
        assert _same(generate_puzzle(seed, 5, 2), generate_puzzle(seed, 5, 2))
    assert not _same(generate_puzzle(1, 5, 2), generate_puzzle(2, 5, 2))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(919, 3, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    cell = tuple(np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0])
    bad["test"][0]["output"][cell] = (int(bad["test"][0]["output"][cell]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect restoration accepted")
