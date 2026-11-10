"""Compact figure extraction from a patterned ARC field."""

import hashlib

import numpy as np


def _counts(rng, nt, ns):
    for value, lo, hi, name in ((nt, 2, 10, "num_train"), (ns, 1, 4, "num_test")):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError(name)
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (nt is None or a == nt) and (ns is None or b == ns)]
    if not choices:
        raise ValueError("incompatible pair counts")
    return choices[int(rng.integers(len(choices)))]


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    seen = {next(iter(points))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if q in points and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return len(seen) == len(points)


def _shape(rng, family):
    if family == "diamond":
        radius = int(rng.integers(2, 5))
        r, c = np.indices((2 * radius + 1, 2 * radius + 1))
        m = abs(r - radius) + abs(c - radius) == radius
    elif family == "box":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        m = np.zeros((h, w), bool)
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
    elif family == "open":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        m = np.zeros((h, w), bool)
        m[[0, -1], :] = True
        m[:, 0] = True
    elif family == "filled":
        m = np.ones((int(rng.integers(2, 7)), int(rng.integers(2, 7))), bool)
    elif family == "cross":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        m = np.zeros((h, w), bool)
        m[h // 2, :] = True
        m[:, w // 2] = True
    elif family == "tee":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        m = np.zeros((h, w), bool)
        m[0, :] = True
        m[:, w // 2] = True
    elif family == "ell":
        h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        m = np.zeros((h, w), bool)
        m[:, 0] = True
        m[-1, :] = True
    elif family == "stair":
        n = int(rng.integers(3, 8))
        m = np.eye(n, dtype=bool)
        if rng.integers(2):
            m[np.arange(n - 1), np.arange(1, n)] = True
    elif family == "zigzag":
        h = int(rng.integers(4, 8))
        w = int(rng.integers(3, 6))
        m = np.zeros((h, w), bool)
        columns = np.rint(np.linspace(0, w - 1, h // 2 + 1)).astype(int)
        walk = np.concatenate((columns, columns[-2::-1]))
        for r in range(h):
            m[r, int(walk[r])] = True
            if r:
                m[r, min(int(walk[r - 1]), int(walk[r])):
                  max(int(walk[r - 1]), int(walk[r])) + 1] = True
        m[0, :2] = True
    elif family == "bend":
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        turn = int(rng.integers(1, h - 1))
        m = np.zeros((h, w), bool)
        m[0, : w // 2 + 1] = True
        m[: turn + 1, w // 2] = True
        m[turn, w // 2 :] = True
        m[turn:, -1] = True
    elif family == "compact":
        style = int(rng.integers(5))
        m = np.zeros((2, 4) if style == 4 else (3, 3), bool)
        if style == 0:
            m[1, :] = True
            m[:, 1] = True
        elif style == 1:
            m[0, :] = True
            m[:, 1] = True
        elif style == 2:
            m[:, 0] = True
            m[-1, :] = True
        elif style == 3:
            m[0, [0, 2]] = True
            m[1, 1] = True
            m[2, [0, 2]] = True
        else:
            m[0, :3] = True
            m[1, 1:] = True
    else:  # overlapping short runs, including the source's wide two-row motif
        h, w = int(rng.integers(2, 5)), int(rng.integers(3, 8))
        run = max(2, w - h + 1)
        m = np.zeros((h, w), bool)
        for r in range(h):
            start = min(r, w - run)
            m[r, start : start + run] = True
    m = np.rot90(m, int(rng.integers(4)))
    rr, cc = np.where(m)
    m = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    assert _connected(m) and min(m.shape) >= 2
    return m


def _canvas(rng, sh, sw, regime):
    square = rng.random() < 0.46
    extension = 8 if regime == "source_train" else 9
    if square:
        side = min(23, max(8, max(sh, sw) + 4 + int(rng.integers(extension + 1))))
        return side, side
    h = min(23, max(8, sh + 4 + int(rng.integers(extension + 1))))
    w = min(23, max(8, sw + 4 + int(rng.integers(extension + 1))))
    if h == w:
        w = w + 1 if w < 23 else w - 1
    return h, w


def _field(rng, h, w, fg, mode):
    permitted = np.array([c for c in range(10) if c != fg], dtype=np.int8)
    if mode == 0:  # one-color field
        k = 1
    elif mode in (1, 2, 6):
        k = int(rng.integers(2, 4))
    elif mode in (3, 4):
        k = int(rng.integers(3, min(8, h, w)))
    else:  # one sparse full-span stripe
        k = 2
    colors = rng.choice(permitted, size=k, replace=False)
    r, c = np.indices((h, w))
    if mode == 0:
        bg = np.full((h, w), colors[0], dtype=np.int8)
    elif mode == 1:
        bg = colors[(r if rng.integers(2) else c) % k]
    elif mode == 2:
        bg = colors[(r + c) % k]
    elif mode == 3:  # whole-width horizontal bands with changed palettes
        row_order = rng.permutation(k)
        bg = colors[row_order[(r + r // max(2, h // 3)) % k]]
    elif mode == 4:  # different periodic fields on the two sides
        cut = int(rng.integers(2, h - 1))
        first = max(1, k // 2)
        bg = np.empty((h, w), np.int8)
        bg[:cut] = colors[(r[:cut] + c[:cut]) % first]
        bg[cut:] = colors[first + (r[cut:] + c[cut:]) % (k - first)]
    elif mode == 5:
        bg = np.full((h, w), colors[0], dtype=np.int8)
        if rng.integers(2):
            bg[int(rng.integers(h)), :] = colors[1]
        else:
            bg[:, int(rng.integers(w))] = colors[1]
    else:  # local changes use existing field colors
        bg = colors[(r + c) % k].copy()
        for _ in range(int(rng.integers(1, 4))):
            rh = int(rng.integers(1, min(4, h - 1)))
            rw = int(rng.integers(2, min(6, w - 1)))
            top = int(rng.integers(1, h - rh))
            left = int(rng.integers(1, w - rw))
            bg[top:top + rh, left:left + rw] = colors[int(rng.integers(k))]
    return np.asarray(bg, dtype=np.int8)


def _infer_foreground(inp):
    boundary = set(map(int, np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))))
    candidates = []
    for color in map(int, np.unique(inp)):
        if color in boundary:
            continue
        mask = inp == color
        rr, cc = np.where(mask)
        if len(rr) >= 3 and _connected(mask) and rr.max() - rr.min() >= 1 and cc.max() - cc.min() >= 1:
            candidates.append(color)
    assert len(candidates) == 1, (inp.shape, candidates)
    return candidates[0]


def _answer(inp, fg):
    rr, cc = np.where(inp == fg)
    crop = inp[int(rr.min()):int(rr.max()) + 1, int(cc.min()):int(cc.max()) + 1]
    return np.where(crop == fg, fg, 0).astype(np.int8)


def _pair(rng, regime, family, fg, need_blank=False):
    for _ in range(80):
        shape = _shape(rng, family)
        sh, sw = shape.shape
        h, w = _canvas(rng, sh, sw, regime)
        top = int(rng.integers(1, h - sh))
        left = int(rng.integers(1, w - sw))
        mode = int(rng.integers(7))
        if need_blank and mode == 0:
            mode = 3
        bg = _field(rng, h, w, fg, mode)
        palette = set(map(int, np.unique(bg)))
        boundary = set(map(int, np.concatenate((bg[0], bg[-1], bg[:, 0], bg[:, -1]))))
        if palette != boundary or fg in palette:
            continue
        patch = bg[top:top + sh, left:left + sw]
        if need_blank and not np.any((~shape) & (patch != 0)):
            continue
        inp = bg.copy()
        inp[top:top + sh, left:left + sw][shape] = fg
        if _infer_foreground(inp) != fg:
            continue
        return {"input": inp, "output": _answer(inp, fg)}
    raise RuntimeError("could not construct a distinct figure and field")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    families = ("diamond", "box", "open", "filled", "cross", "tee",
                "ell", "stair", "zigzag", "bend", "stagger", "compact")
    hollow = ("diamond", "box", "open")
    anchor = int(rng.integers(nt))
    zero_train = int(rng.choice([0, 0, 0, 1])) if nt > 2 else int(rng.choice([0, 0, 1]))
    zero_train_at = int(rng.choice([i for i in range(nt) if i != anchor])) if zero_train else -1
    zero_test_count = int(rng.choice([0, 0, 1, 1, 2])) if ns > 1 else int(rng.integers(2))
    zero_test_at = set(map(int, rng.choice(ns, size=zero_test_count, replace=False)))
    train_families = [str(rng.choice(families)) for _ in range(nt)]
    train_families[anchor] = str(rng.choice(hollow))
    if train_families[0] == train_families[1]:
        train_families[1 if anchor != 1 else 0] = str(rng.choice([f for f in families if f != train_families[0]]))
    test_families = ["compact" if rng.random() < 0.28 else str(rng.choice(families))
                     for _ in range(ns)]
    # A shared rule and output blank are the only puzzle-level conventions.
    # Color identities and field arrangements deliberately vary per pair.
    first_color = int(rng.integers(1, 10))
    train, test = [], []
    for i in range(nt):
        fg = 0 if i == zero_train_at else (first_color if i == anchor else int(rng.integers(1, 10)))
        if i != anchor and fg == first_color and zero_train_at != i:
            fg = 1 + fg % 9
        train.append(_pair(rng, "source_train", train_families[i], fg, i == anchor))
    for i in range(ns):
        fg = 0 if i in zero_test_at else int(rng.integers(1, 10))
        test.append(_pair(rng, "target_test", test_families[i], fg))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(pair):
    digest = hashlib.sha256()
    for key in ("input", "output"):
        grid = pair[key]
        digest.update(bytes(grid.shape))
        digest.update(np.asarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


# Fingerprints of the six solved source pairs, used only to prevent exact copies.
_SOURCE_FINGERPRINTS = frozenset({
    "84abcb300e20a7443872c080d006811d3a310b0cc70ac475706354d0c89f558e",
    "737d30eb69c8cd45273315f239947e688f9baaca8fc54ee4e267ffe383e3119f",
    "5a5327dfa67dd5736121c24c92d4578e417d94d9af02dbc4e922a8fc4288b7f4",
    "29c2609b1c2803738b1162d3bab35cfbe848b3f8ac205dd5798b1f88b95ff971",
    "48cafb81adec83d1900ca67f79500138bf9ba96be4765f4310a4efb21554c89f",
    "44c72fc5ff620e0d3e4e333b15a9b707dc7b0690971ad27267b08a3bf53a8d13",
})


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_fg = set()
    evidence = False
    for regime, pairs in (("source_train", train), ("target_test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            assert _fingerprint(pair) not in _SOURCE_FINGERPRINTS
            fg = _infer_foreground(inp)
            mask = inp == fg
            rr, cc = np.where(mask)
            r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
            assert min(r0, c0) >= 1 and r1 < inp.shape[0] - 1 and c1 < inp.shape[1] - 1
            assert min(out.shape) >= 2 and np.array_equal(out, _answer(inp, fg))
            assert not np.array_equal(inp, out)
            bg = inp[~mask]
            assert len(bg) and fg not in bg
            assert set(map(int, np.unique(bg))) <= set(map(int, np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))))
            if regime == "source_train":
                train_fg.add(fg)
                patch = inp[r0:r1 + 1, c0:c1 + 1]
                evidence |= bool(np.any((~mask[r0:r1 + 1, c0:c1 + 1]) & (patch != 0)))
    assert evidence and any(c != 0 for c in train_fg)
    assert len(train_fg) >= 2
    for pair in test:
        assert np.array_equal(_answer(pair["input"], _infer_foreground(pair["input"])), pair["output"])


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p["train"][0]) for p in puzzles}) >= 40
    dimensions, masks, palettes, zero_sizes = set(), set(), set(), set()
    zero_train = zero_test = small_canvas = large_canvas = 0
    squares = rectangles = nonrare = mono = rich = hollow = filled = margin_one = 0
    small_output = {"train": 0, "test": 0}
    large_output = {"train": 0, "test": 0}
    for p in puzzles:
        validate_generated_puzzle(p)
        zero_train += any(_infer_foreground(x["input"]) == 0 for x in p["train"])
        zero_test += any(_infer_foreground(x["input"]) == 0 for x in p["test"])
        within = {name: set() for name in ("fg", "palette", "mask", "canvas", "place")}
        for group in ("train", "test"):
            for pair in p[group]:
                inp, out = pair["input"], pair["output"]
                fg = _infer_foreground(inp)
                mask = inp == fg
                colors, counts = np.unique(inp, return_counts=True)
                bg_counts = [int(n) for c, n in zip(colors, counts) if c != fg]
                rr, cc = np.where(mask)
                cropped = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                dimensions.add(inp.shape)
                silhouette = (cropped.shape, cropped.tobytes())
                masks.add(silhouette)
                palettes.add(tuple(map(int, colors)))
                within["fg"].add(fg)
                within["palette"].add(tuple(map(int, np.unique(inp[~mask]))))
                within["mask"].add(silhouette)
                within["canvas"].add(inp.shape)
                within["place"].add((int(rr.min()), int(cc.min())))
                squares += inp.shape[0] == inp.shape[1]
                rectangles += inp.shape[0] != inp.shape[1]
                nonrare += int(mask.sum()) >= min(bg_counts)
                mono += len(colors) == 2
                rich += len(colors) >= 5
                hollow += bool(np.any(~cropped))
                filled += bool(np.all(cropped))
                margin_one += min(rr.min(), cc.min(), inp.shape[0] - rr.max() - 1,
                                  inp.shape[1] - cc.max() - 1) == 1
                small_canvas += max(inp.shape) <= 11
                large_canvas += max(inp.shape) >= 17
                small_output[group] += out.size <= 9
                large_output[group] += out.size >= 25
                if fg == 0:
                    zero_sizes.add(out.shape)
                    assert not np.any(inp[~mask] == 0) and not np.any(out)
                assert np.array_equal(out, _answer(inp, fg))
        assert all(len(values) >= 2 for values in within.values())
    assert len(dimensions) >= 25 and len(masks) >= 35 and len(palettes) >= 30
    assert squares >= 20 and rectangles >= 20 and nonrare >= 8
    assert mono >= 8 and rich >= 8 and hollow >= 20 and filled >= 8
    assert margin_one >= 8 and small_canvas >= 8 and large_canvas >= 8
    assert min(small_output.values()) >= 8 and min(large_output.values()) >= 8
    assert zero_train >= 4 and 4 <= zero_test < 42 and len(zero_sizes) >= 8
    for seed, nt, ns in ((7, 2, 1), (8, 10, 1), (9, 10, 4),
                         (10, None, 1), (11, None, 4), (12, 2, None), (13, 10, None)):
        a, b = generate_puzzle(seed, nt, ns), generate_puzzle(seed, nt, ns)
        assert _same(a, b)
        assert nt is None or len(a["train"]) == nt
        assert ns is None or len(a["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError((nt, ns))
