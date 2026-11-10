"""Generate connected-component reflection puzzles in the source ARC family."""

import hashlib
import numpy as np


FAMILIES = ("horizontal", "vertical", "diagonal")
SOURCE_PAIR_DIGESTS = {
    "a3c52b1dd2cb5df874d13cfcda72fc243293134cb5e33729533c271cb15d420c",
    "65e2194c1eb039a607b65d9db2886860a7bf0e5413512f5558bf4d62a9ea2a7a",
    "36f92207068116af59ce9ac49ed5f7e0958bf653781983257fed446f1921b3cc",
    "cbe1f80eb7c5d01849b9c17a0d41e071879f3ebfde0c5701acd1ed523c2153ee",
    "f525929a2da046f58b368fba91472f2a1f507d6e69db44217ae93699ce623d4b",
}


def _counts(rng, train, test):
    for value, lo, hi in ((train, 2, 10), (test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= int(value) <= hi):
            raise ValueError("illegal pair count")
    if test is None and train is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        test = int(test)
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        train = int(train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            cells, stack = [], [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append(cells)
    return result


def _flags(mask):
    h = np.array_equal(mask, mask[::-1, :])
    v = np.array_equal(mask, mask[:, ::-1])
    if mask.shape[0] == mask.shape[1]:
        d1 = np.array_equal(mask, mask.T)
        d2 = np.array_equal(mask, mask[::-1, ::-1].T)
    else:
        d1 = d2 = False
    return h, v, d1, d2


def _family(mask):
    h, v, d1, d2 = _flags(mask)
    if int(h) + int(v) + int(d1 or d2) != 1:
        return None
    return "horizontal" if h else "vertical" if v else "diagonal"


def _valid_mask(mask, family):
    return (mask.sum() >= 5 and mask.sum() < mask.size and
            np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)) and
            len(_components(mask)) == 1 and _family(mask) == family)


def _axis_mask(rng, horizontal, long_size=None, sparse=None):
    """Grow mirrored branches from a continuous spine; gaps arise naturally."""
    for _ in range(120):
        length = long_size or int(rng.integers(4, 11))
        breadth = int(rng.integers(3, min(9, length + 2)))
        h, w = breadth, length
        mask = np.zeros((h, w), bool)
        mask[(h - 1) // 2:h // 2 + 1, :] = True
        density = rng.uniform(0.30, 0.55) if sparse else rng.uniform(0.38, 0.77)
        target = max(int(mask.sum()) + 2, int(round(h * w * density)))
        for _ in range(h * w * 5):
            if int(mask.sum()) >= target and mask[0].any():
                break
            candidates = []
            for r in range((h + 1) // 2):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    rr = h - 1 - r
                    trial = mask.copy()
                    trial[r, c] = trial[rr, c] = True
                    if len(_components(trial)) == 1:
                        candidates.append((r, c, trial))
            if not candidates:
                break
            # Progress toward the outer row before adding random side arms.
            if not mask[0].any() and rng.random() < 0.65:
                minimum = min(x[0] for x in candidates)
                candidates = [x for x in candidates if x[0] == minimum]
            mask = candidates[int(rng.integers(len(candidates)))][2]
        if not horizontal:
            mask = mask.T.copy()
        family = "horizontal" if horizontal else "vertical"
        if _valid_mask(mask, family):
            return mask
    raise RuntimeError("axis mask construction failed")


def _orbit_dual(r, c, n):
    return {(r, c), (c, r), (n - 1 - c, n - 1 - r),
            (n - 1 - r, n - 1 - c)}


def _dual_mask(rng, n, sparse=None):
    """Grow connected orbits under both diagonal reflections."""
    for _ in range(150):
        mask = np.zeros((n, n), bool)
        if n % 2:
            mask[n // 2, n // 2] = True
        else:
            mask[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1] = True
        density = rng.uniform(0.28, 0.48) if sparse else rng.uniform(0.37, 0.70)
        target = int(round(n * n * density))
        for _ in range(n * n):
            if int(mask.sum()) >= target and mask[0].any() and _family(mask) == "diagonal":
                break
            choices = []
            visited = set()
            for r in range(n):
                for c in range(n):
                    orbit = _orbit_dual(r, c, n)
                    key = tuple(sorted(orbit))
                    if key in visited or all(mask[y, x] for y, x in orbit):
                        continue
                    visited.add(key)
                    trial = mask.copy()
                    for y, x in orbit:
                        trial[y, x] = True
                    if len(_components(trial)) == 1 and int(trial.sum()) <= int(0.75 * n * n):
                        choices.append((trial, any(y == 0 for y, x in orbit)))
            if not choices:
                break
            if not mask[0].any() and rng.random() < 0.45:
                edge = [x for x in choices if x[1]]
                if edge:
                    choices = edge
            mask = choices[int(rng.integers(len(choices)))][0]
        if _valid_mask(mask, "diagonal") and _flags(mask)[2:4] == (True, True):
            return mask
    raise RuntimeError("dual diagonal mask construction failed")


def _single_mask(rng, n, slope):
    for _ in range(100):
        rr, cc = np.indices((n, n))
        mask = abs(rr - cc) <= 1
        target = int(round(n * n * rng.uniform(0.33, 0.61)))
        for _ in range(n * n):
            if int(mask.sum()) >= target:
                break
            adjacent = np.zeros(mask.shape, bool)
            adjacent[1:] |= mask[:-1]
            adjacent[:-1] |= mask[1:]
            adjacent[:, 1:] |= mask[:, :-1]
            adjacent[:, :-1] |= mask[:, 1:]
            choices = np.argwhere(adjacent & ~mask)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            mask[r, c] = mask[c, r] = True
        if slope == "anti":
            mask = mask[:, ::-1].copy()
        if (_valid_mask(mask, "diagonal") and
                sum(_flags(mask)[2:4]) == 1):
            return mask
    raise RuntimeError("single diagonal mask construction failed")


def _shape(rng, split, source_shift, large_train):
    if source_shift:
        if split == "train":
            return int(rng.integers(17, 23)), int(rng.integers(18, 24))
        return int(rng.integers(24, 31)), int(rng.integers(24, 31))
    if split == "train" and large_train:
        return int(rng.integers(23, 31)), int(rng.integers(22, 31))
    low = 17 if split == "train" else 20
    return int(rng.integers(low, 31)), int(rng.integers(low, 31))


def _place(rng, masks, height, width):
    order = sorted(range(len(masks)), key=lambda i: masks[i].size, reverse=True)
    for _ in range(90):
        boxes = np.zeros((height, width), bool)
        occupied = np.zeros((height, width), bool)
        placements = [None] * len(masks)
        good = True
        for i in order:
            m = masks[i]
            h, w = m.shape
            if h > height or w > width:
                good = False
                break
            # Complete candidate enumeration avoids the old random-position bias.
            spots = []
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    region = boxes[max(0, r - 1):min(height, r + h + 1),
                                   max(0, c - 1):min(width, c + w + 1)]
                    if not region.any():
                        spots.append((r, c))
            if not spots:
                good = False
                break
            r, c = spots[int(rng.integers(len(spots)))]
            boxes[r:r + h, c:c + w] = True
            occupied[r:r + h, c:c + w] |= m
            placements[i] = (r, c)
        if good:
            return placements, occupied
    raise RuntimeError("mask placement failed")


def _specks(rng, occupied, count):
    used = occupied.copy()
    result = []
    for _ in range(count):
        blocked = used.copy()
        blocked[1:] |= used[:-1]
        blocked[:-1] |= used[1:]
        blocked[:, 1:] |= used[:, :-1]
        blocked[:, :-1] |= used[:, 1:]
        candidates = np.argwhere(~blocked)
        if not len(candidates):
            raise RuntimeError("speck placement failed")
        r, c = map(int, candidates[int(rng.integers(len(candidates)))])
        used[r, c] = True
        result.append((r, c))
    return result


def _spec_mask(rng, spec, split, long_form):
    family, subtype, preset = spec
    if preset is not None:
        return preset.copy()
    if family != "diagonal":
        is_long = long_form and rng.random() < 0.65
        length = int(rng.integers(9, 12)) if is_long else int(rng.integers(4, 9))
        return _axis_mask(rng, family == "horizontal", length, sparse=is_long)
    size = int(rng.choice([5, 6, 7, 8, 9] if split == "test" else [5, 6, 7, 8]))
    if subtype == "dual":
        return _dual_mask(rng, size, sparse=(size >= 7 and rng.random() < 0.7))
    return _single_mask(rng, size, subtype)


def _pair(rng, split, params, background, required, source_shift, large_train, zero_noise):
    for attempt in range(100):
        height, width = _shape(rng, split, source_shift, large_train)
        area = height * width
        cap = min(7 if split == "test" else 5, max(3, area // 100))
        lower = max(len(required), 2 if split == "train" else 3)
        if lower > cap:
            continue
        count = int(rng.integers(lower, cap + 1))
        specs = list(required)
        while len(specs) < count:
            family = FAMILIES[int(rng.choice(3, p=[0.39, 0.39, 0.22]))]
            subtype = ("dual" if rng.random() < 0.78 else
                       ("main" if rng.random() < 0.5 else "anti")) if family == "diagonal" else None
            if subtype in ("main", "anti") and not params["single_enabled"]:
                subtype = "dual"
            specs.append((family, subtype, None))
        rng.shuffle(specs)
        long_form = split == "test" or rng.random() < 0.25
        try:
            masks = [_spec_mask(rng, spec, split, long_form) for spec in specs]
            placements, occupied = _place(rng, masks, height, width)
            if zero_noise:
                speck_count = 0
            elif split == "train":
                speck_count = int(rng.integers(1, 7))
            else:
                speck_count = int(rng.integers(2, 11))
            noise = _specks(rng, occupied, speck_count)
        except RuntimeError:
            continue
        if int(occupied.sum()) * 2 >= area:
            continue
        a = np.full((height, width), background, np.int8)
        b = np.full((height, width), background, np.int8)
        for spec, mask, (r, c) in zip(specs, masks, placements):
            ys, xs = np.where(mask)
            family = _family(mask)
            if family != spec[0]:
                raise AssertionError("constructed mask changed category")
            a[r + ys, c + xs] = params["source"]
            b[r + ys, c + xs] = params["labels"][family]
        for r, c in noise:
            a[r, c] = params["source"]
        return {"input": a, "output": b}
    raise RuntimeError("pair construction failed")


def _analyze(a):
    values, counts = np.unique(a, return_counts=True)
    assert len(values) == 2
    bg = int(values[int(np.argmax(counts))])
    source = int(values[int(np.argmin(counts))])
    assert int(max(counts)) > int(min(counts))
    objects, specks = [], []
    for cells in _components(a == source):
        if len(cells) == 1:
            specks.extend(cells)
            continue
        ys, xs = zip(*cells)
        top, left = min(ys), min(xs)
        mask = np.zeros((max(ys) - top + 1, max(xs) - left + 1), bool)
        for r, c in cells:
            mask[r - top, c - left] = True
        family = _family(mask)
        assert family is not None and _valid_mask(mask, family)
        objects.append((family, mask, cells))
    return bg, source, objects, specks


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    colors = list(map(int, rng.permutation(10)))
    params = {"source": colors[0],
              "labels": dict(zip(FAMILIES, colors[1:4])),
              "single_enabled": bool(rng.random() < 0.48)}
    backgrounds = colors[4:]
    source_shift = bool(rng.random() < 0.48)
    large_train = not source_shift and bool(rng.random() < 0.48)
    repeated = bool(rng.random() < 0.28)
    repeat_mask = _axis_mask(rng, True, int(rng.integers(5, 9))) if repeated else None
    required = [[] for _ in range(train_count + test_count)]
    # All color roles are demonstrated in training. Singles are an extension
    # and both diagonal slopes are taught before they can occur in tests.
    must = [("horizontal", None, repeat_mask if repeated else None),
            ("vertical", None, np.rot90(repeat_mask).copy() if repeated else None),
            ("diagonal", "dual", None)]
    if params["single_enabled"]:
        must += [("diagonal", "main", None), ("diagonal", "anti", None)]
    if repeated:
        required[0].append(must[0])
        required[1].append(must[1])
        must = must[2:]
    for item in must:
        sizes = [len(required[i]) for i in range(train_count)]
        smallest = min(sizes)
        candidates = [i for i, n in enumerate(sizes) if n == smallest]
        required[candidates[int(rng.integers(len(candidates))) ]].append(item)
    # Test diagonal counts vary, including examples without diagonal objects.
    for i in range(train_count, train_count + test_count):
        dual_count = int(rng.choice([0, 1, 2, 3], p=[0.10, 0.43, 0.37, 0.10]))
        required[i].extend([("diagonal", "dual", None)] * dual_count)
        if params["single_enabled"] and rng.random() < 0.43:
            required[i].append(("diagonal", "main" if rng.random() < 0.5 else "anti", None))
    # Pair backgrounds are independent of the object geometry; the first two
    # are deliberately different so the within-puzzle distinction is visible.
    bgs = [int(backgrounds[int(rng.integers(len(backgrounds)))])]
    bgs.append(int(rng.choice([x for x in backgrounds if x != bgs[0]])))
    bgs.extend(int(backgrounds[int(rng.integers(len(backgrounds)))])
               for _ in range(train_count + test_count - 2))
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count)
    zero_noise_index = int(rng.integers(train_count + test_count)) if rng.random() < 0.28 else -1
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(train_count + test_count):
        local = np.random.default_rng(int(pair_seeds[i]))
        split = "train" if i < train_count else "test"
        for _ in range(15):
            pair = _pair(local, split, params, bgs[i], required[i], source_shift,
                         large_train and i == 0, i == zero_noise_index)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("duplicate input construction")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    mapping, backgrounds, sources = {}, set(), set()
    train_subtypes = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            ka, kb = _key(a), _key(b)
            assert ka not in seen_inputs and (ka, kb) not in seen_pairs
            seen_inputs.add(ka)
            seen_pairs.add((ka, kb))
            assert hashlib.sha256(a.astype(np.int8).tobytes() +
                                  b.astype(np.int8).tobytes()).hexdigest() not in SOURCE_PAIR_DIGESTS
            bg, source, objects, specks = _analyze(a)
            backgrounds.add(bg)
            sources.add(source)
            assert (16 if split == "train" else 20) <= a.shape[0] <= 30
            assert (17 if split == "train" else 20) <= a.shape[1] <= 30
            assert (2 if split == "train" else 3) <= len(objects) <= (5 if split == "train" else 7)
            assert 0 <= len(specks) <= (6 if split == "train" else 10)
            covered = np.zeros(a.shape, bool)
            for family, mask, cells in objects:
                ys, xs = zip(*cells)
                colors = np.unique(b[ys, xs])
                assert len(colors) == 1 and int(colors[0]) != bg
                label = int(colors[0])
                if split == "train":
                    assert family not in mapping or mapping[family] == label
                    mapping[family] = label
                    if family == "diagonal":
                        d1, d2 = _flags(mask)[2:]
                        train_subtypes.add("dual" if d1 and d2 else "main" if d1 else "anti")
                covered[ys, xs] = True
            assert np.all(b[~covered] == bg)
            analyses[split].append((a, b, bg, source, objects, specks))
    assert len(sources) == 1 and len(backgrounds) >= 2
    assert set(mapping) == set(FAMILIES) and len(set(mapping.values())) == 3
    assert not (backgrounds & set(mapping.values()))
    assert not (backgrounds & sources) and not (set(mapping.values()) & sources)
    assert "dual" in train_subtypes
    for split in ("train", "test"):
        for a, b, bg, source, objects, specks in analyses[split]:
            expected = np.full(a.shape, bg, np.int8)
            for family, mask, cells in objects:
                if family == "diagonal" and split == "test":
                    d1, d2 = _flags(mask)[2:]
                    subtype = "dual" if d1 and d2 else "main" if d1 else "anti"
                    assert subtype in train_subtypes
                ys, xs = zip(*cells)
                expected[ys, xs] = mapping[family]
            assert np.array_equal(expected, b)


def self_test() -> None:
    a = generate_puzzle(701)
    b = generate_puzzle(701)
    assert all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output"))
    assert _key(generate_puzzle(702)["train"][0]["input"]) != _key(a["train"][0]["input"])
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(800 + i, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    for kw in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
               {"num_test": 5}, {"num_train": 2, "num_test": 2},
               {"num_train": 3, "num_test": 4}, {"num_train": True},
               {"num_train": 3.0}):
        try:
            generate_puzzle(900, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    dual_sizes, single_slopes, backgrounds_varied = set(), set(), 0
    zero_noise, high_noise, long_sparse, repeat_rotations = 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        signature = []
        bgs = []
        axis_masks = {"horizontal": set(), "vertical": set()}
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                bg, source, objects, specks = _analyze(a)
                bgs.append(bg)
                shapes.add(a.shape)
                zero_noise += len(specks) == 0
                high_noise += len(specks) >= (3 if split == "train" else 9)
                signature.append((a.shape, len(objects), len(specks),
                                  tuple(sorted((f, m.shape, int(m.sum())) for f, m, _ in objects))))
                for f, m, _ in objects:
                    if f == "diagonal":
                        d1, d2 = _flags(m)[2:]
                        if d1 and d2:
                            dual_sizes.add(m.shape[0])
                        else:
                            single_slopes.add("main" if d1 else "anti")
                    else:
                        axis_masks[f].add(_key(m))
                        if max(m.shape) >= 9 and m.sum() / m.size <= 0.55:
                            long_sparse += 1
        backgrounds_varied += len(set(bgs)) >= 2
        repeat_rotations += any(_key(np.rot90(np.frombuffer(k[1], dtype=bool).reshape(k[0]))) in axis_masks["vertical"]
                                for k in axis_masks["horizontal"])
        layouts.add(tuple(signature))
        palettes.add(tuple(sorted(set(int(v) for q in p["train"]
                                      for v in np.unique(q["output"])))))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(layouts) >= 40 and len(palettes) >= 20
    assert len(dual_sizes) >= 3 and single_slopes == {"main", "anti"}
    assert backgrounds_varied == 42 and zero_noise >= 2 and high_noise >= 5
    assert long_sparse >= 12 and repeat_rotations >= 3
