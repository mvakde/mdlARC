"""Periodic wallpaper and largest occluder puzzles."""

import hashlib

import numpy as np


_PERIODS = ((3, 3), (4, 4), (2, 3), (3, 4), (4, 3), (3, 2))
_SOURCE_INPUTS = {
    ((22, 20), "0d7522490247ec7654b02c026306395e9f3a55d4035b1015943127d5460203f5"),
    ((23, 19), "2f2a1db515b97813de2736c0981a9addda5d03830d5ca42e2f1ed06b06ecf943"),
    ((21, 22), "dd120a9fd6fe87c640579ef68825ebe1a57e58676b8b24fd30451f500406931c"),
    ((23, 22), "956ceda6646c3c12bd626c10a070fc74915c6bebd047631cc953709d9346768b"),
}


def _seed_value(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    word = _seed_value(seed)
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + word % 4
        choices = list(range(num_test + 1, 11))
        num_train = choices[(3 * word + word // 4) % len(choices)]
    elif num_train is None:
        choices = list(range(num_test + 1, 11))
        num_train = choices[(5 * word + word // 8 + 7) % len(choices)]
    elif num_test is None:
        choices = list(range(1, min(4, num_train - 1) + 1))
        num_test = choices[(7 * word + word // 16 + 3) % len(choices)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _primitive(tile):
    h, w = tile.shape
    return all(not np.array_equal(tile, np.roll(tile, -k, axis=0)) for k in range(1, h)) and all(
        not np.array_equal(tile, np.roll(tile, -k, axis=1)) for k in range(1, w)
    )


def _tile(rng, period, palette, motif):
    h, w = period
    colors = np.asarray(palette, dtype=np.int8)
    for _ in range(80):
        if motif == "cyclic" and h == w and len(colors) == h:
            offsets = (np.arange(h)[:, None] + np.arange(w)[None, :]) % h
            tile = colors[offsets]
        elif motif == "bands":
            tile = rng.choice(colors, (h, w)).astype(np.int8)
            tile[0, :] = colors[0]
        elif motif == "lattice":
            tile = rng.choice(colors, (h, w)).astype(np.int8)
            tile[0, :] = colors[0]
            tile[:, 0] = colors[0]
        else:
            tile = rng.choice(colors, (h, w)).astype(np.int8)
        if set(np.unique(tile)) == set(colors) and _primitive(tile):
            return tile
    return None


def _wallpaper(tile, shape):
    h, w = shape
    return tile[np.arange(h)[:, None] % tile.shape[0], np.arange(w)[None, :] % tile.shape[1]]


def _overlap(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw


def _touch(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    vertical = ar < br + bh and br < ar + ah and (ac + aw == bc or bc + bw == ac)
    horizontal = ac < bc + bw and bc < ac + aw and (ar + ah == br or br + bh == ar)
    return vertical or horizontal


def _place(rng, canvas, sizes, location):
    h, w = canvas
    th, tw = sizes[0]
    columns = list(range(w - tw + 1))
    if location == "left":
        preferred = [c for c in columns if c + tw // 2 < w // 3]
    elif location == "right":
        preferred = [c for c in columns if c + tw // 2 >= 2 * w // 3]
    else:
        preferred = [c for c in columns if w // 3 <= c + tw // 2 < 2 * w // 3]
    if preferred:
        columns = preferred
    for _ in range(30):
        target = (int(rng.integers(h - th + 1)), int(rng.choice(columns)), th, tw)
        rects = [target]
        for rh, rw in sizes[1:]:
            positions = [(r, c, rh, rw) for r in range(h - rh + 1)
                         for c in range(w - rw + 1)
                         if all(not _overlap((r, c, rh, rw), old) for old in rects)]
            if not positions:
                break
            touching = [p for p in positions if any(_touch(p, old) for old in rects)]
            if touching and rng.random() < 0.25:
                positions = touching
            rects.append(positions[int(rng.integers(len(positions)))])
        if len(rects) == len(sizes):
            return rects
    return None


def _candidate(inp, period):
    ph, pw = period
    tile = np.empty((ph, pw), dtype=np.int8)
    for r in range(ph):
        for c in range(pw):
            counts = np.bincount(inp[r::ph, c::pw].ravel().astype(np.int64), minlength=10)
            winner = int(np.argmax(counts))
            if counts[winner] < 3 or counts[winner] <= 2 * int(np.partition(counts, -2)[-2]):
                return None
            tile[r, c] = winner
    if not _primitive(tile):
        return None
    wallpaper = _wallpaper(tile, inp.shape)
    mismatch = inp != wallpaper
    if mismatch.sum() < 2 or mismatch.mean() > 0.52:
        return None
    palette = set(int(x) for x in np.unique(tile))
    mismatch_colors = set(int(x) for x in np.unique(inp[mismatch]))
    if palette & mismatch_colors or len(mismatch_colors) < 2 or len(mismatch_colors) > 5:
        return None
    rectangles = []
    for color in sorted(mismatch_colors):
        points = np.argwhere(inp == color)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0) + 1
        if (r1 - r0) * (c1 - c0) != len(points):
            return None
        if not np.all(inp[r0:r1, c0:c1] == color):
            return None
        rectangles.append((int(r0), int(c0), int(r1 - r0), int(c1 - c0), int(color)))
    if sum(r[2] * r[3] for r in rectangles) != int(mismatch.sum()):
        return None
    areas = sorted((r[2] * r[3] for r in rectangles), reverse=True)
    if areas[0] == areas[1]:
        return None
    target = max(rectangles, key=lambda r: r[2] * r[3])
    return {"period": period, "tile": tile, "wallpaper": wallpaper,
            "palette": tuple(sorted(palette)), "rectangles": rectangles,
            "target": target, "margin": areas[0] - areas[1],
            "ratio": areas[1] / areas[0]}


def _analyze(inp, out=None):
    candidates = [_candidate(inp, p) for p in _PERIODS]
    candidates = [c for c in candidates if c is not None]
    assert candidates, "no recoverable wallpaper and rectangles"
    candidates.sort(key=lambda c: c["period"][0] * c["period"][1])
    best = candidates[0]
    for other in candidates[1:]:
        assert (other["period"][0] % best["period"][0] == 0 and
                other["period"][1] % best["period"][1] == 0), "ambiguous periods"
        assert np.array_equal(other["wallpaper"], best["wallpaper"]), "ambiguous wallpaper"
    r, c, h, w, _ = best["target"]
    expected = best["wallpaper"][r:r+h, c:c+w]
    if out is not None:
        assert np.array_equal(out, expected), "incorrect restored crop"
    best["expected"] = expected
    return best


def _target_size(rng, canvas, kind):
    h, w = canvas
    if kind == "square":
        side = int(rng.integers(5, min(12, h - 3, w - 3) + 1))
        return side, side
    if kind == "wide":
        rh = int(rng.integers(4, min(10, h - 3) + 1))
        rw = int(rng.integers(max(rh + 1, 7), min(w - 3, 16) + 1))
        return rh, rw
    rw = int(rng.integers(4, min(10, w - 3) + 1))
    rh = int(rng.integers(max(rw + 1, 7), min(h - 3, 16) + 1))
    return rh, rw


def _challenger(rng, canvas, target, regime, side_goal):
    area = target[0] * target[1]
    cap_h, cap_w = canvas
    candidates = []
    for h in range(2, min(17, cap_h - 1)):
        for w in range(2, min(17, cap_w - 1)):
            ratio = h * w / area
            if 0.48 <= ratio < 1 and (h, w) != target:
                candidates.append((h, w, ratio))
    if not candidates:
        return None
    band = int(rng.integers(3))
    lo, hi = ((0.48, 0.72), (0.72, 0.88), (0.88, 1.0))[band]
    filtered = [x for x in candidates if lo <= x[2] < hi]
    if filtered:
        candidates = filtered
    if side_goal is not None:
        side_longer = [x for x in candidates if
                       (x[0] > target[0] if side_goal == "taller" else x[1] > target[1])]
        if not side_longer:
            return None
        candidates = side_longer
    elif regime == "source_train" or rng.random() < 0.65:
        side_longer = [x for x in candidates if x[0] > target[0] or x[1] > target[1]]
        if side_longer:
            candidates = side_longer
    choice = candidates[int(rng.integers(len(candidates)))]
    return choice[:2]


def _generate_pair(rng, regime, period, kind, block_count, palette_hint=None, side_goal=None):
    for _ in range(180):
        scale = int(rng.choice((0, 1, 2), p=(0.22, 0.42, 0.36)))
        lo, hi = ((16, 20), (21, 25), (26, 30))[scale]
        canvas = (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
        motifs = ("bands", "lattice", "cyclic", "free") if period[0] == period[1] else (
            "bands", "lattice", "free")
        motif = str(rng.choice(motifs))
        cyclic_size = period[0] if motif == "cyclic" else None
        if palette_hint is not None and (cyclic_size is None or len(palette_hint) == cyclic_size) and rng.random() < 0.5:
            palette = palette_hint
        else:
            palette_size = cyclic_size if cyclic_size is not None else int(
                rng.integers(2, min(4, 10 - block_count) + 1))
            palette = tuple(sorted(int(x) for x in rng.choice(10, palette_size, replace=False)))
        remaining = [c for c in range(10) if c not in palette]
        if len(remaining) < block_count:
            continue
        tile = _tile(rng, period, palette, motif)
        if tile is None:
            continue
        try:
            target_size = _target_size(rng, canvas, kind)
        except ValueError:
            continue
        challenger = _challenger(rng, canvas, target_size, regime, side_goal)
        if challenger is None:
            continue
        area = target_size[0] * target_size[1]
        sizes = [target_size, challenger]
        for _ in range(block_count - 2):
            choices = [(h, w) for h in range(2, 7) for w in range(2, 7)
                       if h * w < min(area * 0.46, challenger[0] * challenger[1])]
            if not choices:
                break
            sizes.append(choices[int(rng.integers(len(choices)))])
        if len(sizes) != block_count:
            continue
        rects = _place(rng, canvas, sizes, str(rng.choice(("left", "center", "right"))))
        if rects is None:
            continue
        wallpaper = _wallpaper(tile, canvas)
        inp = wallpaper.copy()
        colors = rng.choice(remaining, block_count, replace=False)
        for (r, c, h, w), color in zip(rects, colors):
            inp[r:r+h, c:c+w] = color
        r, c, h, w = rects[0]
        out = wallpaper[r:r+h, c:c+w].copy()
        try:
            analysis = _analyze(inp, out)
        except AssertionError:
            continue
        if analysis["period"] != period or len(analysis["rectangles"]) != block_count:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a uniquely recoverable pair")


def _signature(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(_seed_value(seed))
    # The area/crop rule is shared across the puzzle. Colors and periods are
    # visible pair properties, and can vary independently between examples.
    streams = rng.integers(0, np.iinfo(np.int64).max, num_train + num_test, dtype=np.int64)
    train_periods = [(3, 3), (4, 4)]
    rng.shuffle(train_periods)
    train_periods += [_PERIODS[int(rng.choice(len(_PERIODS), p=(.31, .29, .1, .1, .1, .1)))]
                      for _ in range(num_train - 2)]
    train_kinds = ["square", str(rng.choice(("wide", "tall")))]
    rng.shuffle(train_kinds)
    train_kinds += [str(rng.choice(("square", "wide", "tall"))) for _ in range(num_train - 2)]
    train_blocks = [2, int(rng.choice((3, 4)))]
    rng.shuffle(train_blocks)
    train_blocks += [int(rng.choice((2, 3, 4))) for _ in range(num_train - 2)]
    trained_combinations = list(zip(train_periods, train_kinds, train_blocks))
    train_side_goals = ["taller", "wider"] + [None] * (num_train - 2)
    rng.shuffle(train_side_goals)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_palettes = []
    for i in range(num_train + num_test):
        pair_rng = np.random.default_rng(int(streams[i]))
        if i < num_train:
            regime = "source_train"
            period, kind, blocks = train_periods[i], train_kinds[i], train_blocks[i]
            hint = None
            side_goal = train_side_goals[i]
        else:
            regime = "target_test"
            if rng.random() < .35:
                period, kind, blocks = trained_combinations[int(rng.integers(len(trained_combinations)))]
            else:
                period = _PERIODS[int(rng.choice(len(_PERIODS), p=(.35, .30, .09, .09, .09, .08)))]
                kind = str(rng.choice(("square", "wide", "tall")))
                blocks = int(rng.choice((2, 3, 4, 5), p=(.22, .35, .28, .15)))
            hint = train_palettes[int(rng.integers(len(train_palettes)))] if rng.random() < .5 else None
            side_goal = None
        for _ in range(8):
            pair = _generate_pair(pair_rng, regime, period, kind, blocks, hint, side_goal)
            signature = _signature(pair["input"])
            if signature not in seen:
                break
        else:
            raise RuntimeError("duplicate pair construction")
        seen.add(signature)
        puzzle["train" if i < num_train else "test"].append(pair)
        if i < num_train:
            train_palettes.append(_analyze(pair["input"])["palette"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    train_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert not np.array_equal(inp, out)
            key = _signature(inp)
            pair_key = key, _signature(out)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            digest = hashlib.sha256(inp.astype(np.int8).tobytes()).hexdigest()
            assert (inp.shape, digest) not in _SOURCE_INPUTS
            a = _analyze(inp, out)
            rects = a["rectangles"]
            assert 2 <= len(rects) <= (4 if split == "train" else 5)
            assert a["margin"] >= 1 and a["ratio"] >= .48
            assert len(a["palette"]) <= 4
            for r in range(a["period"][0]):
                for c in range(a["period"][1]):
                    visible = inp[r::a["period"][0], c::a["period"][1]]
                    assert int(np.sum(visible == a["tile"][r, c])) >= 3
            assert 16 <= inp.shape[0] <= 30 and 16 <= inp.shape[1] <= 30
            if split == "train":
                train_features.append(a)
    assert {(3, 3), (4, 4)} <= {a["period"] for a in train_features}
    assert any(a["target"][2] == a["target"][3] for a in train_features)
    assert any(a["target"][2] != a["target"][3] for a in train_features)
    assert 2 in {len(a["rectangles"]) for a in train_features}
    assert any(len(a["rectangles"]) >= 3 for a in train_features)
    assert any(any(r[2] > a["target"][2] and r != a["target"]
                   for r in a["rectangles"]) for a in train_features)
    assert any(any(r[3] > a["target"][3] and r != a["target"]
                   for r in a["rectangles"]) for a in train_features)


def _equal(a, b):
    return all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output")) and all(
        len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _equal(generate_puzzle(987654, 6, 3), generate_puzzle(987654, 6, 3))
    assert not _equal(generate_puzzle(111, 5, 2), generate_puzzle(112, 5, 2))
    input_keys, layouts, shapes, outputs, motifs = set(), set(), set(), set(), set()
    phase_zero = aligned = unequal = four_blocks = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _analyze(inp, out)
                r, c, h, w, _ = a["target"]
                input_keys.add(_signature(inp))
                shapes.add(inp.shape)
                outputs.add(out.shape)
                layouts.add((a["period"], len(a["rectangles"]), r, c, h, w))
                motifs.add(_signature(a["tile"]))
                phase_zero += (r % a["period"][0] == 0 and c % a["period"][1] == 0)
                aligned += (h % a["period"][0] == 0 or w % a["period"][1] == 0)
                unequal += (a["period"][0] != a["period"][1])
                four_blocks += (len(a["rectangles"]) >= 4)
    assert len(input_keys) >= 200 and len(layouts) >= 150
    assert len(shapes) >= 35 and len(outputs) >= 30 and len(motifs) >= 100
    assert phase_zero >= 5 and aligned >= 10 and unequal >= 5 and four_blocks >= 5
    for seed, nt, ns in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                         (2004, None, 1), (2005, None, 4), (2006, 2, None),
                         (2007, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for kw in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
               {"num_test": 5}, {"num_train": 2, "num_test": 2},
               {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **kw)
        except ValueError:
            continue
        raise AssertionError(f"accepted illegal counts: {kw}")
