"""Compositional generators for the diagonal-anchor ARC frame."""

import hashlib

import numpy as np


GAPS = (5, 9, 13, 17, 21, 25)
CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
SOURCE_DIGESTS = frozenset({
    ((19, 16), "ba8e6699c6603cc9e8c71f3b40f6816de00b127c59eefb889229dacf413b8934"),
    ((30, 30), "191dd7b68da20ac076c88b5230d8c9722e0a5612dc66410694e41f3759541bde"),
    ((24, 26), "b2468a72b8f7ddd86af1a34243085a2f8c6dd3411120d4adb24262d06bf6c122"),
    ((18, 22), "dd3add19808942fbefe2eccf1b13de26123e6b33a892135d958afcd80f381288"),
})


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _counts(seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    else:
        num_test = 1 + seed % min(4, num_train - 1)
    return num_train, num_test


def _axis(rng, k, gap, force_dimension=None):
    minimum = 2 * k + gap + 2
    assert minimum <= 30
    dimension = int(rng.integers(minimum, 31)) if force_dimension is None else force_dimension
    if not minimum <= dimension <= 30:
        raise ValueError("forced dimension cannot fit geometry")
    extra = dimension - minimum
    before = int(rng.integers(0, extra + 1))
    return before + 1, dimension


def _geometry_options(scales, gaps, used=(), required=None):
    used = set(used)
    options = []
    for k in scales:
        valid = [g for g in gaps if 2 * k + g + 2 <= 30]
        for h in valid:
            for v in valid:
                triple = (k, h, v)
                if triple in used:
                    continue
                if required == "long" and h < 17 and v < 17:
                    continue
                if required == "short" and h != 5 and v != 5:
                    continue
                options.append(triple)
    return options


def _pick_geometry(rng, scales, gaps, used=(), required=None):
    options = _geometry_options(scales, gaps, used, required)
    if not options:
        raise ValueError("no feasible fresh geometry")
    # Select the scale first so a scale with many feasible gap pairs does not dominate.
    choices = sorted({triple[0] for triple in options})
    k = choices[int(rng.integers(len(choices)))]
    at_scale = [triple for triple in options if triple[0] == k]
    return at_scale[int(rng.integers(len(at_scale)))]


def _geometry_schedule(rng, seed, mode, num_train, num_test):
    if mode == 0:  # Source-style unseen anchor scale.
        train_scales, train_gaps = (2, 3), (5, 9, 13, 17)
    elif mode == 2:  # Longer connector in the test.
        train_scales, train_gaps = (1, 2, 3, 4), (5, 9, 13)
    elif mode == 3:  # Shorter connector in the test.
        train_scales, train_gaps = (1, 2, 3, 4, 5, 6), (9, 13, 17, 21)
    else:  # Overlap or new combinations of visible sizes and lengths.
        train_scales, train_gaps = tuple(range(1, 9)), GAPS

    first_scales = list(rng.choice(train_scales, size=2, replace=False))
    training = []
    for index in range(num_train):
        if index < 2:
            scales = (int(first_scales[index]),)
        elif index == 2:
            scales = (int(first_scales[0]),)
        elif index == 3:
            scales = (int(first_scales[1]),)
        else:
            scales = train_scales
        training.append(_pick_geometry(rng, scales, train_gaps, training))
    rng.shuffle(training)

    testing = []
    for index in range(num_test):
        used = training + testing
        if mode == 0:
            scales, gaps, required = (1, 4, 5), (9, 13, 17, 21), None
        elif mode == 1:
            scales, gaps, required = tuple({t[0] for t in training}), GAPS, None
        elif mode == 2:
            scales, gaps, required = (1, 2, 3), GAPS, "long"
        elif mode == 3:
            scales, gaps, required = tuple({t[0] for t in training}), GAPS, "short"
        else:
            scales, gaps, required = tuple({t[0] for t in training}), GAPS, None
        # A few long-gap puzzles deliberately reach the source's 30 x 30 size.
        if mode == 2 and index == 0 and seed % 10 == 2:
            chosen = (1, 25, 25)
            if chosen not in used:
                testing.append(chosen)
                continue
        testing.append(_pick_geometry(rng, scales, gaps, used, required))
    return training, testing


def _prefix_length(rng, gap):
    # Every odd length ends on a visible stripe and leaves at least two cells to complete.
    return 3 + 2 * int(rng.integers(0, (gap - 3) // 2))


def _paint_output(grid, a):
    output = grid.copy()
    k, h, v = a["k"], a["hgap"], a["vgap"]
    r0, r1, c0, c1 = a["r0"], a["r1"], a["c0"], a["c1"]
    anchor, horizontal, vertical, accent = a["anchor"], a["horizontal"], a["vertical"], a["accent"]
    background = a["background"]
    for row in (r0, r1):
        for col in (c0, c1):
            output[row:row+k, col:col+k] = anchor
    for offset in range(h):
        color = horizontal if offset % 4 == 0 else accent if offset % 4 == 2 else background
        output[r0:r0+k, c0+k+offset] = color
        output[r1:r1+k, c0+k+offset] = color
    for offset in range(v):
        color = vertical if offset % 4 == 0 else accent if offset % 4 == 2 else background
        output[r0+k+offset, c0:c0+k] = color
        output[r0+k+offset, c1:c1+k] = color
    output[r0-1, c0+k:c1] = horizontal
    output[r1+k, c0+k:c1] = horizontal
    output[r0+k:r1, c0-1] = vertical
    output[r0+k:r1, c1+k] = vertical
    return output


def _make_pair(rng, geometry, corner, force_shape=None):
    k, hgap, vgap = geometry
    forced_h, forced_w = force_shape if force_shape is not None else (None, None)
    r0, height = _axis(rng, k, vgap, forced_h)
    c0, width = _axis(rng, k, hgap, forced_w)
    r1, c1 = r0 + k + vgap, c0 + k + hgap
    background, anchor, horizontal, vertical, accent = map(int, rng.choice(10, 5, replace=False))
    grid = np.full((height, width), background, dtype=np.int8)
    rb, cb = corner
    sr, sc = (r0, r1)[rb], (c0, c1)[cb]
    orow, ocol = (r0, r1)[1-rb], (c0, c1)[1-cb]
    grid[sr:sr+k, sc:sc+k] = anchor
    grid[orow:orow+k, ocol:ocol+k] = anchor
    hd, vd = (1 if cb == 0 else -1), (1 if rb == 0 else -1)
    hc = sc+k if cb == 0 else sc-1
    vr = sr+k if rb == 0 else sr-1
    hp, vp = _prefix_length(rng, hgap), _prefix_length(rng, vgap)
    for offset in range(hp):
        if offset % 4 == 0:
            grid[sr:sr+k, hc+hd*offset] = horizontal
        elif offset % 4 == 2:
            grid[sr:sr+k, hc+hd*offset] = accent
    for offset in range(vp):
        if offset % 4 == 0:
            grid[vr+vd*offset, sc:sc+k] = vertical
        elif offset % 4 == 2:
            grid[vr+vd*offset, sc:sc+k] = accent
    a = dict(background=background, anchor=anchor, horizontal=horizontal, vertical=vertical,
             accent=accent, k=k, hgap=hgap, vgap=vgap, r0=r0, r1=r1, c0=c0, c1=c1,
             corner=corner, hprefix=hp, vprefix=vp)
    return {"input": grid, "output": _paint_output(grid, a)}


def _analyze(grid):
    background = int(grid[0, 0])
    assert np.all(grid[0] == background) and np.all(grid[-1] == background)
    assert np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)
    occupied = np.argwhere(grid != background)
    assert len(occupied) > 0
    r0, c0 = map(int, occupied.min(axis=0))
    rmax, cmax = map(int, occupied.max(axis=0))
    found = []
    for k in range(1, 12):
        r1, c1 = rmax-k+1, cmax-k+1
        vgap, hgap = r1-r0-k, c1-c0-k
        if vgap not in GAPS or hgap not in GAPS:
            continue
        if r0 < 1 or c0 < 1 or r1+k >= grid.shape[0] or c1+k >= grid.shape[1]:
            continue
        for rb, cb in CORNERS:
            sr, sc = (r0, r1)[rb], (c0, c1)[cb]
            orow, ocol = (r0, r1)[1-rb], (c0, c1)[1-cb]
            anchor = int(grid[sr, sc])
            if anchor == background or not np.all(grid[sr:sr+k, sc:sc+k] == anchor) or not np.all(grid[orow:orow+k, ocol:ocol+k] == anchor):
                continue
            hd, vd = (1 if cb == 0 else -1), (1 if rb == 0 else -1)
            hc = sc+k if cb == 0 else sc-1
            vr = sr+k if rb == 0 else sr-1
            horizontal = int(grid[sr, hc])
            vertical = int(grid[vr, sc])
            accent = int(grid[sr, hc+2*hd])
            if int(grid[vr+2*vd, sc]) != accent:
                continue
            if len({background, anchor, horizontal, vertical, accent}) != 5:
                continue
            hseen = [d for d in range(hgap) if np.any(grid[sr:sr+k, hc+d*hd] != background)]
            vseen = [d for d in range(vgap) if np.any(grid[vr+d*vd, sc:sc+k] != background)]
            if not hseen or not vseen:
                continue
            hp, vp = max(hseen)+1, max(vseen)+1
            if not (3 <= hp <= hgap-2 and hp % 2 == 1 and 3 <= vp <= vgap-2 and vp % 2 == 1):
                continue
            expected = np.full(grid.shape, background, dtype=grid.dtype)
            expected[sr:sr+k, sc:sc+k] = anchor
            expected[orow:orow+k, ocol:ocol+k] = anchor
            for d in range(hp):
                if d % 4 == 0:
                    expected[sr:sr+k, hc+d*hd] = horizontal
                elif d % 4 == 2:
                    expected[sr:sr+k, hc+d*hd] = accent
            for d in range(vp):
                if d % 4 == 0:
                    expected[vr+d*vd, sc:sc+k] = vertical
                elif d % 4 == 2:
                    expected[vr+d*vd, sc:sc+k] = accent
            if np.array_equal(expected, grid):
                found.append(dict(background=background, anchor=anchor, horizontal=horizontal,
                                  vertical=vertical, accent=accent, k=k, hgap=hgap, vgap=vgap,
                                  r0=r0, r1=r1, c0=c0, c1=c1, corner=(rb, cb),
                                  hprefix=hp, vprefix=vp))
    assert len(found) == 1, "input must have one visible seeded-rectangle interpretation"
    return found[0]


def _source_copy(pair):
    inp, out = pair["input"], pair["output"]
    digest = hashlib.sha256(inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()).hexdigest()
    return (inp.shape, digest) in SOURCE_DIGESTS


def _key(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs = set(), set()
        analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = pair["input"], pair["output"]
                for grid in (inp, out):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert np.all((grid >= 0) & (grid <= 9))
                assert inp.shape == out.shape and not np.array_equal(inp, out)
                ikey, okey = _key(inp), _key(out)
                assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
                seen_inputs.add(ikey)
                seen_pairs.add((ikey, okey))
                assert not _source_copy(pair)
                a = _analyze(inp)
                assert np.array_equal(out, _paint_output(inp, a))
                assert np.array_equal(out[inp != a["background"]], inp[inp != a["background"]])
                assert np.all(out[a["r0"]-1, a["c0"]+a["k"]:a["c1"]] == a["horizontal"])
                assert np.all(out[a["r0"]+a["k"]:a["r1"], a["c0"]-1] == a["vertical"])
                analyses[split].append(a)
        assert len({a["k"] for a in analyses["train"]}) >= 2
        train_geometry = {(a["k"], a["hgap"], a["vgap"]) for a in analyses["train"]}
        assert all((a["k"], a["hgap"], a["vgap"]) not in train_geometry for a in analyses["test"])
        # All semantic colors and orientation cues are visible in each input;
        # training establishes their shared four-phase completion rule.
        assert all(a["hprefix"] >= 3 and a["vprefix"] >= 3 for split in analyses.values() for a in split)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = seed % 5
    orientation_regime = seed % 4
    corner_order = [CORNERS[int(i)] for i in rng.permutation(4)]
    training, testing = _geometry_schedule(rng, seed, mode, num_train, num_test)
    puzzle = {"train": [], "test": []}
    for split, geometries in (("train", training), ("test", testing)):
        for index, geometry in enumerate(geometries):
            if orientation_regime == 0:
                corner = corner_order[0]
            elif orientation_regime in (1, 2):
                corner = corner_order[index % 2 if split == "train" else 2 + index % 2]
            else:
                global_index = index if split == "train" else num_train + index
                corner = corner_order[global_index % 4]
            force_shape = (30, 30) if mode == 2 and split == "test" and index == 0 and seed % 10 == 2 else None
            pair = _make_pair(rng, geometry, corner, force_shape)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[key], b[key])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for key in ("input", "output")) and all(len(left[s]) == len(right[s]) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(1234, 6, 3)
    assert _puzzles_equal(a, generate_puzzle(1234, 6, 3))
    assert not _puzzles_equal(a, generate_puzzle(1235, 6, 3))
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4)), 2000):
        p = generate_puzzle(seed, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 3000):
        p = generate_puzzle(seed, *counts)
        if counts[0] is not None:
            assert len(p["train"]) == counts[0]
        if counts[1] is not None:
            assert len(p["test"]) == counts[1]
    for seed, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4)), 4000):
        try:
            generate_puzzle(seed, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    counts, shapes, triples, corners, gaps, scales = set(), set(), set(), set(), set(), set()
    edge_rails = 0
    long_prefixes = 0
    mixed, common = 0, 0
    within_scale_recombination = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        all_a = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                analysis = _analyze(inp)
                all_a.append(analysis)
                scales.add(analysis["k"])
                gaps.update((analysis["hgap"], analysis["vgap"]))
                triples.add((analysis["k"], analysis["hgap"], analysis["vgap"]))
                corners.add(analysis["corner"])
                shapes.add(inp.shape)
                if analysis["r0"] == 1 or analysis["c0"] == 1 or analysis["r1"]+analysis["k"] == inp.shape[0]-1 or analysis["c1"]+analysis["k"] == inp.shape[1]-1:
                    edge_rails += 1
                if analysis["hprefix"] > 3 or analysis["vprefix"] > 3:
                    long_prefixes += 1
        orientations = {x["corner"] for x in all_a}
        mixed += len(orientations) > 1
        common += len(orientations) == 1
        train_a = [_analyze(p["input"]) for p in puzzle["train"]]
        within_scale_recombination += any(x["k"] == y["k"] and (x["hgap"], x["vgap"]) != (y["hgap"], y["vgap"])
                                          for i, x in enumerate(train_a) for y in train_a[i+1:])
    assert len({x[0] for x in counts}) >= 4 and len({x[1] for x in counts}) >= 3
    assert len(shapes) >= 35 and len(triples) >= 35
    assert corners == set(CORNERS) and gaps == set(GAPS) and max(scales) >= 5
    assert edge_rails >= 10 and long_prefixes >= 10
    assert mixed >= 15 and common >= 5 and within_scale_recombination >= 5
