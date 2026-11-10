import hashlib

import numpy as np


# SHA-256 of (height, width, row-major input cells) for the five source inputs.
_SOURCE_INPUT_DIGESTS = {
    "8b6dbe9272c7eb58bf1d6a2df462850ed7672ea4101bb39d7adaaed9230dce4f",
    "b88039b4260201008d117da6ae869e7c57624cc96d8f262e459afde1c06a2bc2",
    "fc642592b2f111cb7795ea6115624da68fc3d2fdc1433162ebc27f3bf61f7f1e",
    "8d9d4a43561fe1e21f5be28bb0a1e5578d0c210c193f5fe1acc815ade4bbd7e6",
    "01dcecd93991310099ccf9d6659a8c6ba7cbcba885890c80751ce901681f669d",
}
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check_count(value, low, high, name):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _pair_counts(rng, num_train, num_test):
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _frame_mask(h, w, gapped, phases):
    """The sides are top, bottom, left, right; checker bands are two cells deep."""
    mask = np.zeros((h, w), dtype=bool)
    for side, row in ((0, 0), (1, h - 1)):
        if gapped[side]:
            columns = np.arange(w)
            outer = (columns + phases[side]) % 2 == 0
            mask[row, :] = outer
            mask[row + (1 if side == 0 else -1), :] = ~outer
        else:
            mask[row, :] = True
    for side, col in ((2, 0), (3, w - 1)):
        if gapped[side]:
            rows = np.arange(2, h - 2)
            outer = (rows + phases[side]) % 2 == 0
            mask[2 : h - 2, col] = outer
            mask[2 : h - 2, col + (1 if side == 2 else -1)] = ~outer
        else:
            mask[:, col] = True
    return mask


def _dimensions(rng, kind):
    # Size and aspect are independent of side pattern and palette.
    aspect = int(rng.choice(5, p=[0.15, 0.25, 0.20, 0.20, 0.20]))
    minimum = 7 if kind == "pattern" else 5
    if aspect == 0:  # compact
        h = int(rng.integers(minimum, 12))
        w = int(rng.integers(minimum, 12))
    elif aspect == 1:  # balanced, including the source's balanced solid test
        h = int(rng.integers(minimum, 23))
        w = int(np.clip(h + rng.integers(-3, 4), minimum, 26))
    elif aspect == 2:  # wide
        h = int(rng.integers(minimum, 15))
        w = int(rng.integers(max(15, h + 5), 27))
    elif aspect == 3:  # tall
        w = int(rng.integers(minimum, 11))
        h = int(rng.integers(max(13, w + 5), 27))
    else:  # shallow, potentially a five-row frame
        h = int(rng.integers(minimum, 9))
        w = int(rng.integers(max(13, h + 5), 27))
    return h, w


def _palette(rng, base, first):
    if first:
        return base
    choice = int(rng.integers(0, 10))
    if choice < 2:
        return base
    if choice < 4:
        return base[0], base[2], base[1]
    if choice < 7:
        colors = rng.choice([c for c in range(10) if c != base[0]], size=2, replace=False)
        return base[0], int(colors[0]), int(colors[1])
    colors = rng.choice(10, size=3, replace=False)
    return tuple(int(c) for c in colors)


def _extend(inp, seed, frame_color, seed_color, background):
    out = inp.copy()
    for dr, dc in _DIRECTIONS:
        r, c = seed[0] + dr, seed[1] + dc
        while 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]:
            if inp[r, c] == frame_color:
                break
            assert inp[r, c] == background
            out[r, c] = seed_color
            r, c = r + dr, c + dc
        else:
            raise AssertionError("frame does not bound a ray")
    return out


def _make_pair(rng, kind, gap_policy, force_four, colors):
    fh, fw = _dimensions(rng, kind)
    top = int(rng.integers(0, min(4, 30 - fh) + 1))
    bottom = int(rng.integers(0, min(4, 30 - fh - top) + 1))
    left = int(rng.integers(0, min(4, 30 - fw) + 1))
    right = int(rng.integers(0, min(4, 30 - fw - left) + 1))
    bg, frame_color, seed_color = colors
    inp = np.full((top + fh + bottom, left + fw + right), bg, dtype=np.int8)

    inset = 2 if kind == "pattern" or force_four else 1
    sr = int(rng.integers(inset, fh - inset))
    sc = int(rng.integers(inset, fw - inset))
    if kind == "pattern":
        n_gapped = int(rng.choice((1, 2, 3, 4), p=(0.35, 0.35, 0.20, 0.10)))
        chosen = rng.choice(4, size=n_gapped, replace=False)
        gapped = tuple(i in chosen for i in range(4))
        phases = [int(x) for x in rng.integers(0, 2, size=4)]
        coordinates = (sc, sc, sr, sr)
        if gap_policy == "no_cross":
            for side in chosen:
                phases[side] = (1 - coordinates[side]) % 2
        elif gap_policy == "cross":
            if not any((coordinates[side] + phases[side]) % 2 == 0 for side in chosen):
                side = int(rng.choice(chosen))
                phases[side] = coordinates[side] % 2
    else:
        gapped = (False,) * 4
        phases = [0] * 4

    mask = _frame_mask(fh, fw, gapped, phases)
    inp[top : top + fh, left : left + fw][mask] = frame_color
    seed = (top + sr, left + sc)
    assert inp[seed] == bg
    inp[seed] = seed_color
    out = _extend(inp, seed, frame_color, seed_color, bg)
    return {"input": inp, "output": out}


def _input_digest(inp):
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()).hexdigest()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _pair_counts(rng, num_train, num_test)

    # Only the role-based rule is shared. The original reassigns literal colors
    # between pairs, so the base palette is a preference rather than a code.
    base = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    profile = int(rng.choice(4, p=(0.20, 0.45, 0.15, 0.20)))
    # 0: all solid; 1: informative patterned train and test;
    # 2: patterned train, solid test; 3: solid train, harmless patterned test.
    puzzle = {"train": [], "test": []}
    seen = set()
    train_pattern_signatures = set()
    for regime, count, key in (("source_train", num_train, "train"),
                               ("target_test", num_test, "test")):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            if regime == "source_train":
                if index == 0:
                    kind = "solid"
                elif index == 1 and profile in (1, 2):
                    kind = "pattern"
                else:
                    kind = "pattern" if profile in (1, 2) and pair_rng.random() < 0.4 else "solid"
                gap_policy = "cross" if kind == "pattern" else "random"
            else:
                if index == 0:
                    kind = "pattern" if profile in (1, 3) else "solid"
                else:
                    kind = "pattern" if profile in (1, 3) and pair_rng.random() < 0.45 else "solid"
                gap_policy = "no_cross" if profile == 3 else "cross"
            colors = _palette(pair_rng, base, regime == "source_train" and index == 0)
            for _ in range(100):
                pair = _make_pair(pair_rng, kind, gap_policy,
                                  regime == "source_train" and index == 0, colors)
                inp = pair["input"]
                identity = (inp.shape, inp.tobytes())
                if identity in seen or _input_digest(inp) in _SOURCE_INPUT_DIGESTS:
                    continue
                if kind == "pattern":
                    info = _analyze_pair(pair)
                    signature = (info["gapped"], info["phases"])
                    if regime == "target_test" and profile == 1 and signature in train_pattern_signatures:
                        continue
                    if regime == "source_train":
                        train_pattern_signatures.add(signature)
                seen.add(identity)
                puzzle[key].append(pair)
                break
            else:
                raise RuntimeError("could not draw a unique pair")
    return puzzle


def _decode_frame(frame):
    h, w = frame.shape
    assert h >= 5 and w >= 5
    gapped = []
    phases = []
    for side in range(4):
        if side < 2:
            outer = frame[0 if side == 0 else h - 1, :]
            inner = frame[1 if side == 0 else h - 2, 2 : w - 2]
            sample = outer[2 : w - 2]
            coordinates = np.arange(2, w - 2)
        else:
            outer = frame[2 : h - 2, 0 if side == 2 else w - 1]
            inner = frame[2 : h - 2, 1 if side == 2 else w - 2]
            sample = outer
            coordinates = np.arange(2, h - 2)
        if np.all(outer) and not np.any(inner):
            gapped.append(False)
            phases.append(0)
        else:
            assert h >= 7 and w >= 7
            matches = [phase for phase in (0, 1)
                       if np.array_equal(sample, (coordinates + phase) % 2 == 0)
                       and np.array_equal(inner, ~sample)]
            assert len(matches) == 1
            gapped.append(True)
            phases.append(matches[0])
    expected = _frame_mask(h, w, gapped, phases)
    assert np.array_equal(frame, expected)
    return tuple(gapped), tuple(phases)


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape
    assert _input_digest(inp) not in _SOURCE_INPUT_DIGESTS
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    unique = [int(v) for v, n in zip(values, counts) if n == 1]
    assert len(unique) == 1
    seed_color = unique[0]
    seed = tuple(int(v[0]) for v in np.where(inp == seed_color))
    candidates = []
    for candidate in values:
        frame_color = int(candidate)
        if frame_color == seed_color:
            continue
        background = int(next(v for v in values if v not in (seed_color, frame_color)))
        rows, cols = np.where(inp == frame_color)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        fh, fw = r1 - r0 + 1, c1 - c0 + 1
        try:
            assert r0 < seed[0] < r1 and c0 < seed[1] < c1
            mask = inp == frame_color
            local = mask[r0 : r1 + 1, c0 : c1 + 1]
            assert np.count_nonzero(mask) == np.count_nonzero(local)
            gapped, phases = _decode_frame(local)
            inset = 2 if any(gapped) else 1
            assert r0 + inset <= seed[0] <= r1 - inset
            assert c0 + inset <= seed[1] <= c1 - inset
            expected = inp.copy()
            lengths = []
            crossings = []
            for side, (dr, dc) in enumerate(_DIRECTIONS):
                r, c = seed[0] + dr, seed[1] + dc
                length = 0
                crossed = False
                while 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and inp[r, c] != frame_color:
                    assert inp[r, c] == background
                    expected[r, c] = seed_color
                    if ((side == 0 and r == r0 + 1) or
                        (side == 1 and r == r1 - 1) or
                        (side == 2 and c == c0 + 1) or
                        (side == 3 and c == c1 - 1)) and gapped[side]:
                        crossed = True
                    length += 1
                    r, c = r + dr, c + dc
                assert 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]
                assert inp[r, c] == frame_color
                lengths.append(length)
                crossings.append(crossed)
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            candidates.append({"kind": "pattern" if any(gapped) else "solid",
                               "gapped": gapped, "phases": phases,
                               "crossings": tuple(crossings), "lengths": tuple(lengths),
                               "colors": (background, frame_color, seed_color),
                               "bounds": (r0, r1, c0, c1), "seed": seed,
                               "frame_size": (fh, fw), "shape": inp.shape,
                               "margins": (r0, inp.shape[0] - r1 - 1,
                                           c0, inp.shape[1] - c1 - 1)})
        except AssertionError:
            continue
    assert len(candidates) == 1
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for key in ("train", "test"):
        for pair in puzzle[key]:
            info = _analyze_pair(pair)
            inp, out = pair["input"], pair["output"]
            input_id = (inp.shape, inp.tobytes())
            pair_id = (input_id, out.tobytes())
            assert input_id not in seen_inputs and pair_id not in seen_pairs
            seen_inputs.add(input_id)
            seen_pairs.add(pair_id)
            infos[key].append(info)
    training = infos["train"]
    testing = infos["test"]
    assert training[0]["kind"] == "solid"
    assert any(all(length > 0 for length in i["lengths"]) for i in training)
    train_cross = any(any(i["crossings"]) for i in training)
    test_cross = any(any(i["crossings"]) for i in testing)
    assert not test_cross or train_cross
    if test_cross:
        train_signatures = {(i["gapped"], i["phases"]) for i in training if i["kind"] == "pattern"}
        assert all((i["gapped"], i["phases"]) not in train_signatures
                   for i in testing if any(i["crossings"]))
    # Palette values and side phases are visible locally; no literal color or
    # checker phase is a hidden puzzle-level convention.


def _same_puzzle(a, b):
    return all(len(a[key]) == len(b[key]) and
               all(np.array_equal(x[g], y[g]) for x, y in zip(a[key], b[key])
                   for g in ("input", "output")) for key in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    shapes, layouts, color_triples, pattern_signatures = set(), set(), set(), set()
    aspect_by_regime = {"train": set(), "test": set()}
    profiles, crossing_sets = set(), set()
    shallow = tall_train = balanced_solid_test = far_margin = zero_arm = False
    nonzero_background = varied_background = swapped_roles = combined = False
    upper_quadrant_train = lower_quadrant_train = False
    source_shape_used = False
    gapped_side_counts = set()
    pattern_phases = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple((key, p["input"].shape, p["input"].tobytes())
                               for key in ("train", "test") for p in puzzle[key]))
        analyses = {key: [_analyze_pair(p) for p in puzzle[key]] for key in ("train", "test")}
        profiles.add((any(i["kind"] == "pattern" for i in analyses["train"]),
                      any(i["kind"] == "pattern" for i in analyses["test"]),
                      any(any(i["crossings"]) for i in analyses["test"])))
        varied_background |= len({i["colors"][0] for key in analyses for i in analyses[key]}) > 1
        roles = [i["colors"][1:] for key in analyses for i in analyses[key]]
        swapped_roles |= any(a == b[::-1] for a in roles for b in roles if a != b)
        for key, collection in analyses.items():
            for i in collection:
                h, w = i["frame_size"]
                aspect = "tall" if h >= w + 4 else "wide" if w >= h + 5 else "balanced"
                aspect_by_regime[key].add(aspect)
                shapes.add(i["shape"])
                layouts.add((i["kind"], i["shape"], i["frame_size"], i["bounds"], i["seed"], i["gapped"], i["phases"]))
                color_triples.add(i["colors"])
                pattern_signatures.add((i["gapped"], i["phases"]))
                crossing_sets.add(i["crossings"])
                if key == "train":
                    midpoint = (i["bounds"][0] + i["bounds"][1]) / 2
                    upper_quadrant_train |= i["seed"][0] < midpoint
                    lower_quadrant_train |= i["seed"][0] > midpoint
                shallow |= h <= 6
                tall_train |= key == "train" and aspect == "tall"
                balanced_solid_test |= key == "test" and i["kind"] == "solid" and aspect == "balanced"
                far_margin |= max(i["margins"]) >= 3
                zero_arm |= 0 in i["lengths"]
                nonzero_background |= i["colors"][0] != 0
                source_shape_used |= i["shape"] in {(14, 14), (6, 20), (14, 15), (17, 8), (12, 12)}
                combined |= (i["kind"] == "pattern" and any(i["crossings"]) and
                             i["colors"][0] != 0 and max(i["margins"]) >= 2 and
                             aspect in ("tall", "wide"))
                if i["kind"] == "pattern":
                    gapped_side_counts.add(sum(i["gapped"]))
                    pattern_phases.update(phase for side, phase in zip(i["gapped"], i["phases"]) if side)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(layouts) >= 100
    assert len(color_triples) >= 40 and len(pattern_signatures) >= 12
    assert aspect_by_regime["train"] == aspect_by_regime["test"] == {"tall", "wide", "balanced"}
    assert (False, False, False) in profiles
    assert (True, True, True) in profiles
    assert (True, False, False) in profiles
    assert (False, True, False) in profiles
    assert len(crossing_sets) >= 10
    assert (True, False, False, False) in crossing_sets
    assert any(c[1] and not any(c[j] for j in (0, 2, 3)) for c in crossing_sets)
    assert any((c[2] or c[3]) and not c[0] and not c[1] for c in crossing_sets)
    assert shallow and tall_train and balanced_solid_test and far_margin and zero_arm
    assert nonzero_background and varied_background and swapped_roles and combined
    assert upper_quadrant_train and lower_quadrant_train and source_shape_used
    assert gapped_side_counts == {1, 2, 3, 4} and pattern_phases == {0, 1}

    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1),
                          (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(12345, ntrain, ntest))
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (1, 1), (10, 10)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
