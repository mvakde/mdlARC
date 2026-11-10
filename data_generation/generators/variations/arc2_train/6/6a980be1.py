"""Solved examples of the framed, guide-bounded projection puzzle."""

import numpy as np


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _make_grids(height, width, orientation, guide_start, thickness, active,
                background, frame, guide, payload):
    """Construct input and ground truth from visible local roles."""
    guide_end = guide_start + thickness + 1
    inp = np.full((height, width), background, dtype=np.int8)
    inp[0, :] = inp[-1, :] = frame
    inp[:, 0] = inp[:, -1] = frame
    out = np.full((height, width), background, dtype=np.int8)
    if orientation == "vertical":
        assert len(active) == height - 2
        inp[1:-1, guide_start] = guide
        inp[1:-1, guide_end] = guide
        out[:, guide_start] = guide
        out[:, guide_end] = guide
        for r, occupied in enumerate(active, 1):
            if occupied:
                inp[r, guide_start + 1:guide_end] = payload
                out[r, guide_start + 1:guide_end] = payload
                out[r, :guide_start] = frame
                out[r, guide_end + 1:] = frame
    elif orientation == "horizontal":
        assert len(active) == width - 2
        inp[guide_start, 1:-1] = guide
        inp[guide_end, 1:-1] = guide
        out[guide_start, :] = guide
        out[guide_end, :] = guide
        for c, occupied in enumerate(active, 1):
            if occupied:
                inp[guide_start + 1:guide_end, c] = payload
                out[guide_start + 1:guide_end, c] = payload
                out[:guide_start, c] = frame
                out[guide_end + 1:, c] = frame
    else:
        raise ValueError("invalid orientation")

    # The second source training pair ends exactly where its next periodic
    # square would start. Its right border keeps the frame color at that phase.
    # This happens only when a regular rhythm reaches the next border cell;
    # long end margins and unequal gaps do not imply a continuation.
    runs = _runs(active)
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    if gaps and len(set(gaps)) == 1:
        left_edge = runs[0][0] == gaps[0]
        right_edge = len(active) - runs[-1][1] == gaps[0]
        if orientation == "vertical":
            if left_edge:
                out[0, :] = frame
                out[0, guide_start] = out[0, guide_end] = guide
                out[0, guide_start + 1:guide_end] = payload
            if right_edge:
                out[-1, :] = frame
                out[-1, guide_start] = out[-1, guide_end] = guide
                out[-1, guide_start + 1:guide_end] = payload
        else:
            if left_edge:
                out[:, 0] = frame
                out[guide_start, 0] = out[guide_end, 0] = guide
                out[guide_start + 1:guide_end, 0] = payload
            if right_edge:
                out[:, -1] = frame
                out[guide_start, -1] = out[guide_end, -1] = guide
                out[guide_start + 1:guide_end, -1] = payload
    return inp, out


def _sample_pattern(rng, k, regime, allow_irregular, force_edge=False,
                    forbid_edge=False):
    # Draw a structural case first. Enumerating all fitting tuples would favor
    # large gaps and suppress uncommon end margins.
    single = not force_edge and regime == "target_test" and rng.random() < 0.17
    if single:
        long_dimension = int(rng.integers(max(8, k + 4), 31))
        available = long_dimension - 2 - k
        lead = int(rng.integers(1, available))
        gaps = []
        blocks = 1
        trail = available - lead
    else:
        irregular = allow_irregular and not force_edge and rng.random() < 0.12
        if irregular:
            # Three squares are enough to exhibit unequal spacings.
            for _ in range(20):
                blocks = int(rng.integers(3, 6))
                gaps = [int(x) for x in rng.integers(1, 8, size=blocks - 1)]
                if len(set(gaps)) > 1 and blocks * k + sum(gaps) <= 28:
                    break
            else:
                irregular = False
        if not irregular:
            gap = int(rng.integers(2 if force_edge else 1,
                                   min(7, 28 - 2 * k) + 1))
            maximum = min(8, (28 // (k + gap) if force_edge
                               else (28 + gap) // (k + gap)))
            blocks = int(rng.integers(2, maximum + 1))
            gaps = [gap] * (blocks - 1)
        used = blocks * k + sum(gaps)
        # Total margins are sampled independently of gap size. They may be
        # asymmetric or longer than the gaps, as in a truncated rhythm.
        if force_edge:
            assert used + gap <= 28
            lead, trail = (gap, 0) if rng.random() < 0.5 else (0, gap)
            margin_total = gap
        else:
            margin_total = int(rng.integers(0, 29 - used))
            lead = int(rng.integers(0, margin_total + 1))
            trail = margin_total - lead
        long_dimension = used + margin_total + 2
        if long_dimension < 8:
            trail += 8 - long_dimension
            long_dimension = 8
        if forbid_edge and not irregular and (lead == gaps[0] or trail == gaps[0]):
            # Keep the sampled rhythm and move only the end margins. A legal
            # no-contact placement always exists within the 30-cell limit.
            choices = [(left, right)
                       for left in range(29 - used)
                       for right in range(29 - used - left)
                       if 6 <= used + left + right <= 28
                       and left != gaps[0] and right != gaps[0]]
            best = min(abs(left - lead) + abs(right - trail)
                       for left, right in choices)
            nearest = [(left, right) for left, right in choices
                       if abs(left - lead) + abs(right - trail) == best]
            lead, trail = nearest[int(rng.integers(0, len(nearest)))]
            long_dimension = used + lead + trail + 2
    active = np.zeros(long_dimension - 2, dtype=bool)
    position = lead
    for index in range(blocks):
        active[position:position + k] = True
        position += k
        if index < blocks - 1:
            position += gaps[index]
    assert position + trail == len(active)
    return active


def _pair_palette(rng, base, mode, index, split, train_frames):
    if mode == "stable":
        background, guide, payload = base[:3]
        available = [c for c in range(10) if c not in (background, guide, payload)]
        if split == "train" and index == 0:
            frame = base[3]
        elif split == "train" and index == 1:
            frame = available[(available.index(base[3]) + 1) % len(available)]
        elif split == "test" and rng.random() < 0.52:
            frame = int(rng.choice(train_frames))
        else:
            frame = int(rng.choice(available))
        return background, frame, guide, payload

    # Variable-role puzzles retain only the geometrically identified rule.
    # The second training example exchanges one role with the frame, so role
    # changes and use of the current input's frame are both visible.
    if split == "train" and index == 0:
        roles = list(base[:4])
    elif split == "train" and index == 1:
        roles = list(base[:4])
        swap = int(rng.integers(0, 3))
        roles[swap], roles[3] = roles[3], roles[swap]
    else:
        roles = [int(x) for x in rng.permutation(10)[:4]]
        if split == "test" and rng.random() < 0.52:
            old_frame = int(rng.choice(train_frames))
            roles = [c for c in roles if c != old_frame]
            while len(roles) < 3:
                candidate = int(rng.integers(0, 10))
                if candidate != old_frame and candidate not in roles:
                    roles.append(candidate)
            roles = roles[:3] + [old_frame]
    background, guide, payload, frame = roles
    return background, frame, guide, payload


def _sample_pair(rng, regime, orientation, k, palette, allow_irregular,
                 force_edge=False, forbid_edge=False):
    active = _sample_pattern(rng, k, regime, allow_irregular,
                             force_edge=force_edge, forbid_edge=forbid_edge)
    long_dimension = len(active) + 2
    minimum_short = max(8, k + 5)
    if long_dimension >= minimum_short and rng.random() < 0.17:
        short_dimension = long_dimension
    else:
        options = [value for value in range(minimum_short, 31)
                   if value != long_dimension]
        short_dimension = int(rng.choice(options))
    latest_start = short_dimension - k - 3
    offset_mode = rng.random()
    if offset_mode < 0.16:
        guide_start = 1
    elif offset_mode < 0.32:
        guide_start = latest_start
    else:
        guide_start = int(rng.integers(1, latest_start + 1))
    height, width = ((long_dimension, short_dimension) if orientation == "vertical"
                     else (short_dimension, long_dimension))
    return _make_grids(height, width, orientation, guide_start, k, active, *palette)


def _array_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _pair_key(pair):
    return _array_key(pair["input"]), _array_key(pair["output"])


def _source_pair_keys():
    # Exact source examples, used solely to reject copies.
    specs = (
        (18, 19, "vertical", 6, 3, 0, 8, 3, 2, (0, 1, 2, 6, 7, 8, 12, 13, 14)),
        (14, 18, "horizontal", 4, 2, 0, 4, 3, 2, (0, 1, 4, 5, 8, 9, 12, 13)),
        (19, 15, "vertical", 4, 1, 0, 6, 3, 2, (1, 4, 7, 10, 13, 16)),
        (19, 19, "horizontal", 6, 4, 0, 1, 3, 2,
         (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)),
    )
    result = set()
    for h, w, orientation, start, k, bg, frame, guide, payload, indices in specs:
        active = np.zeros((h if orientation == "vertical" else w) - 2, dtype=bool)
        active[list(indices)] = True
        inp, out = _make_grids(h, w, orientation, start, k, active,
                               bg, frame, guide, payload)
        result.add((_array_key(inp), _array_key(out)))
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)

    # The transformation has no hidden color map. Stable-role versus
    # variable-role palettes are puzzle-level styles; each pair's local geometry
    # still identifies all four colors. The source-like scale shift is optional.
    base = [int(x) for x in rng.permutation(10)]
    palette_mode = "variable" if rng.random() < 0.62 else "stable"
    scale_shift = rng.random() < 0.28
    edge_lesson = rng.random() < 0.45
    edge_target = int(rng.integers(0, num_test)) if edge_lesson else None
    train_orientations = ["horizontal", "vertical"]
    train_orientations.extend(str(rng.choice(("horizontal", "vertical")))
                              for _ in range(num_train - 2))
    rng.shuffle(train_orientations)
    if scale_shift:
        train_ks = [2, 3] + [int(rng.integers(1, 4)) for _ in range(num_train - 2)]
        test_ks = [int(rng.integers(4, 7)) for _ in range(num_test)]
    else:
        first = int(rng.integers(1, 6))
        second = int(rng.choice([k for k in range(1, 6) if k != first]))
        train_ks = [first, second] + [int(rng.integers(1, 6)) for _ in range(num_train - 2)]
        test_ks = [int(rng.integers(1, 7)) for _ in range(num_test)]
    rng.shuffle(train_ks)
    test_orientations = [str(rng.choice(("horizontal", "vertical")))
                         for _ in range(num_test)]

    puzzle = {"train": [], "test": []}
    seen = set()
    source_keys = _source_pair_keys()
    train_frames = []
    for split, orientations, thicknesses in (
        ("train", train_orientations, train_ks),
        ("test", test_orientations, test_ks),
    ):
        regime = "source_train" if split == "train" else "target_test"
        for index, (orientation, k) in enumerate(zip(orientations, thicknesses)):
            palette = _pair_palette(rng, base, palette_mode, index, split, train_frames)
            if split == "train":
                train_frames.append(palette[1])
            # Each child stream is independent. The bound is generous, while
            # source copies and duplicate pairs are extraordinarily sparse.
            for _ in range(30):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                inp, out = _sample_pair(child_rng, regime, orientation, k,
                                        palette, allow_irregular=(num_train >= 3 and
                                                                  (split == "test" or index >= 2)),
                                        force_edge=(edge_lesson and
                                                    ((split == "train" and index == 0) or
                                                     (split == "test" and index == edge_target))),
                                        forbid_edge=not edge_lesson)
                pair = {"input": inp, "output": out}
                key = _pair_key(pair)
                if key not in seen and key not in source_keys:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _runs(active):
    sequence = np.asarray(active, dtype=bool)
    result = []
    index = 0
    while index < len(sequence):
        if sequence[index]:
            start = index
            while index < len(sequence) and sequence[index]:
                index += 1
            result.append((start, index))
        else:
            index += 1
    return result


def _analyze_input(inp):
    """Recover roles and geometry from cells, without generation metadata."""
    h, w = inp.shape
    frame = int(inp[0, 0])
    assert np.all(inp[0] == frame) and np.all(inp[-1] == frame)
    assert np.all(inp[:, 0] == frame) and np.all(inp[:, -1] == frame)
    colors = set(int(x) for x in np.unique(inp))
    assert len(colors) == 4
    candidates = []
    for orientation in ("vertical", "horizontal"):
        short = w if orientation == "vertical" else h
        long = h if orientation == "vertical" else w
        for guide in colors - {frame}:
            positions = [i for i in range(1, short - 1)
                         if np.all((inp[1:-1, i] if orientation == "vertical"
                                    else inp[i, 1:-1]) == guide)]
            if len(positions) != 2:
                continue
            start, end = positions
            k = end - start - 1
            if not (1 <= k <= 6 and 1 <= start < end <= short - 2):
                continue
            if short < k + 5:
                continue
            for background in colors - {frame, guide}:
                payload = next(iter(colors - {frame, guide, background}))
                # A direct structural template makes both outside regions and
                # all between-guide slices accountable.
                active = []
                valid = True
                for index in range(1, long - 1):
                    cells = (inp[index, start + 1:end] if orientation == "vertical"
                             else inp[start + 1:end, index])
                    if np.all(cells == payload):
                        active.append(True)
                    elif np.all(cells == background):
                        active.append(False)
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                runs = _runs(active)
                if not runs or any(b - a != k for a, b in runs):
                    continue
                gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
                if any(not 1 <= gap <= 7 for gap in gaps):
                    continue
                lead, trail = runs[0][0], len(active) - runs[-1][1]
                if len(runs) == 1 and (lead == 0 or trail == 0):
                    continue
                expected_inp = np.full((h, w), background, dtype=np.int8)
                expected_inp[0, :] = expected_inp[-1, :] = frame
                expected_inp[:, 0] = expected_inp[:, -1] = frame
                if orientation == "vertical":
                    expected_inp[1:-1, start] = guide
                    expected_inp[1:-1, end] = guide
                    for index, occupied in enumerate(active, 1):
                        if occupied:
                            expected_inp[index, start + 1:end] = payload
                else:
                    expected_inp[start, 1:-1] = guide
                    expected_inp[end, 1:-1] = guide
                    for index, occupied in enumerate(active, 1):
                        if occupied:
                            expected_inp[start + 1:end, index] = payload
                if not np.array_equal(inp, expected_inp):
                    continue
                # Independent stencil reconstruction from the observed input.
                expected_out = np.full((h, w), background, dtype=np.int8)
                if orientation == "vertical":
                    expected_out[:, start:end + 1] = inp[:, start:end + 1]
                    expected_out[0, start:end + 1] = background
                    expected_out[-1, start:end + 1] = background
                    expected_out[:, start] = guide
                    expected_out[:, end] = guide
                    for index, occupied in enumerate(active, 1):
                        if occupied:
                            expected_out[index, :start] = frame
                            expected_out[index, end + 1:] = frame
                else:
                    expected_out[start:end + 1, :] = inp[start:end + 1, :]
                    expected_out[start:end + 1, 0] = background
                    expected_out[start:end + 1, -1] = background
                    expected_out[start, :] = guide
                    expected_out[end, :] = guide
                    for index, occupied in enumerate(active, 1):
                        if occupied:
                            expected_out[:start, index] = frame
                            expected_out[end + 1:, index] = frame
                left_edge = bool(gaps and len(set(gaps)) == 1 and lead == gaps[0])
                right_edge = bool(gaps and len(set(gaps)) == 1 and trail == gaps[0])
                if orientation == "vertical":
                    if left_edge:
                        expected_out[0, :] = frame
                        expected_out[0, start] = expected_out[0, end] = guide
                        expected_out[0, start + 1:end] = payload
                    if right_edge:
                        expected_out[-1, :] = frame
                        expected_out[-1, start] = expected_out[-1, end] = guide
                        expected_out[-1, start + 1:end] = payload
                else:
                    if left_edge:
                        expected_out[:, 0] = frame
                        expected_out[start, 0] = expected_out[end, 0] = guide
                        expected_out[start + 1:end, 0] = payload
                    if right_edge:
                        expected_out[:, -1] = frame
                        expected_out[start, -1] = expected_out[end, -1] = guide
                        expected_out[start + 1:end, -1] = payload
                candidates.append({
                    "orientation": orientation, "k": k, "guide_start": start,
                    "guide_end": end, "background": background, "frame": frame,
                    "guide": guide, "payload": payload, "blocks": len(runs),
                    "gaps": tuple(gaps), "lead": lead, "trail": trail,
                    "edge_completion": left_edge or right_edge,
                    "margin_before": start - 1, "margin_after": short - end - 2,
                    "active": tuple(active), "expected_output": expected_out,
                })
    assert len(candidates) == 1, "input has no unique four-role interpretation"
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    source_keys = _source_pair_keys()
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, pair_key = _array_key(inp), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in source_keys
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _analyze_input(inp)
            assert np.array_equal(out, analysis["expected_output"])
            assert set(int(x) for x in np.unique(out)) == set(int(x) for x in np.unique(inp))
            analyses[split].append(analysis)
    train = analyses["train"]
    assert {a["orientation"] for a in train} == {"horizontal", "vertical"}
    assert all(a["blocks"] >= 2 and a["gaps"] for a in train)
    assert len({a["k"] for a in train}) >= 2
    assert len({a["frame"] for a in train}) >= 2
    assert not any(a["edge_completion"] for a in analyses["test"]) or any(
        a["edge_completion"] for a in train)
    # Every training output displays strips on both sides, guide extension,
    # frame removal elsewhere, and unchanged payload/band gaps.
    assert all(any(a["active"]) and not all(a["active"]) for a in train)
    for a in analyses["test"]:
        assert a["blocks"] >= 1


def _puzzles_equal(a, b):
    return (all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test")))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    assert _puzzles_equal(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3))
    for seed, n, m in ((700, 2, 1), (701, 10, 1), (702, 10, 4),
                       (800, None, 1), (801, None, 4),
                       (802, 2, None), (803, 10, None)):
        p = generate_puzzle(seed, n, m)
        validate_generated_puzzle(p)
        assert n is None or len(p["train"]) == n
        assert m is None or len(p["test"]) == m
    for kw in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
               {"num_test": 5}, {"num_train": 2, "num_test": 2},
               {"num_train": 3, "num_test": 4},
               {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(900, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    train_counts, test_counts, puzzle_keys = set(), set(), set()
    layouts, shapes, palettes = set(), set(), set()
    train_k, test_k, gaps, orientations = set(), set(), set(), set()
    first_test_horizontal = first_test_vertical = 0
    square_first = rectangular_first = 0
    familiar_test = thick_test = single_test = 0
    adjacent = long_margin = irregular = color_exchange = frame_reuse = 0
    edge_train_puzzles = edge_test_puzzles = 0
    variable_role_puzzles = stable_role_puzzles = 0
    small = large = landscape = portrait = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        puzzle_keys.add(tuple(tuple(_pair_key(pair) for pair in p[s]) for s in ("train", "test")))
        train = [_analyze_input(pair["input"]) for pair in p["train"]]
        test = [_analyze_input(pair["input"]) for pair in p["test"]]
        edge_train_puzzles += any(a["edge_completion"] for a in train)
        edge_test_puzzles += any(a["edge_completion"] for a in test)
        triples = {(a["background"], a["guide"], a["payload"]) for a in train + test}
        stable_role_puzzles += len(triples) == 1
        variable_role_puzzles += len(triples) > 1
        train_roles = {(a["background"], a["frame"], a["guide"], a["payload"])
                       for a in train}
        test_roles = {(a["background"], a["frame"], a["guide"], a["payload"])
                      for a in test}
        color_exchange += any(len(set(x) & set(y)) > 0 and any(
            x[i] == y[j] for i in range(4) for j in range(4) if i != j)
            for x in train_roles for y in test_roles)
        frame_reuse += any(a["frame"] in {t["frame"] for t in train} for a in test)
        for split, values in (("train", train), ("test", test)):
            for pair, a in zip(p[split], values):
                h, w = pair["input"].shape
                shapes.add((h, w))
                palettes.add((a["background"], a["frame"], a["guide"], a["payload"]))
                orientations.add(a["orientation"])
                layouts.add((a["orientation"], a["k"], h, w, a["guide_start"],
                             a["gaps"], a["lead"], a["trail"], a["blocks"]))
                gaps.update(a["gaps"])
                adjacent += min(a["margin_before"], a["margin_after"]) == 0
                long_margin += a["lead"] > max(a["gaps"], default=0) or a["trail"] > max(a["gaps"], default=0)
                irregular += len(set(a["gaps"])) > 1
                small += min(h, w) <= 12
                large += max(h, w) >= 26
                landscape += w > h
                portrait += h > w
                (train_k if split == "train" else test_k).add(a["k"])
        first = test[0]
        first_test_horizontal += first["orientation"] == "horizontal"
        first_test_vertical += first["orientation"] == "vertical"
        h, w = p["test"][0]["input"].shape
        square_first += h == w
        rectangular_first += h != w
        familiar_test += any(a["k"] in {t["k"] for t in train} for a in test)
        thick_test += any(a["k"] > max(t["k"] for t in train) for a in test)
        single_test += any(a["blocks"] == 1 for a in test)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) >= 40 and len(layouts) >= 100
    assert len(shapes) >= 45 and len(palettes) >= 40
    assert train_k == {1, 2, 3, 4, 5} and test_k == {1, 2, 3, 4, 5, 6}
    assert gaps == set(range(1, 8)) and orientations == {"horizontal", "vertical"}
    assert min(first_test_horizontal, first_test_vertical) >= 8
    assert rectangular_first >= 25 and square_first >= 1
    assert familiar_test >= 8 and thick_test >= 5 and single_test >= 2
    assert min(stable_role_puzzles, variable_role_puzzles) >= 8
    assert color_exchange >= 10 and frame_reuse >= 8
    assert adjacent >= 10 and long_margin >= 10 and irregular >= 1
    assert edge_train_puzzles >= 6 and edge_test_puzzles >= 4
    assert min(small, large, landscape, portrait) >= 10
