"""Complete zero erasures in an L-framed cyclic diagonal pattern."""

import hashlib

import numpy as np


# Exact source input/output hashes, including canvas shape.
SOURCE_PAIR_HASHES = frozenset({
    "fdedf856bb43f89da1295214ef77bbbab9f50c539ece0a7688816fc6cd61234b",
    "fa60766d4c2c05859cab6b5cf342ddbf3b0cb93613a509a67c1f10c6c5cdaace",
    "cbbc0589e769507a626de0809aab30ccb93acd0ce4d5058bbe0e14dc31182e1f",
    "2ea67753bedf3b61fa856e26def42d86711572d19e8acf2c9677bf00ca992e81",
})

CUE_CELLS = ((3, 3), (3, 4), (3, 5), (4, 3), (5, 3))


def _field_cells(shape):
    h, w = shape
    return ((r, c) for r in range(3, h) for c in range(3, w)
            if (r, c) not in CUE_CELLS)


def _render(shape, frame, separator, cue, word):
    rows, cols = np.indices(shape)
    grid = np.asarray(word, dtype=np.int8)[(rows + cols - 8) % len(word)].copy()
    grid[:2, :] = frame
    grid[:, :2] = frame
    grid[2, 2:] = separator
    grid[2:, 2] = separator
    for i, (r, c) in enumerate(CUE_CELLS):
        grid[r, c] = cue[(0, 1, 2, 1, 2)[i]]
    return grid


def _source_hash(inp, out):
    a = np.asarray(inp, dtype=np.uint8)
    b = np.asarray(out, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _recover_from_input(inp):
    """Infer the completed grid from visible cells alone."""
    frame_cells = np.concatenate((inp[:2, :].ravel(), inp[2:, :2].ravel()))
    separator_cells = np.concatenate((inp[2, 2:].ravel(), inp[3:, 2].ravel()))
    frame_values = {int(x) for x in frame_cells if x}
    separator_values = {int(x) for x in separator_cells if x}
    assert len(frame_values) == len(separator_values) == 1
    frame, separator = frame_values.pop(), separator_values.pop()
    assert frame != separator
    assert inp[3, 3] != 0
    cue = [int(inp[3, 3])]
    for top, left in (((3, 4), (4, 3)), ((3, 5), (5, 3))):
        values = {int(inp[p]) for p in (top, left) if inp[p] != 0}
        assert len(values) == 1
        cue.append(values.pop())
    visible = [(r, c, int(inp[r, c])) for r, c in _field_cells(inp.shape)
               if inp[r, c] != 0]
    candidates = []
    for period in range(2, 10):
        word = [None] * period
        for r, c, value in visible:
            phase = (r + c - 8) % period
            if word[phase] is None:
                word[phase] = value
            elif word[phase] != value:
                break
        else:
            if all(v is not None for v in word) and len(set(word)) == period:
                output = _render(inp.shape, frame, separator, cue, word)
                if np.array_equal(output[inp != 0], inp[inp != 0]):
                    candidates.append(output)
    assert len(candidates) == 1, "Visible cells do not identify one cycle."
    return candidates[0]


def _output_features(output):
    frame = int(output[0, 0])
    separator = int(output[2, 2])
    cue = tuple(int(output[3, c]) for c in (3, 4, 5))
    matches = []
    for period in range(2, 10):
        word = tuple(int(v) for v in output[4, 4:4 + period])
        if len(word) == period and len(set(word)) == period:
            if np.array_equal(_render(output.shape, frame, separator, cue, word), output):
                matches.append((period, word))
    assert len(matches) == 1
    return frame, separator, cue, matches[0][0], matches[0][1]


def _resolve_counts(rng, num_train, num_test):
    for count in (num_train, num_test):
        if count is not None and (isinstance(count, (bool, np.bool_)) or
                                  not isinstance(count, (int, np.integer))):
            raise ValueError("Pair counts must be integers.")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be 2 through 10.")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be 1 through 4.")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test.")
    return int(num_train), int(num_test)


def _choose_colors(rng, base_roles, base_word, period):
    colors = np.arange(1, 10)
    if int(rng.integers(4)) == 0 and len(base_word) == period:
        word = tuple(base_word)
    else:
        word = tuple(int(x) for x in rng.choice(colors, size=period, replace=False))
    shift = int(rng.integers(period))
    word = word[shift:] + word[:shift]
    if int(rng.integers(3)) == 0:
        frame, separator, cue = base_roles
    else:
        frame = int(rng.choice(colors))
        separator = int(rng.choice(colors[colors != frame]))
        cue = tuple(int(x) for x in rng.choice(colors, size=3, replace=False))
    return frame, separator, cue, word


def _add_field_motif(mask, rng, style):
    h, w = mask.shape
    if style == "block":
        height = int(rng.integers(2, min(8, h - 3) + 1))
        width = int(rng.integers(2, min(8, w - 3) + 1))
        r = int(rng.integers(3, h - height + 1))
        c = int(rng.integers(3, w - width + 1))
        mask[r:r + height, c:c + width] = True
    elif style == "cut":
        if rng.integers(2):
            length = int(rng.integers(4, min(13, w - 3) + 1))
            r = int(rng.integers(3, h))
            c = int(rng.integers(3, w - length + 1))
            mask[r, c:c + length] = True
        else:
            length = int(rng.integers(4, min(13, h - 3) + 1))
            r = int(rng.integers(3, h - length + 1))
            c = int(rng.integers(3, w))
            mask[r:r + length, c] = True
    elif style == "zigzag":
        r = int(rng.integers(4, h - 3))
        c = int(rng.integers(4, w - 3))
        for step in range(int(rng.integers(5, 12))):
            mask[r, c] = True
            if step % 2:
                r = max(3, min(h - 1, r + int(rng.choice((-1, 1)))))
            else:
                c = max(3, min(w - 1, c + int(rng.choice((-1, 1)))))
    else:
        for _ in range(20):
            r = int(rng.integers(3, h))
            c = int(rng.integers(3, w))
            if (r, c) not in CUE_CELLS and not np.any(
                    mask[max(0, r - 1):min(h, r + 2),
                         max(0, c - 1):min(w, c + 2)]):
                mask[r, c] = True
                return


def _make_mask(rng, shape, region, difficulty, complete_diagonal):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    styles = ("block", "cut", "zigzag", "dot")
    _add_field_motif(mask, rng, styles[int(rng.integers(4))])
    for _ in range(int(rng.integers(0, 2 + difficulty))):
        _add_field_motif(mask, rng, styles[int(rng.integers(4))])
    if not np.any(mask[3:, 3:]):
        mask[h - 2, w - 2] = True
    if region in ("top", "combined"):
        width = int(rng.integers(3, min(11, w - 3) + 1))
        c = int(rng.integers(3, w - width + 1))
        r0 = int(rng.choice((0, 0, 1)))
        r1 = int(rng.integers(4, min(8, h) + 1))
        mask[r0:r1, c:c + width] = True
    if region in ("left", "combined"):
        height = int(rng.integers(3, min(11, h - 3) + 1))
        r = int(rng.integers(3, h - height + 1))
        c0 = int(rng.choice((0, 0, 1)))
        c1 = int(rng.integers(4, min(8, w) + 1))
        mask[r:r + height, c0:c1] = True
    if complete_diagonal:
        middle = (h + w - 2) // 2
        diagonal = int(rng.integers(max(10, middle - 4),
                                    min(h + w - 9, middle + 5)))
        for r, c in _field_cells(shape):
            if r + c == diagonal:
                mask[r, c] = True
    mask[3, 3] = False
    for top, left in (((3, 4), (4, 3)), ((3, 5), (5, 3))):
        if mask[top] and mask[left]:
            mask[left] = False
    return mask


def _train_regions(rng, count):
    if count == 2:
        regions = ["top", "left"] if rng.integers(2) else ["combined", "interior"]
    else:
        regions = ["top", "left", "interior"]
        if rng.integers(2):
            regions[int(rng.integers(2))] = "combined"
        regions += [str(rng.choice(("interior", "top", "left", "combined")))
                    for _ in range(count - 3)]
    return [regions[int(i)] for i in rng.permutation(count)]


def _test_regions(rng, count):
    choices = ("combined", "combined", "top", "left", "interior")
    return [choices[int(rng.integers(len(choices)))] for _ in range(count)]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Geometry, orientation, zero, and the row+column rule are shared. Color
    # defaults create some shared-role repairs, but each input reveals its roles.
    base_colors = rng.permutation(np.arange(1, 10))
    base_roles = (int(base_colors[0]), int(base_colors[1]),
                  tuple(int(x) for x in base_colors[2:5]))
    base_period = int(rng.integers(2, 10))
    base_word = tuple(int(x) for x in rng.choice(np.arange(1, 10),
                                                size=base_period, replace=False))
    train_regions = _train_regions(rng, num_train)
    test_regions = _test_regions(rng, num_test)
    train_periods = [int(x) for x in rng.integers(2, 10, size=num_train)]
    if len(set(train_periods)) == 1:
        train_periods[-1] = 2 + (train_periods[-1] - 1) % 8
    test_periods = []
    for _ in range(num_test):
        if rng.integers(2):
            test_periods.append(int(rng.choice(train_periods)))
        else:
            unseen = [p for p in range(2, 10) if p not in train_periods]
            test_periods.append(int(rng.choice(unseen if unseen else range(2, 10))))
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_outputs = set(), set()
    word_bank = {}
    for side, regions, periods in (("train", train_regions, train_periods),
                                   ("test", test_regions, test_periods)):
        for region, period in zip(regions, periods):
            for _ in range(48):
                shape = ((28, 28) if rng.random() < 0.06 else
                         (int(rng.integers(15, 31)), int(rng.integers(15, 31))))
                frame, separator, cue, word = _choose_colors(
                    rng, base_roles, base_word, period)
                if period in word_bank and rng.random() < 0.4:
                    word = word_bank[period]
                output = _render(shape, frame, separator, cue, word)
                complete = bool(rng.random() < (0.22 if side == "train" else 0.38))
                if side == "test" and region != "combined" and period in train_periods:
                    complete = True
                mask = _make_mask(rng, shape, region, int(side == "test"), complete)
                inp = output.copy()
                inp[mask] = 0
                try:
                    valid = np.array_equal(_recover_from_input(inp), output)
                except AssertionError:
                    valid = False
                key = (shape, inp.tobytes())
                out_key = (shape, output.tobytes())
                if (valid and key not in seen_inputs and out_key not in seen_outputs
                        and _source_hash(inp, output) not in SOURCE_PAIR_HASHES):
                    seen_inputs.add(key)
                    seen_outputs.add(out_key)
                    word_bank.setdefault(period, word)
                    puzzle[side].append({"input": inp, "output": output})
                    break
            else:
                raise AssertionError("Could not construct a uniquely solvable pair.")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair):
    inp, out = pair["input"], pair["output"]
    mask = inp == 0
    frame, separator, cue, period, word = _output_features(out)
    field = np.zeros(inp.shape, dtype=bool)
    for r, c in _field_cells(inp.shape):
        field[r, c] = True
    rows, cols = np.indices(inp.shape)
    diagonal_sums = rows + cols
    full_diagonals = sum(bool(np.any(field & (diagonal_sums == s))) and
                         not np.any((~mask) & field & (diagonal_sums == s))
                         for s in range(6, sum(inp.shape) - 1))
    isolated = sum(bool(mask[r, c] and np.count_nonzero(
        mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2]) == 1)
        for r, c in _field_cells(inp.shape))
    return {
        "shape": inp.shape, "period": period, "word": word,
        "frame": frame, "separator": separator, "cue": cue,
        "top": bool(np.any(mask[:2, :])), "left": bool(np.any(mask[:, :2])),
        "both_edges": bool(np.any(mask[:2, :]) and np.any(mask[:, :2])),
        "row0": bool(np.any(mask[0])), "bottom": bool(np.any(mask[-1])),
        "right": bool(np.any(mask[:, -1])),
        "cue_damage": bool(any(mask[p] for p in CUE_CELLS)),
        "full_diagonals": full_diagonals, "isolated": isolated,
        "mask_size": int(np.count_nonzero(mask)),
        "field_damage": int(np.count_nonzero(mask & field)),
        "role_overlap": len(set((frame, separator) + cue) & set(word)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen_inputs, seen_pairs, seen_outputs = set(), set(), set()
    features = {"train": [], "test": []}
    for side in ("train", "test"):
        for pair in puzzle[side]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            assert 15 <= inp.shape[0] <= 30 and 15 <= inp.shape[1] <= 30
            in_key = (inp.shape, inp.astype(np.uint8).tobytes())
            out_key = (out.shape, out.astype(np.uint8).tobytes())
            assert in_key not in seen_inputs
            assert (in_key, out_key) not in seen_pairs
            assert out_key not in seen_outputs
            seen_inputs.add(in_key)
            seen_pairs.add((in_key, out_key))
            seen_outputs.add(out_key)
            assert _source_hash(inp, out) not in SOURCE_PAIR_HASHES
            mask = inp == 0
            assert np.any(mask) and not np.any(out == 0)
            assert np.array_equal(inp[~mask], out[~mask])
            assert np.array_equal(_recover_from_input(inp), out)
            f = _pair_features(pair)
            assert f["frame"] != f["separator"]
            assert len(set(f["cue"])) == 3
            assert f["field_damage"] > 0
            features[side].append(f)
    train_periods = {f["period"] for f in features["train"]}
    assert len(train_periods) >= 2
    assert any(f["top"] for f in features["train"])
    assert any(f["left"] for f in features["train"])
    assert all(f["both_edges"] or f["full_diagonals"] > 0 or
               f["period"] not in train_periods for f in features["test"])


def _puzzle_signature(puzzle):
    return tuple(tuple((p["input"].shape, p["input"].tobytes(),
                        p["output"].tobytes()) for p in puzzle[side])
                 for side in ("train", "test"))


def self_test() -> None:
    trains, tests, signatures = set(), set(), set()
    global_features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        trains.add(len(puzzle["train"]))
        tests.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        per_puzzle = []
        for side in ("train", "test"):
            for pair in puzzle[side]:
                assert np.array_equal(_recover_from_input(pair["input"]), pair["output"])
                f = _pair_features(pair)
                f["side"] = side
                per_puzzle.append(f)
                global_features.append(f)
        assert len({f["shape"] for f in per_puzzle}) >= 2
        assert len({f["period"] for f in per_puzzle if f["side"] == "train"}) >= 2
    assert len(trains) >= 4 and len(tests) >= 3 and len(signatures) >= 40
    assert set(range(2, 10)).issubset({f["period"] for f in global_features})
    assert len({f["shape"] for f in global_features}) >= 80
    assert len({f["word"] for f in global_features}) >= 100
    assert len({f["mask_size"] for f in global_features}) >= 30
    assert any(f["row0"] for f in global_features)
    assert any(f["bottom"] for f in global_features)
    assert any(f["right"] for f in global_features)
    assert any(f["cue_damage"] for f in global_features)
    assert any(f["full_diagonals"] for f in global_features)
    assert any(f["isolated"] for f in global_features)
    assert any(f["both_edges"] and f["side"] == "train" for f in global_features)
    assert any(not f["both_edges"] and f["side"] == "test" for f in global_features)
    assert len({f["role_overlap"] for f in global_features}) >= 4
    for nt, nv in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1000 + int(nt or 0) * 10 + int(nv or 0), nt, nv)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert nv is None or len(p["test"]) == nv
    assert _puzzle_signature(generate_puzzle(314159, 5, 3)) == _puzzle_signature(
        generate_puzzle(314159, 5, 3))
    for nt, nv in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(7, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted.")
    wrong = generate_puzzle(1234, 3, 1)
    altered = {side: [{role: p[role].copy() for role in ("input", "output")}
                      for p in wrong[side]] for side in ("train", "test")}
    r, c = np.argwhere(altered["test"][0]["input"] == 0)[0]
    old = int(altered["test"][0]["output"][r, c])
    altered["test"][0]["output"][r, c] = old % 9 + 1
    try:
        validate_generated_puzzle(altered)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong restoration was accepted.")
