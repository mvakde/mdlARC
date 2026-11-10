"""Restore one omitted cell in a regular two-colour figure."""

import numpy as np


KINDS = ("filled_rectangle", "thin_frame", "thick_frame", "diamond_v",
         "diamond_h", "triangle_up", "triangle_down", "triangle_left",
         "triangle_right")
TAPERS = KINDS[3:]


def _mask(shape, kind, box, thickness=1):
    height, width = shape
    top, bottom, left, right = box
    if not (0 <= top <= bottom < height and 0 <= left <= right < width):
        return None
    h, w = bottom - top + 1, right - left + 1
    mask = np.zeros(shape, dtype=bool)
    if kind == "filled_rectangle":
        if not (5 <= h <= 18 and 5 <= w <= 20):
            return None
        mask[top:bottom + 1, left:right + 1] = True
    elif kind in ("thin_frame", "thick_frame"):
        if not (6 <= h <= 22 and 6 <= w <= 22):
            return None
        if kind == "thin_frame":
            thickness = 1
        elif not (2 <= thickness <= 4):
            return None
        if h - 2 * thickness < 4 or w - 2 * thickness < 4:
            return None
        mask[top:bottom + 1, left:right + 1] = True
        mask[top + thickness:bottom - thickness + 1,
             left + thickness:right - thickness + 1] = False
    elif kind in TAPERS:
        horizontal = kind in ("diamond_h", "triangle_left", "triangle_right")
        length, breadth = (w, h) if horizontal else (h, w)
        if kind.startswith("diamond"):
            if length % 2 != 1 or not (5 <= length <= 15):
                return None
            cap = breadth - (length - 1)
        else:
            if not (5 <= length <= 10):
                return None
            cap = breadth - 2 * (length - 1)
        if not 1 <= cap <= 3:
            return None
        for i in range(length):
            step = min(i, length - 1 - i) if kind.startswith("diamond") else i
            if kind in ("triangle_down", "triangle_left"):
                step = length - 1 - i
            span = cap + 2 * step
            start = (breadth - span) // 2
            if horizontal:
                mask[top + start:top + start + span, left + i] = True
            else:
                mask[top + i, left + start:left + start + span] = True
    else:
        return None
    return mask


def _completions(inp):
    """Enumerate completions from the visible grid, with either colour in front."""
    colors = np.unique(inp)
    if len(colors) != 2:
        return []
    found = {}
    for foreground in colors:
        observed = inp == foreground
        rows, cols = np.nonzero(observed)
        if len(rows) < 5:
            continue
        box = (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
        background = int(colors[0] if colors[1] == foreground else colors[1])
        for kind in KINDS:
            thicknesses = range(2, 5) if kind == "thick_frame" else (1,)
            for thickness in thicknesses:
                mask = _mask(inp.shape, kind, box, thickness)
                if mask is None or np.any(observed & ~mask):
                    continue
                if np.count_nonzero(mask & ~observed) != 1:
                    continue
                output = np.where(mask, foreground, background).astype(np.int8)
                found[(output.shape, output.tobytes())] = (output, kind, mask)
    return list(found.values())


def _positions(mask, kind, box):
    top, bottom, left, right = box
    groups = {"interior": [], "outer_edge": [], "inner_edge": [], "line_break": []}
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if r in (top, bottom) and c in (left, right):
            continue
        neighbours = [(r + dr, c + dc) for dr, dc in
                      ((-1, 0), (1, 0), (0, -1), (0, 1))]
        same = sum(0 <= a < mask.shape[0] and 0 <= b < mask.shape[1]
                   and mask[a, b] for a, b in neighbours)
        if kind == "thin_frame":
            if top < r < bottom or left < c < right:
                groups["line_break"].append((r, c))
        elif kind == "filled_rectangle":
            if same == 4:
                groups["interior"].append((r, c))
            elif ((r in (top, bottom) and left < c < right) or
                  (c in (left, right) and top < r < bottom)):
                groups["outer_edge"].append((r, c))
        elif kind == "thick_frame":
            if same == 4:
                groups["interior"].append((r, c))
            elif ((r in (top, bottom) and left < c < right) or
                  (c in (left, right) and top < r < bottom)):
                groups["outer_edge"].append((r, c))
            elif (top < r < bottom and left < c < right and same == 3 and
                  any(top < a < bottom and left < b < right and not mask[a, b]
                      for a, b in neighbours)):
                groups["inner_edge"].append((r, c))
        elif top < r < bottom and left < c < right:
            if same == 4:
                groups["interior"].append((r, c))
            elif same >= 2:
                groups["outer_edge"].append((r, c))
    return groups


def _sample_mask(rng, kind):
    special = bool(rng.integers(8) == 0)
    if kind == "filled_rectangle":
        h, w = ((10, 10) if special else
                (int(rng.integers(5, 19)), int(rng.integers(5, 21))))
        thickness = 1
    elif kind == "thin_frame":
        h, w = ((8, 9) if special else
                (int(rng.integers(6, 19)), int(rng.integers(6, 21))))
        thickness = 1
    elif kind == "thick_frame":
        if special:
            thickness, h, w = 2, 10, 15
        else:
            thickness = int(rng.integers(2, 5))
            h = int(rng.integers(2 * thickness + 4, 23))
            w = int(rng.integers(2 * thickness + 4, 23))
    else:
        horizontal = kind in ("diamond_h", "triangle_left", "triangle_right")
        length = ((11 if special else int(rng.choice((5, 7, 9, 11, 13, 15))))
                  if kind.startswith("diamond") else
                  (6 if special else int(rng.integers(5, 11))))
        base = length - 1 if kind.startswith("diamond") else 2 * (length - 1)
        cap = ((2 if kind.startswith("diamond") else 1) if special
               else int(rng.integers(1, 4)))
        breadth = base + cap
        h, w = (breadth, length) if horizontal else (length, breadth)
        thickness = 1
    if special and rng.integers(2) == 0:
        if kind == "filled_rectangle":
            margins = (1, 1, 1, 1)
        elif kind == "thin_frame":
            margins = (2, 2, 2, 1)
        elif kind == "thick_frame" or kind == "diamond_v":
            margins = (0, 0, 0, 0)
        elif kind == "triangle_up":
            margins = (3, 3, 0, 0)
        else:
            margins = tuple(int(rng.integers(0, 5)) for _ in range(4))
    elif kind == "thick_frame" and rng.integers(3) == 0:
        margins = (0, 0, 0, 0)
    elif kind == "thick_frame" and rng.integers(3) == 0:
        margins = tuple(int(rng.integers(1, 5)) for _ in range(4))
    elif kind == "filled_rectangle" and rng.integers(4) == 0:
        margins = tuple(int(rng.integers(0, 2)) for _ in range(4))
    else:
        margins = tuple(int(rng.integers(0, 5)) for _ in range(4))
    if kind == "filled_rectangle" and margins == (0, 0, 0, 0):
        margins = (1, 0, 0, 0)
    top, bottom_margin, left, right_margin = margins
    shape = (h + top + bottom_margin, w + left + right_margin)
    box = (top, top + h - 1, left, left + w - 1)
    return _mask(shape, kind, box, thickness), box


def _make_pair(rng, kind, category, bg, fg):
    for _ in range(90):
        mask, box = _sample_mask(rng, kind)
        if mask is None:
            continue
        positions = _positions(mask, kind, box)[category]
        if not positions:
            continue
        out = np.where(mask, fg, bg).astype(np.int8)
        for index in rng.permutation(len(positions)):
            r, c = positions[int(index)]
            inp = out.copy()
            inp[r, c] = bg
            completions = _completions(inp)
            if len(completions) == 1 and np.array_equal(completions[0][0], out):
                return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a unique completion")


def _select_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError("Invalid pair count")
    options = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or num_train == a)
               and (num_test is None or num_test == b)]
    if not options:
        raise ValueError("Impossible pair counts")
    return options[int(rng.integers(len(options)))]


def _plans(rng, train_count, test_count):
    source_like = bool(rng.integers(0, 4) == 0)
    if source_like:
        train = [("filled_rectangle", "outer_edge"), ("thin_frame", "line_break")]
        shifted = ("diamond_v", "triangle_up", "thick_frame", "diamond_h",
                   "triangle_down", "triangle_left", "triangle_right")
        test = []
        for _ in range(test_count):
            kind = shifted[int(rng.integers(len(shifted)))]
            category = ("interior", "outer_edge")[int(rng.integers(2))]
            test.append((kind, category))
    else:
        solid = ("filled_rectangle",) + TAPERS
        hollow = ("thin_frame", "thick_frame")
        train = [(solid[int(rng.integers(len(solid)))], "outer_edge")]
        hk = hollow[int(rng.integers(len(hollow)))]
        train.append((hk, "inner_edge" if hk == "thick_frame" and rng.integers(2)
                      else "line_break" if hk == "thin_frame" else "outer_edge"))
        test = []
        for _ in range(test_count):
            kind = KINDS[int(rng.integers(len(KINDS)))]
            if kind == "thin_frame":
                category = "line_break"
            elif kind == "thick_frame":
                allowed = (("interior", "outer_edge", "inner_edge") if any(
                    k == "thick_frame" and cat == "inner_edge" for k, cat in train)
                    else ("interior", "outer_edge"))
                category = allowed[int(rng.integers(len(allowed)))]
            else:
                category = ("interior", "outer_edge")[int(rng.integers(2))]
            test.append((kind, category))
        if any(k == "thick_frame" and cat == "inner_edge" for k, cat in train) and rng.integers(2):
            test[int(rng.integers(len(test)))] = ("thick_frame", "inner_edge")
    while len(train) < train_count:
        choices = ("filled_rectangle", "thin_frame") if source_like else KINDS
        kind = choices[int(rng.integers(len(choices)))]
        allowed = (("line_break",) if kind == "thin_frame" else
                   ("interior", "outer_edge", "inner_edge") if kind == "thick_frame"
                   else ("interior", "outer_edge"))
        train.append((kind, allowed[int(rng.integers(len(allowed)))]))
    if not source_like and train_count >= 3 and rng.integers(3) == 0:
        train[2] = ("filled_rectangle", "interior")
        test[0] = ("filled_rectangle", "outer_edge")
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _palettes(rng, count):
    mode = int(rng.integers(3))
    a, b = [int(x) for x in rng.choice(10, size=2, replace=False)]
    pairs = [(a, b)]
    if mode == 0:
        pool = [int(x) for x in rng.choice(
            [x for x in range(10) if x != a], size=4, replace=False)]
        for _ in range(count - 1):
            pairs.append((a, pool[int(rng.integers(len(pool)))]))
    else:
        pairs.append((b, a if mode == 2 else int(rng.choice(
            [x for x in range(10) if x != b]))))
        for _ in range(count - 2):
            if rng.integers(3) == 0:
                pairs.append(pairs[int(rng.integers(len(pairs)))])
            else:
                pairs.append(tuple(int(x) for x in rng.choice(
                    10, size=2, replace=False)))
    return pairs


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _select_counts(rng, num_train, num_test)
    train_plan, test_plan = _plans(rng, train_count, test_count)
    palettes = _palettes(rng, train_count + test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (kind, category) in enumerate(train_plan + test_plan):
        bg, fg = palettes[index]
        for _ in range(16):
            pair = _make_pair(rng, kind, category, bg, fg)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen and not _is_source_pair(pair):
                seen.add(key)
                puzzle["train" if index < train_count else "test"].append(pair)
                break
        else:
            raise RuntimeError("Could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _is_source_pair(pair):
    specs = (
        ((12, 12), "filled_rectangle", (1, 10, 1, 10), 1, (7, 8), 0, 2),
        ((12, 12), "thin_frame", (2, 9, 2, 10), 1, (5, 10), 0, 3),
        ((11, 12), "diamond_v", (0, 10, 0, 11), 1, (3, 8), 0, 7),
        ((12, 11), "triangle_up", (3, 8, 0, 10), 1, (4, 5), 0, 1),
        ((10, 15), "thick_frame", (0, 9, 0, 14), 2, (1, 1), 0, 4),
    )
    for shape, kind, box, thickness, gap, bg, fg in specs:
        if pair["input"].shape != shape:
            continue
        mask = _mask(shape, kind, box, thickness)
        out = np.where(mask, fg, bg)
        if np.array_equal(pair["output"], out):
            inp = out.copy()
            inp[gap] = bg
            if np.array_equal(pair["input"], inp):
                return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    ntr, nte = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntr <= 10 and 1 <= nte <= 4 and ntr > nte
    seen = set()
    train_edge = train_solid_contour = train_hollow = train_thick_inner = False
    test_hollow = test_thick_inner = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            colors = np.unique(out)
            assert len(colors) == 2 and np.array_equal(np.unique(inp), colors)
            diff = np.argwhere(inp != out)
            assert diff.shape == (1, 2)
            r, c = (int(x) for x in diff[0])
            assert int(inp[r, c]) != int(out[r, c])
            completions = _completions(inp)
            assert len(completions) == 1 and np.array_equal(completions[0][0], out)
            _, kind, mask = completions[0]
            assert bool(mask[r, c]) and np.array_equal(mask, out == out[r, c])
            rows, cols = np.nonzero(mask)
            box = (int(rows.min()), int(rows.max()),
                   int(cols.min()), int(cols.max()))
            groups = _positions(mask, kind, box)
            category = next((name for name, positions in groups.items()
                             if (r, c) in positions), None)
            assert category is not None
            assert not _is_source_pair(pair)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            if split == "train":
                train_edge |= category in ("outer_edge", "line_break", "inner_edge")
                train_solid_contour |= (kind == "filled_rectangle" or kind in TAPERS) and category == "outer_edge"
                train_hollow |= kind in ("thin_frame", "thick_frame")
                train_thick_inner |= kind == "thick_frame" and category == "inner_edge"
            else:
                test_hollow |= kind in ("thin_frame", "thick_frame")
                test_thick_inner |= kind == "thick_frame" and category == "inner_edge"
    assert train_edge and train_solid_contour and train_hollow
    assert not test_hollow or train_hollow
    assert not test_thick_inner or train_thick_inner


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[g], y[g]) for x, y in zip(a[s], b[s])
        for g in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = set()
    train_counts, test_counts, shapes, kinds, categories = (
        set(), set(), set(), set(), set())
    backgrounds, layouts = set(), set()
    variable_bg = swapped = mixed_sizes = 0
    measured = {key: 0 for key in (
        "source_like", "harder_revisit", "test_revisit", "rect10", "thin8",
        "rim_full", "rim_inset", "rim_inner_test", "test_four", "test_nonfour",
        "majority_figure", "rotated_edge")}
    split_kinds = {"train": set(), "test": set()}
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        sig = []
        pair_bgs, pair_shapes, palette = [], [], []
        classifications = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                r, c = np.argwhere(inp != out)[0]
                bg, fg = int(inp[r, c]), int(out[r, c])
                pair_bgs.append(bg)
                palette.append((bg, fg))
                backgrounds.add(bg)
                shapes.add(out.shape)
                pair_shapes.append(out.shape)
                completed = _completions(inp)
                assert len(completed) == 1 and np.array_equal(completed[0][0], out)
                kind = completed[0][1]
                kinds.add(kind)
                split_kinds[split].add(kind)
                rows, cols = np.nonzero(completed[0][2])
                box = (int(rows.min()), int(rows.max()),
                       int(cols.min()), int(cols.max()))
                cats = _positions(completed[0][2], kind, box)
                category = next(k for k, pos in cats.items()
                                if (int(r), int(c)) in pos)
                categories.add(category)
                classifications[split].append((kind, category))
                h, w = box[1] - box[0] + 1, box[3] - box[2] + 1
                margin = (box[0], out.shape[0] - box[1] - 1,
                          box[2], out.shape[1] - box[3] - 1)
                measured["rect10"] += kind == "filled_rectangle" and (h, w) == (10, 10)
                measured["thin8"] += kind == "thin_frame" and (h, w) == (8, 9)
                measured["rim_full"] += kind == "thick_frame" and margin == (0, 0, 0, 0)
                measured["rim_inset"] += kind == "thick_frame" and all(x > 0 for x in margin)
                measured["rim_inner_test"] += split == "test" and kind == "thick_frame" and category == "inner_edge"
                measured["majority_figure"] += np.count_nonzero(completed[0][2]) > out.size / 2
                measured["rotated_edge"] += kind in ("triangle_down", "triangle_left", "triangle_right") and category == "outer_edge"
                if split == "test":
                    neighbours = sum(0 <= int(r) + dr < out.shape[0] and
                                     0 <= int(c) + dc < out.shape[1] and
                                     out[int(r) + dr, int(c) + dc] == fg
                                     for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                    measured["test_four"] += neighbours == 4
                    measured["test_nonfour"] += neighbours < 4
                layouts.add((out.shape, (out == fg).tobytes()))
                sig.append((inp.shape, inp.tobytes(), out.tobytes()))
        signatures.add(tuple(sig))
        variable_bg += len(set(pair_bgs)) > 1
        swapped += any(a == d and b == c for i, (a, b) in enumerate(palette)
                       for c, d in palette[i + 1:])
        mixed_sizes += len(set(pair_shapes)) > 1
        train_kinds = {kind for kind, _ in classifications["train"]}
        test_kinds = {kind for kind, _ in classifications["test"]}
        measured["source_like"] += (train_kinds == {"filled_rectangle", "thin_frame"}
                                    and all(kind in TAPERS or kind == "thick_frame"
                                            for kind in test_kinds))
        measured["test_revisit"] += bool(train_kinds & test_kinds)
        measured["harder_revisit"] += (("filled_rectangle", "interior") in classifications["train"]
                                       and ("filled_rectangle", "outer_edge") in classifications["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(backgrounds) >= 7
    assert len(kinds) == len(KINDS)
    assert split_kinds["train"] == set(KINDS) and split_kinds["test"] == set(KINDS)
    assert {"interior", "outer_edge", "inner_edge", "line_break"} <= categories
    assert variable_bg >= 12 and swapped >= 6 and mixed_sizes >= 35
    assert len(layouts) >= 100 and len(signatures) == 42
    assert {(12, 12), (11, 12), (12, 11), (10, 15)} <= shapes
    assert measured["source_like"] >= 4 and measured["harder_revisit"] >= 3
    assert measured["test_revisit"] >= 10
    assert measured["rect10"] >= 2 and measured["thin8"] >= 2
    assert measured["rim_full"] >= 5 and measured["rim_inset"] >= 5
    assert measured["rim_inner_test"] >= 1
    assert measured["test_four"] >= 10 and measured["test_nonfour"] >= 20
    assert measured["majority_figure"] >= 10 and measured["rotated_edge"] >= 10
    for tr, te in ((2, 1), (10, 1), (10, 4), (3, 2)):
        p = generate_puzzle(1000 + tr * 10 + te, tr, te)
        assert (len(p["train"]), len(p["test"])) == (tr, te)
        assert _same_puzzle(p, generate_puzzle(1000 + tr * 10 + te, tr, te))
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (tr or 0) * 10 + (te or 0), tr, te)
        validate_generated_puzzle(p)
        assert tr is None or len(p["train"]) == tr
        assert te is None or len(p["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(123, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
