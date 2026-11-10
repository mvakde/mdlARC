"""Generate seed-marked carrier panels with corrupted regions."""

import hashlib
import numpy as np

SIDES = ("top", "bottom", "left", "right")
SOURCE_HASHES = {
    "88ddb11abdd1eb72c52b470ba3234347ce47a75b5e2699a50051bbba29773a33",
    "9b3acca413f6aa5f6c402feca637bad1b3cca44cd50ac38c9a74be4322a7399f",
    "45a2d8a4e63a96c9999db1b6427bdb836a08abb659e291d1a5a311b7bf9405b1",
    "7009269b5bf2de8cc392b72df0a6560dc43573b0510fe23c6d51e8c4f30c7870",
    "156b8fd3a281a126114e31ea0b31203bfdbadc3f1db3054fbbe7197009ebde7c",
}

def _require(ok, message):
    if not ok:
        raise AssertionError(message)

def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value

def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test

def _point(side, k, x, height, width):
    if side == "top":
        return k, x
    if side == "bottom":
        return height - 1 - k, x
    if side == "left":
        return x, k
    return x, width - 1 - k

def _rectangle(side, start, span, depth, inset, height, width):
    pts = [_point(side, k, x, height, width)
           for k in (inset, depth - 1) for x in (start, start + span - 1)]
    rows, cols = zip(*pts)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1

def _separated(a, b):
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]

def _candidate(rng, side, height, width, ring=False):
    axis = width if side in ("top", "bottom") else height
    inward = height if side in ("top", "bottom") else width
    span = int(rng.integers(7 if ring else 5, min(10, axis - 2) + 1))
    depth = int(rng.integers(6 if ring else 5, min(9, inward - 2) + 1))
    start = int(rng.integers(1, axis - span))
    inset = int(rng.integers(0, 2))
    return {"side": side, "start": start, "span": span, "depth": depth,
            "inset": inset, "rect": _rectangle(side, start, span, depth, inset, height, width)}

def _place(rng, sides, styles, height, width, old=()):
    for _ in range(40):
        chosen = []
        for side, style in zip(sides, styles):
            panel = None
            for _ in range(40):
                proposal = _candidate(rng, side, height, width, style == "ring")
                if all(_separated(proposal["rect"], p["rect"]) for p in (*old, *chosen)):
                    panel = proposal
                    break
            if panel is None:
                break
            chosen.append(panel)
        if len(chosen) == len(sides):
            return chosen
    # A short deterministic geometric search handles crowded legal layouts.
    def search(index, chosen):
        if index == len(sides):
            return chosen
        side, style = sides[index], styles[index]
        span = 7 if style == "ring" else 5
        depth = 6 if style == "ring" else 5
        axis = width if side in ("top", "bottom") else height
        starts = rng.permutation(np.arange(1, axis - span))
        for start in starts:
            for inset in (0, 1):
                panel = {"side": side, "start": int(start), "span": span,
                         "depth": depth, "inset": inset,
                         "rect": _rectangle(side, int(start), span, depth, inset, height, width)}
                if all(_separated(panel["rect"], p["rect"]) for p in (*old, *chosen)):
                    result = search(index + 1, chosen + [panel])
                    if result is not None:
                        return result
        return None
    result = search(0, [])
    if result is not None:
        return result
    raise ValueError("could not place panels")

def _neighbors(point):
    k, x = point
    return ((k - 1, x), (k + 1, x), (k, x - 1), (k, x + 1))

def _holes(cells):
    if not cells:
        return set()
    lo_k, hi_k = min(k for k, _ in cells), max(k for k, _ in cells)
    lo_x, hi_x = min(x for _, x in cells), max(x for _, x in cells)
    todo = [(k, x) for k in range(lo_k, hi_k + 1) for x in (lo_x, hi_x) if (k, x) not in cells]
    todo += [(k, x) for x in range(lo_x, hi_x + 1) for k in (lo_k, hi_k) if (k, x) not in cells]
    outside = set()
    while todo:
        point = todo.pop()
        if point in cells or point in outside:
            continue
        k, x = point
        if lo_k <= k <= hi_k and lo_x <= x <= hi_x:
            outside.add(point)
            todo.extend(_neighbors(point))
    return {(k, x) for k in range(lo_k, hi_k + 1) for x in range(lo_x, hi_x + 1)
            if (k, x) not in cells and (k, x) not in outside}

def _shape(rng, panel, style):
    span, max_k = panel["span"], panel["depth"] - 2
    entry = int(rng.integers(2, span - 2))
    cells = {(1, entry)}
    if style == "single":
        pass
    elif style == "line":
        length = int(rng.integers(2, min(7, max_k) + 1))
        cells.update((k, entry) for k in range(2, length + 1))
    elif style == "bend":
        turn = int(rng.integers(2, max_k + 1))
        cells.update((k, entry) for k in range(2, turn + 1))
        direction = int(rng.choice((-1, 1)))
        limit = min(3, entry - 1 if direction < 0 else span - entry - 2)
        length = int(rng.integers(1, limit + 1))
        cells.update((turn, entry + direction * dx) for dx in range(1, length + 1))
    elif style == "branch":
        length = int(rng.integers(3, max_k + 1))
        cells.update((k, entry) for k in range(2, length + 1))
        for k in range(2, length + 1):
            if rng.random() < .75:
                direction = int(rng.choice((-1, 1)))
                for dx in range(1, int(rng.integers(1, 4)) + 1):
                    x = entry + direction * dx
                    if 1 <= x <= span - 2:
                        cells.add((k, x))
    elif style == "blob":
        cells.add((2, entry))
        target = int(rng.integers(6, min(16, (max_k - 1) * (span - 2)) + 1))
        for _ in range(target * 16):
            if len(cells) >= target:
                break
            base = tuple(cells)[int(rng.integers(len(cells)))]
            options = [(k, x) for k, x in _neighbors(base)
                       if 2 <= k <= max_k and 1 <= x <= span - 2 and (k, x) not in cells]
            if options:
                cells.add(options[int(rng.integers(len(options)))])
        cells |= _holes(cells)
    else:  # ring
        cells.update((k, x) for k in (2, 3, 4) for x in (entry - 1, entry, entry + 1)
                     if (k, x) != (3, entry))
    return entry, cells, _holes(cells) if style == "ring" else set()

def _paint(inp, out, panel, signal, carrier, rng, style):
    h, w = inp.shape
    side = panel["side"]
    for k in range(panel["inset"], panel["depth"]):
        for x in range(panel["start"], panel["start"] + panel["span"]):
            point = _point(side, k, x, h, w)
            inp[point] = out[point] = carrier
    if style is None:
        return
    entry, visible, holes = _shape(rng, panel, style)
    seed = _point(side, 0, panel["start"] + entry, h, w)
    inp[seed] = out[seed] = signal
    wrong = [c for c in range(10) if c not in (carrier, signal)]
    for k, x in visible:
        point = _point(side, k, panel["start"] + x, h, w)
        inp[point] = int(rng.choice(wrong)) if k == 1 or rng.random() >= .17 else signal
        out[point] = signal
    for k, x in holes:
        point = _point(side, k, panel["start"] + x, h, w)
        inp[point], out[point] = carrier, signal

def _generate_pair(rng, regime, sides, styles, signal):
    count = len(sides)
    lower = 27 if count == 3 else 24 if len(set(sides)) < count else 20
    if regime == "source_train":
        h, w = (int(rng.integers(lower, 28)) for _ in range(2))
    elif rng.random() < .38:
        h, w = (int(rng.integers(max(lower, 27), 31)) for _ in range(2))
    else:
        h, w = (int(rng.integers(max(lower, 21), 28)) for _ in range(2))
    try:
        panels = _place(rng, sides, styles, h, w)
    except ValueError:
        h = w = 27 if regime == "source_train" else 30
        panels = _place(rng, sides, styles, h, w)
    inp = rng.integers(0, 10, size=(h, w), dtype=np.int16)
    out = inp.copy()
    carriers = []
    for panel, style in zip(panels, styles):
        carrier = int(rng.choice(carriers)) if carriers and rng.random() < .28 else int(
            rng.choice([c for c in range(10) if c != signal]))
        carriers.append(carrier)
        _paint(inp, out, panel, signal, carrier, rng, style)
    for _ in range(int(rng.integers(0, 3))):
        for _ in range(50):
            side = SIDES[int(rng.integers(4))]
            panel = _candidate(rng, side, h, w)
            if all(_separated(panel["rect"], p["rect"]) for p in panels):
                carrier = int(rng.choice([c for c in range(10) if c != signal]))
                _paint(inp, out, panel, signal, carrier, rng, None)
                panels.append(panel)
                if panel["inset"] == 0 and rng.random() < .45:
                    point = _point(side, 0, panel["start"] + panel["span"] // 2, h, w)
                    inp[point] = out[point] = signal
                break
    # Random mosaic pixels can accidentally resemble a second marker beside a
    # large carrier field. Remove only the inward false entry, keeping its
    # incidental boundary color and every real seed.
    for _ in range(3):
        false_entries = [(side, x, carrier) for side, x, carrier, _ in _input_candidates(inp)
                         if not (inp != out)[_point(side, 1, x, h, w)]]
        if not false_entries:
            break
        for side, x, carrier in false_entries:
            point = _point(side, 1, x, h, w)
            inp[point] = out[point] = carrier
    return inp, out

def _seeds(inp, changed, signal):
    h, w = inp.shape
    found = []
    for side in SIDES:
        axis = w if side in ("top", "bottom") else h
        for x in range(1, axis - 1):
            if inp[_point(side, 0, x, h, w)] == signal and changed[_point(side, 1, x, h, w)]:
                found.append((side, x))
    return found

def _trace(inp, side, x0, signal):
    h, w = inp.shape
    left = _point(side, 1, x0 - 1, h, w)
    right = _point(side, 1, x0 + 1, h, w)
    carrier = int(inp[left])
    _require(carrier == int(inp[right]) and carrier != signal,
             "entry lacks a carrier-colored surround")
    visible, todo = set(), [(1, x0)]
    inward = h if side in ("top", "bottom") else w
    axis = w if side in ("top", "bottom") else h
    while todo:
        k, x = todo.pop()
        if (k, x) in visible or not (1 <= k < inward and 0 <= x < axis):
            continue
        if int(inp[_point(side, k, x, h, w)]) == carrier:
            continue
        visible.add((k, x))
        _require(len(visible) <= 40, "candidate escapes the carrier panel")
        todo.extend(_neighbors((k, x)))
    _require(visible, "empty target")
    lo_k, hi_k = min(k for k, _ in visible), max(k for k, _ in visible)
    lo_x, hi_x = min(x for _, x in visible), max(x for _, x in visible)
    _require(hi_k + 1 < inward and lo_x > 0 and hi_x + 1 < axis,
             "target lacks a carrier margin")
    for k in range(lo_k, hi_k + 1):
        for x in (lo_x - 1, hi_x + 1):
            _require(int(inp[_point(side, k, x, h, w)]) == carrier,
                     "target lacks a rectangular carrier field")
    for x in range(lo_x, hi_x + 1):
        _require(int(inp[_point(side, hi_k + 1, x, h, w)]) == carrier,
                 "target lacks an inward carrier field")
    holes = _holes(visible)
    shape = visible | holes
    return {"side": side, "carrier": carrier, "signal": signal,
            "seed": _point(side, 0, x0, h, w),
            "shape": {_point(side, k, x, h, w) for k, x in shape},
            "size": len(shape), "holes": len(holes),
            "correct": sum(int(inp[_point(side, k, x, h, w)]) == signal for k, x in visible),
            "thin": len({x for _, x in visible}) == 1 or len({k for k, _ in visible}) == 1}

def _input_candidates(inp):
    """Detect plausible edge-to-panel regions using the input alone."""
    h, w = inp.shape
    found = []
    for side in SIDES:
        axis = w if side in ("top", "bottom") else h
        for x in range(1, axis - 1):
            signal = int(inp[_point(side, 0, x, h, w)])
            carrier = int(inp[_point(side, 1, x - 1, h, w)])
            if (carrier == signal or carrier != int(inp[_point(side, 1, x + 1, h, w)])
                    or carrier == int(inp[_point(side, 1, x, h, w)])):
                continue
            try:
                target = _trace(inp, side, x, signal)
            except AssertionError:
                continue
            found.append((side, x, carrier, target))
    return found

def _analyze_pair(inp, out):
    _require(inp.shape == out.shape, "dimensions changed")
    changed = inp != out
    _require(bool(changed.any()), "pair has no repair")
    colors = np.unique(out[changed])
    _require(len(colors) == 1, "targets in a pair use different signals")
    signal = int(colors[0])
    targets = [_trace(inp, side, x, signal) for side, x in _seeds(inp, changed, signal)]
    _require(1 <= len(targets) <= 3, "pair needs one to three marked panels")
    plausible = _input_candidates(inp)
    _require(len(plausible) == len(targets) and
             all(changed[_point(side, 1, x, *inp.shape)] and
                 int(inp[_point(side, 0, x, *inp.shape)]) == signal
                 for side, x, _, _ in plausible),
             "input offers another plausible edge-to-panel interpretation")
    predicted, occupied = inp.copy(), set()
    for target in targets:
        _require(not (occupied & target["shape"]), "targets overlap")
        occupied |= target["shape"]
        _require(any(changed[p] for p in target["shape"]), "marked panel has no corruption")
        for point in target["shape"]:
            predicted[point] = signal
    _require(np.array_equal(predicted, out), "output violates seed-connected recoloring")
    return signal, targets

def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # Shared convention: a boundary marker supplies the color of its panel's
    # connected payload. Literal signal and carrier colors are pair properties.
    signals = [int(c) for c in rng.permutation(10)]
    sides = list(SIDES)
    rng.shuffle(sides)
    n = 1 if rng.random() < .4 else 2
    train_sides = [sides[:n], sides[n:]]
    for _ in range(2, train_count):
        train_sides.append([SIDES[int(rng.integers(4))]
                            for _ in range(int(rng.integers(1, 4)))])
    rng.shuffle(train_sides)
    test_sides = [[SIDES[int(rng.integers(4))]
                   for _ in range(int(rng.integers(1, 4)))] for _ in range(test_count)]
    styles = ("single", "line", "bend", "branch", "blob", "ring")
    train_styles = [[styles[int(rng.integers(len(styles)))] for _ in row]
                    for row in train_sides]
    slots = [(i, j) for i, row in enumerate(train_styles) for j in range(len(row))]
    rng.shuffle(slots)
    train_styles[slots[0][0]][slots[0][1]] = "ring"
    train_styles[slots[1][0]][slots[1][1]] = "single" if rng.random() < .5 else "line"
    test_styles = [[styles[int(rng.integers(len(styles)))] for _ in row]
                   for row in test_sides]
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    for split, side_rows, style_rows, offset in (
        ("train", train_sides, train_styles, 0),
        ("test", test_sides, test_styles, train_count),
    ):
        for i, (side_row, style_row) in enumerate(zip(side_rows, style_rows)):
            child = np.random.default_rng(int(child_seeds[offset + i]))
            signal = signals[i % 10] if split == "train" else signals[(train_count + i) % 10]
            inp, out = _generate_pair(child, "source_train" if split == "train" else "target_test",
                                      side_row, style_row, signal)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle

def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= size <= 30 for size in grid.shape), "grid size out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid color out of range")
            inp, out = pair["input"], pair["output"]
            key_in = (inp.shape, inp.tobytes())
            key_pair = (key_in, out.shape, out.tobytes())
            _require(key_in not in seen_inputs and key_pair not in seen_pairs,
                     "duplicate generated pair")
            seen_inputs.add(key_in)
            seen_pairs.add(key_pair)
            a, b = np.asarray(inp, dtype=np.int8), np.asarray(out, dtype=np.int8)
            digest = hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()
            _require(digest not in SOURCE_HASHES, "source pair was copied")
            analyses[split].append(_analyze_pair(inp, out))
    trained = {t["side"] for _, ts in analyses["train"] for t in ts}
    _require(trained == set(SIDES), "training must show all four entry directions")
    _require(len({s for s, _ in analyses["train"]}) >= 2, "training must vary signal color")
    _require(any(t["holes"] for _, ts in analyses["train"] for t in ts),
             "training must show carrier-colored enclosed cells")
    for _, targets in analyses["test"]:
        _require(all(t["side"] in trained for t in targets), "test direction was not trained")

def _same(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))

def self_test() -> None:
    a = generate_puzzle(314, 6, 3)
    _require(_same(a, generate_puzzle(314, 6, 3)), "nondeterministic generation")
    _require(not _same(a, generate_puzzle(315, 6, 3)), "seeds had no effect")
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1001 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(p)
        _require(train is None or len(p["train"]) == train, "train count ignored")
        _require(test is None or len(p["test"]) == test, "test count ignored")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, dimensions = set(), set(), set()
    target_counts, side_sets, carriers, sizes, holes = set(), set(), set(), set(), set()
    saw_correct = saw_thin = saw_single = saw_same_side = saw_opposite = False
    measurements = {name: 0 for name in (
        "flush", "inset", "incidental", "plain_line", "shared_carrier",
        "novel_test_carrier", "large_test", "comparable_test", "combined",
        "varied_counts_within", "varied_dimensions_within")}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        within_counts, within_dimensions = set(), set()
        trained_carriers = set()
        largest_train = max(pair["input"].size for pair in p["train"])
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                dimensions.add(inp.shape)
                within_dimensions.add(inp.shape)
                signal, ts = _analyze_pair(inp, out)
                within_counts.add(len(ts))
                # Reconstruct the result from the input alone, using the
                # trained edge-marker rule, rather than the difference mask.
                inferred = _input_candidates(inp)
                independent = inp.copy()
                _require(len(inferred) == len(ts), "input-only target count changed")
                for _, _, _, target in inferred:
                    for point in target["shape"]:
                        independent[point] = target["signal"]
                _require(np.array_equal(independent, out), "input-only prediction failed")
                local_carriers = {t["carrier"] for t in ts}
                if split == "test":
                    measurements["novel_test_carrier"] += bool(local_carriers - trained_carriers)
                    measurements["large_test"] += inp.shape[0] >= 28 and inp.shape[1] >= 28
                    measurements["comparable_test"] += inp.size <= largest_train
                else:
                    trained_carriers |= local_carriers
                measurements["shared_carrier"] += len(local_carriers) < len(ts)
                h, w = inp.shape
                border = ({(0, x) for x in range(w)} | {(h - 1, x) for x in range(w)}
                          | {(y, 0) for y in range(h)} | {(y, w - 1) for y in range(h)})
                seeds = {t["seed"] for t in ts}
                incidental = sum(int(inp[point]) == signal and point not in seeds for point in border)
                measurements["incidental"] += incidental > 0
                measurements["combined"] += len(ts) > 1 and incidental > 0 and any(t["holes"] for t in ts)
                _require(not any("0123456789" in "".join(str(int(v)) for v in row) for row in inp),
                         "synthetic noise stamp appeared")
                target_counts.add(len(ts))
                side_sets.add(tuple(sorted(t["side"] for t in ts)))
                carriers.update(t["carrier"] for t in ts)
                sizes.update(t["size"] for t in ts)
                holes.update(t["holes"] for t in ts)
                saw_correct |= any(t["correct"] > 0 for t in ts)
                saw_thin |= any(t["thin"] for t in ts)
                saw_single |= any(t["size"] == 1 for t in ts)
                sides = [t["side"] for t in ts]
                saw_same_side |= len(set(sides)) < len(sides)
                saw_opposite |= {"top", "bottom"} <= set(sides) or {"left", "right"} <= set(sides)
                for t in ts:
                    side = t["side"]
                    x = t["seed"][1] if side in ("top", "bottom") else t["seed"][0]
                    flush = (int(inp[_point(side, 0, x - 1, h, w)]) == t["carrier"]
                             and int(inp[_point(side, 0, x + 1, h, w)]) == t["carrier"])
                    measurements["flush" if flush else "inset"] += 1
                    rows = {point[0] for point in t["shape"]}
                    columns = {point[1] for point in t["shape"]}
                    measurements["plain_line"] += t["size"] > 1 and (len(rows) == 1 or len(columns) == 1)
        measurements["varied_counts_within"] += len(within_counts) > 1
        measurements["varied_dimensions_within"] += len(within_dimensions) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    _require(len(dimensions) >= 30 and target_counts == {1, 2, 3}, "layout diversity too low")
    _require(len(side_sets) >= 12 and len(carriers) == 10, "side or carrier diversity too low")
    _require(len(sizes) >= 12 and 0 in holes and any(v > 0 for v in holes),
             "shape diversity too low")
    _require(all((saw_correct, saw_thin, saw_single, saw_same_side, saw_opposite)),
             "reviewed cases were not exercised")
    _require(measurements["flush"] >= 20 and measurements["inset"] >= 20,
             "boundary placement diversity too low")
    _require(measurements["incidental"] >= 20 and measurements["plain_line"] >= 20,
             "incidental markers or straight lines absent")
    _require(measurements["shared_carrier"] >= 10 and measurements["novel_test_carrier"] >= 10,
             "carrier variation too low")
    _require(measurements["large_test"] >= 5 and measurements["comparable_test"] >= 5,
             "test scale mix too narrow")
    _require(measurements["combined"] >= 10 and
             measurements["varied_counts_within"] >= 30 and
             measurements["varied_dimensions_within"] == 42,
             "within-puzzle or combined-case diversity too low")
