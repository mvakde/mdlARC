"""Generate puzzles about one global, coordinate-preserving color substitution."""

import numpy as np


_SOURCE_ROWS = (
    (("7776", "6676", "7767", "7677", "7676", "6667"),
     ("7772", "2272", "7727", "7277", "7272", "2227")),
    (("6676", "6677", "7767"), ("2272", "2277", "7727")),
    (("776666", "676777", "767767"),
     ("772222", "272777", "727727")),
    (("6776", "6767", "7776", "7676"),
     ("2772", "2727", "7772", "7272")),
)


def _fingerprint(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_grid(rows):
    return np.array([[int(x) for x in row] for row in rows], dtype=np.int8)


_SOURCE_INPUTS = frozenset(_fingerprint(_source_grid(a)) for a, _ in _SOURCE_ROWS)
_SOURCE_PAIRS = frozenset(
    (_fingerprint(_source_grid(a)), _fingerprint(_source_grid(b)))
    for a, b in _SOURCE_ROWS
)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _counts(seed, num_train, num_test):
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


def _palette(rng):
    colors = [int(x) for x in rng.permutation(10)]
    roll = int(rng.integers(100))
    if roll < 60:
        mode = "fixed"
    elif roll < 84:
        mode = "vary_preserved"
    elif roll < 96:
        mode = "vary_mutable"
    else:
        mode = "merged"
    mutable, preserved = colors[:2]
    replacement = preserved if mode == "merged" else colors[2]
    pool = [x for x in colors if x not in {mutable, preserved, replacement}]
    return mode, mutable, preserved, replacement, pool


def _pair_colors(rng, palette, train_count, test_count):
    mode, mutable, preserved, replacement, pool = palette
    total = train_count + test_count
    roles = [(mutable, preserved) for _ in range(total)]
    if mode in ("vary_preserved", "vary_mutable"):
        # Two visibly different demonstrations identify which role is fixed.
        alternatives = [preserved if mode == "vary_preserved" else mutable] + pool
        order = [int(x) for x in rng.permutation(alternatives)]
        assigned = [order[0], order[1]]
        assigned += [int(rng.choice(order)) for _ in range(train_count - 2)]
        novel = [x for x in order if x not in assigned]
        for _ in range(test_count):
            if novel and rng.random() < 0.7:
                assigned.append(int(rng.choice(novel)))
            else:
                assigned.append(int(rng.choice(order)))
        if mode == "vary_preserved":
            roles = [(mutable, x) for x in assigned]
        else:
            roles = [(x, preserved) for x in assigned]
    return roles


def _shape(rng, first=False):
    category = int(rng.integers(100))
    if first and category < 25:
        category = 40
    if category < 12:
        shape = (1, int(rng.integers(2, 21)))
        if rng.random() < 0.5:
            shape = shape[::-1]
    elif category < 32:
        shape = (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
    elif category < 85:
        shape = (int(rng.integers(2, 13)), int(rng.integers(2, 13)))
    else:
        shape = (int(rng.integers(13, 25)), int(rng.integers(3, 21)))
        if rng.random() < 0.5:
            shape = shape[::-1]
    if first and shape[0] * shape[1] < 10:
        shape = (int(rng.integers(3, 8)), int(rng.integers(4, 9)))
    return shape


def _shapes(rng, train_count, test_count):
    train = [_shape(rng, first=True)]
    for _ in range(train_count - 1):
        candidate = _shape(rng)
        for _ in range(12):
            if candidate != train[0]:
                break
            candidate = _shape(rng)
        if candidate == train[0]:
            candidate = (train[0][0], 1 + train[0][1] % 24)
        train.append(candidate)

    relation = int(rng.integers(4))
    if relation == 0:
        first_test = train[int(rng.integers(train_count))]
    elif relation == 1:
        first_test = _shape(rng)
        for _ in range(16):
            if first_test not in train:
                break
            first_test = _shape(rng)
        if first_test in train:
            first_test = (1, 23)
    elif relation == 2:
        maximum = max(h * w for h, w in train)
        choices = [(h, w) for h in range(1, 6) for w in range(2, 9)
                   if 2 <= h * w < maximum]
        first_test = choices[int(rng.integers(len(choices)))] if choices else train[0]
    else:
        maximum = max(h * w for h, w in train)
        choices = [(h, w) for h in range(9, 29) for w in range(9, 29)
                   if h * w > maximum]
        first_test = choices[int(rng.integers(len(choices)))] if choices else (29, 29)
    test = [first_test] + [_shape(rng) for _ in range(test_count - 1)]
    return train + test


_STYLES = ("scatter", "block", "line", "stripes", "checker", "border",
           "diagonal", "islands", "holes", "cluster", "combined")


def _base_mask(rng, shape, style):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if style == "scatter":
        mask = rng.random(shape) < float(rng.beta(1.5, 1.5))
    elif style == "block":
        bh = int(rng.integers(1, h + 1))
        bw = int(rng.integers(1, w + 1))
        y = int(rng.integers(h - bh + 1))
        x = int(rng.integers(w - bw + 1))
        mask[y:y + bh, x:x + bw] = True
    elif style == "line":
        if rng.random() < 0.5:
            y = int(rng.integers(h))
            x1 = int(rng.integers(w))
            x2 = int(rng.integers(x1 + 1, w + 1))
            mask[y, x1:x2] = True
        else:
            x = int(rng.integers(w))
            y1 = int(rng.integers(h))
            y2 = int(rng.integers(y1 + 1, h + 1))
            mask[y1:y2, x] = True
    elif style == "stripes":
        vertical = bool(rng.integers(2))
        period = int(rng.integers(2, 6))
        thick = int(rng.integers(1, period))
        phase = int(rng.integers(period))
        coordinates = np.arange(w if vertical else h)
        selected = (coordinates + phase) % period < thick
        mask = np.broadcast_to(selected[None, :] if vertical else selected[:, None], shape).copy()
    elif style == "checker":
        cell = int(rng.integers(1, 4))
        yy, xx = np.indices(shape)
        mask = ((yy // cell + xx // cell + int(rng.integers(2))) % 2 == 0)
    elif style == "border":
        thick = int(rng.integers(1, max(2, min(h, w) // 3 + 1)))
        mask[:thick] = True
        mask[-thick:] = True
        mask[:, :thick] = True
        mask[:, -thick:] = True
    elif style == "diagonal":
        yy, xx = np.indices(shape)
        shift = int(rng.integers(-h + 1, w))
        slope = 1 if rng.random() < 0.5 else -1
        mask = np.abs(xx - (slope * yy + shift)) <= int(rng.integers(1, 3)) - 1
    elif style == "islands":
        for _ in range(int(rng.integers(2, 7))):
            bh = int(rng.integers(1, min(h, 4) + 1))
            bw = int(rng.integers(1, min(w, 4) + 1))
            y = int(rng.integers(h - bh + 1))
            x = int(rng.integers(w - bw + 1))
            mask[y:y + bh, x:x + bw] = True
    elif style == "holes":
        mask[:] = True
        for _ in range(int(rng.integers(1, 5))):
            bh = int(rng.integers(1, max(2, h // 2 + 1)))
            bw = int(rng.integers(1, max(2, w // 2 + 1)))
            y = int(rng.integers(h - bh + 1))
            x = int(rng.integers(w - bw + 1))
            mask[y:y + bh, x:x + bw] = False
    elif style == "cluster":
        y, x = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(2, max(3, h * w // 2 + 1)))):
            mask[y, x] = True
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            y, x = min(h - 1, max(0, y + dy)), min(w - 1, max(0, x + dx))
    else:
        left = _base_mask(rng, shape, ("block", "line", "stripes", "diagonal")[int(rng.integers(4))])
        right = _base_mask(rng, shape, ("block", "line", "islands", "border")[int(rng.integers(4))])
        mask = left | right if rng.random() < 0.65 else left ^ right
    return mask


def _adjust_count(rng, mask, desired):
    flat = mask.reshape(-1).copy()
    current = int(np.count_nonzero(flat))
    if current > desired:
        positions = rng.permutation(np.flatnonzero(flat))[:current - desired]
        flat[positions] = False
    elif current < desired:
        positions = rng.permutation(np.flatnonzero(~flat))[:desired - current]
        flat[positions] = True
    return flat.reshape(mask.shape)


def _mask(rng, shape, style, first_train):
    mask = _base_mask(rng, shape, style)
    if rng.random() < 0.5:
        mask = ~mask
    n = mask.size
    density = int(rng.integers(100))
    if not first_train and density < 8:
        mask = _adjust_count(rng, mask, int(rng.integers(1, min(4, n))))
    elif not first_train and density < 16:
        other = int(rng.integers(1, min(4, n)))
        mask = _adjust_count(rng, mask, n - other)
    elif density < 28:
        target = int(rng.integers(max(1, n // 7), max(2, n // 3 + 1)))
        if rng.random() < 0.5:
            target = n - target
        mask = _adjust_count(rng, mask, target)
    low = 3 if first_train else 1
    high = n - (2 if first_train else 1)
    mask = _adjust_count(rng, mask, min(high, max(low, int(np.count_nonzero(mask)))))
    return mask


def _new_pair(rng, shape, style, mutable, preserved, replacement, first_train,
              seen, seen_layouts):
    for attempt in range(48):
        current_shape = shape if attempt < 32 else (7 + attempt - 32, 11)
        current_style = style if attempt < 24 else _STYLES[int(rng.integers(len(_STYLES)))]
        mask = _mask(rng, current_shape, current_style, first_train)
        inp = np.where(mask, mutable, preserved).astype(np.int8)
        out = np.where(mask, replacement, preserved).astype(np.int8)
        key = _fingerprint(inp)
        layout = (current_shape, tuple(bool(x) for x in mask.flat))
        if key not in seen and key not in _SOURCE_INPUTS and layout not in seen_layouts:
            seen.add(key)
            seen_layouts.add(layout)
            return {"input": inp, "output": out}
    raise RuntimeError("unable to construct a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = _palette(rng)
    roles = _pair_colors(rng, palette, train_count, test_count)
    shapes = _shapes(rng, train_count, test_count)
    styles = [_STYLES[int(x)] for x in rng.integers(len(_STYLES), size=train_count + test_count)]
    if styles[1] == styles[0]:
        styles[1] = _STYLES[(_STYLES.index(styles[0]) + int(rng.integers(1, len(_STYLES)))) % len(_STYLES)]
    child_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen, seen_layouts = set(), set()
    for index, ((mutable, preserved), shape, style, child_seed) in enumerate(
        zip(roles, shapes, styles, child_seeds)
    ):
        pair = _new_pair(np.random.default_rng(int(child_seed)), shape, style,
                         mutable, preserved, palette[3], index == 0, seen,
                         seen_layouts)
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _learn_rule(train):
    changes, unchanged, replacements = set(), set(), set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape
        colors = set(int(x) for x in np.unique(inp))
        assert len(colors) == 2
        altered = inp != out
        assert np.any(altered) and np.any(~altered)
        before = set(int(x) for x in np.unique(inp[altered]))
        after = set(int(x) for x in np.unique(out[altered]))
        stable = set(int(x) for x in np.unique(inp[~altered]))
        assert len(before) == len(after) == len(stable) == 1
        mutable, = before
        preserved, = stable
        replacement, = after
        assert mutable != preserved and mutable != replacement
        assert np.array_equal(altered, inp == mutable)
        assert np.array_equal(out, np.where(inp == mutable, replacement, preserved))
        changes.add(mutable)
        unchanged.add(preserved)
        replacements.add(replacement)
    assert len(replacements) == 1
    replacement, = replacements
    assert len(changes) == 1 or len(unchanged) == 1
    assert not (len(changes) > 1 and len(unchanged) > 1)
    if len(changes) == 1:
        mutable, = changes
        assert replacement not in changes
        if len(unchanged) > 1:
            assert replacement not in unchanged
            return "vary_preserved", mutable, None, replacement
        preserved, = unchanged
        return "fixed", mutable, preserved, replacement
    preserved, = unchanged
    assert replacement != preserved and replacement not in changes
    return "vary_mutable", None, preserved, replacement


def _predict(rule, inp):
    mode, mutable, preserved, replacement = rule
    colors = set(int(x) for x in np.unique(inp))
    assert len(colors) == 2
    if mode == "vary_mutable":
        assert preserved in colors and replacement not in colors
        mutable, = colors - {preserved}
    elif mode == "vary_preserved":
        assert mutable in colors and replacement not in colors
        preserved, = colors - {mutable}
    else:
        assert colors == {mutable, preserved}
    return np.where(inp == mutable, replacement, preserved).astype(inp.dtype)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= length <= 30 for length in grid.shape)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        assert len(np.unique(inp)) == 2
        key = _fingerprint(inp)
        pair_key = (key, _fingerprint(out))
        assert key not in seen_inputs and pair_key not in seen_pairs
        layout = (inp.shape, tuple(bool(x) for x in (inp != out).flat))
        assert layout not in seen_layouts
        assert key not in _SOURCE_INPUTS and pair_key not in _SOURCE_PAIRS
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        seen_layouts.add(layout)
    assert train[0]["input"].size >= 10
    first_changes = int(np.count_nonzero(train[0]["input"] != train[0]["output"]))
    assert first_changes >= 3
    assert len({pair["input"].shape for pair in train}) >= 2
    rule = _learn_rule(train)
    for pair in train + test:
        assert np.array_equal(_predict(rule, pair["input"]), pair["output"])
    # Training demonstrates a changed and an unchanged role on each example.
    # For changing-role conventions, two distinct training colors establish
    # that the fixed role and replacement, rather than one literal pair, govern tests.
    if rule[0] == "vary_preserved":
        assert len({int(np.unique(p["input"][p["input"] == p["output"]])[0])
                    for p in train}) >= 2
    elif rule[0] == "vary_mutable":
        assert len({int(np.unique(p["input"][p["input"] != p["output"]])[0])
                    for p in train}) >= 2


def _equal_puzzles(left, right):
    for section in ("train", "test"):
        assert len(left[section]) == len(right[section])
        for a, b in zip(left[section], right[section]):
            assert np.array_equal(a["input"], b["input"])
            assert np.array_equal(a["output"], b["output"])


def _component_count(mask):
    remaining = set(zip(*np.where(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x),
                             (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _has_enclosed_region(mask):
    h, w = mask.shape
    remaining = set(zip(*np.where(mask)))
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x),
                             (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= 2 and all(
            0 < y < h - 1 and 0 < x < w - 1 for y, x in component
        ):
            return True
    return False


def _visible_motifs(mask):
    h, w = mask.shape
    flags = set()
    for role in (mask, ~mask):
        ys, xs = np.where(role)
        box = role[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if box.size >= 4 and box.size < role.size and np.all(box):
            flags.add("rectangle")
            if min(box.shape) == 1:
                flags.add("thin_line")
        if h >= 3 and w >= 3:
            yy, xx = np.indices(role.shape)
            if np.array_equal(role, (yy + xx) % 2 == 0) or np.array_equal(role, (yy + xx) % 2 == 1):
                flags.add("checker")
            frame = np.zeros_like(role)
            frame[0] = frame[-1] = True
            frame[:, 0] = frame[:, -1] = True
            if np.array_equal(role, frame):
                flags.add("border")
            if h >= 4 and w >= 4 and np.count_nonzero(role) >= 4:
                for shift in range(-h, w + h):
                    if (np.array_equal(role, xx - yy == shift) or
                            np.array_equal(role, xx + yy == shift)):
                        flags.add("diagonal")
                        break
        if np.all(role == role[:, :1]) and len(np.unique(role[:, 0])) == 2:
            flags.add("horizontal_bands")
        if np.all(role == role[:1, :]) and len(np.unique(role[0])) == 2:
            flags.add("vertical_bands")
        if _has_enclosed_region(role):
            flags.add("enclosed_region")
    if _component_count(mask) >= 3 or _component_count(~mask) >= 3:
        flags.add("multiple_components")
    return flags


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    _equal_puzzles(generate_puzzle(731), generate_puzzle(731))
    _equal_puzzles(generate_puzzle(912, 7, 3), generate_puzzle(912, 7, 3))
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1),
                                          (1003, 10, 4), (1101, None, 1),
                                          (1102, None, 4), (1103, 2, None),
                                          (1104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (True, None), (None, 1.5)):
        try:
            generate_puzzle(1200, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    shapes, layouts, palettes, density_bins = set(), set(), set(), set()
    modes = set()
    puzzle_signatures = set()
    motif_counts = {name: 0 for name in ("rectangle", "thin_line", "checker",
                                        "border", "diagonal", "enclosed_region",
                                        "horizontal_bands", "vertical_bands",
                                        "multiple_components")}
    shared = unseen = smaller = line = large = lone_changed = lone_stable = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_signatures.add(tuple(
            tuple((_fingerprint(pair["input"]), _fingerprint(pair["output"]))
                  for pair in puzzle[section])
            for section in ("train", "test")
        ))
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        modes.add(_learn_rule(puzzle["train"])[0] +
                  ("_merged" if len(np.unique(puzzle["train"][0]["output"])) == 1 else ""))
        triples = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                altered = inp != out
                for motif in _visible_motifs(altered):
                    motif_counts[motif] += 1
                assert np.array_equal(_predict(_learn_rule(puzzle["train"]), inp), out)
                shape = inp.shape
                shapes.add(shape)
                layouts.add((shape, tuple(bool(x) for x in altered.flat)))
                changed = int(np.count_nonzero(altered))
                stable = inp.size - changed
                density_bins.add(min(9, 10 * changed // inp.size))
                lone_changed += changed == 1
                lone_stable += stable == 1
                line += min(shape) == 1
                large += max(shape) >= 13
                triples.add((int(np.unique(inp[altered])[0]),
                             int(np.unique(inp[~altered])[0]),
                             int(np.unique(out[altered])[0])))
                assert _fingerprint(inp) not in _SOURCE_INPUTS
        palettes.update(triples)
        for pair in puzzle["test"]:
            shape = pair["input"].shape
            shared += shape in train_shapes
            unseen += shape not in train_shapes
            smaller += shape[0] * shape[1] < max(h * w for h, w in train_shapes)
    assert len(puzzle_signatures) >= 40
    assert len(shapes) >= 35 and len(layouts) >= 150
    assert len(palettes) >= 25 and len(density_bins) >= 8
    assert line >= 8 and large >= 15
    assert lone_changed >= 3 and lone_stable >= 3
    assert shared >= 5 and unseen >= 15 and smaller >= 10
    assert {"fixed", "vary_preserved", "vary_mutable", "fixed_merged"} <= modes
    assert motif_counts["rectangle"] >= 30
    assert motif_counts["thin_line"] >= 5
    assert motif_counts["checker"] >= 2 and motif_counts["border"] >= 3
    assert motif_counts["diagonal"] >= 2 and motif_counts["enclosed_region"] >= 30
    assert motif_counts["horizontal_bands"] >= 15
    assert motif_counts["vertical_bands"] >= 15
    assert motif_counts["multiple_components"] >= 50
