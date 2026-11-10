"""Generate ARC puzzles about four-connected enclosed background regions."""

import numpy as np


_SOURCE_INPUTS = (
    "00000000000 00000000000 04400000000 04440000000 04444000000 04444400000 04444000000 00000000000 00000000000 00000000000",
    "000000000000 000444444440 004400000040 044000000040 040000000040 040000000040 040000000040 040000000440 044000000400 004444440400 000000044440 000000000000",
    "0000044044000 0000044444000 0044444444400 0044444444400 0444444444400 0444444444400 0044444444440 0004444444440 0000004444400 0000000000000",
    "004400 040440 040044 040004 044444 000440",
    "000000000000 000444400000 004400400000 004000400000 000400400000 000400400000 004400400000 004000400000 004000400000 004004400000 004444000000 000000000000",
    "000000044000 000000444400 000044444400 000044444000 000044444000 444444444000 044444444000 004444444000 004444000000 000000000000 000000000000",
    "00444440000000000000 00400044000444400000 00400004004400400000 00400000444400444400 00444440000000000400 00000044444444444400",
)


def _regions(background):
    h, w = background.shape
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if not background[r, c] or seen[r, c]:
                continue
            stack, cells, edge = [(r, c)], [], False
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                edge |= y in (0, h - 1) or x in (0, w - 1)
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and background[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            regions.append((cells, edge))
    return regions


def _fill(inp, bg, fg):
    out = inp.copy()
    for cells, edge in _regions(inp == bg):
        if not edge:
            for r, c in cells:
                out[r, c] = fg
    return out


def _dilate(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _spans(rng, h, w, anchor=False):
    """Variable orthogonal spans produce connected, hole-free stepped regions."""
    if h == 1 or w == 1 or rng.random() < .18:
        return np.ones((h, w), dtype=bool)
    transpose = h > w
    length_axis, depth = (h, w) if transpose else (w, h)
    mask = np.zeros((depth, length_axis), dtype=bool)
    minimum = 2 if depth >= 3 else 1
    size = int(rng.integers(minimum, depth + 1))
    start = 0 if anchor else int(rng.integers(depth - size + 1))
    for k in range(length_axis):
        if k:
            size = max(minimum, min(depth, size + int(rng.choice((-1, 0, 0, 1)))))
            start = max(0, min(depth - size, start + int(rng.choice((-1, 0, 0, 1)))))
        mask[start:start + size, k] = True
    return mask.T if transpose else mask


def _dimensions(rng, category, split):
    if category in ("closed", "open", "solid") and rng.random() < .10:
        return ((6, 6), (6, 20), (20, 6))[int(rng.integers(3))]
    if category in ("mixed", "multi"):
        ranges = ((8, 15, 15, 27), (15, 27, 8, 15), (11, 22, 11, 22))
        weights = None
    else:
        ranges = ((6, 9, 6, 10), (9, 17, 9, 17),
                  (6, 11, 16, 27), (16, 27, 6, 11), (17, 30, 17, 30))
        weights = ((3, 4, 2, 2, 1) if split == "train" else (1, 3, 4, 3, 1))
        weights = np.array(weights) / sum(weights)
    a, b, c, d = ranges[int(rng.choice(len(ranges), p=weights))]
    return int(rng.integers(a, b + 1)), int(rng.integers(c, d + 1))


def _gate(fg, cavity, rng):
    """Cut independently positioned gates; the pocket remains four-connected to outside."""
    ys, xs = np.nonzero(cavity)
    gate_count = 3 if min(fg.shape) >= 12 else 2
    for side in rng.permutation(4)[:int(rng.integers(1, gate_count + 1))]:
        if side == 0:
            options = [(int(y), int(x)) for y, x in zip(ys, xs) if y == ys.min()]
            dy, dx = -1, 0
        elif side == 1:
            options = [(int(y), int(x)) for y, x in zip(ys, xs) if y == ys.max()]
            dy, dx = 1, 0
        elif side == 2:
            options = [(int(y), int(x)) for y, x in zip(ys, xs) if x == xs.min()]
            dy, dx = 0, -1
        else:
            options = [(int(y), int(x)) for y, x in zip(ys, xs) if x == xs.max()]
            dy, dx = 0, 1
        center = int(rng.integers(len(options)))
        width = int(rng.integers(1, min(3 if min(fg.shape) >= 12 else 2,
                                        len(options)) + 1))
        first = min(max(0, center - width // 2), len(options) - width)
        for y, x in options[first:first + width]:
            y, x = y + dy, x + dx
            while 0 <= y < fg.shape[0] and 0 <= x < fg.shape[1] and fg[y, x]:
                fg[y, x] = False
                y, x = y + dy, x + dx


def _tabs(fg, cavity, rng):
    for _ in range(int(rng.integers(0, 5))):
        ys, xs = np.nonzero(fg)
        if not len(ys):
            return
        k = int(rng.integers(len(ys)))
        y, x = int(ys[k]), int(xs[k])
        dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
        for step in range(1, int(rng.integers(2, 5))):
            yy, xx = y + dy * step, x + dx * step
            if not (0 <= yy < fg.shape[0] and 0 <= xx < fg.shape[1]) or cavity[yy, xx]:
                break
            fg[yy, xx] = True


def _shape(rng, category, split, diagonal=False):
    h, w = _dimensions(rng, category, split)
    fg = np.zeros((h, w), dtype=bool)
    if category == "solid":
        rh, rw = int(rng.integers(3, h - 1)), int(rng.integers(3, w - 1))
        top, left = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        fg[top:top + rh, left:left + rw] = _spans(rng, rh, rw)
        _tabs(fg, np.zeros_like(fg), rng)
    else:
        count = 1 if category in ("closed", "open") else int(rng.integers(2, 4))
        horizontal = w >= h
        bounds = np.linspace(0, w if horizontal else h, count + 1, dtype=int)
        open_witnesses = []
        open_indices = (set(range(count)) if category == "open" else
                        {int(rng.integers(count))} if category == "mixed" else set())
        for index in range(count):
            lo, hi = int(bounds[index]), int(bounds[index + 1])
            zone_h, zone_w = (h, hi - lo) if horizontal else (hi - lo, w)
            if zone_h < 5 or zone_w < 5:
                raise ValueError("zone too small")
            cavity = np.zeros((h, w), dtype=bool)
            if diagonal:
                rh = int(rng.integers(2, min(5, h - 2) + 1))
                rw = int(rng.integers(2, min(7, w - 3) + 1))
                top, left = 1, int(rng.integers(2, w - rw - 1))
                local = _spans(rng, rh, rw, anchor=True)
            else:
                if rng.random() < .24:
                    if rng.random() < .5:
                        rh, rw = 1, int(rng.integers(3, zone_w - 1))
                    else:
                        rh, rw = int(rng.integers(3, zone_h - 1)), 1
                else:
                    rh, rw = int(rng.integers(2, zone_h - 1)), int(rng.integers(2, zone_w - 1))
                if count == 1 and max(h, w) / min(h, w) >= 2 and rng.random() < .55:
                    if w >= h:
                        rw = zone_w - 2
                    else:
                        rh = zone_h - 2
                top = int(rng.integers(1 + (0 if horizontal else lo),
                                       (h if horizontal else hi) - rh))
                left = int(rng.integers(1 + (lo if horizontal else 0),
                                        (hi if horizontal else w) - rw))
                local = _spans(rng, rh, rw)
            cavity[top:top + rh, left:left + rw] = local
            thickness = 1 if diagonal else int(rng.choice((1, 1, 1, 2)))
            ring = cavity.copy()
            for _ in range(thickness):
                ring = _dilate(ring)
            obj = ring & ~cavity
            if index in open_indices:
                _gate(obj, cavity, rng)
            _tabs(obj, cavity, rng)
            if index in open_indices:
                witness = tuple(int(v) for v in np.argwhere(cavity)[0])
                open_witnesses.append(witness)
            fg |= obj
        if open_witnesses:
            exterior = np.zeros_like(fg)
            for cells, edge in _regions(~fg):
                if edge:
                    for y, x in cells:
                        exterior[y, x] = True
            if not all(exterior[y, x] for y, x in open_witnesses):
                raise ValueError("gate does not reach exterior")
    if rng.random() < .5:
        fg = np.fliplr(fg)
    if rng.random() < .5:
        fg = np.flipud(fg)
    if rng.random() < .28:
        fg = fg.T
    return fg.copy()


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError("illegal pair count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("train count must exceed test count")
    return int(num_train), int(num_test)


def _features(fg):
    exterior = np.zeros_like(fg)
    enclosed = np.zeros_like(fg)
    holes = 0
    for cells, edge in _regions(~fg):
        target = exterior if edge else enclosed
        if not edge:
            holes += 1
        for r, c in cells:
            target[r, c] = True
    ys, xs = np.nonzero(fg)
    pocket = False
    if len(ys):
        top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
        pocket = bool(np.any(exterior[top + 1:bottom, left + 1:right]))
    diagonal = any(np.any(enclosed[max(0, -dy):fg.shape[0] - max(0, dy),
                                    max(0, -dx):fg.shape[1] - max(0, dx)] &
                          exterior[max(0, dy):fg.shape[0] - max(0, -dy),
                                   max(0, dx):fg.shape[1] - max(0, -dx)])
                   for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
    return holes, pocket, bool(diagonal)


def _source_copy(inp):
    if set(np.unique(inp).tolist()) != {0, 4}:
        return False
    return any(inp.shape == (len(rows), len(rows[0])) and
               np.array_equal(inp, np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8))
               for rows in (block.split() for block in _SOURCE_INPUTS))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    bg, fg = map(int, rng.choice(10, size=2, replace=False))
    categories = ("closed", "open", "solid", "mixed", "multi")
    train_plan = [("closed", True), ("open", False)]
    train_plan += [(str(rng.choice(categories, p=(.30, .16, .12, .25, .17))), False)
                   for _ in range(n_train - 2)]
    rng.shuffle(train_plan)
    test_plan = [(str(rng.choice(categories, p=(.36, .13, .10, .27, .14))), False)
                 for _ in range(n_test)]
    puzzle, used = {"train": [], "test": []}, set()
    for split, plan in (("train", train_plan), ("test", test_plan)):
        for category, diagonal in plan:
            for _ in range(160):
                try:
                    mask = _shape(rng, category, split, diagonal)
                except ValueError:
                    continue
                holes, pocket, contact = _features(mask)
                if category != "solid" and mask.mean() > .66:
                    continue
                if category in ("solid", "open") and holes != 0:
                    continue
                if category == "closed" and holes != 1:
                    continue
                if category == "multi" and holes < 2:
                    continue
                if category == "mixed" and (holes < 1 or not pocket):
                    continue
                if category == "open" and not pocket:
                    continue
                if diagonal and not contact:
                    continue
                inp = np.where(mask, fg, bg).astype(np.int8)
                key = (inp.shape, inp.tobytes())
                if key in used or _source_copy(inp):
                    continue
                out = _fill(inp, bg, fg)
                if bg not in out:
                    continue
                used.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not build a valid unique shape")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((grid >= 0) & (grid <= 9)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    changes = set()
    for pair in train:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        _check_grid(pair["input"])
        _check_grid(pair["output"])
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape
        changes.update((int(a), int(b)) for a, b in zip(inp[inp != out], out[inp != out]))
    assert len(changes) == 1
    bg, fg = next(iter(changes))
    assert bg != fg
    seen_inputs, seen_pairs = set(), set()
    changed = unchanged = diagonal = open_pocket = 0
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape
            assert set(np.unique(inp).tolist()) == {bg, fg}
            assert set(np.unique(out).tolist()) == {bg, fg}
            assert np.array_equal(out, _fill(inp, bg, fg))
            assert not _source_copy(inp)
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            if split == "train":
                holes, pocket, contact = _features(inp == fg)
                changed += holes > 0
                unchanged += holes == 0
                diagonal += contact
                open_pocket += pocket
    assert changed >= 1 and unchanged >= 1 and diagonal >= 1 and open_pocket >= 1


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def _independent_fill(inp, bg, fg):
    """Exterior-first check, separate from the region-first output builder."""
    h, w = inp.shape
    exterior = np.zeros((h, w), dtype=bool)
    pending = [(r, c) for r in range(h) for c in range(w)
               if r in (0, h - 1) or c in (0, w - 1)]
    while pending:
        r, c = pending.pop()
        if exterior[r, c] or inp[r, c] != bg:
            continue
        exterior[r, c] = True
        pending.extend((y, x) for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                       if 0 <= y < h and 0 <= x < w and not exterior[y, x])
    out = inp.copy()
    out[(inp == bg) & ~exterior] = fg
    return out


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert all(_same(p, generate_puzzle(seed)) for seed, p in enumerate(puzzles))
    shapes, structures, palettes, counts = set(), set(), set(), set()
    split_holes = {"train": set(), "test": set()}
    split_aspects = {"train": set(), "test": set()}
    mixed = diagonal = edge_touch = narrow = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        exemplar = next(p for p in puzzle["train"] if np.any(p["input"] != p["output"]))
        changed = exemplar["input"] != exemplar["output"]
        bg, fg = int(exemplar["input"][changed][0]), int(exemplar["output"][changed][0])
        palettes.add((bg, fg))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                mask = inp == fg
                holes, pocket, contact = _features(mask)
                assert np.array_equal(pair["output"], _independent_fill(inp, bg, fg))
                shapes.add(inp.shape)
                structures.add((inp.shape, mask.tobytes()))
                split_holes[split].add(min(holes, 2))
                ratio = max(inp.shape) / min(inp.shape)
                split_aspects[split].add("elongated" if ratio >= 2 else "compact")
                mixed += holes > 0 and pocket
                diagonal += contact
                edge_touch += bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
                narrow += any(len({r for r, _ in cells}) == 1 or len({c for _, c in cells}) == 1
                              for cells, edge in _regions(~mask) if not edge)
    assert len(shapes) >= 40 and len(structures) >= 140 and len(palettes) >= 12
    assert len(counts) >= 10
    assert split_holes["train"] == split_holes["test"] == {0, 1, 2}
    assert all(len(v) == 2 for v in split_aspects.values())
    assert mixed >= 12 and diagonal >= 42 and edge_touch >= 30 and narrow >= 12
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(91, a, b)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(91, a, b))
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError((a, b))
