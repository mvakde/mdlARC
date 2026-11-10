"""Generate connected partial rectangles completed by color bounding boxes."""

import numpy as np


_SOURCE_ROWS = (
    ("010000", "010000", "011100", "011133", "000030", "000030"),
    ("000666", "000606", "444600", "444000", "400000", "400000"),
    ("555500", "005500", "005500", "000000", "007000", "007770"),
    ("000000", "000800", "000800", "088800", "000002", "000022"),
    ("6666666666", "6666666666", "6600000066", "6600000066",
     "6600000066", "6600000066", "0000000066", "0000000066",
     "0000000066", "0000000066"),
)
_SOURCE_INPUTS = tuple(np.array([[int(c) for c in row] for row in rows],
                                dtype=np.int8) for rows in _SOURCE_ROWS)


def _resolve_counts(seed, rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _connected(mask):
    cells = set(zip(*np.where(mask)))
    if not cells:
        return False
    pending = [next(iter(cells))]
    reached = set()
    while pending:
        r, c = pending.pop()
        if (r, c) in reached:
            continue
        reached.add((r, c))
        for near in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if near in cells and near not in reached:
                pending.append(near)
    return len(reached) == len(cells)


def _valid_mask(mask):
    h, w = mask.shape
    return (0 < int(np.count_nonzero(~mask))
            and np.any(mask[0]) and np.any(mask[-1])
            and np.any(mask[:, 0]) and np.any(mask[:, -1])
            and _connected(mask))


def _spine_mask(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    row, col = int(rng.integers(h)), int(rng.integers(w))
    mask[row, :] = True
    mask[:, col] = True
    # Each extension touches a spine. Independent lengths make unequal arms.
    for _ in range(int(rng.integers(0, 1 + max(2, (h+w)//2)))):
        if rng.integers(2):
            r = int(rng.integers(h))
            end = int(rng.integers(w))
            mask[r, min(col, end):max(col, end)+1] = True
        else:
            c = int(rng.integers(w))
            end = int(rng.integers(h))
            mask[min(row, end):max(row, end)+1, c] = True
    return mask


def _rim_mask(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    for _ in range(int(rng.integers(0, 1 + max(2, min(h, w))))):
        if rng.integers(2):
            r = int(rng.integers(1, h-1)) if h > 2 else 0
            end = int(rng.integers(1, w))
            if rng.integers(2):
                mask[r, :end] = True
            else:
                mask[r, w-end:] = True
        else:
            c = int(rng.integers(1, w-1)) if w > 2 else 0
            end = int(rng.integers(1, h))
            if rng.integers(2):
                mask[:end, c] = True
            else:
                mask[h-end:, c] = True
    return mask


def _carve_mask(rng, h, w, dense=False):
    area = h*w
    upper = max(2, int(area * (0.25 if dense else 0.72)))
    target = int(rng.integers(1, upper))
    mask = np.ones((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    holes = {start}
    while len(holes) < target:
        edge = set()
        for r, c in holes:
            for near in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= near[0] < h and 0 <= near[1] < w and near not in holes:
                    edge.add(near)
        if not edge:
            break
        choices = sorted(edge)
        holes.add(choices[int(rng.integers(len(choices)))])
    for r, c in holes:
        mask[r, c] = False
    return mask


def _partial_mask(rng, h, w, style):
    for _ in range(50):
        kind = style if style != "mixed" else ("spine", "carve", "rim")[
            int(rng.choice(3, p=(0.45, 0.40, 0.15)))]
        if kind == "spine":
            mask = _spine_mask(rng, h, w)
        elif kind == "rim":
            mask = _rim_mask(rng, h, w)
        else:
            mask = _carve_mask(rng, h, w, dense=style == "dense")
        if _valid_mask(mask):
            return mask
    # A one-cell-wide L always gives a connected, incomplete rectangle.
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _dimensions(rng, count, full):
    low = 6 if full else 8 if count >= 3 else 5
    high = 19 if count >= 3 else 17
    if rng.random() < 0.18 and count <= 2 and not full:
        return 6, 6
    bands = ((low, min(high, 8)), (max(low, 8), min(high, 13)),
             (max(low, 12), high))
    row_band = bands[int(rng.choice(3, p=(0.32, 0.40, 0.28)))]
    col_band = bands[int(rng.choice(3, p=(0.32, 0.40, 0.28)))]
    return (int(rng.integers(row_band[0], row_band[1]+1)),
            int(rng.integers(col_band[0], col_band[1]+1)))


def _regions(rng, h, w, count):
    regions = [(0, 0, h, w)]
    while len(regions) < count:
        possible = [i for i, (r, c, hh, ww) in enumerate(regions)
                    if hh >= 4 or ww >= 4]
        i = possible[int(rng.integers(len(possible)))]
        r, c, hh, ww = regions.pop(i)
        axes = ([0] if hh >= 4 else []) + ([1] if ww >= 4 else [])
        axis = axes[int(rng.integers(len(axes)))]
        if axis == 0:
            cut = int(rng.integers(2, hh-1))
            regions.extend(((r, c, cut, ww), (r+cut, c, hh-cut, ww)))
        else:
            cut = int(rng.integers(2, ww-1))
            regions.extend(((r, c, hh, cut), (r, c+cut, hh, ww-cut)))
    return regions


def _box_in_region(rng, region, prefer_large=False):
    r, c, hh, ww = region
    def extent(length):
        if length == 2:
            return 2
        if prefer_large or rng.random() < 0.55:
            lo = max(2, int(np.ceil(length * float(rng.uniform(0.45, 0.75)))))
            return int(rng.integers(lo, length+1))
        return int(rng.integers(2, length+1))
    h, w = extent(hh), extent(ww)
    return (r + int(rng.integers(hh-h+1)),
            c + int(rng.integers(ww-w+1)), h, w)


def _pair(rng, background, count, style, full=False, reuse=(), dims=None):
    h, w = dims if dims is not None else _dimensions(rng, count, full)
    for _ in range(65):
        boxes = [(0, 0, h, w)] if full else [
            _box_in_region(rng, region, prefer_large=(i == 0))
            for i, region in enumerate(_regions(rng, h, w, count))]
        if style != "mixed" and not any(bh >= 3 and bw >= 3
                                             for _, _, bh, bw in boxes):
            continue
        colors = []
        available = [c for c in range(10) if c != background]
        if reuse and rng.random() < 0.65:
            old = [c for c in reuse if c in available]
            if old:
                chosen = old[int(rng.integers(len(old)))]
                colors.append(chosen)
                available.remove(chosen)
        rng.shuffle(available)
        colors.extend(available[:count-len(colors)])
        inp = np.full((h, w), background, dtype=np.int8)
        for (r, c, bh, bw), color in zip(boxes, colors):
            local_style = style if len(colors) == 1 or rng.random() < 0.65 else "mixed"
            mask = _partial_mask(rng, bh, bw, local_style)
            inp[r:r+bh, c:c+bw][mask] = color
        if np.count_nonzero(inp == background) <= inp.size/2:
            continue
        return {"input": inp, "output": _complete(inp, background)}
    raise RuntimeError("could not construct a recognizable partial rectangle pair")


def _complete(inp, background):
    out = inp.copy()
    for color in np.unique(inp):
        if color != background:
            rows, cols = np.where(inp == color)
            out[rows.min():rows.max()+1, cols.min():cols.max()+1] = color
    return out


def _background(inp):
    colors, counts = np.unique(inp, return_counts=True)
    assert counts.max() > inp.size/2
    return int(colors[np.argmax(counts)])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(int(seed), rng, num_train, num_test)
    # The background is always the strict input majority. A subset of puzzles
    # teaches this role with two different background colors in training.
    backgrounds = list(map(int, rng.choice(10, size=2, replace=False)))
    if rng.random() >= 0.32:
        backgrounds = backgrounds[:1]
    full_slots = (0,) + tuple(range(2, num_train))
    full_train = (full_slots[int(rng.integers(len(full_slots)))]
                  if rng.random() < 0.30 else -1)
    familiar_tests = rng.random() < 0.18
    multi_index = (1-full_train if full_train in (0, 1)
                   else int(rng.integers(2)))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_colors = ()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            if split == "train" and index < 2:
                count = 2 if index == multi_index else int(rng.choice((1, 3)))
                style = "spine" if index == 0 else "dense"
                full = index == full_train and count == 1
                if index == full_train and count != 1:
                    count, full = 1, True
                background = backgrounds[index % len(backgrounds)]
            elif split == "test" and familiar_tests:
                count, style, full = 2, "mixed", False
                background = backgrounds[int(rng.integers(len(backgrounds)))]
            else:
                count = int(rng.choice((1, 2, 3, 4), p=(0.28, 0.37, 0.25, 0.10)))
                style = "mixed"
                full = count == 1 and rng.random() < (0.21 if split == "test" else 0.13)
                background = backgrounds[int(rng.integers(len(backgrounds)))]
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                pair = _pair(pair_rng, background, count, style, full,
                             first_colors if split == "train" and index == 1 else (),
                             (6, 6) if split == "test" and familiar_tests else None)
                key = _key(pair["input"])
                if key in seen or any(np.array_equal(pair["input"], source)
                                      for source in _SOURCE_INPUTS):
                    continue
                if split == "train" and index == 1 and pair["input"].shape == puzzle["train"][0]["input"].shape:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_colors = tuple(int(c) for c in np.unique(pair["input"])
                                         if c != background)
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    ntr, nte = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntr <= 10 and 1 <= nte <= 4 and ntr > nte
    seen, backgrounds, train_multi = set(), set(), False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            assert not any(np.array_equal(inp, source) for source in _SOURCE_INPUTS)
            bg = _background(inp)
            colors = [int(c) for c in np.unique(inp) if c != bg]
            assert 1 <= len(colors) <= 4
            if split == "train":
                backgrounds.add(bg)
                train_multi |= len(colors) >= 2
            else:
                assert bg in backgrounds
            boxes = []
            expected = inp.copy()
            for color in colors:
                rows, cols = np.where(inp == color)
                r0, r1 = int(rows.min()), int(rows.max())
                c0, c1 = int(cols.min()), int(cols.max())
                assert r1 > r0 and c1 > c0
                mask = inp[r0:r1+1, c0:c1+1] == color
                assert _valid_mask(mask)
                boxes.append((r0, c0, r1, c1))
                expected[r0:r1+1, c0:c1+1] = color
            for i, (r0, c0, r1, c1) in enumerate(boxes):
                for rr0, cc0, rr1, cc1 in boxes[i+1:]:
                    assert r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0
            assert np.array_equal(expected, out)
    assert train_multi
    assert puzzle["train"][0]["input"].shape != puzzle["train"][1]["input"].shape
    assert len(np.unique(puzzle["train"][0]["input"])) != len(
        np.unique(puzzle["train"][1]["input"]))
    assert len(backgrounds) <= 2
    if len(backgrounds) == 2:
        assert _background(puzzle["train"][0]["input"]) != _background(
            puzzle["train"][1]["input"])


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, shapes, structures = set(), set(), set(), set()
    object_counts, hole_fractions, backgrounds = set(), set(), set()
    train_single = test_multi = train_full = test_full = False
    changed_background = constant_background = reused_color = False
    rectangular = train_tiny = test_tiny = large_box = adjacent = False
    familiar_tests = test_large_multi = interior_gap = asymmetric = False
    small_hole = large_missing = False
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_two_colors = []
        local_backgrounds = set()
        test_familiar = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                local_backgrounds.add(bg)
                backgrounds.add(bg)
                shapes.add(inp.shape)
                rectangular |= inp.shape[0] != inp.shape[1]
                colors = [int(c) for c in np.unique(inp) if c != bg]
                if split == "train" and len(first_two_colors) < 2:
                    first_two_colors.append(set(colors))
                object_counts.add(len(colors))
                train_single |= split == "train" and len(colors) == 1
                test_multi |= split == "test" and len(colors) >= 2
                test_large_multi |= split == "test" and len(colors) >= 3 and max(inp.shape) >= 9
                if split == "test":
                    test_familiar.append(inp.shape == (6, 6) and len(colors) == 2)
                parts = []
                boxes = []
                for color in colors:
                    rows, cols = np.where(inp == color)
                    r0, r1, c0, c1 = map(int, (rows.min(), rows.max(),
                                                cols.min(), cols.max()))
                    box = inp[r0:r1+1, c0:c1+1]
                    local_mask = box == color
                    fraction = float(np.mean(box == bg))
                    hole_fractions.add(round(fraction, 1))
                    if split == "train":
                        train_tiny |= box.shape == (2, 2)
                    else:
                        test_tiny |= box.shape == (2, 2)
                    large_box |= box.size >= 40
                    small_hole |= int(np.count_nonzero(~local_mask)) <= 2
                    large_missing |= fraction >= 0.5
                    interior_gap |= bool(np.any(~local_mask[1:-1, 1:-1])) if min(box.shape) >= 3 else False
                    asymmetric |= (not np.array_equal(local_mask, np.flip(local_mask, 0))
                                   and not np.array_equal(local_mask, np.flip(local_mask, 1)))
                    boxes.append((r0, c0, r1, c1))
                    parts.append((box.shape, tuple((box == color).flat)))
                    assert np.all(out[r0:r1+1, c0:c1+1] == color)
                structures.add((inp.shape, tuple(sorted(parts))))
                adjacent |= any((r1+1 == rr0 or rr1+1 == r0) and
                                not (c1 < cc0 or cc1 < c0) or
                                (c1+1 == cc0 or cc1+1 == c0) and
                                not (r1 < rr0 or rr1 < r0)
                                for i, (r0, c0, r1, c1) in enumerate(boxes)
                                for rr0, cc0, rr1, cc1 in boxes[i+1:])
                full = len(colors) == 1 and boxes[0] == (0, 0, inp.shape[0]-1, inp.shape[1]-1)
                train_full |= split == "train" and full
                test_full |= split == "test" and full
        changed_background |= len(local_backgrounds) > 1
        constant_background |= len(local_backgrounds) == 1
        reused_color |= bool(first_two_colors[0] & first_two_colors[1])
        familiar_tests |= all(test_familiar)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(structures) >= 100
    assert object_counts == {1, 2, 3, 4}
    assert len(hole_fractions) >= 7 and len(backgrounds) >= 7
    assert all((train_single, test_multi, train_full, test_full,
                changed_background, constant_background, reused_color,
                rectangular, train_tiny, test_tiny, large_box, adjacent,
                familiar_tests, test_large_multi, interior_gap, asymmetric,
                small_hole, large_missing))
    for ntr, nte in ((2, 1), (10, 1), (10, 4), (3, 2)):
        puzzle = generate_puzzle(12345, ntr, nte)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntr, nte)
        assert _same_puzzle(puzzle, generate_puzzle(12345, ntr, nte))
    for ntr, nte in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(54321, ntr, nte))
    for ntr, nte in ((1, None), (11, None), (None, 0), (None, 5),
                     (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(0, ntr, nte)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {ntr}, {nte}")
