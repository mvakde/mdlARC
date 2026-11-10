"""Synthetic puzzles for minimal object cropping and horizontal duplication."""

import numpy as np


_SOURCE_ROWS = (
    (("00000000", "00000000", "00000000", "00000000", "00000000", "00020000", "00222000", "00220000"), ("020020", "222222", "220220")),
    (("00000000", "08800000", "00800000", "08880000", "00000000", "00000000", "00000000", "00000000"), ("880880", "080080", "888888")),
    (("00000000", "00000110", "00001000", "00000100", "00000000", "00000000", "00000000", "00000000"), ("011011", "100100", "010010")),
    (("00000000", "00000000", "00000000", "00000000", "00030000", "03330000", "03000000", "00000000"), ("003003", "333333", "300300")),
)


def _key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


_SOURCE_PAIRS = frozenset(
    (
        _key(np.array([[int(x) for x in row] for row in inp], dtype=int)),
        _key(np.array([[int(x) for x in row] for row in out], dtype=int)),
    )
    for inp, out in _SOURCE_ROWS
)


def _counts(seed, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    reached = {next(iter(points))}
    frontier = list(reached)
    while frontier:
        row, col = frontier.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (row + dr, col + dc)
                if neighbor in points and neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
    return len(reached) == len(points)


def _asymmetric(mask):
    if mask.shape[0] < 3 or mask.shape[1] < 3 or np.all(mask):
        return False
    variants = (np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2))
    if mask.shape[0] == mask.shape[1]:
        variants += (np.rot90(mask), np.rot90(mask, 3), mask.T,
                     np.flipud(np.fliplr(mask.T)))
    return all(not np.array_equal(mask, other) for other in variants)


def _dimensions(rng, anchor=False):
    if anchor:
        options = ((3, 3), (3, 4), (4, 3), (4, 4))
        return options[int(rng.integers(len(options)))]
    mode = int(rng.integers(10))
    if mode < 2:
        return 3, 3
    if mode < 6:
        return int(rng.integers(2, 6)), int(rng.integers(2, 7))
    if mode < 8:
        if rng.integers(2):
            return int(rng.integers(5, 10)), int(rng.integers(2, 5))
        return int(rng.integers(2, 6)), int(rng.integers(5, 12))
    if mode == 8:
        return 1, int(rng.integers(3, 11))
    return int(rng.integers(3, 11)), 1


def _path(mask, start, end, rng):
    row, col = start
    mask[row, col] = True
    while (row, col) != end:
        dr = int(np.sign(end[0] - row))
        dc = int(np.sign(end[1] - col))
        if dr and dc and rng.random() < 0.40:
            if rng.integers(2):
                dr = 0
            else:
                dc = 0
        row += dr
        col += dc
        mask[row, col] = True


def _motif(rng, height, width, style, require_asym=False):
    if height == 1 or width == 1:
        return np.ones((height, width), dtype=bool)
    for _ in range(48):
        mask = np.zeros((height, width), dtype=bool)
        if style == "sparse":
            # A true diagonal chain can be disconnected under four-neighbor
            # connectivity while remaining one eight-connected object.
            row, col = 0, (0 if rng.integers(2) else width - 1)
            dc = 1 if col == 0 else -1
            while True:
                mask[row, col] = True
                if row == height - 1 and col in (0, width - 1):
                    break
                if row < height - 1:
                    row += 1
                if 0 <= col + dc < width:
                    col += dc
            if rng.random() < 0.55:
                neighbors = [(r, c) for r in range(height) for c in range(width)
                             if not mask[r, c] and np.any(mask[max(0, r-1):r+2, max(0, c-1):c+2])]
                r, c = neighbors[int(rng.integers(len(neighbors)))]
                mask[r, c] = True
        else:
            # Boundary anchors are joined by eight-neighbor paths; no separate
            # distractor object is ever introduced.
            anchors = ((0, int(rng.integers(width))),
                       (height - 1, int(rng.integers(width))),
                       (int(rng.integers(height)), 0),
                       (int(rng.integers(height)), width - 1))
            order = rng.permutation(4)
            for a, b in zip(order[:-1], order[1:]):
                _path(mask, anchors[int(a)], anchors[int(b)], rng)
        if style == "solid":
            target = height * width
        else:
            densities = {"sparse": (0.20, 0.42), "branch": (0.40, 0.65),
                         "thick": (0.63, 0.83), "dense": (0.82, 0.96)}
            lo, hi = densities[style]
            target = max(int(mask.sum()), int(round(height * width * rng.uniform(lo, hi))))
        while int(mask.sum()) < target:
            candidates = []
            for row in range(height):
                for col in range(width):
                    if not mask[row, col] and np.any(mask[max(0, row-1):row+2, max(0, col-1):col+2]):
                        candidates.append((row, col))
            row, col = candidates[int(rng.integers(len(candidates)))]
            mask[row, col] = True
        if not require_asym or _asymmetric(mask):
            return mask
    # A deterministic, connected asymmetric witness for the training set.
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :2] = True
    mask[1, 1] = True
    mask[2:, 2] = True
    mask[-1, 1:] = True
    mask[1, 0] = True
    assert _asymmetric(mask)
    return mask


def _canvas_shape(rng, mask):
    height, width = mask.shape
    if height <= 6 and width <= 6 and rng.random() < 0.16:
        rows, cols = 8, 8
    else:
        rows = min(30, height + int(rng.integers(2, 15)))
        cols = min(30, width + int(rng.integers(2, 15)))
    # The background is uniquely dominant even for thick or solid motifs.
    while rows * cols < int(mask.sum()) * 2.4:
        if (rows - height <= cols - width and rows < 30) or cols == 30:
            rows += 1
        else:
            cols += 1
    return rows, cols


def _position(rng, canvas, motif, holdout, split):
    mr, mc = canvas[0] - motif[0], canvas[1] - motif[1]
    candidates = [(r, c) for r in range(mr + 1) for c in range(mc + 1)]
    if holdout is not None:
        axis, test_high = holdout
        limit = (mr, mc)[axis]
        if limit >= 2:
            want_high = test_high if split == "test" else not test_high
            subset = [p for p in candidates if ((p[axis] / limit >= 0.60) if want_high else (p[axis] / limit <= 0.40))]
            if subset:
                candidates = subset
    category = int(rng.choice(4, p=[0.26, 0.31, 0.24, 0.19]))
    if category == 0:
        chosen = [p for p in candidates if p[0] in (0, mr) and p[1] in (0, mc)]
    elif category == 1:
        chosen = [p for p in candidates if p[0] in (0, mr) or p[1] in (0, mc)]
    elif category == 2:
        chosen = [p for p in candidates if 0 < p[0] < mr and 0 < p[1] < mc]
    else:
        chosen = candidates
    if not chosen:
        chosen = candidates
    return chosen[int(rng.integers(len(chosen)))]


def _paint(rng, mask, background, colors):
    crop = np.full(mask.shape, background, dtype=np.int64)
    colors = colors[:int(mask.sum())]
    if len(colors) == 1:
        crop[mask] = colors[0]
    else:
        # Color patches vary independently of the connected union geometry.
        points = np.argwhere(mask)
        centers = points[rng.choice(len(points), size=len(colors), replace=False)]
        for row, col in points:
            distances = [abs(int(row)-int(r)) + abs(int(col)-int(c)) for r, c in centers]
            crop[row, col] = colors[int(np.argmin(distances))]
    return crop


def _build_pair(rng, shape, style, background, colors, holdout, split, anchor=False):
    mask = _motif(rng, *shape, style, require_asym=anchor)
    canvas = _canvas_shape(rng, mask)
    row, col = _position(rng, canvas, shape, holdout, split)
    crop = _paint(rng, mask, background, colors)
    inp = np.full(canvas, background, dtype=np.int64)
    inp[row:row+shape[0], col:col+shape[1]] = crop
    return {"input": inp, "output": np.concatenate((crop, crop), axis=1)}


def _background(inp):
    colors, counts = np.unique(inp, return_counts=True)
    order = np.argsort(counts)
    assert len(order) >= 2 and counts[order[-1]] > counts[order[-2]]
    background = int(colors[order[-1]])
    assert counts[order[-1]] * 2 > inp.size
    return background


def _features(pair):
    inp, out = pair["input"], pair["output"]
    background = _background(inp)
    mask = inp != background
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    crop = inp[r0:r1, c0:c1]
    assert crop.shape[1] <= 15 and np.array_equal(out, np.concatenate((crop, crop), axis=1))
    assert _connected(mask)
    assert (r1-r0, c1-c0) == out.shape[:1] + (out.shape[1] // 2,)
    return {"background": background, "shape": crop.shape, "position": (int(r0), int(c0)),
            "mask": mask[r0:r1, c0:c1], "foreground": set(map(int, np.unique(inp[mask]))),
            "multicolor": len(np.unique(inp[mask])) > 1}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    # These are presentation regimes; crop and two-copy output are shared by
    # every pair. No hidden convention is needed to interpret a test input.
    holdout = (int(rng.integers(2)), bool(rng.integers(2))) if rng.random() < 0.43 else None
    multicolor_puzzle = bool(rng.random() < 0.38)
    styles = ("sparse", "branch", "thick", "dense", "solid")
    train, test = [], []
    used_inputs = set()
    used_masks = set()
    train_foreground = set()
    anchor_shape = _dimensions(rng, anchor=True)
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(80):
                if split == "train" and index == 0:
                    shape = anchor_shape
                elif split == "train" and index == 1:
                    shape = _dimensions(rng)
                    if shape == anchor_shape:
                        continue
                else:
                    shape = _dimensions(rng)
                style = styles[int(rng.integers(len(styles)))]
                background = int(rng.integers(10))
                available = [x for x in range(10) if x != background]
                if split == "test" and train_foreground and rng.random() < 0.50:
                    familiar = sorted(train_foreground.intersection(available))
                    if familiar:
                        first = int(rng.choice(familiar))
                    else:
                        first = int(rng.choice(available))
                else:
                    first = int(rng.choice(available))
                colors = [first]
                if multicolor_puzzle and ((split == "train" and index == 1) or (split == "test" and index == 0) or rng.random() < 0.27):
                    other = [x for x in available if x != first]
                    colors.append(int(rng.choice(other)))
                    if rng.random() < 0.32:
                        other = [x for x in other if x != colors[1]]
                        colors.append(int(rng.choice(other)))
                pair = _build_pair(rng, shape, style, background, colors, holdout, split, anchor=(split == "train" and index == 0))
                key = _key(pair["input"])
                pairkey = (key, _key(pair["output"]))
                info = _features(pair)
                maskkey = (info["shape"], tuple(int(x) for x in info["mask"].flat))
                if key in used_inputs or pairkey in _SOURCE_PAIRS or maskkey in used_masks:
                    continue
                if split == "train" and index == 1 and info["background"] == _features(train[0])["background"]:
                    continue
                break
            else:
                raise AssertionError("bounded motif construction exhausted")
            used_inputs.add(key)
            used_masks.add(maskkey)
            if split == "train":
                train.append(pair)
                train_foreground.update(info["foreground"])
            else:
                test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[x], list) for x in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, maskkeys = set(), set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inkey, outkey = _key(pair["input"]), _key(pair["output"])
            assert inkey not in seen_inputs and (inkey, outkey) not in seen_pairs
            assert (inkey, outkey) not in _SOURCE_PAIRS
            assert not np.array_equal(pair["input"], pair["output"])
            seen_inputs.add(inkey)
            seen_pairs.add((inkey, outkey))
            info = _features(pair)
            maskkey = (info["shape"], tuple(int(x) for x in info["mask"].flat))
            assert maskkey not in maskkeys
            maskkeys.add(maskkey)
            features[split].append(info)
    train = features["train"]
    assert _asymmetric(train[0]["mask"])
    assert len({x["shape"] for x in train}) >= 2
    assert len({x["background"] for x in train}) >= 2
    first_crop = puzzle["train"][0]["output"][:, :train[0]["shape"][1]]
    assert not np.array_equal(puzzle["train"][0]["output"], np.repeat(first_crop, 2, axis=1))
    if any(x["multicolor"] for x in features["test"]):
        assert any(x["multicolor"] for x in train)
    # A non-symmetric example proves that both halves preserve orientation.
    # Distinct training crop sizes prove that the bounding box is pair-local.
    for pair in puzzle["train"]:
        half = pair["output"].shape[1] // 2
        assert np.array_equal(pair["output"][:, :half], pair["output"][:, half:])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    sizes, canvases, bgs, positions, structures, foregrounds = set(), set(), set(), set(), set(), set()
    multicolor = diagonal_only = lines = solids = familiar = new_color = 0
    sparse = dense = holes = edge = corner = interior = 0
    tall = wide = zero_background = zero_foreground = combined = 0
    familiar_size = new_size = 0
    quadrants = set()
    holdout = overlap = 0
    for p in automatic:
        validate_generated_puzzle(p)
        train_fg = set()
        train_pos = []
        test_pos = []
        train_shapes = {x["shape"] for x in [_features(pair) for pair in p["train"]]}
        for split in ("train", "test"):
            for pair in p[split]:
                info = _features(pair)
                inp = pair["input"]
                canvas = inp.shape
                size = info["shape"]
                row, col = info["position"]
                max_row, max_col = canvas[0] - size[0], canvas[1] - size[1]
                normalized = (row / max_row, col / max_col)
                sizes.add(size)
                canvases.add(canvas)
                bgs.add(info["background"])
                positions.add(info["position"])
                structures.add((size, tuple(int(x) for x in info["mask"].flat)))
                foregrounds.update(info["foreground"])
                multicolor += info["multicolor"]
                lines += min(size) == 1
                solids += bool(np.all(info["mask"]))
                sparse += np.mean(info["mask"]) <= 0.4
                dense += np.mean(info["mask"]) >= 0.8
                holes += bool(_hole_sizes(info["mask"]))
                on_edge = row in (0, max_row) or col in (0, max_col)
                edge += on_edge
                corner += row in (0, max_row) and col in (0, max_col)
                interior += not on_edge
                tall += canvas[0] > canvas[1]
                wide += canvas[1] > canvas[0]
                zero_background += info["background"] == 0
                zero_foreground += 0 in info["foreground"]
                combined += info["multicolor"] and max(size) >= 6 and on_edge and info["background"] != 0
                if not _connected_four(info["mask"]):
                    diagonal_only += 1
                # Reconstruct from the visible input rather than metadata or
                # the generated output, then compare with the label.
                values, counts = np.unique(inp, return_counts=True)
                bg = int(values[np.argmax(counts)])
                cells = np.argwhere(inp != bg)
                lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
                crop = inp[lo[0]:hi[0], lo[1]:hi[1]]
                assert np.array_equal(pair["output"], np.tile(crop, (1, 2)))
                if split == "train":
                    train_fg.update(info["foreground"])
                    train_pos.append(normalized)
                else:
                    familiar += bool(info["foreground"] & train_fg)
                    new_color += bool(info["foreground"] - train_fg)
                    familiar_size += size in train_shapes
                    new_size += size not in train_shapes
                    quadrants.add((row * 2 > max_row, col * 2 > max_col))
                    test_pos.append(normalized)
        separated = any(
            (all(a[axis] <= 0.4 for a in train_pos) and all(b[axis] >= 0.6 for b in test_pos))
            or (all(a[axis] >= 0.6 for a in train_pos) and all(b[axis] <= 0.4 for b in test_pos))
            for axis in (0, 1)
        )
        holdout += separated
        overlap += not separated
    assert len(sizes) >= 35 and len(canvases) >= 80
    assert len(bgs) == 10 and len(positions) >= 45
    assert len(structures) >= 170 and len(foregrounds) == 10
    assert multicolor >= 15 and diagonal_only >= 3 and lines >= 8 and solids >= 8
    assert familiar >= 15 and new_color >= 15
    assert familiar_size >= 8 and new_size >= 15
    assert sparse >= 10 and dense >= 30 and holes >= 10
    assert edge >= 30 and corner >= 10 and interior >= 20
    assert tall >= 30 and wide >= 30 and zero_background >= 10 and zero_foreground >= 10
    assert combined >= 3 and len(quadrants) == 4
    assert overlap >= 8 and holdout >= 8
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(13123 + (args[0] or 0) + (args[1] or 0), *args)
        validate_generated_puzzle(p)
        if args[0] is not None:
            assert len(p["train"]) == args[0]
        if args[1] is not None:
            assert len(p["test"]) == args[1]
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(7, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    assert _puzzle_key(generate_puzzle(99, 7, 3)) == _puzzle_key(generate_puzzle(99, 7, 3))
    assert _puzzle_key(generate_puzzle(99)) != _puzzle_key(generate_puzzle(100))


def _connected_four(mask):
    points = set(map(tuple, np.argwhere(mask)))
    reached = {next(iter(points))}
    frontier = list(reached)
    while frontier:
        r, c = frontier.pop()
        for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if neighbor in points and neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return len(reached) == len(points)


def _hole_sizes(mask):
    height, width = mask.shape
    remaining = {(r, c) for r in range(height) for c in range(width) if not mask[r, c]}
    sizes = []
    while remaining:
        start = remaining.pop()
        reached = {start}
        frontier = [start]
        while frontier:
            r, c = frontier.pop()
            for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    reached.add(neighbor)
                    frontier.append(neighbor)
        if all(0 < r < height-1 and 0 < c < width-1 for r, c in reached):
            sizes.append(len(reached))
    return sizes
