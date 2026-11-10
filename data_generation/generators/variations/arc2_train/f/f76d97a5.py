"""Generate ARC puzzles that extract a gray mask in its companion color."""

import numpy as np


MARKER = 5
BACKGROUND = 0
COMPANIONS = (1, 2, 3, 4, 6, 7, 8, 9)


def _rows(*rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


SOURCE = (
    (_rows("55666", "65566", "66556", "66655", "56665"),
     _rows("66000", "06600", "00660", "00066", "60006")),
    (_rows("454", "555", "454"), _rows("040", "444", "040")),
    (_rows("95999", "99559", "95999", "99599", "99955"),
     _rows("09000", "00990", "09000", "00900", "00099")),
    (_rows("33353", "35333", "35535", "33353", "55533"),
     _rows("00030", "03000", "03303", "00030", "33300")),
)


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return int(value)


def _counts(rng, train, test):
    train, test = _count(train, "num_train", 2, 10), _count(test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None:
        choices = [n for n in range(2, 11) if test is None or n > test]
        weights = np.array([3 if n <= 4 else 1 for n in choices], dtype=float)
        train = int(rng.choice(choices, p=weights / weights.sum()))
    if test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        weights = np.array([3 if n == 1 else 1 for n in choices], dtype=float)
        test = int(rng.choice(choices, p=weights / weights.sum()))
    return train, test


def _shape(rng, source_like):
    if source_like:
        n = (3, 5)[int(rng.integers(2))]
        return n, n
    branch = int(rng.choice(5, p=[.18, .17, .38, .17, .10]))
    if branch == 0:
        n = (3, 5)[int(rng.integers(2))]
        return n, n
    if branch == 1:
        n = int(rng.integers(3, 11))
        return (1, n) if rng.random() < .5 else (n, 1)
    if branch == 2:
        h, w = int(rng.integers(2, 10)), int(rng.integers(2, 10))
        if h == w:
            w = 2 + (w - 1) % 8
        return h, w
    if branch == 3:
        return int(rng.integers(7, 13)), int(rng.integers(7, 13))
    n = int(rng.integers(2, 7))
    return n, n


def _base_mask(rng, h, w, mode):
    mask = np.zeros((h, w), dtype=bool)
    n = h * w
    if min(h, w) == 1:
        mask.flat[rng.choice(n, size=max(1, n // 2), replace=False)] = True
        return mask
    if mode == 0:  # stepped diagonal
        for r in range(h):
            c = min(w - 1, r * w // h)
            mask[r, c] = True
            if rng.random() < .6:
                mask[r, min(w - 1, c + 1)] = True
    elif mode == 1:  # cross
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, :] = True
        mask[:, c] = True
    elif mode == 2:  # bent run
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, :c + 1] = True
        mask[r:, c] = True
    elif mode == 3:  # hollow outline
        if min(h, w) >= 3:
            top, left = int(rng.integers(h - 2)), int(rng.integers(w - 2))
            bottom = int(rng.integers(top + 2, h))
            right = int(rng.integers(left + 2, w))
            mask[top, left:right + 1] = True
            mask[bottom, left:right + 1] = True
            mask[top:bottom + 1, left] = True
            mask[top:bottom + 1, right] = True
        else:
            mask[0, :] = True
            mask[:, -1] = True
    elif mode == 4:  # edge-touching partial frame
        mask[0, :] = rng.random(w) < .75
        mask[-1, :] |= rng.random(w) < .45
        mask[:, 0] |= rng.random(h) < .55
        mask[:, -1] |= rng.random(h) < .45
    elif mode == 5:  # connected growth plus separate cells
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        while int(mask.sum()) < max(2, n // 4):
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((rr, cc))
            rr, cc = sorted(frontier)[int(rng.integers(len(frontier)))]
            mask[rr, cc] = True
        mask.flat[rng.choice(n, size=max(1, n // 9), replace=False)] = True
    elif mode == 6:  # checkerboard with omissions
        parity = int(rng.integers(2))
        for r in range(h):
            for c in range(w):
                mask[r, c] = (r + c) % 2 == parity and rng.random() < .85
    elif mode == 7:  # one or two solid blocks
        for _ in range(int(rng.integers(1, 3))):
            top, left = int(rng.integers(h)), int(rng.integers(w))
            bottom, right = int(rng.integers(top + 1, h + 1)), int(rng.integers(left + 1, w + 1))
            mask[top:bottom, left:right] = True
    elif mode == 8:  # irregular mixture
        mask[:] = rng.random((h, w)) < rng.uniform(.25, .7)
    else:  # intersecting or separated segments
        for _ in range(int(rng.integers(2, 5))):
            if rng.random() < .5:
                r, left = int(rng.integers(h)), int(rng.integers(w))
                mask[r, left:int(rng.integers(left + 1, w + 1))] = True
            else:
                c, top = int(rng.integers(w)), int(rng.integers(h))
                mask[top:int(rng.integers(top + 1, h + 1)), c] = True
    return mask


def _mask(rng, height, width, clear_roles=False):
    total = height * width
    mode = int(rng.integers(10))
    transpose = bool(rng.integers(2))
    h, w = (width, height) if transpose else (height, width)
    mask = _base_mask(rng, h, w, mode)
    if transpose:
        mask = mask.T
    if rng.random() < .5:
        mask = np.flip(mask, axis=0)
    if rng.random() < .5:
        mask = np.flip(mask, axis=1)
    occupied = np.argwhere(mask)
    if len(occupied) and len(occupied) < total:
        top, left = occupied.min(axis=0)
        bottom, right = occupied.max(axis=0)
        dr = int(rng.integers(-int(top), height - int(bottom)))
        dc = int(rng.integers(-int(left), width - int(right)))
        moved = np.zeros_like(mask)
        moved[occupied[:, 0] + dr, occupied[:, 1] + dc] = True
        mask = moved
    density = "natural" if clear_roles else str(rng.choice(
        ["single", "one_hole", "sparse", "natural", "dense"],
        p=[.15, .10, .15, .42, .18]))
    if density == "single":
        target = 1
    elif density == "one_hole":
        target = total - 1
    elif density == "sparse":
        target = int(rng.integers(2, max(3, total // 4 + 1)))
    elif density == "dense":
        low = max(2, int(np.ceil(.72 * total)))
        target = total - 1 if low >= total else int(rng.integers(low, total))
    else:
        target = min(total - 2, max(2, int(mask.sum())))
    target = min(total - 1, max(1, target))
    current = int(mask.sum())
    if current < target:
        mask.flat[rng.choice(np.flatnonzero(~mask), size=target - current, replace=False)] = True
    elif current > target:
        mask.flat[rng.choice(np.flatnonzero(mask), size=current - target, replace=False)] = False
    return mask


def _pair(mask, color):
    return {"input": np.where(mask, MARKER, color).astype(np.int8),
            "output": np.where(mask, color, BACKGROUND).astype(np.int8)}


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _mask_signature(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _source_copy(pair):
    return any(np.array_equal(pair["input"], inp) and np.array_equal(pair["output"], out)
               for inp, out in SOURCE)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    # Gray and black are fixed source conventions. Shape style is sampled once.
    source_like = bool(rng.random() < .25)
    palette_size = int(rng.integers(2, min(5, train_count) + 1))
    palette = [int(c) for c in rng.choice(COMPANIONS, size=palette_size, replace=False)]
    train_colors = palette[:2] + [int(rng.choice(palette)) for _ in range(train_count - 2)]
    puzzle = {"train": [], "test": []}
    masks, used_inputs = [], set()
    for index in range(train_count + test_count):
        training = index < train_count
        if training:
            color = train_colors[index]
        else:
            observed = set(train_colors)
            choices = sorted(observed) if rng.random() < .5 else [
                c for c in COMPANIONS if c not in observed]
            color = int(rng.choice(choices))
        reuse = training and index >= 2 and rng.random() < .24
        candidates = [m for i, m in enumerate(masks) if train_colors[i] != color] if reuse else []
        if candidates:
            mask = candidates[int(rng.integers(len(candidates)))].copy()
            shape = mask.shape
        else:
            familiar_shape = not training and rng.random() < .43
            if familiar_shape:
                shape = masks[int(rng.integers(train_count))].shape
            else:
                shape = _shape(rng, source_like)
                if index == 1 or (not training and not source_like):
                    previous = {m.shape for m in (masks[:1] if index == 1 else masks[:train_count])}
                    for _ in range(8):
                        if shape not in previous:
                            break
                        shape = _shape(rng, source_like)
            mask = _mask(rng, *shape, clear_roles=training and index < 2)
        for attempt in range(33):
            pair = _pair(mask, color)
            sig = _signature(pair["input"])
            duplicate_foundation = training and index == 1 and (
                _mask_signature(mask) == _mask_signature(masks[0]))
            duplicate_test_mask = not training and any(
                _mask_signature(mask) == _mask_signature(old) for old in masks)
            if (sig not in used_inputs and not _source_copy(pair)
                    and not duplicate_foundation and not duplicate_test_mask):
                break
            if attempt == 31:
                shape = 17 + index, 16 + index
            mask = _mask(rng, *shape, clear_roles=training and index < 2)
        else:
            raise RuntimeError("could not construct a unique pair")
        used_inputs.add(sig)
        puzzle["train" if training else "test"].append(pair)
        masks.append(mask)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, train_colors, train_masks, test_masks = set(), set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            colors = {int(c) for c in np.unique(inp)}
            assert len(colors) == 2 and MARKER in colors
            color = next(iter(colors - {MARKER}))
            assert color in COMPANIONS
            mask = inp == MARKER
            assert np.array_equal(out, np.where(mask, color, BACKGROUND))
            assert {int(c) for c in np.unique(out)} == {BACKGROUND, color}
            assert not np.array_equal(inp, out) and not _source_copy(pair)
            input_sig, output_sig = _signature(inp), _signature(out)
            assert input_sig not in inputs and (input_sig, output_sig) not in pairs
            inputs.add(input_sig)
            pairs.add((input_sig, output_sig))
            mask_sig = _mask_signature(mask)
            if split == "train":
                train_colors.add(color)
                train_masks.add(mask_sig)
            else:
                assert mask_sig not in train_masks and mask_sig not in test_masks
                test_masks.add(mask_sig)
    assert len(train_colors) >= 2 and len(train_masks) >= 2


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _predict_from_training(train, test_input):
    # Infer candidate color rules from training grids alone.
    candidates = []
    for marker in range(10):
        for background in range(10):
            if all(marker in np.unique(p["input"]) and len(np.unique(p["input"])) == 2
                   and np.array_equal(p["output"], np.where(
                       p["input"] == marker,
                       next(c for c in np.unique(p["input"]) if c != marker),
                       background)) for p in train):
                colors = {int(c) for c in np.unique(test_input)}
                if marker in colors and len(colors) == 2:
                    color = next(iter(colors - {marker}))
                    candidates.append(np.where(test_input == marker, color, background))
    assert candidates and all(np.array_equal(candidates[0], c) for c in candidates)
    return candidates[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            row, col = pending.pop()
            for neighbor in ((row + 1, col), (row - 1, col),
                             (row, col + 1), (row, col - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
    return count


def _has_hole(mask):
    empty = ~mask
    height, width = mask.shape
    boundary = {(row, col) for row, col in np.argwhere(empty)
                if row in (0, height - 1) or col in (0, width - 1)}
    visited = set(boundary)
    pending = list(boundary)
    while pending:
        row, col = pending.pop()
        for rr, cc in ((row + 1, col), (row - 1, col),
                       (row, col + 1), (row, col - 1)):
            neighbor = (rr, cc)
            if (0 <= rr < height and 0 <= cc < width and empty[rr, cc]
                    and neighbor not in visited):
                visited.add(neighbor)
                pending.append(neighbor)
    return bool(set(map(tuple, np.argwhere(empty))) - visited)


def self_test() -> None:
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None), (7, 3)):
        seed = 12045 + len(str(args))
        p = generate_puzzle(seed, *args)
        validate_generated_puzzle(p)
        assert args[0] is None or len(p["train"]) == args[0]
        assert args[1] is None or len(p["test"]) == args[1]
        assert _equal(p, generate_puzzle(seed, *args))
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (3.0, 1)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {args}")
    train_counts, test_counts, shapes, palettes, structures, regimes = (set() for _ in range(6))
    rectangular = strips = large = singleton = one_hole = reuse = source_like = 0
    connected = disconnected = crosses = hollows = interior = 0
    first = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if first is None:
            first = p
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        train_palette = {next(iter(set(map(int, np.unique(x["input"]))) - {MARKER}))
                         for x in p["train"]}
        train_shapes = {x["input"].shape for x in p["train"]}
        train_masks = [_mask_signature(x["input"] == MARKER) for x in p["train"]]
        reuse += len(train_masks) - len(set(train_masks))
        source_like += int(all(x["input"].shape in ((3, 3), (5, 5))
                               for s in ("train", "test") for x in p[s]))
        for s in ("train", "test"):
            for x in p[s]:
                inp, out = x["input"], x["output"]
                h, w = inp.shape
                shapes.add((h, w))
                palettes.add(tuple(map(int, np.unique(inp))))
                structures.add(_mask_signature(inp == MARKER))
                rectangular += int(h != w)
                strips += int(min(h, w) == 1)
                large += int(max(h, w) >= 9)
                n = int(np.count_nonzero(inp == MARKER))
                mask = inp == MARKER
                singleton += int(n == 1)
                one_hole += int(n == inp.size - 1)
                components = _components(mask)
                connected += int(components == 1)
                disconnected += int(components >= 3)
                crosses += int(mask.all(axis=0).any() and mask.all(axis=1).any()
                               and not mask.all())
                hollows += int(_has_hole(mask))
                interior += int(not (mask[0].any() or mask[-1].any()
                                     or mask[:, 0].any() or mask[:, -1].any()))
                assert np.array_equal(out, np.where(inp == MARKER,
                    next(c for c in np.unique(inp) if c != MARKER), BACKGROUND))
                if s == "test":
                    color = next(c for c in np.unique(inp) if c != MARKER)
                    regimes.add((int(color) in train_palette, (h, w) in train_shapes))
                    assert np.array_equal(_predict_from_training(p["train"], inp.copy()), out)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(palettes) >= 7 and len(structures) >= 130
    assert rectangular >= 35 and strips >= 8 and large >= 20
    assert singleton >= 4 and one_hole >= 4 and reuse >= 3
    assert connected >= 30 and disconnected >= 12
    assert crosses >= 5 and hollows >= 5 and interior >= 2
    assert source_like >= 2 and len(regimes) == 4
    assert not _equal(first, generate_puzzle(123456))
