"""Generators for the maximal neutral square ARC family."""

import numpy as np


_SOURCE_INPUT_TEXT = (
    ("0500500000", "5500055050", "0005500555", "0050550050",
     "0500000050", "5050055505", "0005055050", "0050555000",
     "5055055050", "0000000500"),
    ("5500000500", "0055000055", "5505000050", "0000555505",
     "0505055050", "5000050055", "5550500005", "0500005550",
     "5000050055", "5000005500"),
    ("0050000505", "0500050005", "0050500000", "0000500000",
     "0550055505", "5005050000", "5555055500", "0005000050",
     "0000555555", "0000050050"),
    ("5000505050", "5005055000", "5505500550", "5000000500",
     "5000550005", "0000005500", "0055005500", "5050505005",
     "0550500555", "0005550000"),
)
_SOURCE_OUTPUT_TEXT = (
    ("0511500000", "5511055050", "1105500555", "1150551150",
     "0501101150", "5051155505", "1105055050", "1150555000",
     "5055055050", "0000000500"),
    ("5500111500", "0055111055", "5505111050", "0000555505",
     "0505055050", "5000051155", "5550501105", "0511105550",
     "5011150055", "5011105500"),
    ("0051100505", "0501151115", "1150501110", "1100501110",
     "0550055505", "5005050011", "5555055511", "1115000050",
     "1110555555", "1110050050"),
    ("5110505050", "5115055000", "5505511550", "5111011500",
     "5111550005", "0111115511", "0055115511", "5050505005",
     "0550500555", "0005550000"),
)


def _source_grid(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_INPUTS = tuple(map(_source_grid, _SOURCE_INPUT_TEXT))
_SOURCE_OUTPUTS = tuple(map(_source_grid, _SOURCE_OUTPUT_TEXT))


def _resolve_counts(rng, num_train, num_test):
    for value, name, low, high in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _maximal_rectangles(mask):
    """All solid rectangles that cannot grow by a complete side."""
    height, width = mask.shape
    result = []
    for top in range(height):
        common = np.ones(width, dtype=bool)
        for bottom in range(top, height):
            common &= mask[bottom]
            padded = np.concatenate(([False], common, [False]))
            changes = np.flatnonzero(padded[1:] != padded[:-1])
            for start, stop in zip(changes[::2], changes[1::2]):
                left, right = int(start), int(stop) - 1
                if top and mask[top - 1, left:right + 1].all():
                    continue
                if bottom + 1 < height and mask[bottom + 1, left:right + 1].all():
                    continue
                result.append((top, left, bottom, right))
    return result


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _touch(a, b):
    rows = max(a[0], b[0]) <= min(a[2], b[2])
    cols = max(a[1], b[1]) <= min(a[3], b[3])
    return ((rows and (a[3] + 1 == b[1] or b[3] + 1 == a[1]))
            or (cols and (a[2] + 1 == b[0] or b[2] + 1 == a[0])))


def _side(rect):
    return rect[2] - rect[0] + 1


def _candidates(mask):
    return [r for r in _maximal_rectangles(mask)
            if _side(r) >= 2 and _side(r) == r[3] - r[1] + 1]


def _unambiguous(candidates):
    degree = [0] * len(candidates)
    for i, a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if _overlap(a, b):
                if _side(a) == _side(b):
                    return False
                degree[i] += 1
                degree[j] += 1
                if degree[i] > 1 or degree[j] > 1:
                    return False
    return True


def _selected(candidates):
    """The source test resolves an overlapping 2x2 in favor of a 3x3."""
    accepted = []
    for rect in sorted(candidates, key=lambda r: (-_side(r), r[0], r[1])):
        if not any(_overlap(rect, larger) for larger in accepted):
            accepted.append(rect)
    return accepted


def _make_output(grid, background, highlight):
    output = grid.copy()
    for top, left, bottom, right in _selected(_candidates(grid == background)):
        output[top:bottom + 1, left:right + 1] = highlight
    return output


def _dimensions(rng, minimum):
    band = int(rng.choice(3, p=[0.30, 0.38, 0.32]))
    ranges = ((8, 14), (14, 23), (23, 31))
    lo, hi = ranges[band]
    lo = max(lo, minimum)
    h = int(rng.integers(lo, hi))
    w = int(rng.integers(lo, hi))
    if rng.random() < 0.30:
        if rng.random() < 0.5:
            h = int(rng.integers(max(8, minimum), 16))
        else:
            w = int(rng.integers(max(8, minimum), 16))
    return h, w


def _add_path(mask, square, rng, protected):
    top, left, bottom, right = square
    direction = int(rng.integers(4))
    if direction < 2:
        col = int(rng.integers(left, right + 1))
        row = top - 1 if direction == 0 else bottom + 1
        dr, dc = (-1, 0) if direction == 0 else (1, 0)
    else:
        row = int(rng.integers(top, bottom + 1))
        col = left - 1 if direction == 2 else right + 1
        dr, dc = (0, -1) if direction == 2 else (0, 1)
    for step in range(int(rng.integers(1, 7))):
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]):
            break
        if protected[row, col]:
            break
        mask[row, col] = True
        if step >= 1 and rng.random() < 0.22:
            dr, dc = (dc, dr) if rng.random() < 0.5 else (-dc, -dr)
        row += dr
        col += dc


def _build_mask(rng, height, width, density, primary_side, max_side, mode,
                extra_count):
    mask = rng.random((height, width)) < density
    squares = []
    margin = max(primary_side + 3, 6)
    if mode == "overlap":
        if height < primary_side + 4 or width < primary_side + 4:
            return None
        top = int(rng.integers(2, height - primary_side - 1))
        left = int(rng.integers(2, width - primary_side - 1))
        squares = [(top, left, primary_side),
                   (top + primary_side - 1, left - 1, 2)]
    elif mode in ("touch", "chain"):
        other = int(rng.integers(2, min(max_side, 5) + 1))
        if height < primary_side + other + 1 or width < primary_side + other + 1:
            return None
        top = int(rng.integers(1, height - primary_side - other + 1))
        left = int(rng.integers(1, width - primary_side - other + 1))
        squares = [(top, left, primary_side),
                   (top + primary_side - 1, left + primary_side, other)]
        if mode == "chain":
            third = int(rng.integers(2, min(max_side, 4) + 1))
            t, l, s = squares[-1]
            if t + s - 1 + third < height and l + s + third <= width:
                squares.append((t + s - 1, l + s, third))
    else:
        if height < margin or width < margin:
            return None
        top = int(rng.integers(1, height - primary_side))
        left = int(rng.integers(1, width - primary_side))
        squares = [(top, left, primary_side)]

    for _ in range(extra_count):
        side = int(rng.integers(2, min(max_side, min(height, width) - 2) + 1))
        for _ in range(30):
            top = int(rng.integers(0, height - side + 1))
            left = int(rng.integers(0, width - side + 1))
            rect = (top, left, top + side - 1, left + side - 1)
            if all(not _overlap(rect, (r, c, r + s - 1, c + s - 1))
                   for r, c, s in squares):
                squares.append((top, left, side))
                break

    union = np.zeros_like(mask)
    for top, left, side in squares:
        union[top:top + side, left:left + side] = True
    near = union.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs = slice(max(0, dr), height + min(0, dr))
            cs = slice(max(0, dc), width + min(0, dc))
            rt = slice(max(0, -dr), height + min(0, -dr))
            ct = slice(max(0, -dc), width + min(0, -dc))
            near[rs, cs] |= union[rt, ct]
    mask[near] = False
    mask[union] = True

    if mode != "overlap":
        for top, left, side in squares:
            rect = (top, left, top + side - 1, left + side - 1)
            for _ in range(int(rng.integers(0, 5))):
                _add_path(mask, rect, rng, union)

    if rng.random() < 0.55:
        rh = int(rng.integers(1, min(4, height) + 1))
        rw = int(rng.integers(max(4, rh + 1), min(9, width) + 1))
        if rng.random() < 0.5:
            rh, rw = rw, rh
        if rh <= height and rw <= width:
            if rng.random() < 0.45:
                anchor = squares[0]
                top = int(np.clip(anchor[0] + rng.integers(-2, 3), 0, height - rh))
                left = int(np.clip(anchor[1] + rng.integers(-3, 4), 0, width - rw))
            else:
                top = int(rng.integers(0, height - rh + 1))
                left = int(rng.integers(0, width - rw + 1))
            mask[top:top + rh, left:left + rw] = True
    intended = [(r, c, r + s - 1, c + s - 1) for r, c, s in squares]
    return mask, intended


def _pair(rng, background, highlight, blockers, max_side, regime, index,
          overlap_enabled, density, allowed_sides=None):
    for _ in range(180):
        if regime == "source_train" and index == 0 and overlap_enabled:
            mode, primary_side = "overlap", 3
        else:
            modes = ["separate", "touch", "chain"]
            weights = [0.49, 0.37, 0.14]
            if overlap_enabled and (regime == "target_test" or index > 0):
                modes.append("overlap")
                weights = [0.41, 0.31, 0.12, 0.16]
            mode = str(rng.choice(modes, p=weights))
            if mode == "overlap":
                primary_side = int(rng.integers(3, min(max_side, 5) + 1))
            elif regime == "source_train" and index == 0:
                primary_side = 2
            elif regime == "source_train" and index == 1:
                primary_side = max_side
            else:
                primary_side = int(rng.integers(2, max_side + 1))
        minimum = max(primary_side + 4, 8)
        if mode in ("touch", "chain"):
            minimum = max(minimum, primary_side + min(max_side, 5) + 1)
        height, width = _dimensions(rng, minimum)
        extra = int(rng.integers(0, min(5, height * width // 65 + 1)))
        if regime == "source_train" and index == 0 and overlap_enabled:
            extra = max(extra, 1)
        made = _build_mask(rng, height, width, density, primary_side,
                           max_side, mode, extra)
        if made is None:
            continue
        mask, intended = made
        candidates = _candidates(mask)
        if not _unambiguous(candidates):
            continue
        if not overlap_enabled and any(
            _overlap(a, b) for i, a in enumerate(candidates)
            for b in candidates[i + 1:]
        ):
            continue
        chosen = _selected(candidates)
        if not chosen or len(chosen) > 16:
            continue
        if allowed_sides is not None and not set(map(_side, chosen)) <= allowed_sides:
            continue
        if intended[0] not in chosen:
            continue
        if mode in ("touch", "chain") and (
            intended[1] not in chosen or not _touch(intended[0], intended[1])
        ):
            continue
        if mode == "overlap" and (
            intended[1] not in candidates or intended[1] in chosen
            or not _overlap(intended[0], intended[1])
        ):
            continue
        if regime == "source_train" and index == 0 and overlap_enabled:
            if not any(_side(r) == 2 for r in chosen):
                continue
        if regime == "source_train" and index == 1 and max_side not in (
            _side(r) for r in chosen
        ):
            continue
        palette = rng.choice(blockers, size=height * width).reshape(height, width)
        grid = palette.astype(np.int8)
        grid[mask] = background
        output = _make_output(grid, background, highlight)
        return {"input": grid, "output": output}
    raise RuntimeError("pair construction exhausted bounded attempts")


def _key(grid):
    return (grid.shape, grid.tobytes())


def _roles(puzzle):
    changed = [(p["input"], p["output"]) for p in puzzle["train"]
               if np.any(p["input"] != p["output"])]
    assert changed, "training does not show a highlight"
    from_colors = set()
    to_colors = set()
    for before, after in changed:
        difference = before != after
        from_colors.update(map(int, np.unique(before[difference])))
        to_colors.update(map(int, np.unique(after[difference])))
    assert len(from_colors) == len(to_colors) == 1, "inconsistent color roles"
    background, highlight = from_colors.pop(), to_colors.pop()
    assert background != highlight
    return background, highlight


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        before, after = pair["input"], pair["output"]
        for grid in (before, after):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= size <= 30 for size in grid.shape)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        assert before.shape == after.shape and before.dtype == after.dtype
        key = _key(before)
        assert key not in seen, "duplicate input"
        seen.add(key)
        assert not any(np.array_equal(before, source) for source in _SOURCE_INPUTS
                       if before.shape == source.shape), "source copy"
    background, highlight = _roles(puzzle)
    training_blockers = set()
    train_sizes = set()
    train_has_overlap = False
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            before, after = pair["input"], pair["output"]
            input_colors = set(map(int, np.unique(before)))
            assert background in input_colors and highlight not in input_colors
            blockers = input_colors - {background}
            assert blockers, "blocker role absent"
            if split == "train":
                training_blockers |= blockers
            else:
                assert blockers <= training_blockers, "test introduces blocker color"
            candidates = _candidates(before == background)
            assert _unambiguous(candidates), "unresolved square conflict"
            selected = _selected(candidates)
            assert selected, "missing selected square"
            assert np.array_equal(after, _make_output(before, background, highlight))
            assert np.any(before != after)
            overlaps = any(_overlap(a, b) for i, a in enumerate(candidates)
                           for b in candidates[i + 1:])
            if split == "train":
                train_sizes.update(map(_side, selected))
                train_has_overlap |= overlaps
            else:
                assert set(map(_side, selected)) <= train_sizes, "unshown square size"
                assert not overlaps or train_has_overlap, "overlap absent from training"
            assert not any((after != before)[before != background])
    assert 2 in train_sizes and len(train_sizes) >= 2, "training lacks size variation"


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    colors = rng.choice(10, size=10, replace=False).tolist()
    background, highlight = int(colors[0]), int(colors[1])
    pool_size = int(rng.choice([1, 2, 3], p=[0.43, 0.35, 0.22]))
    blocker_pool = [int(c) for c in colors[2:2 + pool_size]]
    max_side = int(rng.choice([3, 4, 5, 6], p=[0.24, 0.30, 0.29, 0.17]))
    overlap_enabled = bool(rng.random() < 0.57)
    densities = ["sparse", "dense"] + [
        str(rng.choice(["sparse", "middle", "dense"]))
        for _ in range(train_count - 2)
    ]
    rng.shuffle(densities)
    puzzle = {"train": [], "test": []}
    used = set()
    demonstrated_sides = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            profile = densities[index] if split == "train" else str(
                rng.choice(["sparse", "middle", "dense"], p=[0.2, 0.3, 0.5]))
            lo, hi = {"sparse": (0.22, 0.38), "middle": (0.39, 0.54),
                      "dense": (0.55, 0.70)}[profile]
            density = float(rng.uniform(lo, hi))
            if split == "train" and index == 0:
                chosen_blockers = blocker_pool
            elif split == "train" and len(blocker_pool) > 1:
                chosen_blockers = [blocker_pool[index % len(blocker_pool)]]
            else:
                n = int(rng.integers(1, len(blocker_pool) + 1))
                chosen_blockers = list(rng.choice(blocker_pool, size=n, replace=False))
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _pair(pair_rng, background, highlight, chosen_blockers,
                             max_side, "source_train" if split == "train"
                             else "target_test", index, overlap_enabled, density,
                             None if split == "train" else demonstrated_sides)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        demonstrated_sides.update(map(
                            _side, _selected(_candidates(pair["input"] == background))))
                    break
            else:
                raise RuntimeError("could not make unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(np.array_equal(x[k], y[k])
               for split in ("train", "test")
               for x, y in zip(a[split], b[split])
               for k in ("input", "output"))


def self_test() -> None:
    for source, answer in zip(_SOURCE_INPUTS, _SOURCE_OUTPUTS):
        assert np.array_equal(_make_output(source, 0, 1), answer)
    observed_train, observed_test = set(), set()
    dimensions, sizes, densities, touching, overlaps = set(), set(), set(), 0, 0
    sparse = dense = small = large = elongated = few = many = 0
    multicolor = nonsquare_overlap = 0
    within_density_spread = 0
    train_touch_test_separate = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        observed_train.add(len(puzzle["train"]))
        observed_test.add(len(puzzle["test"]))
        signatures.add(tuple(_key(pair["input"]) for pair in
                             puzzle["train"] + puzzle["test"]))
        background, highlight = _roles(puzzle)
        puzzle_densities = []
        train_touch = False
        test_separate = False
        for pair in puzzle["train"] + puzzle["test"]:
            before = pair["input"]
            dimensions.add(before.shape)
            density = float(np.mean(before == background))
            puzzle_densities.append(density)
            densities.add(round(density, 1))
            sparse += density < 0.40
            dense += density >= 0.59
            small += max(before.shape) <= 13
            large += 30 in before.shape
            elongated += max(before.shape) / min(before.shape) >= 1.7
            multicolor += len(np.unique(before)) >= 3
            candidates = _candidates(before == background)
            chosen = _selected(candidates)
            sizes.update(map(_side, chosen))
            few += len(chosen) <= 2
            many += len(chosen) >= 6
            is_touching = any(_touch(a, b) for i, a in enumerate(chosen)
                              for b in chosen[i + 1:])
            touching += int(is_touching)
            if any(pair is training_pair for training_pair in puzzle["train"]):
                train_touch |= is_touching
            else:
                test_separate |= not is_touching
            overlaps += int(any(_overlap(a, b) for i, a in enumerate(candidates)
                                for b in candidates[i + 1:]))
            nonsquares = [r for r in _maximal_rectangles(before == background)
                          if r[2] - r[0] != r[3] - r[1]]
            nonsquare_overlap += int(any(_overlap(a, b) for a in nonsquares
                                         for b in chosen))
            assert np.array_equal(pair["output"],
                                  _make_output(before, background, highlight))
        within_density_spread += max(puzzle_densities) - min(puzzle_densities) >= 0.15
        train_touch_test_separate += train_touch and test_separate
    assert len(observed_train) >= 4 and len(observed_test) >= 3
    assert len(signatures) == 42 and len(dimensions) >= 30
    assert {2, 3, 4, 5} <= sizes and len(densities) >= 5
    assert touching >= 8 and overlaps >= 5
    assert sparse >= 20 and dense >= 20
    assert small >= 20 and large >= 1 and elongated >= 8
    assert few >= 10 and many >= 10
    assert multicolor >= 5 and nonsquare_overlap >= 20
    assert within_density_spread >= 35 and train_touch_test_separate >= 5
    for n, m in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2468, n, m)
        validate_generated_puzzle(puzzle)
        if n is not None:
            assert len(puzzle["train"]) == n
        if m is not None:
            assert len(puzzle["test"]) == m
    for n, m in ((1, 1), (11, 1), (2, 0), (6, 5),
                 (2, 2), (2, 5), (4, 4)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
