"""Generate two-object ARC puzzles with translation to first colored-cell contact."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "85a968c14e6d0a59761b20fa19d9ed8dd985421ddda8e6a9657260b7d21f9cd6",
    "17014a120c29dd6fd05fe436e86a2291944c6f723890522c85d1bca5d186e742",
    "fa31314a9449362672064210cba4be0c006ab4ce7d2e04965e4f89636ef7d94a",
    "54c763356b1d3b940f8a420a87cc3800e2fe7c1aef7659d975515960fbec9f2b",
    "64ff22f9170dc1abcba21109d93db00a4ad6f421075362e48499b347fc769902",
}
_DIRECTIONS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _border(h, w):
    a = np.zeros((h, w), bool)
    a[0] = a[-1] = True
    a[:, 0] = a[:, -1] = True
    return a


def _local_mask(cells):
    low = cells.min(axis=0)
    a = np.zeros(tuple(cells.max(axis=0) - low + 1), bool)
    a[tuple((cells - low).T)] = True
    return a


def _mover_family(a):
    h, w = a.shape
    if a.all():
        return "bar" if min(h, w) == 1 else "solid"
    if np.array_equal(a, _border(h, w)):
        return "frame"
    for turns in range(4):
        b = np.rot90(a, turns)
        width = b.shape[1]
        if ((b[0].all() or b[-1].all()) and
                (b[:, 0].all() or b[:, -1].all())):
            return "ell"
        if ((b[0].all() or b[-1].all()) and
                any(b[:, x].all() for x in range(1, width - 1))):
            return "tee"
    raise AssertionError("unknown movable shape")


def _step_shape(h, w, bend_y, bend_x, mirror):
    a = np.zeros((h, w), bool)
    a[:, 0] = True
    a[0, :bend_x + 1] = True
    a[1:bend_y + 1, bend_x] = True
    a[bend_y, bend_x:] = True
    a[bend_y + 1:, -1] = True
    a[-1] = True
    return a[:, ::-1].copy() if mirror else a


def _target_family(a, direction):
    a = np.rot90(a, -_DIRECTIONS.index(direction))
    h, w = a.shape
    if np.array_equal(a, _border(h, w)):
        return "frame"
    u = np.zeros((h, w), bool)
    u[0] = u[-1] = True
    u[:, -1] = True
    if np.array_equal(a, u):
        return "u"
    v = np.zeros((h, w), bool)
    v[0] = True
    for y in range(1, h):
        x = round(y * (w // 2) / (h - 1))
        v[y, x] = v[y, w - 1 - x] = True
    if np.array_equal(a, v):
        return "v"
    for bend_y in range(2, h - 2):
        for bend_x in range(2, w - 2):
            for mirror in (False, True):
                if np.array_equal(a, _step_shape(h, w, bend_y, bend_x, mirror)):
                    return "step"
    raise AssertionError("unknown stationary shape")


def _mover(rng, kind):
    if kind == "solid":
        return np.ones((int(rng.integers(2, 6)), int(rng.integers(2, 6))), bool)
    if kind == "frame":
        return _border(int(rng.integers(3, 7)), int(rng.integers(3, 7)))
    if kind == "bar":
        length = int(rng.integers(2, 8))
        return np.ones((1, length) if rng.integers(2) else (length, 1), bool)
    if kind == "ell":
        h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        a = np.zeros((h, w), bool)
        a[:, 0 if rng.integers(2) else -1] = True
        a[0 if rng.integers(2) else -1, :] = True
        return a
    h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    a = np.zeros((h, w), bool)
    a[0 if rng.integers(2) else -1, :] = True
    a[:, int(rng.integers(1, w - 1))] = True
    return a


def _target(rng, style, small_h):
    if style == "u":
        h = int(rng.integers(max(7, small_h + 4), 14))
        w = int(rng.integers(5, 13))
        a = np.zeros((h, w), bool)
        a[0] = a[-1] = True
        a[:, -1] = True
        return a
    if style == "frame":
        return _border(int(rng.integers(5, 14)), int(rng.integers(5, 14)))
    if style == "v":
        h = int(rng.integers(6, 13))
        w = int(rng.choice((7, 9, 11, 13)))
        a = np.zeros((h, w), bool)
        a[0] = True
        for y in range(1, h):
            x = round(y * (w // 2) / (h - 1))
            a[y, x] = a[y, w - 1 - x] = True
        return a
    h, w = int(rng.integers(6, 14)), int(rng.integers(6, 14))
    bend_y = int(rng.integers(2, h - 2))
    bend_x = int(rng.integers(2, w - 2))
    return _step_shape(h, w, bend_y, bend_x, bool(rng.integers(2)))


def _components(grid, background):
    occupied = grid != background
    visited = np.zeros(grid.shape, bool)
    result = []
    for y, x in np.argwhere(occupied):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        cells = []
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]
                        and occupied[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        result.append(np.asarray(cells, dtype=np.int16))
    return result


def _analyze(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    parts = _components(grid, background)
    if len(parts) != 2:
        raise AssertionError("input needs two connected figures")
    parts.sort(key=len)
    small, big = parts
    if len(small) >= len(big) or len(small) * 3 > len(big) * 2:
        raise AssertionError("figure sizes do not distinguish roles")
    for cells in parts:
        if len(np.unique(grid[tuple(cells.T)])) != 1:
            raise AssertionError("each figure needs one color")
    sy0, sx0 = small.min(axis=0)
    sy1, sx1 = small.max(axis=0)
    by0, bx0 = big.min(axis=0)
    by1, bx1 = big.max(axis=0)
    if sx1 + 1 < bx0 and sy0 <= by1 and by0 <= sy1:
        direction = (0, 1)
        gap = int(bx0 - sx1 - 1)
    elif bx1 + 1 < sx0 and sy0 <= by1 and by0 <= sy1:
        direction = (0, -1)
        gap = int(sx0 - bx1 - 1)
    elif sy1 + 1 < by0 and sx0 <= bx1 and bx0 <= sx1:
        direction = (1, 0)
        gap = int(by0 - sy1 - 1)
    elif by1 + 1 < sy0 and sx0 <= bx1 and bx0 <= sx1:
        direction = (-1, 0)
        gap = int(sy0 - by1 - 1)
    else:
        raise AssertionError("figures need one clear axial gap")
    return background, small, big, direction, gap


def _predict(grid):
    background, small, big, direction, gap = _analyze(grid)
    fixed = set(map(tuple, big))
    delta = np.asarray(direction)
    for travel in range(1, max(grid.shape) + 1):
        moved = small + travel * delta
        if np.any(moved < 0) or np.any(moved >= np.asarray(grid.shape)):
            break
        if any(tuple(c) in fixed for c in moved):
            break
        if any((int(y + dy), int(x + dx)) in fixed
               for y, x in moved for dy, dx in _DIRECTIONS):
            output = grid.copy()
            color = int(grid[tuple(small[0])])
            output[tuple(small.T)] = background
            output[tuple(moved.T)] = color
            return output, travel, gap
    raise AssertionError("no reachable first contact")


def _digest(pair):
    hash_value = hashlib.sha256()
    for field in ("input", "output"):
        a = pair[field]
        hash_value.update(bytes(a.shape))
        hash_value.update(np.asarray(a, dtype=np.int8).tobytes())
    return hash_value.hexdigest()


def _make_pair(rng, style, direction, palette, same_color, require_recess):
    for _ in range(500):
        small = _mover(rng, rng.choice(("solid", "frame", "bar", "ell", "tee")))
        sh, sw = small.shape
        big = _target(rng, style, sh)
        bh, bw = big.shape
        if small.sum() * 3 > big.sum() * 2 or small.size >= big.size:
            continue
        if style == "u":
            if bh - sh < 4:
                continue
            offset = int(rng.integers(2, bh - sh - 1))
        else:
            offset = int(rng.integers(-sh + 1, bh))
        gap = int(rng.integers(1, 9))
        margin_limit = 2 if rng.random() < 0.28 else 6
        top, bottom, left, right = map(int, rng.integers(0, margin_limit, size=4))
        by = top + max(0, -offset)
        sy = by + offset
        sx = left
        bx = sx + sw + gap
        h = max(sy + sh, by + bh) + bottom
        w = bx + bw + right
        if h > 30 or w > 30 or h < 4 or w < 4:
            continue
        background = int(rng.choice(palette))
        objects = [int(c) for c in palette if c != background]
        small_color = int(rng.choice(objects))
        big_color = small_color if same_color else int(rng.choice([c for c in objects if c != small_color]))
        grid = np.full((h, w), background, dtype=np.int8)
        grid[sy:sy + sh, sx:sx + sw][small] = small_color
        grid[by:by + bh, bx:bx + bw][big] = big_color
        turns = _DIRECTIONS.index(direction)
        # rot90 maps canonical right to up, left, then down.
        grid = np.rot90(grid, turns).copy()
        try:
            result, travel, box_gap = _predict(grid)
            _, a, b, inferred_direction, _ = _analyze(grid)
        except AssertionError:
            continue
        if inferred_direction != direction or len(a) >= len(b):
            continue
        if bool(travel > box_gap) != require_recess:
            continue
        if _digest({"input": grid, "output": result}) in _SOURCE_DIGESTS:
            continue
        return {"input": grid, "output": result}
    raise RuntimeError("could not construct a clear contact pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError("illegal pair count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("training pairs must outnumber test pairs")

    # The puzzle's only shared convention is the geometric rule. A common
    # palette and curriculum are sampled here; colors have no fixed roles.
    palette = (0,) + tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False)))
    curriculum = str(rng.choice(("simple", "mixed", "recessed")))
    train_styles = ["frame", "u" if curriculum == "recessed" else
                    str(rng.choice(("v", "step"))) if curriculum == "mixed" else "frame"]
    train_styles += [str(rng.choice(("frame", "v", "step", "u")
                                     if curriculum == "recessed" else
                                     ("frame", "v", "step")))
                     for _ in range(num_train - 2)]
    test_styles = (["u"] if curriculum == "recessed" else
                   [str(rng.choice(("v", "step")))]) + [
        str(rng.choice(("frame", "v", "step", "u")
                       if curriculum == "recessed" else ("frame", "v", "step")))
        for _ in range(num_test - 1)]
    if curriculum == "simple" and rng.integers(2):
        train_directions = [(0, 1), (1, 0)]
    else:
        chosen = rng.choice(4, 2, replace=False)
        train_directions = [_DIRECTIONS[int(i)] for i in chosen]
    train_directions += [_DIRECTIONS[int(rng.integers(4))]
                         for _ in range(num_train - 2)]
    test_directions = [_DIRECTIONS[int(rng.integers(4))] for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for bucket, styles, directions in (("train", train_styles, train_directions),
                                       ("test", test_styles, test_directions)):
        for index, (style, direction) in enumerate(zip(styles, directions)):
            same = index == 1 if bucket == "train" else bool(rng.random() < 0.38)
            if bucket == "train" and index >= 2:
                same = bool(rng.random() < 0.38)
            for _ in range(20):
                pair = _make_pair(rng, style, direction, palette, same,
                                  style == "u")
                if (bucket == "train" and index == 1
                        and _analyze(pair["input"])[0] ==
                        _analyze(puzzle["train"][0]["input"])[0]):
                    continue
                if bucket == "train" and index == 1:
                    _, previous_small, _, _, _ = _analyze(puzzle["train"][0]["input"])
                    _, current_small, _, _, _ = _analyze(pair["input"])
                    if (_mover_family(_local_mask(previous_small)) ==
                            _mover_family(_local_mask(current_small))):
                        continue
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[bucket].append(pair)
                    break
            else:
                raise RuntimeError("could not find unique input")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert all(type(puzzle[k]) is list for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_directions, train_depths, test_depths = set(), [], []
    train_families, train_movers = set(), set()
    test_families = set()
    train_equal, train_distinct = False, False
    for bucket in ("train", "test"):
        for pair in puzzle[bucket]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = (a.shape, a.tobytes())
            pair_key = (key, b.tobytes())
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert _digest(pair) not in _SOURCE_DIGESTS
            background, small, big, direction, gap = _analyze(a)
            mover_family = _mover_family(_local_mask(small))
            target_family = _target_family(_local_mask(big), direction)
            predicted, travel, measured_gap = _predict(a)
            assert np.array_equal(predicted, b) and gap == measured_gap
            assert 1 <= gap <= 8 and travel >= gap
            assert np.count_nonzero(a != background) == np.count_nonzero(b != background)
            assert set(np.unique(a)) == set(np.unique(b))
            depth = travel - gap
            if bucket == "train":
                train_directions.add(direction)
                train_depths.append(depth)
                train_families.add(target_family)
                train_movers.add(mover_family)
                equal = a[tuple(small[0])] == a[tuple(big[0])]
                train_equal |= bool(equal)
                train_distinct |= not bool(equal)
            else:
                test_depths.append(depth)
                test_families.add(target_family)
    assert len(train_directions) >= 2
    assert train_equal and train_distinct
    assert len({_analyze(p["input"])[0] for p in puzzle["train"]}) >= 2
    assert len(train_movers) >= 2 and "frame" in train_families
    assert bool(test_families & {"v", "step", "u"})
    assert ("u" not in test_families or "u" in train_families)
    assert not any(test_depths) or any(train_depths)


def _same_puzzle(first, second):
    return all(len(first[b]) == len(second[b]) and all(
        np.array_equal(p1[k], p2[k])
        for p1, p2 in zip(first[b], second[b]) for k in ("input", "output"))
        for b in ("train", "test"))


def self_test() -> None:
    counts, backgrounds, directions, dimensions = set(), set(), set(), set()
    deep_train = deep_test = same_test = 0
    partial = gap_one = compact = spacious = 0
    train_styles, test_styles, mover_families = set(), set(), set()
    direction_mover, target_direction, step_mirrors = set(), set(), set()
    mover_sizes, target_sizes = set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for b in ("train", "test") for p in puzzle[b]))
        for bucket in ("train", "test"):
            for pair in puzzle[bucket]:
                grid = pair["input"]
                bg, small, big, direction, gap = _analyze(grid)
                output, travel, _ = _predict(grid)
                assert np.array_equal(output, pair["output"])
                backgrounds.add(bg)
                directions.add(direction)
                dimensions.add(grid.shape)
                small_mask, big_mask = _local_mask(small), _local_mask(big)
                mover = _mover_family(small_mask)
                style = _target_family(big_mask, direction)
                (train_styles if bucket == "train" else test_styles).add(style)
                mover_families.add(mover)
                direction_mover.add((direction, mover))
                target_direction.add((direction, style))
                mover_sizes.add(small_mask.shape)
                target_sizes.add(big_mask.shape)
                if style == "step":
                    canonical = np.rot90(big_mask, -_DIRECTIONS.index(direction))
                    step_mirrors.add(bool(canonical[:, -1].all()))
                axis = 0 if direction[0] == 0 else 1
                partial += (small[:, axis].min() < big[:, axis].min()
                            or small[:, axis].max() > big[:, axis].max())
                gap_one += gap == 1
                compact += min(grid.shape) <= 8
                spacious += max(grid.shape) >= 25
                if travel > gap:
                    if bucket == "train":
                        deep_train += 1
                    else:
                        deep_test += 1
                if bucket == "test" and grid[tuple(small[0])] == grid[tuple(big[0])]:
                    same_test += 1
    assert len({x for x, _ in counts}) >= 4
    assert len({y for _, y in counts}) >= 3
    assert len(signatures) == 42 and len(dimensions) >= 35
    assert len(backgrounds) >= 5 and len(directions) == 4
    assert deep_train >= 5 and deep_test >= 2 and same_test >= 5
    assert train_styles == test_styles == {"frame", "v", "step", "u"}
    assert mover_families == {"solid", "frame", "bar", "ell", "tee"}
    assert len(direction_mover) == 20 and len(target_direction) == 16
    assert step_mirrors == {False, True}
    assert len(mover_sizes) >= 20 and len(target_sizes) >= 35
    assert partial >= 30 and gap_one >= 10
    assert compact >= 8 and spacious >= 30
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        p = generate_puzzle(123, nt, ns)
        validate_generated_puzzle(p)
        assert (nt is None or len(p["train"]) == nt)
        assert (ns is None or len(p["test"]) == ns)
        assert _same_puzzle(p, generate_puzzle(123, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    broken = generate_puzzle(9, 2, 1)
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted incorrect output")
