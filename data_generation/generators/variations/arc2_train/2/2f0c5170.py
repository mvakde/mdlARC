import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "ecfec832effe86795cc8e973d4c8a7a319dd2cfc6acbd05671813a2555990260",
    "04d91f6a6ca5721c8ead732cf38331040dfda6176ac0ae304ba33dc64aabfa54",
    "d09d651b1317f21f075540ea3eaa989814c87939f242289bfd90dcb85bc8b256",
    "3b2e06434607517c1e7ad6a52f69f2498f996e22d5403e4615a437a502c55932",
}
_LAYOUTS = ("NW", "NE", "SW", "SE", "L", "R", "U", "D")
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _count(value, low, high, name):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _digest(pair):
    h = hashlib.sha256()
    for part in ("input", "output"):
        grid = pair[part]
        h.update(bytes(grid.shape))
        h.update(grid.astype(np.uint8).tobytes())
    return h.hexdigest()


def _connected(cells):
    pending = [next(iter(cells))]
    seen = {pending[0]}
    while pending:
        r, c = pending.pop()
        for dr, dc in _NEIGHBORS:
            q = (r + dr, c + dc)
            if q in cells and q not in seen:
                seen.add(q)
                pending.append(q)
    return len(seen) == len(cells)


def _motif(rng, source_size, regime):
    sh, sw = source_size
    inset = sh >= 5 and sw >= 5 and rng.random() < 0.36
    lo_r, hi_r = (1, sh - 2) if inset else (0, sh - 1)
    lo_c, hi_c = (1, sw - 2) if inset else (0, sw - 1)
    capacity = (hi_r - lo_r + 1) * (hi_c - lo_c + 1)
    minimum = 7
    maximum = min(16, capacity - 1, sh * sw - 4)
    if maximum < minimum:
        return None
    if regime == "target_test" and rng.random() < 0.55:
        minimum = min(9, maximum)
    size = int(rng.integers(minimum, maximum + 1))

    touch = not inset and rng.random() < 0.68
    if touch:
        side = int(rng.integers(4))
        if side == 0:
            start = (lo_r, int(rng.integers(lo_c, hi_c + 1)))
        elif side == 1:
            start = (hi_r, int(rng.integers(lo_c, hi_c + 1)))
        elif side == 2:
            start = (int(rng.integers(lo_r, hi_r + 1)), lo_c)
        else:
            start = (int(rng.integers(lo_r, hi_r + 1)), hi_c)
    else:
        start = (
            int(rng.integers(lo_r, hi_r + 1)),
            int(rng.integers(lo_c, hi_c + 1)),
        )

    cells = {start}
    growth = int(rng.integers(5))
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for dr, dc in _NEIGHBORS:
                q = (r + dr, c + dc)
                if lo_r <= q[0] <= hi_r and lo_c <= q[1] <= hi_c and q not in cells:
                    frontier.add(q)
        frontier = sorted(frontier)
        weights = []
        for r, c in frontier:
            adjacent = sum((r + dr, c + dc) in cells for dr, dc in _NEIGHBORS)
            if growth == 0:
                weight = float(adjacent ** 2)
            elif growth == 1:
                weight = 1.0 / adjacent
            elif growth == 2:
                weight = 1.0 + 2.0 * abs(c - start[1])
            elif growth == 3:
                weight = 1.0 + 2.0 * abs(r - start[0])
            else:
                weight = 1.0
            weights.append(weight)
        probabilities = np.asarray(weights) / sum(weights)
        cells.add(frontier[int(rng.choice(len(frontier), p=probabilities))])

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    rmin, rmax, cmin, cmax = min(rows), max(rows), min(cols), max(cols)
    tips = [
        q for q in cells
        if sum((q[0] + dr, q[1] + dc) in cells for dr, dc in _NEIGHBORS) == 1
    ]
    edge = [
        q for q in cells
        if q[0] in (rmin, rmax) or q[1] in (cmin, cmax)
    ]
    interior = [q for q in cells if q not in edge]
    groups = [group for group in (tips, edge, interior) if group]
    group = groups[int(rng.integers(len(groups)))]
    anchor = sorted(group)[int(rng.integers(len(group)))]
    return cells, anchor


def _placement(rng, cells, source_anchor, dest_size, want_crop):
    dh, dw = dest_size
    candidates = {}
    n = len(cells)
    for row in range(dh):
        for col in range(dw):
            clipped = sum(
                not (0 <= row + r - source_anchor[0] < dh
                     and 0 <= col + c - source_anchor[1] < dw)
                for r, c in cells
            )
            if want_crop:
                if not 1 <= clipped <= max(2, n // 3):
                    continue
            elif clipped:
                continue
            dr, dc = row - source_anchor[0], col - source_anchor[1]
            direction = (
                "0" if dr == dc == 0 else
                ("N" if dr < 0 else "S" if dr > 0 else "")
                + ("W" if dc < 0 else "E" if dc > 0 else "")
            )
            candidates.setdefault(direction, []).append((row, col))
    if not candidates:
        return None
    directions = sorted(candidates)
    weights = np.asarray([0.35 if d == "0" else 1.0 for d in directions])
    weights /= weights.sum()
    direction = directions[int(rng.choice(len(directions), p=weights))]
    group = candidates[direction]
    return group[int(rng.integers(len(group)))]


def _layout(rng, source_size, dest_size, direction):
    sh, sw = source_size
    dh, dw = dest_size
    rgap, cgap = [int(v) for v in rng.integers(1, 5, size=2)]
    if direction in ("NW", "NE", "SW", "SE"):
        source_above = direction[0] == "N"
        source_left = direction[1] == "W"
        source_at = (
            0 if source_above else dh + rgap,
            0 if source_left else dw + cgap,
        )
        dest_at = (
            sh + rgap if source_above else 0,
            sw + cgap if source_left else 0,
        )
    elif direction in ("L", "R"):
        delta = int(rng.integers(-4, 5))
        source_at = (max(0, delta), 0 if direction == "L" else dw + cgap)
        dest_at = (max(0, -delta), sw + cgap if direction == "L" else 0)
    else:
        delta = int(rng.integers(-4, 5))
        source_at = (0 if direction == "U" else dh + rgap, max(0, delta))
        dest_at = (sh + rgap if direction == "U" else 0, max(0, -delta))

    top, bottom, left, right = [int(v) for v in rng.integers(1, 4, size=4)]
    sr, sc = source_at
    dr, dc = dest_at
    height = max(sr + sh, dr + dh) + top + bottom
    width = max(sc + sw, dc + dw) + left + right
    if height > 30 or width > 30:
        return None
    return (height, width), (sr + top, sc + left), (dr + top, dc + left)


def _make_pair(rng, regime, palette, direction, want_crop):
    exterior, background, carrier, marker = palette
    for _ in range(160):
        source_size = (
            int(rng.integers(4, 11)),
            int(rng.integers(4, 11)),
        )
        result = _motif(rng, source_size, regime)
        if result is None:
            continue
        cells, source_anchor = result
        dest_size = (
            int(rng.integers(4, 12)),
            int(rng.integers(4, 12)),
        )
        dest_anchor = _placement(rng, cells, source_anchor, dest_size, want_crop)
        if dest_anchor is None:
            continue
        layout = _layout(rng, source_size, dest_size, direction)
        if layout is None:
            continue
        grid_size, source_at, dest_at = layout
        source = np.full(source_size, background, dtype=np.int64)
        for r, c in cells:
            source[r, c] = marker if (r, c) == source_anchor else carrier
        destination = np.full(dest_size, background, dtype=np.int64)
        destination[dest_anchor] = marker
        grid = np.full(grid_size, exterior, dtype=np.int64)
        sr, sc = source_at
        dr, dc = dest_at
        grid[sr:sr + source_size[0], sc:sc + source_size[1]] = source
        grid[dr:dr + dest_size[0], dc:dc + dest_size[1]] = destination
        output = np.full(dest_size, background, dtype=np.int64)
        for r, c in cells:
            yy = dest_anchor[0] + r - source_anchor[0]
            xx = dest_anchor[1] + c - source_anchor[1]
            if 0 <= yy < dest_size[0] and 0 <= xx < dest_size[1]:
                output[yy, xx] = source[r, c]
        pair = {"input": grid, "output": output}
        if _digest(pair) not in _SOURCE_DIGESTS:
            return pair
    raise RuntimeError("could not construct a valid pair")


def _rectangles(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    boxes = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        pending = [(r, c)]
        seen[r, c] = True
        points = []
        while pending:
            y, x = pending.pop()
            points.append((y, x))
            for dy, dx in _NEIGHBORS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                    and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    pending.append((yy, xx))
        r0 = min(y for y, _ in points)
        r1 = max(y for y, _ in points)
        c0 = min(x for _, x in points)
        c1 = max(x for _, x in points)
        assert len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)
        boxes.append((r0, r1, c0, c1))
    return boxes


def _inspect(pair):
    grid, output = pair["input"], pair["output"]
    exterior = int(grid[0, 0])
    assert np.all(grid[0] == exterior)
    assert np.all(grid[-1] == exterior)
    assert np.all(grid[:, 0] == exterior)
    assert np.all(grid[:, -1] == exterior)
    boxes = _rectangles(grid != exterior)
    assert len(boxes) == 2
    panels = [grid[r0:r1 + 1, c0:c1 + 1] for r0, r1, c0, c1 in boxes]
    sets = [set(map(int, np.unique(panel))) for panel in panels]
    assert sorted(map(len, sets)) == [2, 3]
    si = 0 if len(sets[0]) == 3 else 1
    di = 1 - si
    source, dest = panels[si], panels[di]
    sbox, dbox = boxes[si], boxes[di]
    assert sets[di] < sets[si] and exterior not in sets[si]
    candidates = [v for v in sets[di] if np.count_nonzero(dest == v) == 1
                  and np.count_nonzero(source == v) == 1]
    assert len(candidates) == 1
    marker = candidates[0]
    background = next(iter(sets[di] - {marker}))
    carrier = next(iter(sets[si] - sets[di]))
    assert np.count_nonzero(source == carrier) >= 6
    cells = {tuple(map(int, q)) for q in np.argwhere(source != background)}
    assert _connected(cells)
    sa = tuple(map(int, np.argwhere(source == marker)[0]))
    da = tuple(map(int, np.argwhere(dest == marker)[0]))
    expected = np.full(dest.shape, background, dtype=np.int64)
    clipped = 0
    for r, c in cells:
        yy, xx = da[0] + r - sa[0], da[1] + c - sa[1]
        if 0 <= yy < dest.shape[0] and 0 <= xx < dest.shape[1]:
            expected[yy, xx] = source[r, c]
        else:
            clipped += 1
    assert output.shape == dest.shape
    assert np.array_equal(output, expected)
    assert np.count_nonzero(output == carrier) >= 4
    assert _digest(pair) not in _SOURCE_DIGESTS
    sr = -1 if sbox[1] < dbox[0] else 1 if dbox[1] < sbox[0] else 0
    sc = -1 if sbox[3] < dbox[2] else 1 if dbox[3] < sbox[2] else 0
    assert (sr, sc) != (0, 0)
    assert abs(sbox[0] - dbox[0]) + abs(sbox[2] - dbox[2]) > 0
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    edge = min(rows) == 0 or max(rows) == source.shape[0] - 1 or min(cols) == 0 or max(cols) == source.shape[1] - 1
    degrees = {
        q: sum((q[0] + dr, q[1] + dc) in cells for dr, dc in _NEIGHBORS)
        for q in cells
    }
    tip = degrees[sa] == 1
    interior = min(rows) < sa[0] < max(rows) and min(cols) < sa[1] < max(cols)
    return {
        "marker": marker, "base": (exterior, background, carrier),
        "layout": (sr, sc), "crop": clipped, "size": len(cells),
        "source_shape": source.shape, "dest_shape": dest.shape,
        "span": (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1),
        "tips_count": sum(degree == 1 for degree in degrees.values()),
        "branch_count": sum(degree >= 3 for degree in degrees.values()),
        "density": len(cells) / ((max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)),
        "move": (da[0] - sa[0], da[1] - sa[1]),
        "source_edge": edge, "tip": tip, "interior": interior,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs = set()
    seen_pairs = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = (pair["input"].shape, pair["input"].tobytes())
            pkey = (ikey, pair["output"].shape, pair["output"].tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert not np.array_equal(pair["input"], pair["output"])
            info[split].append(_inspect(pair))
    train, test = info["train"], info["test"]
    assert len({m["marker"] for m in train}) >= 2
    assert any(m["crop"] == 0 for m in train)
    assert any(m["move"] != (0, 0) for m in train)
    if any(m["crop"] for m in test):
        assert any(m["crop"] for m in train)
    assert len({m["marker"] for m in train + test}) < ntrain + ntest


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _count(num_train, 2, 10, "num_train")
    _count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))

    # Literal marker colors may change; the shared semantic roles do not.
    marker_pool = [int(v) for v in rng.choice(10, size=int(rng.integers(3, 6)), replace=False)]
    base_pool = [v for v in range(10) if v not in marker_pool]
    stable_base = rng.random() < 0.72
    base = tuple(map(int, rng.choice(base_pool, size=3, replace=False)))
    markers = [marker_pool[0], marker_pool[1]]
    if num_train >= 3:
        markers.append(marker_pool[0])
    while len(markers) < num_train + num_test:
        markers.append(marker_pool[int(rng.integers(len(marker_pool)))])
    if num_train == 2:
        markers[2] = marker_pool[0]
    # With larger puzzles the repeat can occur at several training positions.
    if num_train > 3:
        j = int(rng.integers(2, num_train))
        markers[2], markers[j] = markers[j], markers[2]

    allow_crop = rng.random() < 0.55
    train_crops = [False] * num_train
    if allow_crop:
        train_crops[int(rng.integers(num_train))] = True
        for i in range(num_train):
            if not train_crops[i] and rng.random() < 0.14:
                train_crops[i] = True
        if all(train_crops):
            train_crops[0] = False
    test_crops = [allow_crop and rng.random() < 0.28 for _ in range(num_test)]

    familiar = rng.random() < 0.19
    if familiar:
        direction = _LAYOUTS[int(rng.integers(4))]
        layouts = [direction] * (num_train + num_test)
    else:
        layouts = [_LAYOUTS[int(rng.integers(len(_LAYOUTS)))] for _ in range(num_train)]
        for _ in range(num_test):
            layouts.append(
                layouts[int(rng.integers(num_train))]
                if rng.random() < 0.52
                else _LAYOUTS[int(rng.integers(len(_LAYOUTS)))]
            )

    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(num_train + num_test):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        crop = train_crops[local_index] if split == "train" else test_crops[local_index]
        for _ in range(50):
            if stable_base:
                pair_base = base
            else:
                pair_base = tuple(map(int, rng.choice(base_pool, size=3, replace=False)))
            palette = (*pair_base, markers[index])
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                              palette, layouts[index], crop)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(left, right):
    return all(
        len(left[split]) == len(right[split])
        and all(np.array_equal(a[part], b[part])
                for a, b in zip(left[split], right[split])
                for part in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert all(_same(p, generate_puzzle(seed)) for seed, p in enumerate(puzzles))
    signatures = {
        (p["train"][0]["input"].shape, p["train"][0]["input"].tobytes())
        for p in puzzles
    }
    assert len(signatures) >= 40
    metas = [
        [(_inspect(pair), split) for split in ("train", "test") for pair in p[split]]
        for p in puzzles
    ]
    all_meta = [m for ms in metas for m, _ in ms]
    assert len({m["layout"] for m in all_meta}) == 8
    assert {m["crop"] == 0 for m in all_meta} == {True, False}
    assert {m["source_edge"] for m in all_meta} == {True, False}
    assert {m["tip"] for m in all_meta} == {True, False}
    assert {m["interior"] for m in all_meta} == {True, False}
    assert len({m["span"] for m in all_meta}) >= 15
    assert len({m["tips_count"] for m in all_meta}) >= 4
    assert len({m["branch_count"] for m in all_meta}) >= 4
    assert len({m["move"] for m in all_meta}) >= 20
    assert len({m["dest_shape"] for m in all_meta}) >= 25
    assert any(m["source_shape"][0] == 4 for m in all_meta)
    assert any(m["dest_shape"][0] > m["source_shape"][0] for m in all_meta)
    assert any(m["dest_shape"][0] < m["source_shape"][0] for m in all_meta)
    assert any(len({m["layout"] for m, _ in ms}) >= 3 for ms in metas)
    assert any(len({m["base"] for m, _ in ms}) >= 2 for ms in metas)
    for ntrain, ntest in (
        (2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
        (2, None), (10, None), (6, 3)
    ):
        p = generate_puzzle(801, ntrain, ntest)
        validate_generated_puzzle(p)
        assert ntrain is None or len(p["train"]) == ntrain
        assert ntest is None or len(p["test"]) == ntest
        assert _same(p, generate_puzzle(801, ntrain, ntest))
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
