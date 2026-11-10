"""Hollow tile fields with clockwise, stable edge gravity."""

import hashlib

import numpy as np


_EDGES = ("top", "right", "bottom", "left")
_SOURCE_HASHES = {
    "310b07d103e7f0930f90c97c8dac4ca4d87ec876dcf5682971fcce4878e28ca3",
    "69de7138442f2060af9236f1f651e698be367242f3db4b0fe6ba6e44c688b2d6",
    "44510d3e17526224ee4b152abc085657dd1e3702b1efaf22ad1ea3fc448abd7d",
    "93af3408bca9a8efe2b45f7b7a1a6245f3c94843397d593452fe45c5c22cf60c",
}


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _counts(rng, seed, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt is None and ne is None:
        ne = 1 + seed % 4
        nt = max(2, ne + 1) + (seed // 4) % (11 - max(2, ne + 1))
    elif nt is None:
        nt = int(rng.integers(max(2, ne + 1), 11))
    elif ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _size(g):
    s, p, extra = g["side"], g["pitch"], g["extra"]
    along = (g["a"] + g["b"] - 1) * p + extra + s
    cross = (g["cross"] - 1) * p + s
    return (along + g["along_before"] + g["along_after"],
            cross + g["cross_before"] + g["cross_after"])


def _geometry(rng, dense=False, small=False):
    for _ in range(160):
        if dense:
            a = b = 3
            cross = int(rng.integers(6, 8))
            side, pitch = 3, 4
        else:
            a = int(rng.integers(2, 4 if small else 5))
            b = int(rng.integers(2, 4 if small else 5))
            if a + b > 6:
                continue
            cross = int(rng.integers(3, 6 if small else 8))
            side = int(rng.integers(3, 5))
            pitch = side + int(rng.integers(1, 3))
        g = {"a": a, "b": b, "cross": cross, "side": side,
             "pitch": pitch, "extra": int(rng.integers(2, pitch)),
             "along_before": int(rng.integers(1, 4)),
             "along_after": int(rng.integers(1, 4)),
             "cross_before": int(rng.integers(1, 4)),
             "cross_after": int(rng.integers(1, 4))}
        along, transverse = _size(g)
        if along <= 30 and transverse <= 30:
            return g
    return {"a": 3, "b": 3, "cross": 6 if dense else 4,
            "side": 3, "pitch": 4, "extra": 2,
            "along_before": 1, "along_after": 1,
            "cross_before": 1, "cross_after": 1}


def _starts(g):
    along = [g["along_before"] + i * g["pitch"] +
             (g["extra"] if i >= g["a"] else 0)
             for i in range(g["a"] + g["b"])]
    cross = [g["cross_before"] + i * g["pitch"] for i in range(g["cross"])]
    return along, cross


def _render(board, background, marker, edge, g):
    along_extent, cross_extent = _size(g)
    shape = (cross_extent, along_extent) if edge in ("top", "bottom") else (along_extent, cross_extent)
    grid = np.full(shape, background, dtype=np.int8)
    if edge == "top":
        grid[0, :] = marker
    elif edge == "bottom":
        grid[-1, :] = marker
    elif edge == "right":
        grid[:, -1] = marker
    else:
        grid[:, 0] = marker
    along, cross = _starts(g)
    for ci, rr in enumerate(cross):
        for ai, cc in enumerate(along):
            color = int(board[ci, ai] if edge in ("top", "bottom") else board[ai, ci])
            if color == background:
                continue
            r, c = (rr, cc) if edge in ("top", "bottom") else (cc, rr)
            s = g["side"]
            grid[r:r+s, c:c+s] = color
            grid[r+1:r+s-1, c+1:c+s-1] = background
    return grid


def _compact(board, background, edge):
    result = np.full(board.shape, background, dtype=np.int8)
    if edge in ("top", "bottom"):
        for r in range(board.shape[0]):
            values = board[r, board[r] != background]
            if edge == "top":
                result[r, board.shape[1]-len(values):] = values
            else:
                result[r, :len(values)] = values
    else:
        for c in range(board.shape[1]):
            values = board[board[:, c] != background, c]
            if edge == "right":
                result[board.shape[0]-len(values):, c] = values
            else:
                result[:len(values), c] = values
    return result


def _anchors(cross, length, a):
    return {(0, 0), (0, 1), (1, 0), (cross-1, 0),
            (0, length-1), (cross-1, length-1), (0, a-1), (0, a)}


def _mask(rng, g, mode, strong, high):
    cross, length, a = g["cross"], g["a"] + g["b"], g["a"]
    total = cross * length
    anchors = _anchors(cross, length, a)
    mask = np.zeros((cross, length), dtype=bool)
    if mode in ("sparse", "random", "dense"):
        if mode == "sparse":
            target = max(len(anchors), int(rng.integers(7, 11)))
        elif mode == "dense":
            target = int(rng.integers(31, total + 1))
        else:
            target = int(rng.integers(max(8, total // 3), max(9, 3 * total // 4 + 1)))
        for r, c in anchors:
            mask[r, c] = True
        free = np.flatnonzero(~mask.ravel())
        add = min(target - int(mask.sum()), len(free))
        if add > 0:
            mask.ravel()[rng.choice(free, size=add, replace=False)] = True
    elif mode == "full":
        mask[:] = True
    elif mode == "one_line":
        mask[:] = True
        line = int(rng.integers(1, cross-1))
        first_hole = -1 if high else (1 if line == 1 else 0)
        mask[line, first_hole] = False
        if length > 4 and rng.random() < 0.5:
            mask[line, -2 if high else first_hole + 1] = False
    else:
        fill = 0.72 if mode == "cluster" else (0.68 if mode in ("seam_blank", "outer_blank") else 0.83)
        mask[:] = rng.random((cross, length)) < fill
        if mode == "cluster":
            r = int(rng.integers(0, cross-1))
            c = int(rng.integers(0, length-1))
            mask[r:r+2, c:c+2] = False
        elif mode == "aligned_holes":
            c = int(rng.integers(0, length))
            mask[:, c] = False
            if length >= 5 and rng.random() < 0.6:
                mask[1:-1, (c+2) % length] = False
        elif mode == "nearly_full":
            holes = int(rng.integers(1, max(3, total // 7 + 1)))
            mask.ravel()[rng.choice(total, size=holes, replace=False)] = False
        for r, c in anchors:
            mask[r, c] = True
        if mode == "seam_blank":
            if rng.random() < 0.5:
                mask[:, a-1] = False
                mask[0, a-2] = mask[0, a] = True
            else:
                mask[:, a] = False
                mask[0, a-1] = mask[0, a+1] = True
        elif mode == "outer_blank":
            if rng.random() < 0.5:
                mask[0, :] = False
                mask[1, 0] = mask[2, 0] = True
                mask[1, a-1:a+1] = True
            else:
                mask[-1, :] = False
                mask[-2, 0] = mask[-2, -1] = True
    if strong:
        mask[2, :] = False
        mask[2, a-1:a+1] = True
    return mask


def _color_board(rng, mask, colors, mode, g, background, strong, must_use):
    cross, length = mask.shape
    base = np.full(mask.shape, background, dtype=np.int8)
    if len(colors) == 1:
        base[mask] = colors[0]
    elif mode == "mixed":
        base[mask] = rng.choice(np.asarray(colors, dtype=np.int8), size=int(mask.sum()))
    elif mode == "fields":
        base[:, :g["a"]][mask[:, :g["a"]]] = colors[0]
        base[:, g["a"]:][mask[:, g["a"]:]] = colors[1]
    elif mode == "bands":
        shift = int(rng.integers(0, len(colors)))
        band = int(rng.integers(1, 3))
        for r in range(cross):
            base[r, mask[r]] = colors[(r // band + shift) % len(colors)]
    elif mode == "runs":
        for c in range(length):
            base[mask[:, c], c] = colors[min(len(colors)-1, c * len(colors) // length)]
    else:
        base[mask] = colors[0]
        r = int(rng.integers(0, cross-1))
        c = int(rng.integers(0, length-1))
        patch = base[r:r+2, c:c+2]
        patch[mask[r:r+2, c:c+2]] = colors[1]
    if strong:
        base[2, g["a"]-1] = colors[0]
        base[2, g["a"]] = colors[1]
    if must_use:
        positions = [(r, c) for r, c in zip(*np.nonzero(mask)) if not strong or r != 2]
        for color, index in zip(must_use, rng.permutation(len(positions))):
            r, c = positions[int(index)]
            base[r, c] = color
    return base


def _pair(rng, edge, train_palette, novel_palette, demonstrated, strong, test_mode):
    mode = "random" if strong else str(rng.choice(np.asarray(
        ("sparse", "random", "dense", "nearly_full", "full", "one_line",
         "aligned_holes", "cluster", "seam_blank", "outer_blank"),
        dtype=object), p=(0.12, 0.16, 0.09, 0.11, 0.07, 0.09, 0.12, 0.11, 0.07, 0.06)))
    g = _geometry(rng, dense=mode == "dense", small=mode == "sparse")
    if strong or mode == "outer_blank":
        for _ in range(20):
            if g["cross"] >= 4:
                break
            g = _geometry(rng)
        else:
            g = {"a": 2, "b": 2, "cross": 4, "side": 3, "pitch": 4,
                 "extra": 2, "along_before": 1, "along_after": 1,
                 "cross_before": 1, "cross_after": 1}
    wanted_new = 0 if test_mode is None or test_mode == "familiar" else (2 if test_mode == "new_two" else 1)
    valid_roles = []
    for background in range(10):
        for marker in range(10):
            if background == marker:
                continue
            familiar = sorted(set(train_palette if test_mode is None else demonstrated) - {background, marker})
            novel = sorted(set(novel_palette) - {background, marker})
            if len(familiar) >= (2 if strong else 1) and len(novel) >= wanted_new:
                valid_roles.append((background, marker))
    background, marker = valid_roles[int(rng.integers(0, len(valid_roles)))]
    familiar = sorted(set(train_palette if test_mode is None else demonstrated) - {background, marker})
    novel = sorted(set(novel_palette) - {background, marker})
    if test_mode == "one_color":
        color = int(rng.choice(novel if rng.random() < 0.5 else familiar))
        colors = (color,)
        must_use = (color,) if color in novel else ()
    elif test_mode is None:
        n = int(rng.integers(2 if strong else 1, min(4, len(familiar)) + 1))
        colors = tuple(int(x) for x in rng.choice(familiar, size=n, replace=False))
        must_use = ()
    else:
        new = tuple(int(x) for x in rng.choice(novel, size=wanted_new, replace=False)) if wanted_new else ()
        nfam = int(rng.integers(1, min(3, len(familiar)) + 1))
        fam = tuple(int(x) for x in rng.choice(familiar, size=nfam, replace=False))
        colors = fam + new
        must_use = new
    high = edge in ("top", "right")
    mask = _mask(rng, g, mode, strong, high)
    color_style = str(rng.choice(np.asarray(("mixed", "fields", "bands", "runs", "patch"), dtype=object)))
    oriented = _color_board(rng, mask, colors, color_style, g, background, strong, must_use)
    board = oriented if edge in ("top", "bottom") else oriented.T.copy()
    return {"input": _render(board, background, marker, edge, g),
            "output": _compact(board, background, edge)}


def _signature(grid):
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


def _source_hash(pair):
    h = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name].astype(np.int8, copy=False)
        h.update(bytes(grid.shape))
        h.update(grid.tobytes())
    return h.hexdigest()


def _choose_edges(rng, train_count, test_count):
    start = int(rng.integers(0, 4))
    training = [_EDGES[start], _EDGES[(start + 1) % 4]]
    if train_count == 2:
        edge_count = 2
    elif train_count == 3:
        edge_count = int(rng.choice((2, 3)))
    else:
        edge_count = int(rng.choice((2, 3, 4), p=(0.27, 0.43, 0.30)))
    remaining = [e for e in _EDGES if e not in training]
    if edge_count > 2:
        training.extend(str(x) for x in rng.choice(np.asarray(remaining),
                                                    size=edge_count-2, replace=False))
    allowed = tuple(dict.fromkeys(training))
    while len(training) < train_count:
        training.append(str(rng.choice(np.asarray(allowed))))
    seen = set(training)
    unseen = [e for e in _EDGES if e not in seen]
    tests = []
    for i in range(test_count):
        if i == 1 and rng.random() < 0.65:
            candidates = [e for e in _EDGES if e != tests[0]]
        elif unseen and rng.random() < 0.5:
            candidates = unseen
        else:
            candidates = [e for e in _EDGES if e in seen]
        tests.append(str(rng.choice(np.asarray(candidates))))
    return training, tests


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    # Color reservoirs affect sampling only. Tile decoding and clockwise gravity are fixed.
    palette = [int(x) for x in rng.permutation(10)]
    train_palette, novel_palette = tuple(palette[:5]), tuple(palette[5:])
    train_edges, test_edges = _choose_edges(rng, train_count, test_count)
    seen_inputs = set()
    puzzle = {"train": [], "test": []}
    demonstrated = set()
    for split, edges in (("train", train_edges), ("test", test_edges)):
        for index, edge in enumerate(edges):
            if split == "test":
                modes = ("familiar", "new_one", "new_two", "one_color")
                test_mode = modes[(int(seed) + index) % len(modes)]
            else:
                test_mode = None
            for _ in range(12):
                pair = _pair(rng, edge, train_palette, novel_palette,
                             demonstrated, split == "train" and index < 2, test_mode)
                sig = _signature(pair["input"])
                if sig not in seen_inputs and _source_hash(pair) not in _SOURCE_HASHES:
                    break
            else:
                raise AssertionError("could not construct a unique pair")
            seen_inputs.add(sig)
            puzzle[split].append(pair)
            if split == "train":
                background = int(np.bincount(pair["input"].ravel(), minlength=10).argmax())
                demonstrated.update(set(int(x) for x in np.unique(pair["output"])) - {background})
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, background, marker):
    remaining = set(zip(*np.nonzero((grid != background) & (grid != marker))))
    tiles = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        cells = {start}
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        rows, cols = zip(*cells)
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        side = r1-r0+1
        assert side in (3, 4) and c1-c0+1 == side
        color = int(grid[r0, c0])
        assert color not in (background, marker)
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                wanted = color if r in (r0, r1) or c in (c0, c1) else background
                assert int(grid[r, c]) == wanted
        assert len(cells) == 4 * side - 4
        tiles.append((r0, c0, color, side))
    assert tiles
    return tiles


def _decode(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(counts.argmax())
    assert counts[background] > max(int(counts[i]) for i in range(10) if i != background)
    outer = (("top", grid[0]), ("right", grid[:, -1]),
             ("bottom", grid[-1]), ("left", grid[:, 0]))
    markers = [(name, int(line[0])) for name, line in outer
               if int(line[0]) != background and np.all(line == line[0])]
    assert len(markers) == 1
    edge, marker = markers[0]
    tiles = _components(grid, background, marker)
    sides = {tile[3] for tile in tiles}
    assert len(sides) == 1
    side = sides.pop()
    horizontal = edge in ("top", "bottom")
    along = sorted({t[1] if horizontal else t[0] for t in tiles})
    cross = sorted({t[0] if horizontal else t[1] for t in tiles})
    differences = [y-x for values in (along, cross) for x, y in zip(values, values[1:])]
    pitch = min(differences)
    assert pitch in (side+1, side+2)
    assert len(cross) >= 2 and cross[1]-cross[0] == pitch
    assert all((x-cross[0]) % pitch == 0 for x in cross)
    along_extent = grid.shape[1] if horizontal else grid.shape[0]
    cross_extent = grid.shape[0] if horizontal else grid.shape[1]
    raw_before = cross[0]
    raw_after = cross_extent - (cross[-1] + side)
    cross_before, cross_after = raw_before % pitch, raw_after % pitch
    assert 1 <= cross_before <= 3 and 1 <= cross_after <= 3
    skip_before = (raw_before-cross_before) // pitch
    skip_after = (raw_after-cross_after) // pitch
    assert skip_before + skip_after <= 1
    cross_count = (cross[-1]-cross[0]) // pitch + 1 + skip_before + skip_after
    assert 3 <= cross_count <= 7
    seams = [i for i, (x, y) in enumerate(zip(along, along[1:])) if (y-x) % pitch]
    assert len(seams) == 1
    si = seams[0]
    gap = along[si+1]-along[si]
    extra = gap % pitch
    seam_missing = (gap-extra) // pitch - 1
    assert 2 <= extra < pitch and 0 <= seam_missing <= 1
    left_last = (along[si]-along[0]) // pitch
    length = (along[-1]-along[0]-extra) // pitch + 1
    candidates = [(left_last+1+k, length-left_last-1-k)
                  for k in range(seam_missing+1)]
    candidates = [(a, b) for a, b in candidates
                  if 2 <= a <= 4 and 2 <= b <= 4 and a+b <= 6]
    assert candidates
    a, b = candidates[0]
    g = {"a": a, "b": b, "cross": cross_count, "side": side,
         "pitch": pitch, "extra": extra,
         "along_before": along[0],
         "cross_before": cross_before}
    g["along_after"] = along_extent - (along[-1] + side)
    g["cross_after"] = cross_after
    assert all(1 <= g[key] <= 3 for key in ("along_before", "along_after",
                                             "cross_before", "cross_after"))
    assert _size(g) == (along_extent, cross_extent)
    oriented = np.full((cross_count, a+b), background, dtype=np.int8)
    for r, c, color, _ in tiles:
        along_coord, cross_coord = (c, r) if horizontal else (r, c)
        ci = skip_before + (cross_coord-cross[0]) // pitch
        if along_coord <= along[si]:
            ai = (along_coord-along[0]) // pitch
        else:
            ai = left_last + seam_missing + 1 + (along_coord-along[si+1]) // pitch
        assert 0 <= ci < cross_count and 0 <= ai < a+b
        assert oriented[ci, ai] == background
        oriented[ci, ai] = color
    assert oriented[skip_before, 0] != background
    assert oriented[skip_before+1, 0] != background
    assert oriented[cross_count-skip_after-1, 0] != background
    assert oriented[skip_before, -1] != background or skip_before == 1
    assert oriented[cross_count-skip_after-1, -1] != background
    if skip_before:
        assert np.all(oriented[0] == background)
    if skip_after:
        assert np.all(oriented[-1] == background)
    if not seam_missing and not skip_before and not skip_after:
        assert all(oriented[r, c] != background for r, c in _anchors(cross_count, a+b, a))
    board = oriented if horizontal else oriented.T.copy()
    assert np.array_equal(grid, _render(board, background, marker, edge, g))
    return edge, background, marker, board, g


def _moving_lines(board, background, edge):
    lines = board if edge in ("top", "bottom") else board.T
    target = _compact(board, background, edge)
    target_lines = target if edge in ("top", "bottom") else target.T
    return sum(not np.array_equal(x, y) for x, y in zip(lines, target_lines))


def _strong_evidence(board, background, edge, g):
    lines = board if edge in ("top", "bottom") else board.T
    a = g["a"]
    high = edge in ("top", "right")
    for line in lines:
        positions = np.flatnonzero(line != background)
        if len(positions) < 2 or not any(x < a for x in positions) or not any(x >= a for x in positions):
            continue
        values = line[positions]
        if len(np.unique(values)) < 2:
            continue
        targets = np.arange(len(line)-len(values), len(line)) if high else np.arange(len(values))
        if np.array_equal(positions, targets):
            continue
        if any((x < a) != (y < a) for x, y in zip(positions, targets)):
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            key = _signature(pair["input"])
            assert key not in seen
            seen.add(key)
            assert _source_hash(pair) not in _SOURCE_HASHES
            edge, background, marker, board, g = _decode(pair["input"])
            assert background != marker
            assert np.array_equal(pair["output"], _compact(board, background, edge))
            assert set(int(x) for x in np.unique(pair["output"])) == set(int(x) for x in np.unique(board))
            assert marker not in pair["output"]
            if split == "train":
                training.append((edge, background, board, g, pair["output"]))
    assert _EDGES.index(training[1][0]) == (_EDGES.index(training[0][0]) + 1) % 4
    possible_offsets = set(range(4))
    for edge, background, board, g, observed in training:
        possible_offsets = {offset for offset in possible_offsets
                            if np.array_equal(
                                _compact(board, background,
                                         _EDGES[(_EDGES.index(edge)+offset) % 4]),
                                observed)}
    assert possible_offsets == {0}
    for edge, background, board, g, _ in training[:2]:
        assert _strong_evidence(board, background, edge, g)


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(i) for i in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000+seed, *counts)
        validate_generated_puzzle(p)
        if counts[0] is not None:
            assert len(p["train"]) == counts[0]
        if counts[1] is not None:
            assert len(p["test"]) == counts[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _same_puzzle(generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3))
    assert not _same_puzzle(generate_puzzle(123, 7, 3), generate_puzzle(124, 7, 3))
    edges_train, edges_test, shapes, occupancies, sides, pitches, gaps = set(), set(), set(), set(), set(), set(), set()
    sparse = dense = full = one_line = one_color = blank_outer = blank_seam = 0
    changed_roles = same_edge = unseen_edge = mixed_tests = all_edges = 0
    varying_shapes = varying_splits = varying_sides = sparse_dense_puzzles = 0
    familiar_tests = one_new_tests = multiple_new_tests = role_crossovers = 0
    field_colors = bands = runs = aligned_holes = clustered_holes = 0
    unseen_sides = set()
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        records = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                edge, bg, marker, board, g = _decode(pair["input"])
                records.append((split, edge, bg, marker, board, g))
                (edges_train if split == "train" else edges_test).add(edge)
                shapes.add((tuple(pair["input"].shape), tuple(board.shape)))
                count = int(np.count_nonzero(board != bg))
                occupancies.add(count)
                sides.add(g["side"]); pitches.add(g["pitch"]); gaps.add(g["extra"])
                sparse += count <= 10
                dense += count > 30
                full += count == board.size
                one_line += _moving_lines(board, bg, edge) == 1
                one_color += len(set(int(x) for x in np.unique(board)) - {bg}) == 1
                oriented = board if edge in ("top", "bottom") else board.T
                blank_outer += bool(np.all(oriented[0] == bg) or np.all(oriented[-1] == bg))
                blank_seam += bool(np.all(oriented[:, g["a"]-1] == bg)
                                   or np.all(oriented[:, g["a"]] == bg))
                first = set(int(x) for x in oriented[:, :g["a"]].flat) - {bg}
                second = set(int(x) for x in oriented[:, g["a"]:].flat) - {bg}
                field_colors += len(first) == len(second) == 1 and first != second
                row_colors = [set(int(x) for x in line) - {bg} for line in oriented]
                column_colors = [set(int(x) for x in line) - {bg} for line in oriented.T]
                row_singles = [next(iter(x)) for x in row_colors if len(x) == 1]
                column_singles = [next(iter(x)) for x in column_colors if len(x) == 1]
                bands += len(set(row_singles)) >= 2
                runs += len(set(column_singles)) >= 2
                holes = oriented == bg
                aligned_holes += bool(np.any(np.sum(holes, axis=0) >= 2))
                clustered_holes += any(np.all(holes[r:r+2, c:c+2])
                                       for r in range(holes.shape[0]-1)
                                       for c in range(holes.shape[1]-1))
        tr = {x[1] for x in records if x[0] == "train"}
        te = [x[1] for x in records if x[0] == "test"]
        changed_roles += len({(x[2], x[3]) for x in records}) > 1
        same_edge += any(x in tr for x in te)
        unseen_edge += any(x not in tr for x in te)
        unseen_sides.update(x for x in te if x not in tr)
        mixed_tests += len(set(te)) > 1
        all_edges += len(tr) == 4
        role_crossovers += any(x[2] == y[3] for x in records for y in records)
        varying_shapes += len({tuple(x[4].shape) for x in records}) > 1
        varying_splits += len({(x[5]["a"], x[5]["b"]) for x in records}) > 1
        varying_sides += len({x[5]["side"] for x in records}) > 1
        sparse_dense_puzzles += (any(np.count_nonzero(x[4] != x[2]) <= 10 for x in records)
                                 and any(np.count_nonzero(x[4] != x[2]) > 30 for x in records))
        training_colors = set().union(*(set(int(c) for c in np.unique(x[4])) - {x[2]}
                                        for x in records if x[0] == "train"))
        for x in records:
            if x[0] == "test":
                new = (set(int(c) for c in np.unique(x[4])) - {x[2]}) - training_colors
                familiar_tests += len(new) == 0
                one_new_tests += len(new) == 1
                multiple_new_tests += len(new) >= 2
    assert edges_train == edges_test == set(_EDGES)
    assert len(shapes) >= 30 and len(occupancies) >= 20
    assert sides == {3, 4} and len(pitches) >= 3 and len(gaps) >= 3
    assert sparse >= 5 and dense >= 5 and full >= 5 and one_line >= 5 and one_color >= 5
    assert blank_outer >= 5 and blank_seam >= 5
    assert changed_roles >= 35 and same_edge >= 10 and unseen_edge >= 5
    assert mixed_tests >= 5 and all_edges >= 3 and unseen_sides == set(_EDGES)
    assert varying_shapes >= 35 and varying_splits >= 35 and varying_sides >= 20
    assert sparse_dense_puzzles >= 5 and role_crossovers >= 20
    assert familiar_tests >= 5 and one_new_tests >= 5 and multiple_new_tests >= 5
    assert min(field_colors, bands, runs, aligned_holes, clustered_holes) >= 5
