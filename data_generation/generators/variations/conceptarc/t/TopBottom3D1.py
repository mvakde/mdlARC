"""Synthetic perpendicular-band occlusion puzzles."""

import hashlib

import numpy as np


_SOURCE_HASHES = frozenset({
    "23d8107f84d2c592a00b0c503366893b1bb5357b4a3b993bcb278a05adab79b8",
    "cafbd4714d01e303d692824250814e0dc4ec6e3549d107a21b8f9f0c176b9cf0",
    "089a7e57a4218617ff25233c0b09266bcbbc0bb505e95009762e9fa05c113d75",
    "335851a6f3ac77ef2ab13b0c28e311718d299fa23595316ca3c7d310d04d7e79",
    "26e6f783e13ce810068200ea5f712d1781a109ce090fd970c708cfa0cf9da462",
    "3a165dc2cc03c200d3d56e0b8af5c195abf6bc8deb92a63535fd207d274b0aa0",
})


def _pick(rng, low, high):
    assert low <= high
    return int(rng.integers(low, high + 1))


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("pair counts must be integers")
            if not low <= value <= high:
                raise ValueError("pair count out of range")
    if num_train is None:
        lower = max(2, int(num_test) + 1) if num_test is not None else 2
        num_train = _pick(rng, lower, 10)
    if num_test is None:
        num_test = _pick(rng, 1, min(4, int(num_train) - 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _size(rng):
    category = _pick(rng, 0, 9)
    if category < 3:
        return _pick(rng, 8, 11)
    if category < 7:
        return _pick(rng, 12, 20)
    return _pick(rng, 21, 30)


def _dimensions(rng, mode):
    if mode == "separate":
        axis = _pick(rng, 0, 3)
        if axis == 0:
            return _size(rng), _pick(rng, 17, 30), axis
        if axis == 2:
            return _pick(rng, 17, 30), _size(rng), axis
        return _pick(rng, 17, 30), _pick(rng, 17, 30), axis
    choice = _pick(rng, 0, 11)
    if choice == 0:
        return 8, 8, None
    if choice == 1:
        return 10, 10, None
    return _size(rng), _size(rng), None


def _bands(rng, length, count, maximum, edge=None, margin=1):
    """Disjoint intervals with visible gaps and independently sampled widths."""
    left_min = 0 if edge == "start" else margin
    right_min = 0 if edge == "end" else margin
    budget = length - left_min - right_min - (count - 1)
    assert budget >= count
    widths = [1] * count
    order = list(rng.permutation(count))
    spare = budget - count
    for index in order:
        extra = _pick(rng, 0, min(maximum - 1, spare))
        widths[index] += extra
        spare -= extra
    bins = count + 1
    eligible = [i for i in range(bins) if not ((i == 0 and edge == "start") or (i == count and edge == "end"))]
    allocations = [0] * bins
    if spare:
        weights = rng.dirichlet(np.ones(len(eligible)))
        draw = rng.multinomial(spare, weights)
        for index, amount in zip(eligible, draw):
            allocations[index] = int(amount)
    gaps = [left_min + allocations[0]]
    gaps += [1 + allocations[i] for i in range(1, count)]
    gaps += [right_min + allocations[count]]
    result = []
    position = gaps[0]
    for index, width in enumerate(widths):
        result.append((position, position + width))
        position += width + gaps[index + 1]
    assert position == length
    return result


def _colors(rng, background, avoid=None):
    values = [c for c in range(10) if c != background]
    options = [(a, b) for a in values for b in values if a != b]
    if avoid is not None:
        options = [item for item in options if frozenset(item) != frozenset(avoid)]
    return options[_pick(rng, 0, len(options) - 1)]


def _tiles(rng, height, width, split_kind):
    if split_kind == 0:  # side by side
        cut = _pick(rng, 8, width - 9)
        return [(0, height, 0, cut), (0, height, cut + 1, width)]
    if split_kind == 2:  # stacked
        cut = _pick(rng, 8, height - 9)
        return [(0, cut, 0, width), (cut + 1, height, 0, width)]
    if split_kind == 1:  # the other diagonal
        cut_r = _pick(rng, 8, height - 9)
        cut_c = _pick(rng, 8, width - 9)
        return [(0, cut_r, cut_c + 1, width), (cut_r + 1, height, 0, cut_c)]
    cut_r = _pick(rng, 8, height - 9)
    cut_c = _pick(rng, 8, width - 9)
    return [(0, cut_r, 0, cut_c), (cut_r + 1, height, cut_c + 1, width)]


def _arrangement(rng, tile, grid_shape, h_count, v_count, colors, edge, inset, alignment=None):
    tr0, tr1, tc0, tc1 = tile
    height, width = grid_shape
    tile_h, tile_w = tr1 - tr0, tc1 - tc0
    choices = []
    if tr0 == 0:
        choices.append("top")
    if tr1 == height:
        choices.append("bottom")
    if tc0 == 0:
        choices.append("left")
    if tc1 == width:
        choices.append("right")
    boundary = choices[_pick(rng, 0, len(choices) - 1)] if edge and choices and not inset else None
    if boundary in ("top", "bottom") and tile_w < 2 * v_count + 3:
        boundary = None
    if boundary in ("left", "right") and tile_h < 2 * h_count + 3:
        boundary = None
    row_edge = "start" if boundary == "top" else "end" if boundary == "bottom" else None
    col_edge = "start" if boundary == "left" else "end" if boundary == "right" else None
    rows = [alignment[1]] if alignment is not None and alignment[0] == "row" else _bands(rng, tile_h, h_count, 6, row_edge, 2 if col_edge or inset else 1)
    cols = [alignment[1]] if alignment is not None and alignment[0] == "col" else _bands(rng, tile_w, v_count, 5, col_edge, 2 if row_edge or inset else 1)
    h_color, v_color = colors
    horizontal, vertical = [], []
    for row0, row1 in rows:
        low = 1 if inset or (tr0 == 0 and row0 == 0 and tc0 == 0) else 0
        c0 = tc0 + _pick(rng, low, cols[0][0] - 1) if cols[0][0] else tc0
        c1 = tc0 + _pick(rng, cols[-1][1] + 1, tile_w - int(inset)) if cols[-1][1] < tile_w else tc1
        horizontal.append((tr0 + row0, tr0 + row1, c0, c1, h_color))
    for col0, col1 in cols:
        low = 1 if inset or (tc0 == 0 and col0 == 0 and tr0 == 0) else 0
        r0 = tr0 + _pick(rng, low, rows[0][0] - 1) if rows[0][0] else tr0
        r1 = tr0 + _pick(rng, rows[-1][1] + 1, tile_h - int(inset)) if rows[-1][1] < tile_h else tr1
        vertical.append((r0, r1, tc0 + col0, tc0 + col1, v_color))
    return horizontal, vertical


def _make_pair(rng, background, base_colors, mode, order, edge, inset):
    height, width, split_kind = _dimensions(rng, mode)
    tiles = _tiles(rng, height, width, split_kind) if mode == "separate" else [(0, height, 0, width)]
    counts = {
        "single": (1, 1), "parallel_v": (1, 2), "parallel_h": (2, 1),
        "mesh": (2, 2), "separate": (1, 1),
        "triple_v": (1, 3), "triple_h": (3, 1),
    }
    inset = inset and all(
        tile[1] - tile[0] >= 2 * counts[mode][0] + 3
        and tile[3] - tile[2] >= 2 * counts[mode][1] + 3
        for tile in tiles
    )
    horizontal, vertical, groups = [], [], []
    for index, tile in enumerate(tiles):
        colors = base_colors if index == 0 or _pick(rng, 0, 1) == 0 else _colors(rng, background)
        alignment = None
        if index == 1 and mode == "separate" and split_kind == 0:
            alignment = ("row", (groups[0][0][0][0] - tiles[0][0], groups[0][0][0][1] - tiles[0][0]))
        if index == 1 and mode == "separate" and split_kind == 2:
            alignment = ("col", (groups[0][1][0][2] - tiles[0][2], groups[0][1][0][3] - tiles[0][2]))
        hs, vs = _arrangement(rng, tile, (height, width), *counts[mode], colors, edge and (index == 0 or _pick(rng, 0, 1) == 0), inset, alignment)
        horizontal.extend(hs)
        vertical.extend(vs)
        groups.append((hs, vs))
    crossings = [(h, v) for hs, vs in groups for h in hs for v in vs]
    if order == "mixed":
        assert len(crossings) >= 2
        fronts = ["H", "V"] + ["H" if _pick(rng, 0, 1) == 0 else "V" for _ in crossings[2:]]
        rng.shuffle(fronts)
    elif order in ("H", "V"):
        fronts = [order] * len(crossings)
    else:
        fronts = ["H" if _pick(rng, 0, 1) == 0 else "V" for _ in crossings]
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for r0, r1, c0, c1, color in horizontal + vertical:
        inp[r0:r1, c0:c1] = color
        out[r0:r1, c0:c1] = color
    for (h, v), front in zip(crossings, fronts):
        r0, r1 = h[0], h[1]
        c0, c1 = v[2], v[3]
        inp[r0:r1, c0:c1] = h[4] if front == "H" else v[4]
        out[r0:r1, c0:c1] = v[4] if front == "H" else h[4]
    return inp, out


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    background = 0 if _pick(rng, 0, 3) else _pick(rng, 1, 9)
    base_colors = _colors(rng, background)
    simple_front = "H" if _pick(rng, 0, 1) == 0 else "V"
    puzzle = {"train": [], "test": []}
    seen = set()
    modes = ["single", "parallel_v", "parallel_h", "mesh", "separate", "triple_v", "triple_h"]
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train" and index == 0:
                mode, order = "single", simple_front
                colors = base_colors
            elif section == "train" and index == 1:
                mode, order = modes[_pick(rng, 1, len(modes) - 1)], "mixed"
                colors = _colors(rng, background, base_colors)
            else:
                mode = modes[_pick(rng, 0, len(modes) - 1)]
                order = "H" if section == "test" and _pick(rng, 0, 2) == 0 else "random"
                colors = base_colors if _pick(rng, 0, 2) == 0 else _colors(rng, background)
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = _make_pair(pair_rng, background, colors, mode, order, _pick(rng, 0, 2) == 0, _pick(rng, 0, 3) == 0)
                key = _grid_key(inp)
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append({"input": inp, "output": out})
                    break
            else:
                raise AssertionError("unable to generate a unique pair")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    stack.append((rr, cc))
        result.append(component)
    return result


def _span(flags, position):
    assert flags[position]
    start, end = position, position + 1
    while start and flags[start - 1]:
        start -= 1
    while end < len(flags) and flags[end]:
        end += 1
    return start, end


def _uniform_neighbor(grid, r0, r1, c0, c1):
    values = np.unique(grid[r0:r1, c0:c1])
    assert len(values) == 1
    return int(values[0])


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    height, width = inp.shape
    background = int(inp[0, 0])
    assert int(out[0, 0]) == background
    changed = inp != out
    components = _components(changed)
    assert components
    crossings = []
    h_rects, v_rects = set(), set()
    for component in components:
        rows, cols = zip(*component)
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1
        assert len(component) == (r1 - r0) * (c1 - c0)
        in_color = _uniform_neighbor(inp, r0, r1, c0, c1)
        out_color = _uniform_neighbor(out, r0, r1, c0, c1)
        hs = []
        vs = []
        if c0:
            hs.append(_uniform_neighbor(inp, r0, r1, c0 - 1, c0))
        if c1 < width:
            hs.append(_uniform_neighbor(inp, r0, r1, c1, c1 + 1))
        if r0:
            vs.append(_uniform_neighbor(inp, r0 - 1, r0, c0, c1))
        if r1 < height:
            vs.append(_uniform_neighbor(inp, r1, r1 + 1, c0, c1))
        assert hs and vs and len(set(hs)) == len(set(vs)) == 1
        h_color, v_color = hs[0], vs[0]
        assert len({background, h_color, v_color}) == 3
        if (in_color, out_color) == (h_color, v_color):
            front = "H"
        else:
            assert (in_color, out_color) == (v_color, h_color)
            front = "V"
        h_support = np.all((inp[r0:r1, :] == h_color) | (out[r0:r1, :] == h_color), axis=0)
        v_support = np.all((inp[:, c0:c1] == v_color) | (out[:, c0:c1] == v_color), axis=1)
        hc0, hc1 = _span(h_support, c0)
        vr0, vr1 = _span(v_support, r0)
        assert (hc0 < c0 or hc1 > c1) and (vr0 < r0 or vr1 > r1)
        h_rects.add((r0, r1, hc0, hc1, h_color))
        v_rects.add((vr0, vr1, c0, c1, v_color))
        crossings.append((r0, r1, c0, c1, in_color, out_color, front))
    h_rects, v_rects = sorted(h_rects), sorted(v_rects)
    h_mask = np.zeros_like(changed)
    v_mask = np.zeros_like(changed)
    base = np.full_like(inp, background)
    for r0, r1, c0, c1, color in h_rects:
        assert not np.any(h_mask[r0:r1, c0:c1])
        h_mask[r0:r1, c0:c1] = True
        base[r0:r1, c0:c1] = color
    for r0, r1, c0, c1, color in v_rects:
        assert not np.any(v_mask[r0:r1, c0:c1])
        v_mask[r0:r1, c0:c1] = True
        base[r0:r1, c0:c1] = color
    assert np.array_equal(h_mask & v_mask, changed)
    expected_in, expected_out = base.copy(), base.copy()
    for r0, r1, c0, c1, in_color, out_color, _ in crossings:
        expected_in[r0:r1, c0:c1] = in_color
        expected_out[r0:r1, c0:c1] = out_color
    assert np.array_equal(inp, expected_in)
    assert np.array_equal(out, expected_out)
    return {
        "background": background, "h_rects": h_rects, "v_rects": v_rects,
        "fronts": tuple(item[6] for item in crossings), "crossing_count": len(crossings),
        "boundary_crossing": any(r0 == 0 or r1 == height or c0 == 0 or c1 == width for r0, r1, c0, c1, *_ in crossings),
        "edge_band": any(r0 == 0 or r1 == height or c0 == 0 or c1 == width for r0, r1, c0, c1, _ in h_rects + v_rects),
    }


def _source_digest(inp, out):
    key = f"{inp.shape[0]}x{inp.shape[1]}:" + "".join(str(int(x)) for x in inp.flat)
    key += ":" + "".join(str(int(x)) for x in out.flat)
    return hashlib.sha256(key.encode()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    backgrounds = set()
    train_fronts = set()
    mixed_train = False
    train_palettes = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in inputs and (input_key, output_key) not in pairs
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            assert _source_digest(inp, out) not in _SOURCE_HASHES
            info = _analyze_pair(inp, out)
            backgrounds.add(info["background"])
            if section == "train":
                train_fronts.update(info["fronts"])
                mixed_train |= len(set(info["fronts"])) == 2
                train_palettes.add(frozenset(int(x) for x in np.unique(inp) if x != info["background"]))
    assert len(backgrounds) == 1
    assert train_fronts == {"H", "V"} and mixed_train
    assert len(train_palettes) >= 2


def _fingerprint(puzzle):
    return tuple((section, tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    shapes, palettes, layouts, counts, h_widths, v_widths, backgrounds = set(), set(), set(), set(), set(), set(), set()
    boundary, inset, mixed_test = False, False, False
    split_shapes = {"train": set(), "test": set()}
    split_counts = {"train": set(), "test": set()}
    split_boundary = {"train": False, "test": False}
    split_inset = {"train": False, "test": False}
    simple_fronts = set()
    for puzzle in puzzles:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_pair(inp, out)
                shapes.add(inp.shape)
                split_shapes[section].add(inp.shape)
                palettes.add(frozenset(int(x) for x in np.unique(inp) if x != info["background"]))
                backgrounds.add(info["background"])
                layouts.add((inp.shape, tuple(r[:4] for r in info["h_rects"]), tuple(r[:4] for r in info["v_rects"])))
                kind = (len(info["h_rects"]), len(info["v_rects"]), info["crossing_count"])
                counts.add(kind)
                split_counts[section].add(kind)
                h_widths.update(r[1] - r[0] for r in info["h_rects"])
                v_widths.update(r[3] - r[2] for r in info["v_rects"])
                boundary |= info["boundary_crossing"]
                inset |= not info["edge_band"]
                split_boundary[section] |= info["boundary_crossing"]
                split_inset[section] |= not info["edge_band"]
                if kind == (1, 1, 1):
                    simple_fronts.update(info["fronts"])
                if section == "test":
                    mixed_test |= len(set(info["fronts"])) == 2
                assert not np.array_equal(inp, out)
    assert len(shapes) >= 30 and len(palettes) >= 20 and len(layouts) >= 100
    assert len(counts) == 7 and max(h_widths) >= 6 and max(v_widths) >= 4
    assert len(backgrounds) >= 2 and simple_fronts == {"H", "V"}
    for section in ("train", "test"):
        assert (8, 8) in split_shapes[section] and (10, 10) in split_shapes[section]
        assert len(split_counts[section]) == 7
        assert split_boundary[section] and split_inset[section]
    assert boundary and inset and mixed_test
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 5, 3)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, nt, ns))
    for seed, nt, ns in ((201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (0, 1), (10, 10)):
        try:
            generate_puzzle(999, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
