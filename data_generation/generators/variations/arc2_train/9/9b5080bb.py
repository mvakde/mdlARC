"""Generate two-panel ARC scenes with sparse defects on shared-color motifs."""

import hashlib

import numpy as np


SIDES = ("top", "right", "bottom", "left")
SOURCE_HASHES = frozenset({
    "26d01e479a7b4de2a28d716eddcacfc644384abb2baf254ce95f5f8660c39bda",
    "068ed10735fdc9dc86e19697111ac502183c6909b612346b65a54b8ca16c1bae",
    "2f983ac37f864b3d24f831c701aa2980de5d6b1770a8ffe9b4dc1b5f1d4da6d5",
    "04772b8fdfc680107c80c14bc7cee95fea07725a54393c8c41ced66512f0ff79",
})


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _components(grid, color):
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row, col in np.argwhere(grid == color):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack, cells = [(row, col)], []
        seen[row, col] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]
                        and not seen[ny, nx] and grid[ny, nx] == color):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(cells)
    return result


def _panel_layout(rng, height, width, counts, fallback=False):
    if fallback:
        height, width = max(height, 24), max(width, 24)
        return height, width, ((1, 11, 1, width - 2),
                               (12, height - 2, 1, width - 2)), (0, 1)
    horizontal = bool(rng.integers(0, 2))
    length, cross = (height, width) if horizontal else (width, height)
    middle = length // 2
    first_start = _ri(rng, 0, min(3, length - 10))
    first_end = _ri(rng, max(first_start + 9, middle - 3),
                    min(length - 1, middle + 4))
    second_start = _ri(rng, max(0, middle - 4), min(length - 10, middle + 3))
    second_end = length - 1 - _ri(rng, 0, min(3, length - second_start - 10))
    spans = ((first_start, first_end), (second_start, second_end))
    panels = []
    for owner, (a, b) in enumerate(spans):
        minimum = 16 if counts[owner] == 2 else 12
        if cross < minimum:
            return None
        extent = _ri(rng, minimum, cross)
        offset = _ri(rng, 0, cross - extent)
        panels.append((a, b, offset, offset + extent - 1) if horizontal
                      else (offset, offset + extent - 1, a, b))
    order = (0, 1) if rng.integers(0, 2) else (1, 0)
    return height, width, tuple(panels), order


def _place_motifs(rng, base, owner, count, blocked=(), min_first=3):
    height, width = base.shape
    mask = (base == owner).astype(np.int16)
    prefix = np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    placed = list(blocked)
    for index in range(count):
        chosen = None
        for attempt in range(36):
            lower = min_first if index == 0 else 3
            if count == 2 and attempt >= 22:
                rh, rw = _ri(rng, lower, 5), _ri(rng, lower, 6)
            else:
                rh = _ri(rng, lower, min(11, height - 2))
                rw = _ri(rng, lower, min(13, width - 2))
            hh, ww = rh + 2, rw + 2
            if hh > height or ww > width:
                continue
            sums = (prefix[hh:, ww:] - prefix[:-hh, ww:]
                    - prefix[hh:, :-ww] + prefix[:-hh, :-ww])
            valid = sums == hh * ww
            for y0, y1, x0, x1 in placed:
                ys = np.arange(valid.shape[0])[:, None] + 1
                xs = np.arange(valid.shape[1])[None, :] + 1
                valid &= ~((ys - 1 <= y1 + 2) & (ys + rh >= y0 - 2)
                           & (xs - 1 <= x1 + 2) & (xs + rw >= x0 - 2))
            candidates = np.argwhere(valid)
            if len(candidates):
                top, left = (int(v) for v in candidates[_ri(rng, 0, len(candidates) - 1)])
                chosen = (top + 1, top + rh, left + 1, left + rw)
                break
        if chosen is None:
            return None
        placed.append(chosen)
    return placed[len(blocked):]


def _defects(rng, rect, forced_kind=None, forced_side=None):
    y0, y1, x0, x1 = rect
    options = []
    for x in range(x0 + 1, x1):
        options.extend((("top", (y0, x), (y0 - 1, x)),
                        ("bottom", (y1, x), (y1 + 1, x))))
    for y in range(y0 + 1, y1):
        options.extend((("left", (y, x0), (y, x0 - 1)),
                        ("right", (y, x1), (y, x1 + 1))))
    rng.shuffle(options)
    if forced_side is not None:
        options.sort(key=lambda item: item[0] != forced_side)
    target = _ri(rng, 1, 4)
    selected = []
    for item in options:
        side, (y, x), _ = item
        if sum(old[0] == side for old in selected) >= 2:
            continue
        if any(abs(y - old[1][0]) + abs(x - old[1][1]) < 3
               for old in selected):
            continue
        selected.append(item)
        if len(selected) == target:
            break
    kinds = [forced_kind] if forced_kind is not None else []
    kinds.extend("tab" if rng.integers(0, 2) else "notch"
                 for _ in range(len(selected) - len(kinds)))
    result = []
    for (side, boundary, outward), kind in zip(selected, kinds):
        if kind == "notch" and ((side in ("top", "bottom") and x1 - x0 + 1 == 3)
                                or (side in ("left", "right") and y1 - y0 + 1 == 3)):
            kind = "tab"
        result.append((side, kind, boundary, outward))
    return result


def _generate_pair(rng, regime, palette, training_index):
    if regime == "source_train":
        height, width = _ri(rng, 19, 27), _ri(rng, 19, 27)
        counts = tuple(1 + int(rng.random() < 0.42) for _ in range(2))
    else:
        height, width = _ri(rng, 21, 30), _ri(rng, 21, 30)
        counts = tuple(1 + int(rng.random() < 0.68) for _ in range(2))
    colors = tuple(int(v) for v in rng.permutation(palette)[:4])
    canvas, owners, motif = colors[0], colors[1:3], colors[3]
    for attempt in range(101):
        layout = _panel_layout(rng, height, width, counts, fallback=attempt == 100)
        if layout is None:
            continue
        hh, ww, panels, order = layout
        base = np.full((hh, ww), canvas, dtype=np.int8)
        for i in order:
            a, b, c, d = panels[i]
            base[a:b + 1, c:d + 1] = owners[i]
        if np.count_nonzero(base == canvas) < 20 or any(len(_components(base, color)) != 1
               or np.count_nonzero(base == color) < 55 for color in owners):
            continue
        minimum = 4 if regime == "source_train" and training_index < 2 else 3
        first = _place_motifs(rng, base, owners[0], counts[0], min_first=minimum)
        second = (None if first is None else
                  _place_motifs(rng, base, owners[1], counts[1], first, minimum))
        if first is None or second is None:
            continue
        rects = [first, second]
        break
    else:
        raise AssertionError("bounded panel construction failed")
    inp, out = base.copy(), base.copy()
    for owner_rects in rects:
        for a, b, c, d in owner_rects:
            inp[a:b + 1, c:d + 1] = motif
            out[a:b + 1, c:d + 1] = motif
    for owner_index, owner_rects in enumerate(rects):
        for j, rect in enumerate(owner_rects):
            forced = None
            forced_side = None
            if regime == "source_train" and training_index < 2 and j == 0:
                forced = ("tab", "notch")[(training_index + owner_index) % 2]
                forced_side = (("top", "right"), ("bottom", "left"))[training_index][owner_index]
            for _, kind, boundary, outward in _defects(rng, rect, forced, forced_side):
                if kind == "tab":
                    inp[outward] = motif
                else:
                    inp[boundary] = owners[owner_index]
                out[boundary] = owners[1 - owner_index]
                out[outward] = owners[1 - owner_index]
    return inp, out


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = _ri(rng, 1, 4)
        num_train = _ri(rng, max(2, num_test + 1), 10)
    elif num_train is None:
        num_train = _ri(rng, max(2, num_test + 1), 10)
    elif num_test is None:
        num_test = _ri(rng, 1, min(4, num_train - 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    # The opposite-carrier two-cell rule is the sole puzzle-wide convention.
    # A small pool relates the pairs without fixing their literal palettes.
    pool = tuple(int(v) for v in rng.permutation(10)[:_ri(rng, 5, 7)])
    shared_palette = tuple(int(v) for v in rng.choice(pool, 4, replace=False))
    fixed_palette = rng.random() < 0.20
    pair_seeds = rng.integers(0, 2**63 - 1, size=num_train + num_test,
                              dtype=np.int64)
    puzzle = {"train": [], "test": []}
    for split, count, offset, regime in (("train", num_train, 0, "source_train"),
                                         ("test", num_test, num_train, "target_test")):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + index]))
            palette = (shared_palette if fixed_palette or pair_rng.random() < 0.28
                       else tuple(int(v) for v in pair_rng.choice(pool, 4, replace=False)))
            inp, out = _generate_pair(pair_rng, regime, palette, index)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_with_motif(inp, out, motif, canvas, carriers):
    height, width = inp.shape
    comps = _components(inp, motif)
    assert 2 <= len(comps) <= 4
    expected = inp.copy()
    owner_counts = {v: 0 for v in carriers}
    types, sides, areas, rectangles = set(), set(), [], []
    repeated_side = pure = False
    for comp in comps:
        ys = np.array([p[0] for p in comp], dtype=int)
        xs = np.array([p[1] for p in comp], dtype=int)
        ycounts = np.bincount(ys, minlength=height)
        xcounts = np.bincount(xs, minlength=width)
        dense_y = np.flatnonzero(ycounts >= 3)
        dense_x = np.flatnonzero(xcounts >= 3)
        assert len(dense_y) >= 2 and len(dense_x) >= 2
        y0, y1 = int(dense_y[0]), int(dense_y[-1])
        x0, x1 = int(dense_x[0]), int(dense_x[-1])
        assert 0 < y0 < y1 < height - 1 and 0 < x0 < x1 < width - 1
        assert 3 <= y1 - y0 + 1 <= 13 and 3 <= x1 - x0 + 1 <= 15
        areas.append((y1 - y0 + 1) * (x1 - x0 + 1))
        ring = []
        for x in range(x0, x1 + 1):
            ring.extend((int(inp[y0 - 1, x]), int(inp[y1 + 1, x])))
        for y in range(y0, y1 + 1):
            ring.extend((int(inp[y, x0 - 1]), int(inp[y, x1 + 1])))
        others = set(ring) - {motif}
        assert len(others) == 1 and others <= carriers
        owner = next(iter(others))
        assert set(ring) <= {owner, motif}
        opposite = next(iter(carriers - {owner}))
        owner_counts[owner] += 1
        defects = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if inp[y, x] == motif:
                    continue
                assert inp[y, x] == owner
                boundary_sides = []
                if y == y0 and x0 < x < x1:
                    boundary_sides.append(("top", (y - 1, x)))
                if y == y1 and x0 < x < x1:
                    boundary_sides.append(("bottom", (y + 1, x)))
                if x == x0 and y0 < y < y1:
                    boundary_sides.append(("left", (y, x - 1)))
                if x == x1 and y0 < y < y1:
                    boundary_sides.append(("right", (y, x + 1)))
                assert len(boundary_sides) == 1
                side, outward = boundary_sides[0]
                defects.append((side, "notch", (y, x), outward))
        for y, x in comp:
            if y0 <= y <= y1 and x0 <= x <= x1:
                continue
            if y == y0 - 1 and x0 < x < x1:
                side, boundary = "top", (y0, x)
            elif y == y1 + 1 and x0 < x < x1:
                side, boundary = "bottom", (y1, x)
            elif x == x0 - 1 and y0 < y < y1:
                side, boundary = "left", (y, x0)
            elif x == x1 + 1 and y0 < y < y1:
                side, boundary = "right", (y, x1)
            else:
                raise AssertionError("invalid tab")
            defects.append((side, "tab", boundary, (y, x)))
        assert 1 <= len(defects) <= 4
        rectangles.append((y0, y1, x0, x1, owner, len(defects)))
        local_types, local_sides = set(), []
        for side, kind, boundary, outward in defects:
            assert inp[boundary] == (owner if kind == "notch" else motif)
            assert inp[outward] == (owner if kind == "notch" else motif)
            expected[boundary] = opposite
            expected[outward] = opposite
            types.add(kind)
            sides.add((side, kind))
            local_types.add(kind)
            local_sides.append(side)
        repeated_side |= len(local_sides) != len(set(local_sides))
        pure |= len(local_types) == 1
    assert set(owner_counts.values()) <= {1, 2}
    assert np.array_equal(expected, out)
    assert np.count_nonzero(inp == canvas) >= 1
    return {"counts": tuple(sorted(owner_counts.values())), "types": types,
            "sides": sides, "areas": areas, "repeated_side": repeated_side,
            "pure": pure, "motifs": len(comps), "canvas": canvas,
            "carriers": carriers, "motif": motif, "rectangles": rectangles}


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    colors = set(int(v) for v in np.unique(inp))
    assert len(colors) == 4 and set(int(v) for v in np.unique(out)) == colors
    diff = inp != out
    assert np.any(diff)
    carriers = set(int(v) for v in np.unique(out[diff]))
    assert len(carriers) == 2
    assert all(np.count_nonzero(inp == v) >= 25 for v in carriers)
    unused = colors - carriers
    valid = []
    for motif in unused:
        canvas = next(iter(unused - {motif}))
        try:
            valid.append(_analyze_with_motif(inp, out, motif, canvas, carriers))
        except (AssertionError, IndexError, StopIteration):
            pass
    assert len(valid) == 1
    return valid[0]


def _pair_hash(inp, out):
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()
                          + out.astype(np.uint8).tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    signatures = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            signature = (inp.shape, inp.tobytes())
            assert signature not in signatures
            signatures.add(signature)
            assert _pair_hash(inp, out) not in SOURCE_HASHES
            infos[split].append(_inspect_pair(pair))
    assert {"tab", "notch"} <= set().union(*(v["types"] for v in infos["train"]))
    assert set(SIDES) <= {side for info in infos["train"] for side, _ in info["sides"]}
    assert all(info["counts"][0] >= 1 for info in infos["train"])


def _same_puzzle(a, b):
    return all(len(a[k]) == len(b[k]) for k in ("train", "test")) and all(
        np.array_equal(pa[name], pb[name])
        for split in ("train", "test") for pa, pb in zip(a[split], b[split])
        for name in ("input", "output"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, palettes, structures = set(), set(), set()
    motif_counts, areas, side_pairs = set(), set(), set()
    train_motifs, test_motifs, defect_counts = set(), set(), set()
    motif_heights, motif_widths = set(), set()
    consistent_palettes = varied_palettes = varied_counts = 0
    role_rotation = nonrectangular = edge_contacts = adjacent = 0
    tall_motifs = wide_motifs = square_motifs = 0
    repeated = pure = smaller_test = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_areas, test_areas = [], []
        puzzle_palettes, puzzle_motifs = set(), set()
        color_roles = {color: set() for color in range(10)}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _inspect_pair(pair)
                shapes.add(inp.shape)
                palettes.add(frozenset(np.unique(inp).tolist()))
                puzzle_palettes.add(frozenset(np.unique(inp).tolist()))
                puzzle_motifs.add(info["motifs"])
                (train_motifs if split == "train" else test_motifs).add(info["motifs"])
                color_roles[info["canvas"]].add("canvas")
                color_roles[info["motif"]].add("motif")
                for owner in info["carriers"]:
                    color_roles[owner].add("carrier")
                structures.add((inp.shape, info["motifs"], tuple(sorted(info["areas"])),
                                tuple(sorted(info["sides"]))))
                motif_counts.add(info["motifs"])
                areas.update(info["areas"])
                side_pairs.update(info["sides"])
                repeated |= info["repeated_side"]
                pure |= info["pure"]
                (train_areas if split == "train" else test_areas).extend(info["areas"])
                regions = []
                parts = _components(inp, info["motif"])
                for owner in info["carriers"]:
                    mask = inp == owner
                    for y0, y1, x0, x1, own, defects in info["rectangles"]:
                        if own != owner:
                            continue
                        mask[y0:y1 + 1, x0:x1 + 1] = True
                        for part in parts:
                            if any(y0 <= y <= y1 and x0 <= x <= x1 for y, x in part):
                                for y, x in part:
                                    mask[y, x] = True
                        height, width = y1 - y0 + 1, x1 - x0 + 1
                        motif_heights.add(height)
                        motif_widths.add(width)
                        defect_counts.add(defects)
                        tall_motifs += height > width
                        wide_motifs += width > height
                        square_motifs += height == width
                    yy, xx = np.where(mask)
                    region = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
                    nonrectangular += not np.all(region)
                    edge_contacts += (yy.min() == 0 or xx.min() == 0
                                      or yy.max() == inp.shape[0] - 1
                                      or xx.max() == inp.shape[1] - 1)
                    regions.append(mask)
                adjacent += (np.any(regions[0][1:] & regions[1][:-1])
                             or np.any(regions[1][1:] & regions[0][:-1])
                             or np.any(regions[0][:, 1:] & regions[1][:, :-1])
                             or np.any(regions[1][:, 1:] & regions[0][:, :-1]))
        smaller_test |= min(test_areas) < max(train_areas)
        consistent_palettes += len(puzzle_palettes) == 1
        varied_palettes += len(puzzle_palettes) > 1
        varied_counts += len(puzzle_motifs) > 1
        role_rotation += any(len(roles) > 1 for roles in color_roles.values())
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(palettes) >= 20 and len(structures) >= 100
    assert motif_counts == {2, 3, 4} and len(areas) >= 25
    assert len(side_pairs) == 8 and repeated and pure and smaller_test
    assert train_motifs == test_motifs == {2, 3, 4}
    assert defect_counts == {1, 2, 3, 4}
    assert len(motif_heights) >= 7 and len(motif_widths) >= 9
    assert tall_motifs >= 50 and wide_motifs >= 50 and square_motifs >= 30
    assert nonrectangular >= 20 and edge_contacts >= 50 and adjacent >= 20
    assert consistent_palettes >= 1 and varied_palettes >= 5
    assert varied_counts >= 20 and role_rotation >= 20
    for nt, nq in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        seed = 700 + int(nt or 0) * 10 + int(nq or 0)
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert nq is None or len(puzzle["test"]) == nq
        assert _same_puzzle(puzzle, generate_puzzle(seed, nt, nq))
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 4), (4, 4)):
        try:
            generate_puzzle(999, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
