"""Generate ARC scenes where the filled rectangle determines a blank output."""

import numpy as np

_SOURCE_INPUTS = {
    tuple(s.split()) for s in (
        "0000000000 0002220000 0002020000 0002220000 0000000000 0330011110 0330010010 0330011110 0000000000 0000000000",
        "0000000000 0000044440 0000044440 0777044440 0707044440 0707000000 0707088800 0707080800 0777088800 0000000000",
        "0666660888 0600060808 0666660888 0000000000 0000004440 0222004040 0222004040 0222004440 0222000000 0222000000",
        "0000008880 0222008880 0202008880 0202008880 0202008880 0222008880 0000000000 0555005555 0505005005 0555005555",
        "0000000000000 0555555500000 0555555500000 0555555500000 0555555500000 0000000000000",
        "0111111110 0100000010 0109999010 0109999010 0100000010 0111111110",
    )
}


def _randint(rng, low, high):
    return int(rng.integers(low, high + 1))


def _canvas(rng, mode):
    if mode == "compact":
        if rng.random() < 0.30:
            return ((10, 10), (6, 10), (6, 13))[_randint(rng, 0, 2)]
        return _randint(rng, 7, 15), _randint(rng, 7, 17)
    if mode == "middle":
        return _randint(rng, 12, 23), _randint(rng, 12, 24)
    return _randint(rng, 20, 30), _randint(rng, 20, 30)


def _outline_side(rng, limit):
    cap = min(10, limit)
    if cap < 3:
        return None
    if rng.random() < 0.72:
        cap = min(5, cap)
    return _randint(rng, 3, cap)


def _object_specs(rng, height, width, count, anchor, desired_size):
    if desired_size is not None:
        solid_h, solid_w = desired_size
    elif anchor == 0:
        solid_h, solid_w = _randint(rng, 2, 3), _randint(rng, 2, 4)
    elif anchor == 1:
        solid_h, solid_w = _randint(rng, 4, 7), _randint(rng, 4, 7)
    else:
        choice = rng.random()
        if choice < 0.28:
            solid_h, solid_w = _randint(rng, 6, 10), _randint(rng, 2, 6)
        elif choice < 0.56:
            solid_h, solid_w = _randint(rng, 2, 6), _randint(rng, 6, 10)
        else:
            solid_h, solid_w = _randint(rng, 2, 8), _randint(rng, 2, 8)
    if solid_h > height or solid_w > width:
        return None
    nested = count >= 2 and anchor is None and rng.random() < 0.32
    if nested:
        max_h, max_w = min(height, solid_h + 7), min(width, solid_w + 7)
        if max_h < solid_h + 4 or max_w < solid_w + 4:
            return None
        frame_h = _randint(rng, solid_h + 4, max_h)
        frame_w = _randint(rng, solid_w + 4, max_w)
    outlines = []
    for index in range(count - 1):
        if nested and index == 0:
            outlines.append((frame_h, frame_w))
        elif anchor == 0 and index == 0:
            outlines.append((_randint(rng, 5, min(8, height)),
                             _randint(rng, 5, min(8, width))))
        elif anchor == 1 and index == 0:
            outlines.append((3, 3))
        else:
            outlines.append((_outline_side(rng, height),
                             _outline_side(rng, width)))
    if any(h is None or w is None for h, w in outlines):
        return None
    if anchor == 0 and outlines[0][0] * outlines[0][1] <= solid_h * solid_w:
        return None
    if anchor == 1 and outlines[0][0] * outlines[0][1] >= solid_h * solid_w:
        return None
    return (solid_h, solid_w), outlines, nested


def _colors(rng, palette, background, count, anchor_colors, anchor):
    available = [c for c in palette if c != background]
    if anchor == 0:
        solid, first = anchor_colors
    elif anchor == 1:
        first, solid = anchor_colors
    else:
        solid = available[_randint(rng, 0, len(available) - 1)]
        first = None
    outlines = []
    for index in range(count - 1):
        if index == 0 and first is not None:
            outlines.append(first)
        else:
            choices = [c for c in available if c != solid and c not in outlines]
            outlines.append(choices[_randint(rng, 0, len(choices) - 1)])
    if anchor is None and outlines:
        if len(outlines) >= 2 and rng.random() < 0.42:
            i, j = rng.choice(len(outlines), 2, replace=False)
            outlines[int(j)] = outlines[int(i)]
        if rng.random() < 0.24:
            outlines[_randint(rng, 0, len(outlines) - 1)] = solid
    return solid, outlines


def _separate(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah < br or br + bh < ar or
            ac + aw < bc or bc + bw < ac)


def _place(rng, height, width, solid_size, outlines, nested, colors,
           edge_bias):
    solid_h, solid_w = solid_size
    solid_color, outline_colors = colors
    if nested:
        footprints = [(outlines[0][0], outlines[0][1], "frame", 0)]
        start = 1
    else:
        footprints = [(solid_h, solid_w, "solid", -1)]
        start = 0
    for index in range(start, len(outlines)):
        footprints.append((*outlines[index], "outline", index))
    footprints.sort(key=lambda item: item[0] * item[1], reverse=True)
    for _ in range(5):
        chosen = []
        order = footprints.copy()
        if edge_bias and not nested:
            order.sort(key=lambda item: item[2] != "solid")
        for h, w, kind, index in order:
            positions = [(r, c) for r in range(height - h + 1)
                         for c in range(width - w + 1) if not (r == 0 and c == 0)]
            rng.shuffle(positions)
            if edge_bias and not chosen:
                positions.sort(key=lambda rc: not (
                    rc[0] == 0 or rc[1] == 0 or
                    rc[0] + h == height or rc[1] + w == width))
            found = None
            for r, c in positions:
                candidate = (r, c, h, w)
                if all(_separate(candidate, item[0]) for item in chosen):
                    found = candidate
                    break
            if found is None:
                break
            chosen.append((found, kind, index))
        if len(chosen) == len(order):
            break
    else:
        return None
    grid = np.full((height, width), -1, dtype=np.int8)
    solid_position = None
    for (r, c, h, w), kind, index in chosen:
        if kind == "solid":
            solid_position = (r, c)
        elif kind == "frame":
            solid_position = (_randint(rng, r + 2, r + h - solid_h - 2),
                              _randint(rng, c + 2, c + w - solid_w - 2))
        if kind != "solid":
            color = outline_colors[index]
            grid[r, c:c + w] = color
            grid[r + h - 1, c:c + w] = color
            grid[r:r + h, c] = color
            grid[r:r + h, c + w - 1] = color
    r, c = solid_position
    grid[r:r + solid_h, c:c + solid_w] = solid_color
    return grid


def _make_pair(rng, background, palette, anchor=None,
               anchor_colors=None, desired_size=None, regime="train"):
    for _ in range(180):
        if anchor is not None:
            mode = rng.choice(("compact", "middle", "spacious"),
                              p=(0.70, 0.22, 0.08))
            count = 3 if anchor == 0 else 4
        else:
            mode = rng.choice(("compact", "middle", "spacious"),
                              p=(0.70, 0.22, 0.08))
            weights = (0.10, 0.19, 0.38, 0.33) if regime == "train" else (
                0.16, 0.17, 0.29, 0.38)
            count = int(rng.choice((1, 2, 3, 4), p=weights))
        height, width = _canvas(rng, mode)
        specs = _object_specs(rng, height, width, count, anchor, desired_size)
        if specs is None:
            continue
        solid_size, outlines, nested = specs
        colors = _colors(rng, palette, background, count, anchor_colors, anchor)
        grid = _place(rng, height, width, solid_size, outlines, nested,
                      colors, edge_bias=rng.random() < 0.53)
        if grid is None:
            continue
        grid[grid == -1] = background
        values, frequencies = np.unique(grid, return_counts=True)
        background_cells = int(frequencies[np.flatnonzero(values == background)[0]])
        if any(int(n) >= background_cells for color, n in
               zip(values, frequencies) if int(color) != background):
            continue
        output = np.full(solid_size, background, dtype=np.int8)
        pair = {"input": grid, "output": output}
        try:
            _pair_info(pair)
        except AssertionError:
            continue
        if tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS:
            continue
        return pair
    raise RuntimeError("Could not place a valid rectangle scene")


def _find_objects(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = set()
            while stack:
                rr, cc = stack.pop()
                cells.add((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc),
                               (rr, cc - 1), (rr, cc + 1)):
                    if (0 <= nr < height and 0 <= nc < width and
                            not seen[nr, nc] and grid[nr, nc] == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, r1 = min(x for x, _ in cells), max(x for x, _ in cells)
            c0, c1 = min(x for _, x in cells), max(x for _, x in cells)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            full = {(y, x) for y in range(r0, r1 + 1)
                    for x in range(c0, c1 + 1)}
            border = {(y, x) for y, x in full
                      if y in (r0, r1) or x in (c0, c1)}
            if cells == full and h >= 2 and w >= 2:
                kind = "solid"
            elif (cells == border and h >= 3 and w >= 3 and
                  np.any(grid[r0 + 1:r1, c0 + 1:c1] == background)):
                kind = "outline"
            else:
                raise AssertionError("Malformed rectangle")
            objects.append({"kind": kind, "color": color,
                            "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                            "height": h, "width": w})
    return objects


def _pair_info(pair):
    assert type(pair) is dict and set(pair) == {"input", "output"}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    grid, output = pair["input"], pair["output"]
    background = int(grid[0, 0])
    values, frequencies = np.unique(grid, return_counts=True)
    background_cells = int(frequencies[np.flatnonzero(values == background)[0]])
    assert all(int(n) < background_cells for color, n in
               zip(values, frequencies) if int(color) != background)
    objects = _find_objects(grid, background)
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    assert len(solids) == 1 and 1 <= len(objects) <= 4
    solid = solids[0]
    assert output.shape == (solid["height"], solid["width"])
    assert np.all(output == background)
    assert not np.array_equal(grid, output)
    nested = False
    for i, first in enumerate(objects):
        for second in objects[i + 1:]:
            overlap = (first["r0"] <= second["r1"] and
                       second["r0"] <= first["r1"] and
                       first["c0"] <= second["c1"] and
                       second["c0"] <= first["c1"])
            if overlap:
                outer, inner = ((first, second) if first["kind"] == "outline"
                                else (second, first))
                assert not nested and outer["kind"] == "outline"
                assert inner["kind"] == "solid"
                assert (inner["r0"] >= outer["r0"] + 2 and
                        inner["r1"] <= outer["r1"] - 2 and
                        inner["c0"] >= outer["c0"] + 2 and
                        inner["c1"] <= outer["c1"] - 2)
                nested = True
            else:
                assert (first["r1"] + 1 < second["r0"] or
                        second["r1"] + 1 < first["r0"] or
                        first["c1"] + 1 < second["c0"] or
                        second["c1"] + 1 < first["c0"])
    return background, objects, solid, nested


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    infos, seen = [], set()
    for pair in train + test:
        info = _pair_info(pair)
        grid = pair["input"]
        key = (grid.shape, grid.tobytes())
        assert key not in seen
        seen.add(key)
        rows = tuple("".join(str(int(v)) for v in row) for row in grid)
        assert rows not in _SOURCE_INPUTS
        infos.append(info)
    a, b = infos[:2]
    assert len(a[1]) >= 2 and len(b[1]) >= 2
    assert not a[3] and not b[3]
    assert a[2]["color"] in {o["color"] for o in b[1]
                               if o["kind"] == "outline"}
    assert b[2]["color"] in {o["color"] for o in a[1]
                               if o["kind"] == "outline"}
    area_a = a[2]["height"] * a[2]["width"]
    area_b = b[2]["height"] * b[2]["width"]
    assert any(o["height"] * o["width"] > area_a for o in a[1]
               if o["kind"] == "outline")
    assert any(o["height"] * o["width"] < area_b for o in b[1]
               if o["kind"] == "outline")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    for value, minimum, maximum in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                not isinstance(value, (int, np.integer)) or
                not minimum <= value <= maximum):
            raise ValueError("Illegal pair count")
    if num_test is None:
        num_test = _randint(rng, 1, min(4, num_train - 1)) if num_train else _randint(rng, 1, 4)
    if num_train is None:
        num_train = _randint(rng, max(2, num_test + 1), 10)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    palette = tuple(int(c) for c in rng.permutation(10))
    base = palette[0]
    mixed_backgrounds = rng.random() < 0.43
    backgrounds = [base] * (num_train + num_test)
    if mixed_backgrounds:
        backgrounds[1] = palette[1]
        for index in range(2, len(backgrounds)):
            if rng.random() < 0.48:
                backgrounds[index] = palette[_randint(rng, 1, 9)]
    anchor_colors = [c for c in palette if c not in backgrounds[:2]][:2]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes = []
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            position = index if regime == "train" else num_train + index
            anchor = index if regime == "train" and index < 2 else None
            desired_size = None
            if regime == "test" and rng.random() < 0.34:
                desired_size = train_sizes[_randint(rng, 0, len(train_sizes) - 1)]
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, backgrounds[position], palette,
                                  anchor, anchor_colors, desired_size, regime)
                grid = pair["input"]
                key = (grid.shape, grid.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    if regime == "train":
                        train_sizes.append(pair["output"].shape)
                    break
            else:
                raise RuntimeError("Could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(first, second):
    return all(len(first[k]) == len(second[k]) and all(
        np.array_equal(a[t], b[t]) for a, b in zip(first[k], second[k])
        for t in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    counts = {"train": set(), "test": set()}
    shapes, solid_sizes, backgrounds, layouts = set(), set(), set(), set()
    edge = compact = nested = repeated = shared = single = four_repeat = 0
    overlap = 0
    novel_size = source_canvas = dense_compact_four = small_nested_test = 0
    nested_extra = mixed_puzzles = 0
    train_single = train_nested = held_single = held_nested = 0
    outline_shapes = set()
    nested_offsets = set()
    repeat_patterns = set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        sizes = {p["output"].shape for p in puzzle["train"]}
        train_infos = [_pair_info(p) for p in puzzle["train"]]
        test_infos = [_pair_info(p) for p in puzzle["test"]]
        has_single = any(len(info[1]) == 1 for info in train_infos)
        has_nested = any(info[3] for info in train_infos)
        train_single += has_single
        train_nested += has_nested
        held_single += not has_single and any(len(info[1]) == 1
                                              for info in test_infos)
        held_nested += not has_nested and any(info[3] for info in test_infos)
        mixed_puzzles += len({info[0] for info in train_infos + test_infos}) > 1
        assert len({len(info[1]) for info in train_infos + test_infos}) > 1
        assert len({pair["input"].shape for regime in ("train", "test")
                    for pair in puzzle[regime]}) > 1
        for regime in ("train", "test"):
            counts[regime].add(len(puzzle[regime]))
            for pair in puzzle[regime]:
                grid = pair["input"]
                bg, objects, solid, is_nested = _pair_info(pair)
                assert pair["output"].shape == (solid["height"], solid["width"])
                assert np.all(pair["output"] == bg)
                shapes.add(grid.shape)
                solid_sizes.add(pair["output"].shape)
                backgrounds.add(bg)
                layouts.add((len(objects), is_nested, grid.shape))
                edge += any(o["r0"] == 0 or o["c0"] == 0 or
                            o["r1"] == grid.shape[0] - 1 or
                            o["c1"] == grid.shape[1] - 1 for o in objects)
                compact += grid.shape[0] <= 15 and grid.shape[1] <= 17
                source_canvas += grid.shape in ((10, 10), (6, 10), (6, 13))
                dense_compact_four += (len(objects) == 4 and
                    grid.shape[0] <= 15 and grid.shape[1] <= 17 and
                    np.mean(grid != bg) >= 0.24)
                nested += is_nested
                single += len(objects) == 1
                outlines = [o for o in objects if o["kind"] == "outline"]
                outline_shapes.update((o["height"], o["width"]) for o in outlines)
                if is_nested:
                    nested_extra += len(objects) >= 3
                    outer = next(o for o in outlines if
                        o["r0"] < solid["r0"] < solid["r1"] < o["r1"] and
                        o["c0"] < solid["c0"] < solid["c1"] < o["c1"])
                    nested_offsets.add((solid["r0"] - outer["r0"] - 1,
                                        solid["c0"] - outer["c0"] - 1))
                    if regime == "test":
                        small_nested_test += max(pair["output"].shape) <= 5
                rep = len({o["color"] for o in outlines}) < len(outlines)
                repeated += rep
                four_repeat += rep and len(objects) == 4
                if rep and len(objects) == 4:
                    repeat_patterns.add(tuple(i for i, o in enumerate(outlines)
                        if sum(x["color"] == o["color"] for x in outlines) > 1))
                shared += any(o["color"] == solid["color"] for o in outlines)
                if regime == "test":
                    overlap += pair["output"].shape in sizes
                    novel_size += pair["output"].shape not in sizes
    assert len(counts["train"]) >= 4 and len(counts["test"]) >= 3
    assert len(shapes) >= 45 and len(solid_sizes) >= 25
    assert len(backgrounds) >= 8 and len(layouts) >= 80
    assert min(edge, compact, nested, repeated, shared, single,
               four_repeat, overlap) > 0
    assert min(novel_size, source_canvas, dense_compact_four,
               small_nested_test, nested_extra, mixed_puzzles,
               train_single, train_nested, held_single, held_nested) > 0
    assert len(outline_shapes) >= 30 and len(nested_offsets) >= 4
    assert len(repeat_patterns) == 3
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(115, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
        assert _same_puzzle(p, generate_puzzle(115, nt, ns))
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(116, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(117, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
