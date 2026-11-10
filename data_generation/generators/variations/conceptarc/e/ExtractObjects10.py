"""Generate repeated-rectangle extraction puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ("11111333221111133322", "11111333221111133322", "11111333331111133333", "11111333331111133333", "33322111113332211111", "33322111113332211111", "33333111113333311111", "33333111113333311111"),
    ("00555500", "00533500", "00000000", "00000000", "00555500", "00533500", "00000000", "00000000", "00555500", "00533500", "00000000", "00000000"),
    ("888887777788888", "888887777788888", "888887747788888", "888887777788888", "888887777788888", "777778888877777", "777778888877777", "774778888877477", "777778888877777", "777778888877777"),
    ("616444", "661444", "661444", "444616", "444661", "444661", "616444", "661444", "661444"),
    ("555111555111", "555111555111", "555111555111", "555191555191", "555191555191", "555111555111", "555111555111", "555111555111"),
}


def _key(grid):
    return (grid.shape, grid.tobytes())


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _recover(grid):
    """Find complete, foreground-filled rectangle tilings without generation state."""
    height, width = grid.shape
    answers = []
    for background in np.unique(grid):
        mask = grid != background
        cells = np.argwhere(mask)
        area = len(cells)
        if area < 8:
            continue
        top, left = map(int, cells[0])
        for mh in range(2, height - top + 1):
            for mw in range(2, width - left + 1):
                tile_area = mh * mw
                if area % tile_area or not 2 <= area // tile_area <= 9:
                    continue
                motif = grid[top:top + mh, left:left + mw]
                if np.any(motif == background) or len(np.unique(motif)) < 2:
                    continue
                covered = np.zeros(grid.shape, dtype=bool)
                origins = []
                for rr, cc in cells:
                    r, c = int(rr), int(cc)
                    if covered[r, c]:
                        continue
                    if r + mh > height or c + mw > width:
                        break
                    patch = grid[r:r + mh, c:c + mw]
                    if not np.array_equal(patch, motif):
                        break
                    if np.any(covered[r:r + mh, c:c + mw]):
                        break
                    covered[r:r + mh, c:c + mw] = True
                    origins.append((r, c))
                if len(origins) == area // tile_area and np.array_equal(covered, mask):
                    answers.append((int(background), motif.copy(), tuple(origins)))
    return answers


def _layout(origins):
    ys = {r for r, _ in origins}
    xs = {c for _, c in origins}
    if len(ys) == 1:
        return "horizontal"
    if len(xs) == 1:
        return "vertical"
    if len(origins) == len(ys) * len(xs):
        return "grid"
    if len(ys) <= 4 and len(xs) <= 4 and len(origins) >= 3:
        return "staggered"
    return "irregular"


def _gap(rng, tight=False):
    if rng.random() < (0.72 if tight else 0.22):
        return 0
    return int(rng.integers(1, 5))


def _positions(rng, kind, mh, mw):
    if kind in ("horizontal", "vertical"):
        count = int(rng.integers(2, 7))
        step = mw if kind == "horizontal" else mh
        offsets = [0]
        for _ in range(count - 1):
            offsets.append(offsets[-1] + step + _gap(rng))
        if kind == "horizontal":
            return [(0, x) for x in offsets]
        return [(y, 0) for y in offsets]

    if kind == "grid":
        nr, nc = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        ys, xs = [0], [0]
        for _ in range(nr - 1):
            ys.append(ys[-1] + mh + _gap(rng))
        for _ in range(nc - 1):
            xs.append(xs[-1] + mw + _gap(rng))
        return [(y, x) for y in ys for x in xs]

    if kind == "staggered":
        nr, nc = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        ys, xs = [0], [0]
        for _ in range(nr - 1):
            ys.append(ys[-1] + mh + _gap(rng, True))
        for _ in range(nc - 1):
            xs.append(xs[-1] + mw + _gap(rng, True))
        parity = int(rng.integers(2))
        places = [(ys[r], xs[c]) for r in range(nr) for c in range(nc) if (r + c) % 2 == parity]
        if len(places) > 8:
            places = places[:8]
        return places

    # Free placement gives unequal horizontal and vertical offsets.
    count = int(rng.integers(3, 7))
    span_h = min(30, max(2 * mh + 2, 4 * mh + 3))
    span_w = min(30, max(2 * mw + 2, 4 * mw + 3))
    places = [(0, 0)]
    for _ in range(100):
        if len(places) == count:
            break
        y = int(rng.integers(0, span_h - mh + 1))
        x = int(rng.integers(0, span_w - mw + 1))
        if all(y + mh <= a or a + mh <= y or x + mw <= b or b + mw <= x for a, b in places):
            places.append((y, x))
    if len(places) != count:
        raise ValueError("Could not place irregular copies")
    return places


def _motif(rng, height, width, colors):
    main, accent = colors[:2]
    result = np.full((height, width), main, dtype=np.int8)
    style = int(rng.integers(0, 8))
    cells = set()
    if style == 0:
        cells.add((int(rng.integers(height)), int(rng.integers(width))))
    elif style in (1, 2):
        if style == 1:
            row = int(rng.integers(height))
            length = int(rng.integers(2, width + 1))
            start = int(rng.integers(width - length + 1))
            cells = {(row, c) for c in range(start, start + length)}
        else:
            col = int(rng.integers(width))
            length = int(rng.integers(2, height + 1))
            start = int(rng.integers(height - length + 1))
            cells = {(r, col) for r in range(start, start + length)}
    elif style == 3:
        ph = int(rng.integers(1, height + 1))
        pw = int(rng.integers(1, width + 1))
        r = int(rng.integers(height - ph + 1))
        c = int(rng.integers(width - pw + 1))
        cells = {(y, x) for y in range(r, r + ph) for x in range(c, c + pw)}
    elif style == 4:
        r = int(rng.integers(height - 1))
        c = int(rng.integers(width - 1))
        arm_h = int(rng.integers(2, height - r + 1))
        arm_w = int(rng.integers(2, width - c + 1))
        cells = {(y, c) for y in range(r, r + arm_h)} | {(r, x) for x in range(c, c + arm_w)}
    elif style == 5:
        for _ in range(int(rng.integers(2, 5))):
            ph = int(rng.integers(1, max(2, height // 2 + 1)))
            pw = int(rng.integers(1, max(2, width // 2 + 1)))
            r = int(rng.integers(height - ph + 1))
            c = int(rng.integers(width - pw + 1))
            cells.update((y, x) for y in range(r, r + ph) for x in range(c, c + pw))
    elif style == 6:
        # A broad asymmetric region, often close to half the motif.
        if rng.random() < 0.5:
            cut = int(rng.integers(1, width))
            cells = {(r, c) for r in range(height) for c in range(cut)}
        else:
            cut = int(rng.integers(1, height))
            cells = {(r, c) for r in range(cut) for c in range(width)}
        for _ in range(max(1, height * width // 8)):
            cell = (int(rng.integers(height)), int(rng.integers(width)))
            if cell in cells:
                cells.remove(cell)
            else:
                cells.add(cell)
    else:
        density = float(rng.uniform(0.35, 0.6))
        cells = {(r, c) for r in range(height) for c in range(width) if rng.random() < density}
    if not cells or len(cells) == height * width:
        cells = {(height // 2, width // 2)}
    for r, c in cells:
        result[r, c] = accent
    if len(colors) == 3:
        third = colors[2]
        accent_cells = sorted(cells)
        pick = rng.choice(len(accent_cells), size=max(1, len(accent_cells) // 3), replace=False)
        for index in np.atleast_1d(pick):
            r, c = accent_cells[int(index)]
            result[r, c] = third
    return result


def _palette(rng, previous):
    size = 4 if rng.random() < 0.16 else 3
    if previous and rng.random() < 0.3:
        keep = list(rng.choice(previous[int(rng.integers(len(previous)))], size=int(rng.integers(1, min(3, size))), replace=False))
        rest = [v for v in range(10) if v not in keep]
        colors = keep + list(rng.choice(rest, size=size - len(keep), replace=False))
        rng.shuffle(colors)
        return tuple(map(int, colors))
    return tuple(map(int, rng.choice(10, size=size, replace=False)))


def _make_pair(rng, kind, previous):
    palette = _palette(rng, previous)
    background, colors = palette[0], palette[1:]
    mh = int(rng.integers(2, 11))
    mw = int(rng.integers(2, 11))
    origins = _positions(rng, kind, mh, mw)
    if len(origins) < 2 or len(origins) > 9:
        raise ValueError("Unsupported copy count")
    core_h = max(r + mh for r, _ in origins)
    core_w = max(c + mw for _, c in origins)
    if core_h > 30 or core_w > 30:
        raise ValueError("Layout exceeds grid")
    margins = []
    for core in (core_h, core_w):
        free = 30 - core
        a = 0
        if free and rng.random() >= 0.57:
            a = int(rng.integers(1, min(4, free) + 1))
        b = 0
        if free > a and rng.random() >= 0.57:
            b = int(rng.integers(1, min(4, free - a) + 1))
        margins.append((a, b))
    (top, bottom), (left, right) = margins
    motif = _motif(rng, mh, mw, colors)
    canvas = np.full((top + core_h + bottom, left + core_w + right), background, dtype=np.int8)
    for r, c in origins:
        canvas[top + r:top + r + mh, left + c:left + c + mw] = motif
    return canvas, motif.copy(), palette


def _valid_count(value, low, high, label):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or not low <= int(value) <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _valid_count(num_train, 2, 10, "num_train")
    if num_test is not None:
        num_test = _valid_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_train = 2 + int(seed) % 9
        num_test = 1 + (int(seed) // 9) % min(4, num_train - 1)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    kinds = ["horizontal", "vertical", "grid", "staggered", "irregular"]
    train_kinds = list(rng.choice(kinds, size=num_train))
    if train_kinds[1] == train_kinds[0]:
        train_kinds[1] = kinds[(kinds.index(train_kinds[0]) + int(rng.integers(1, 5))) % 5]
    test_kinds = list(rng.choice(kinds, size=num_test, p=[0.17, 0.17, 0.18, 0.32, 0.16]))
    puzzle = {"train": [], "test": []}
    used = set()
    palettes = []
    for split, schedule in (("train", train_kinds), ("test", test_kinds)):
        for index, kind in enumerate(schedule):
            for _ in range(240):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                try:
                    inp, out, palette = _make_pair(pair_rng, kind, palettes)
                except ValueError:
                    continue
                key = _key(inp)
                if key in used or _rows(inp) in _SOURCE_INPUTS:
                    continue
                if split == "train" and index == 1:
                    first = puzzle["train"][0]
                    if out.shape == first["output"].shape or set(palette) == set(np.unique(first["input"])):
                        continue
                recovered = _recover(inp)
                if len(recovered) != 1 or not np.array_equal(recovered[0][1], out):
                    continue
                if _layout(recovered[0][2]) != kind:
                    continue
                puzzle[split].append({"input": inp, "output": out})
                palettes.append(palette)
                used.add(key)
                break
            else:
                raise RuntimeError(f"Could not generate an unambiguous {kind} pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_shapes, train_palettes, train_layouts = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert _rows(inp) not in _SOURCE_INPUTS
            candidates = _recover(inp)
            assert len(candidates) == 1
            background, motif, origins = candidates[0]
            assert np.array_equal(out, motif)
            assert background not in out
            assert 2 <= len(origins) <= 9
            assert out.shape != inp.shape
            assert len(np.unique(out)) in (2, 3)
            if split == "train":
                train_shapes.add(out.shape)
                train_palettes.add(frozenset(map(int, np.unique(inp))))
                train_layouts.add(_layout(origins))
    assert len(train_shapes) >= 2
    assert len(train_palettes) >= 2
    assert len(train_layouts) >= 2


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[g], b[g]) for a, b in zip(first[s], second[s]) for g in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    layouts, copy_counts, motif_shapes, palettes, structures = set(), set(), set(), set(), set()
    accent_masks = set()
    dense = multi = edge = tight = 0
    two_vertical = long_horizontal = grid_2x2 = all_zero_stagger = unequal_linear = 0
    reused = changed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local_palettes = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                candidates = _recover(inp)
                assert len(candidates) == 1 and np.array_equal(candidates[0][1], out)
                bg, _, origins = candidates[0]
                layout = _layout(origins)
                layouts.add(layout)
                copy_counts.add(len(origins))
                motif_shapes.add(out.shape)
                palette = frozenset(map(int, np.unique(inp)))
                palettes.add(palette)
                local_palettes.append(palette)
                structures.add((layout, inp.shape, out.shape, tuple(origins)))
                counts = np.unique(out, return_counts=True)[1]
                values = np.unique(out)
                rare = values[int(np.argmin(counts))]
                accent_masks.add((out.shape, (out == rare).tobytes()))
                dense += int(min(counts) / out.size >= 0.34)
                multi += int(len(np.unique(out)) == 3)
                edge += int(np.any(inp[0] != bg) and np.any(inp[-1] != bg) and np.any(inp[:, 0] != bg) and np.any(inp[:, -1] != bg))
                tight += int(layout == "staggered" and any(abs(r - a) == out.shape[0] and abs(c - b) == out.shape[1] for r, c in origins for a, b in origins if (r, c) != (a, b)))
                ys = sorted({r for r, _ in origins})
                xs = sorted({c for _, c in origins})
                two_vertical += int(layout == "vertical" and len(origins) == 2)
                long_horizontal += int(layout == "horizontal" and len(origins) >= 3)
                grid_2x2 += int(layout == "grid" and len(ys) == len(xs) == 2)
                all_zero_stagger += int(layout == "staggered" and all(b - a == out.shape[0] for a, b in zip(ys, ys[1:])) and all(b - a == out.shape[1] for a, b in zip(xs, xs[1:])))
                positions = xs if layout == "horizontal" else ys if layout == "vertical" else []
                unequal_linear += int(bool(positions) and len({b - a for a, b in zip(positions, positions[1:])}) > 1)
        reused += int(len(set(local_palettes)) < len(local_palettes))
        changed += int(len(set(local_palettes)) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert layouts == {"horizontal", "vertical", "grid", "staggered", "irregular"}
    assert len(copy_counts) >= 5 and max(copy_counts) >= 5
    assert len(motif_shapes) >= 15 and len(palettes) >= 20 and len(structures) >= 80
    assert len(accent_masks) >= 30
    assert dense >= 10 and multi >= 10 and edge >= 10 and tight >= 10
    assert two_vertical >= 5 and long_horizontal >= 5 and grid_2x2 >= 5
    assert all_zero_stagger >= 5 and unequal_linear >= 5
    assert reused >= 2 and changed == 42
    for offset, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + offset, nt, ne)
        assert _same(puzzle, generate_puzzle(1000 + offset, nt, ne))
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ne is not None:
            assert len(puzzle["test"]) == ne
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {kwargs}")
