import numpy as np


_SOURCE_INPUTS = {
    tuple(s.split()) for s in (
        """0000000
0333330
0300030
0300030
0300030
0333330
0000000""",
        """0000000000
0444000000
0404044444
0444040004
0000040004
0000040004
0000044444
0000000000
0000000000
0000000000""",
        """00000000000
00000000000
00022222000
00020002000
00020002000
00020002000
00022222000
00000000000
00000000000
00000000000
00000000000""",
        """000000000000000
000000000000000
066600006666666
060600006000006
066600006000006
000000006000006
000000006000006
066666006000006
060006006666666
060006000000000
060006000000000
066666000000000
000000000000000
000000000000000
000000000000000""",
        """00000000000000000000
00000010000000909090
00000101000000000000
00001000100000900090
00010000010000000000
00100000001000909090
00010000010000000000
00001000100000000000
00000101000000000000
00000010000777770000
00000000000700070000
00000800000700070000
00008080000700070000
00880008800777770000
00800000800000000000
00800000800000000000
00800000800000000000
00880008800000000000
00008080000000000000
00000800000000000000""",
    )
}


def _mask(kind, h, w, option=0):
    m = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    if kind == "frame":
        m[0, :] = m[-1, :] = True
        m[:, 0] = m[:, -1] = True
    elif kind == "diamond":
        for y in range(h):
            x = int(round(cx * (1 - abs(y - cy) / cy)))
            m[y, cx - x] = m[y, cx + x] = True
    elif kind == "stepped":
        shoulder = 1 + option % 2
        for y in range(h):
            d = min(y, h - 1 - y)
            if d == 0:
                m[y, cx] = True
            elif d == 1:
                m[y, cx - 1] = m[y, cx + 1] = True
            elif d == 2:
                m[y, :shoulder + 1] = True
                m[y, w - shoulder - 1:] = True
            else:
                m[y, 0] = m[y, -1] = True
    elif kind == "inset":
        depth = 1 + option % 2
        neck = 1 + option // 2
        m[0, :] = m[-1, :] = True
        for y in range(1, h - 1):
            x = depth if abs(y - cy) <= neck else 0
            m[y, x] = m[y, w - 1 - x] = True
    elif kind == "sparse":
        xs = {0, w - 1, cx}
        ys = {0, h - 1, cy}
        if option & 1 and cx > 1:
            xs.update((1, w - 2))
        if option & 2 and cy > 1:
            ys.update((1, h - 2))
        for x in xs:
            m[0, x] = m[-1, x] = True
        for y in ys:
            m[y, 0] = m[y, -1] = True
    else:
        raise ValueError(kind)
    if m[cy, cx] or not np.array_equal(m, m[::-1, :]) or not np.array_equal(m, m[:, ::-1]):
        raise ValueError((kind, h, w, option))
    return m


def _build_shapes():
    shapes = {}
    for h in (3, 5, 7, 9, 11):
        for w in (3, 5, 7, 9, 11):
            for kind in ("frame", "diamond", "stepped", "inset", "sparse"):
                if kind == "diamond" and min(h, w) < 5:
                    continue
                if kind == "stepped" and (h < 7 or w < 7):
                    continue
                if kind == "inset" and (h < 7 or w < 7):
                    continue
                if kind == "sparse" and min(h, w) < 5:
                    continue
                options = range(4) if kind in ("inset", "sparse") else range(2) if kind == "stepped" else range(1)
                for option in options:
                    try:
                        mask = _mask(kind, h, w, option)
                    except ValueError:
                        continue
                    key = (mask.shape, mask.tobytes())
                    shapes.setdefault(key, (kind, mask))
    return tuple(shapes.values())


_SHAPES = _build_shapes()
_BY_KIND = {
    kind: tuple(mask for name, mask in _SHAPES if name == kind)
    for kind in ("frame", "diamond", "stepped", "inset", "sparse")
}


def _grid_key(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _extract_motifs(grid):
    bg = _background(grid)
    foreground = set(map(tuple, np.argwhere(grid != bg)))
    assert foreground
    by_cell = {cell: [] for cell in foreground}
    candidates = []
    colors = sorted(set(int(v) for v in grid.flat) - {bg})
    for color in colors:
        points = np.argwhere(grid == color)
        for kind, mask in _SHAPES:
            h, w = mask.shape
            if h > grid.shape[0] or w > grid.shape[1]:
                continue
            first = np.argwhere(mask)[0]
            expected = np.where(mask, color, bg)
            offsets = np.argwhere(mask)
            for point in points:
                r, c = int(point[0] - first[0]), int(point[1] - first[1])
                if r < 0 or c < 0 or r + h > grid.shape[0] or c + w > grid.shape[1]:
                    continue
                if not np.array_equal(grid[r:r+h, c:c+w], expected):
                    continue
                cells = frozenset((r + int(y), c + int(x)) for y, x in offsets)
                candidate = (kind, color, (r + h // 2, c + w // 2), (r, c, r + h - 1, c + w - 1), cells)
                index = len(candidates)
                candidates.append(candidate)
                for cell in cells:
                    by_cell[cell].append(index)

    solutions = []

    def separate(a, b, same_color):
        gap = 1 if same_color else 0
        return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]

    def search(remaining, chosen):
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(tuple(chosen))
            return
        best = None
        for cell in remaining:
            options = [i for i in by_cell[cell] if candidates[i][4] <= remaining and all(separate(candidates[i][3], candidates[j][3], candidates[i][1] == candidates[j][1]) for j in chosen)]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
            if len(best) == 1:
                break
        for i in best:
            search(remaining - candidates[i][4], chosen + [i])

    search(foreground, [])
    assert len(solutions) == 1, "motif grouping is ambiguous"
    return [candidates[i][:4] for i in solutions[0]]


def _choose_masks(rng, count, regime, pair_index):
    masks = []
    if regime == "train":
        if pair_index == 0:
            sizes = [int(rng.choice((5, 7)))]
        elif pair_index == 1:
            sizes = list(rng.choice((3, 5, 7), size=2, replace=False))
        else:
            sizes = [int(v) for v in rng.choice((3, 5, 7), size=count)]
        for size in sizes:
            masks.append(("frame", _mask("frame", size, size)))
    else:
        kinds = [str(v) for v in rng.choice(("frame", "diamond", "stepped", "inset", "sparse"), size=count, replace=True)]
        if pair_index == 0 and count >= 3 and all(k == "frame" for k in kinds):
            kinds[-1] = "stepped"
        for kind in kinds:
            options = _BY_KIND[kind]
            mask = options[int(rng.integers(len(options)))]
            masks.append((kind, mask))
    return masks


def _choose_colors(rng, count, bg, regime, pair_index):
    available = np.array([c for c in range(10) if c != bg], dtype=np.int8)
    if regime == "train":
        ncolors = 1
    else:
        ncolors = int(rng.integers(1, min(count, 4) + 1))
    chosen = [int(v) for v in rng.choice(available, size=ncolors, replace=False)]
    return [chosen[i % ncolors] for i in rng.permutation(count)]


def _place_pair(rng, motifs, bg):
    count = len(motifs)
    layouts = {
        1: ((1, 1),),
        2: ((1, 2), (2, 1), (2, 2)),
        3: ((1, 3), (3, 1), (2, 2)),
        4: ((1, 4), (4, 1), (2, 2)),
    }[count]
    for _ in range(24):
        rows, cols = layouts[int(rng.integers(len(layouts)))]
        slots = list(range(rows * cols))
        rng.shuffle(slots)
        slots = slots[:count]
        assignments = list(zip(slots, motifs))
        row_heights = [max((mask.shape[0] for slot, (_, mask, _) in assignments if slot // cols == y), default=3) + int(rng.integers(0, 3)) for y in range(rows)]
        col_widths = [max((mask.shape[1] for slot, (_, mask, _) in assignments if slot % cols == x), default=3) + int(rng.integers(0, 3)) for x in range(cols)]
        row_gaps = [int(rng.integers(1, 6)) for _ in range(rows - 1)]
        col_gaps = [int(rng.integers(1, 6)) for _ in range(cols - 1)]
        core_h = sum(row_heights) + sum(row_gaps)
        core_w = sum(col_widths) + sum(col_gaps)
        if core_h > 28 or core_w > 28:
            continue
        top = int(rng.integers(1, min(5, 30 - core_h)))
        left = int(rng.integers(1, min(5, 30 - core_w)))
        bottom = int(rng.integers(1, min(5, 31 - core_h - top)))
        right = int(rng.integers(1, min(5, 31 - core_w - left)))
        height, width = top + core_h + bottom, left + core_w + right
        inp = np.full((height, width), bg, dtype=np.int8)
        out = inp.copy()
        for slot, (_, mask, color) in assignments:
            sy, sx = divmod(slot, cols)
            base_y = top + sum(row_heights[:sy]) + sum(row_gaps[:sy])
            base_x = left + sum(col_widths[:sx]) + sum(col_gaps[:sx])
            y = base_y + int(rng.integers(0, row_heights[sy] - mask.shape[0] + 1))
            x = base_x + int(rng.integers(0, col_widths[sx] - mask.shape[1] + 1))
            h, w = mask.shape
            inp[y:y+h, x:x+w][mask] = color
            out[y:y+h, x:x+w][mask] = color
            out[y+h//2, x+w//2] = color
        return {"input": inp, "output": out}
    raise RuntimeError("unable to fit motifs")


def _valid_count(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(label)
    return int(value)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _valid_count(num_train, 2, 10, "num_train")
    if num_test is not None:
        num_test = _valid_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    bg = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                if regime == "train":
                    count = 1 if i == 0 else 2 if i == 1 else int(pair_rng.integers(1, 3))
                else:
                    count = int(pair_rng.integers(1, 5))
                chosen = _choose_masks(pair_rng, count, regime, i)
                colors = _choose_colors(pair_rng, count, bg, regime, i)
                motifs = [(kind, mask, color) for (kind, mask), color in zip(chosen, colors)]
                try:
                    pair = _place_pair(pair_rng, motifs, bg)
                except RuntimeError:
                    continue
                key = _grid_key(pair["input"])
                rows = tuple("".join(str(v) for v in row) for row in key)
                if key in seen or rows in _SOURCE_INPUTS:
                    continue
                try:
                    found = _extract_motifs(pair["input"])
                except AssertionError:
                    continue
                if len(found) != count:
                    continue
                expected = pair["input"].copy()
                for _, color, (y, x), _ in found:
                    expected[y, x] = color
                if not np.array_equal(expected, pair["output"]):
                    continue
                seen.add(key)
                puzzle[regime].append(pair)
                break
            else:
                raise RuntimeError(("could not generate pair", regime, i))
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, backgrounds = set(), set(), set()
    train_details = []
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert grid.dtype.kind in "iu" and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            assert tuple("".join(str(v) for v in row) for row in ikey) not in _SOURCE_INPUTS
            bg = _background(inp)
            backgrounds.add(bg)
            motifs = _extract_motifs(inp)
            expected = inp.copy()
            for _, color, (y, x), box in motifs:
                assert inp[y, x] == bg and color != bg
                assert (box[0] + box[2]) // 2 == y and (box[1] + box[3]) // 2 == x
                expected[y, x] = color
            assert np.array_equal(out, expected) and not np.array_equal(inp, out)
            if regime == "train":
                assert 1 <= len(motifs) <= 2 and all(kind == "frame" for kind, _, _, _ in motifs)
                assert all(box[2] - box[0] == box[3] - box[1] for _, _, _, box in motifs)
                assert len({color for _, color, _, _ in motifs}) == 1
                train_details.append(motifs)
            else:
                assert 1 <= len(motifs) <= 4
    assert len(backgrounds) == 1
    assert len(train_details[0]) == 1 and train_details[0][0][3][2] - train_details[0][0][3][0] >= 4
    assert len(train_details[1]) == 2
    assert len({m[3][2] - m[3][0] for m in train_details[1]}) == 2


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[regime]) for regime in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    shapes, mask_families, layouts, palettes, backgrounds, pair_counts = set(), set(), set(), set(), set(), set()
    motif_sizes, arrangement_types = set(), set()
    within_count, within_shape, within_color = 0, 0, 0
    within_test_family, within_test_color, within_palette = 0, 0, 0
    simple_tests = monochrome_multi_tests = four_color_tests = new_color_tests = 0
    combined_tests = nonsquare_motifs = same_color_close_pairs = 0
    all_keys = set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        key = _puzzle_key(puzzle)
        assert key == _puzzle_key(generate_puzzle(seed))
        all_keys.add(key)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        counts, dimensions, color_counts, puzzle_palettes = set(), set(), set(), set()
        test_families, test_color_counts = set(), set()
        train_colors = {
            int(v)
            for pair in puzzle["train"]
            for v in pair["input"].flat
            if int(v) != _background(pair["input"])
        }
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                backgrounds.add(bg)
                motifs = _extract_motifs(inp)
                counts.add(len(motifs))
                pair_counts.add(len(motifs))
                shapes.add(inp.shape)
                dimensions.add(inp.shape)
                color_counts.add(len({m[1] for m in motifs}))
                palette = tuple(sorted(set(int(v) for v in inp.flat)))
                palettes.add(palette)
                puzzle_palettes.add(palette)
                mask_families.update(m[0] for m in motifs)
                layouts.add(tuple(sorted((m[0], m[2][0] - motifs[0][2][0], m[2][1] - motifs[0][2][1]) for m in motifs)))
                families = {m[0] for m in motifs}
                colors = {m[1] for m in motifs}
                for _, _, _, box in motifs:
                    height, width = box[2] - box[0] + 1, box[3] - box[1] + 1
                    motif_sizes.add((height, width))
                    nonsquare_motifs += height != width
                if len(motifs) > 1:
                    ys = {m[2][0] for m in motifs}
                    xs = {m[2][1] for m in motifs}
                    arrangement_types.add(
                        "row" if len(ys) == 1 else
                        "column" if len(xs) == 1 else
                        "two_by_two" if len(ys) == 2 and len(xs) == 2 else
                        "staggered"
                    )
                    for i, a in enumerate(motifs):
                        for b in motifs[i + 1:]:
                            aa, bb = a[3], b[3]
                            row_overlap = max(aa[0], bb[0]) <= min(aa[2], bb[2])
                            col_overlap = max(aa[1], bb[1]) <= min(aa[3], bb[3])
                            row_gap = max(aa[0] - bb[2] - 1, bb[0] - aa[2] - 1)
                            col_gap = max(aa[1] - bb[3] - 1, bb[1] - aa[3] - 1)
                            same_color_close_pairs += a[1] == b[1] and (
                                (row_overlap and col_gap == 1) or
                                (col_overlap and row_gap == 1)
                            )
                if regime == "test":
                    test_families.add(tuple(sorted(families)))
                    test_color_counts.add(len(colors))
                    simple_tests += families == {"frame"}
                    monochrome_multi_tests += len(motifs) > 1 and len(colors) == 1
                    four_color_tests += len(motifs) == 4 and len(colors) == 4
                    new_color_tests += bool(colors - train_colors)
                    combined_tests += (
                        len(motifs) == 4 and len(colors) == 4 and
                        {"stepped", "sparse"} <= families and bool(colors - train_colors)
                    )
                differences = np.argwhere(inp != out)
                assert len(differences) == len(motifs)
                assert {(int(y), int(x)) for y, x in differences} == {m[2] for m in motifs}
        within_count += len(counts) > 1
        within_shape += len(dimensions) > 1
        within_color += len(color_counts) > 1
        within_test_family += len(test_families) > 1
        within_test_color += len(test_color_counts) > 1
        within_palette += len(puzzle_palettes) > 1
    assert len(all_keys) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 40 and len(palettes) >= 25
    assert mask_families == {"frame", "diamond", "stepped", "inset", "sparse"}
    assert pair_counts == {1, 2, 3, 4} and len(backgrounds) >= 4
    assert within_count >= 25 and within_shape >= 35 and within_color >= 10
    assert len(motif_sizes) >= 20 and len(arrangement_types) == 4
    assert nonsquare_motifs >= 20 and same_color_close_pairs >= 3
    assert simple_tests >= 1 and monochrome_multi_tests >= 5
    assert four_color_tests >= 2 and new_color_tests >= 20 and combined_tests >= 1
    assert within_test_family >= 15 and within_test_color >= 10 and within_palette >= 35
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        a = generate_puzzle(700 + i, nt, ne)
        b = generate_puzzle(700 + i, nt, ne)
        validate_generated_puzzle(a)
        assert _puzzle_key(a) == _puzzle_key(b)
        assert nt is None or len(a["train"]) == nt
        assert ne is None or len(a["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(999, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError((nt, ne))
