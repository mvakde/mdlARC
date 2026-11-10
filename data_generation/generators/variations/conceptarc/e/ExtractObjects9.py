import numpy as np


_SOURCE_INPUTS = {
    "0000000000/0222222220/0222222220/0244444420/0242222420/"
    "0244444420/0244444420/0244444420/0222222220/0222222220/"
    "0222222220/0222222220/0000000000",
    "000000000000000/000333333333300/000333322223300/"
    "000333323223300/000333323333300/000333323333300/"
    "000333322223300/000333333333300/000333333333300/"
    "000000000000000",
    "000000000000/000000000000/000555555500/000577777500/"
    "000577577500/000575557500/000577777500/000555555500/"
    "000000000000/000000000000/000000000000",
    "0000000000/0000000000/0000000000/0777777770/0771777170/"
    "0771111170/0771171170/0777111770/0777777770/0777777770/"
    "0777777770/0000000000/0000000000/0000000000/0000000000",
    "09998999990000/09988999990000/09888888890000/"
    "09988999990000/09998999990000/09999999990000",
}


def _bbox(mask):
    rows, cols = np.nonzero(mask)
    assert len(rows)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            row, col = pending.pop()
            for cell in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    pending.append(cell)
    return count


def _has_enclosed_gap(mask):
    height, width = mask.shape
    pending = [(r, c) for r in range(height) for c in range(width)
               if not mask[r, c] and
               (r in (0, height - 1) or c in (0, width - 1))]
    reached = set(pending)
    while pending:
        row, col = pending.pop()
        for nr, nc in ((row - 1, col), (row + 1, col),
                       (row, col - 1), (row, col + 1)):
            if (0 <= nr < height and 0 <= nc < width
                    and not mask[nr, nc] and (nr, nc) not in reached):
                reached.add((nr, nc))
                pending.append((nr, nc))
    return int(np.count_nonzero(~mask)) > len(reached)


def _grid_key(grid):
    return grid.shape, tuple(map(int, grid.flat))


def _rows(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("illegal pair count")
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("train count must exceed test count")
    # Give the small, source-like puzzles a fair share of automatic seeds:
    # sampling uniformly from legal (train, test) tuples heavily favors
    # large train counts because they have more compatible test counts.
    train_options = sorted({a for a, _ in choices})
    train_count = train_options[int(rng.integers(len(train_options)))]
    test_options = [b for a, b in choices if a == train_count]
    test_count = test_options[int(rng.integers(len(test_options)))]
    return train_count, test_count


def _motif(rng, height, width, style):
    h, w = height, width
    mask = np.zeros((h, w), dtype=bool)
    if style == "perforated":
        mask[:] = True
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(1, h - 1))
            c = int(rng.integers(1, w - 1))
            rh = int(rng.integers(1, min(3, h - r)))
            cw = int(rng.integers(1, min(4, w - c)))
            mask[r:r + rh, c:c + cw] = False
        if (~mask).sum() < 2:
            mask[1, 1:min(w - 1, 3)] = False
    elif style == "banded":
        mask[:] = True
        row = int(rng.integers(1, h - 1))
        mask[row, 1:-1] = False
        for _ in range(int(rng.integers(0, 3))):
            r = int(rng.integers(1, h - 1))
            c = int(rng.integers(1, w - 1))
            length = int(rng.integers(1, min(4, w - c)))
            mask[r, c:c + length] = False
    elif style == "lattice":
        row = int(rng.integers(1, h - 1))
        col = int(rng.integers(1, w - 1))
        mask[row, :] = True
        mask[:, col] = True
        if rng.random() < 0.45 and row + 1 < h:
            mask[row + 1, :] = True
        if rng.random() < 0.3 and col + 1 < w:
            mask[:, col + 1] = True
        for _ in range(int(rng.integers(2, 6))):
            if rng.random() < 0.5:
                r = int(rng.integers(h))
                end = int(rng.integers(1, w + 1))
                mask[r, :max(col + 1, end)] = True
            else:
                r = int(rng.integers(h))
                start = int(rng.integers(0, col + 1))
                mask[r, start:] = True
    elif style == "comb":
        mask[1, :] = True
        left = int(rng.integers(1, max(2, w // 3 + 1)))
        right = int(rng.integers(1, max(2, w // 3 + 1)))
        mask[0, :left] = True
        mask[0, w - right:] = True
        col = int(rng.integers(1, w - 1))
        mask[1:, col] = True
        start = int(rng.integers(0, col + 1))
        end = int(rng.integers(col + 1, w + 1))
        mask[-1, start:end] = True
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.integers(2, h))
            if rng.random() < 0.5:
                mask[r, :col + 1] = True
            else:
                mask[r, col:] = True
    elif style == "strokes":
        row = int(rng.integers(1, h - 1))
        col = int(rng.integers(1, w - 1))
        mask[row, :] = True
        mask[:, col] = True
        for _ in range(int(rng.integers(3, 8))):
            r = int(rng.integers(h))
            c = int(rng.integers(w))
            if rng.random() < 0.5:
                mask[r, min(c, col):max(c, col) + 1] = True
            else:
                mask[min(r, row):max(r, row) + 1, c] = True
                mask[row, min(c, col):max(c, col) + 1] = True
    elif style == "islands":
        rh = int(rng.integers(1, min(4, h // 2 + 1)))
        cw = int(rng.integers(1, min(4, w // 2 + 1)))
        mask[:rh, :cw] = True
        mask[-rh:, -cw:] = True
        if rh > 1 and cw > 1 and rng.random() < 0.5:
            mask[0, cw - 1] = False
        if rng.random() < 0.5:
            mask = mask[:, ::-1].copy()
    else:
        raise ValueError("unknown motif style")
    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    assert (~mask).sum() >= 2
    return mask


def _insets(rng, dense):
    if dense and rng.random() < 0.6:
        margins = [1, 1, 1, 1]
    else:
        margins = [int(rng.choice([0, 1, 1, 2, 2, 3, 4, 5]))
                   for _ in range(4)]
        while sum(x > 0 for x in margins) < 2:
            margins[int(rng.integers(4))] = int(rng.integers(1, 4))
    return margins


def _outer(rng):
    outer = [int(rng.integers(1, 5)) for _ in range(4)]
    mode = int(rng.choice(4, p=[0.43, 0.27, 0.18, 0.12]))
    if mode == 1:
        outer[int(rng.integers(4))] = 0
    elif mode == 2:
        pair = ((0, 1), (2, 3))[int(rng.integers(2))]
        outer[pair[0]] = outer[pair[1]] = 0
    elif mode == 3:
        pair = ((0, 2), (0, 3), (1, 2), (1, 3))[
            int(rng.integers(4))]
        outer[pair[0]] = outer[pair[1]] = 0
    return outer


def _infer_roles(input_grid):
    corners = [int(input_grid[r, c]) for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    background = max(set(corners), key=corners.count)
    assert corners.count(background) >= 3
    colors = set(map(int, np.unique(input_grid)))
    assert len(colors) == 3
    top, bottom, left, right = _bbox(input_grid != background)
    field = input_grid[top:bottom + 1, left:right + 1]
    assert np.all(field != background)
    perimeter = np.concatenate((field[0], field[-1], field[:, 0],
                                field[:, -1]))
    other = list(colors - {background})
    counts = [int(np.count_nonzero(perimeter == color)) for color in other]
    assert max(counts) > len(perimeter) * 0.58
    field_color = other[int(np.argmax(counts))]
    motif_color = next(iter(colors - {background, field_color}))
    return background, field_color, motif_color, (top, bottom, left, right)


def _solve_input(input_grid):
    background, _, motif_color, _ = _infer_roles(input_grid)
    top, bottom, left, right = _bbox(input_grid == motif_color)
    crop = input_grid[top:bottom + 1, left:right + 1]
    return np.where(crop == motif_color, motif_color, background).astype(np.int8)


def _generate_pair(rng, regime, style):
    if regime == "source_train":
        height = int(rng.integers(4, 9))
        width = int(rng.integers(4, 10))
    else:
        height = int(rng.integers(4, 10))
        width = int(rng.integers(4, 12))
        if rng.random() < 0.25:
            width = int(rng.integers(7, 12))
    if style == "islands":
        height = max(height, 5)
        width = max(width, 5)
    dense = style in ("banded", "perforated")
    try:
        motif = _motif(rng, height, width, style)
    except AssertionError:
        return None
    if style != "islands" and _components(motif) != 1:
        return None
    margins = _insets(rng, dense)
    outer = _outer(rng)
    canvas_h = height + margins[0] + margins[1] + outer[0] + outer[1]
    canvas_w = width + margins[2] + margins[3] + outer[2] + outer[3]
    if canvas_h > 30 or canvas_w > 30:
        return None
    background, field_color, motif_color = map(
        int, rng.choice(10, size=3, replace=False))
    field_h = height + margins[0] + margins[1]
    field_w = width + margins[2] + margins[3]
    input_grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
    ft, fl = outer[0], outer[2]
    input_grid[ft:ft + field_h, fl:fl + field_w] = field_color
    mt, ml = ft + margins[0], fl + margins[2]
    input_grid[mt:mt + height, ml:ml + width][motif] = motif_color
    output_grid = np.where(motif, motif_color, background).astype(np.int8)
    try:
        assert _infer_roles(input_grid)[:3] == (
            background, field_color, motif_color)
    except AssertionError:
        return None
    return input_grid, output_grid


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    # Disconnected pieces use the same motif color and one shared crop.
    # When enabled, a training example always demonstrates this convention.
    islands_enabled = bool(rng.random() < 0.35)
    island_demo = int(rng.integers(train_count)) if islands_enabled else -1
    styles = ("perforated", "banded", "lattice", "comb", "strokes")
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, key in (
        ("source_train", train_count, "train"),
        ("target_test", test_count, "test"),
    ):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(300):
                if regime == "source_train" and index == island_demo:
                    style = "islands"
                elif regime == "target_test" and islands_enabled and (
                    pair_rng.random() < 0.18
                ):
                    style = "islands"
                else:
                    style = str(pair_rng.choice(styles))
                generated = _generate_pair(pair_rng, regime, style)
                if generated is None:
                    continue
                input_grid, output_grid = generated
                signature = _grid_key(input_grid)
                if signature in seen or _rows(input_grid) in _SOURCE_INPUTS:
                    continue
                if regime == "source_train" and index > 0:
                    first = puzzle["train"][0]
                    first_mc = _infer_roles(first["input"])[2]
                    new_mc = _infer_roles(input_grid)[2]
                    if _grid_key(output_grid == new_mc) == _grid_key(
                        first["output"] == first_mc
                    ):
                        continue
                break
            else:
                raise RuntimeError("could not construct a unique valid pair")
            seen.add(signature)
            puzzle[key].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
        n, m = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
        seen_inputs = set()
        train_has_islands = False
        test_has_islands = False
        train_masks = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {
                    "input", "output"}
                inp, out = pair["input"], pair["output"]
                for grid in (inp, out):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert 1 <= grid.shape[0] <= 30
                    assert 1 <= grid.shape[1] <= 30
                    assert np.all((grid >= 0) & (grid <= 9))
                assert inp.shape != out.shape
                key = _grid_key(inp)
                assert key not in seen_inputs
                seen_inputs.add(key)
                assert _rows(inp) not in _SOURCE_INPUTS
                bg, field_color, motif_color, box = _infer_roles(inp)
                assert len({bg, field_color, motif_color}) == 3
                assert set(map(int, np.unique(out))) == {bg, motif_color}
                assert np.array_equal(out, _solve_input(inp))
                mask = out == motif_color
                assert mask[0].any() and mask[-1].any()
                assert mask[:, 0].any() and mask[:, -1].any()
                assert int((~mask).sum()) >= 2
                assert box[1] - box[0] + 1 >= out.shape[0]
                assert box[3] - box[2] + 1 >= out.shape[1]
                separated = _components(mask) > 1
                if split == "train":
                    train_has_islands |= separated
                    train_masks.add(_grid_key(mask))
                else:
                    test_has_islands |= separated
        assert len(train_masks) >= 2
        assert not test_has_islands or train_has_islands
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid puzzle") from exc


def _equal(first, second):
    return all(
        len(first[k]) == len(second[k]) and
        all(np.array_equal(a[name], b[name])
            for a, b in zip(first[k], second[k])
            for name in ("input", "output"))
        for k in ("train", "test"))


def self_test() -> None:
    counts = set()
    test_counts = set()
    masks = set()
    shapes = set()
    roles = set()
    backgrounds_vary = 0
    motif_majority = 0
    edge_field_inset_motif = 0
    padded_field_edge_motif = 0
    two_edge_fields = 0
    islands_puzzles = 0
    train_height_variation = 0
    enclosed_gaps = 0
    filled_bands = 0
    separated_top = 0
    sparse_extremes = 0
    simple_train = 0
    familiar_test = 0
    edge_train = 0
    padded_test = 0
    top_left_occupied = 0
    dense_edge = 0
    role_exchanges = 0
    for seed in range(6000, 6042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        bgs = set()
        heights = set()
        train_islands = False
        test_islands = False
        train_roles = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, fc, mc, (ft, fb, fl, fr) = _infer_roles(inp)
                bgs.add(bg)
                roles.add((bg, fc, mc))
                shapes.add((inp.shape, out.shape))
                masks.add(_grid_key(out == mc))
                motif_box = _bbox(inp == mc)
                touched = [ft == 0, fb == inp.shape[0] - 1,
                           fl == 0, fr == inp.shape[1] - 1]
                motif_edge = [motif_box[0] == ft, motif_box[1] == fb,
                              motif_box[2] == fl, motif_box[3] == fr]
                if sum(touched) >= 2:
                    two_edge_fields += 1
                if any(touched) and not any(motif_edge):
                    edge_field_inset_motif += 1
                if not any(touched) and any(motif_edge):
                    padded_field_edge_motif += 1
                if np.count_nonzero(inp == mc) >= np.count_nonzero(inp == fc):
                    motif_majority += 1
                    if any(touched):
                        dense_edge += 1
                mask = out == mc
                enclosed_gaps += _has_enclosed_gap(mask)
                full_rows = [r for r in range(mask.shape[0])
                             if mask[r].all()]
                filled_bands += any(
                    b > a + 1 and not np.all(mask[a + 1:b])
                    for a in full_rows for b in full_rows if b > a)
                top_runs = int(mask[0, 0]) + int(np.count_nonzero(
                    mask[0, 1:] & ~mask[0, :-1]))
                separated_top += (
                    top_runs >= 2 and mask[1].all()
                    and 0 < int(mask[-1].sum()) < mask.shape[1])
                sparse_extremes += max(
                    int(mask[0].sum()), int(mask[-1].sum()),
                    int(mask[:, 0].sum()), int(mask[:, -1].sum())) <= 2
                top_left_occupied += int(inp[0, 0]) != bg
                if split == "train":
                    heights.add(out.shape[0])
                    train_islands |= _components(out == mc) > 1
                    train_roles.append((bg, fc, mc))
                    simple_train += out.shape[0] == 5 and out.shape[1] <= 6
                    edge_train += any(touched)
                else:
                    test_islands |= _components(out == mc) > 1
                    familiar_test += out.shape[0] == 5
                    padded_test += not any(touched)
                assert np.array_equal(out, _solve_input(inp))
        assert not test_islands or train_islands
        islands_puzzles += train_islands
        backgrounds_vary += len(bgs) > 1
        train_height_variation += len(heights) > 1
        role_exchanges += any(
            a[1] == b[2] or a[2] == b[1]
            for a in train_roles for b in train_roles if a != b)
        assert _equal(puzzle, generate_puzzle(seed))
    assert len(counts) >= 4 and len(test_counts) >= 3
    assert len(masks) >= 80 and len(shapes) >= 70 and len(roles) >= 80
    assert backgrounds_vary >= 30 and train_height_variation >= 20
    assert motif_majority >= 3 and two_edge_fields >= 10
    assert edge_field_inset_motif >= 5 and padded_field_edge_motif >= 5
    assert islands_puzzles >= 5
    assert enclosed_gaps >= 20 and filled_bands >= 30
    assert separated_top >= 5 and sparse_extremes >= 5
    assert simple_train >= 10 and familiar_test >= 3
    assert edge_train >= 30 and padded_test >= 10
    assert top_left_occupied >= 2 and dense_edge >= 5
    assert role_exchanges >= 10
    for a, b in ((2, 1), (10, 1), (10, 4),
                 (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(42, a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
        assert _equal(puzzle, generate_puzzle(42, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(42, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
