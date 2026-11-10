"""Connected-object puzzles whose highest occupied row changes color."""

import numpy as np

_COUNT_OPTIONS = (
    (2, 1), (3, 1), (4, 2), (5, 1), (5, 2), (6, 3),
    (7, 2), (7, 3), (8, 4), (9, 3), (10, 4), (10, 1),
)
_SOURCE_INPUTS = {
    tuple(s.splitlines()) for s in (
        "00000000\n00000000\n00000000\n04440000\n04440000\n04440000\n00000000\n00000000",
        "0000000000\n0000000000\n0777777770\n0777777770\n0777777770\n0777777770\n0777777770\n0000000000\n0000000000\n0000000000",
        "00000000\n00033000\n00333300\n03333330\n33333333\n33333333\n33333333\n00000000",
        "000000000\n000000000\n005555500\n005555500\n005555500\n005555500\n005555500\n000000000\n000000000",
        "0000000\n0000000\n0033300\n0033300\n0033300\n0033300\n0033300\n0033300\n0033300\n0033300\n0033300\n0033300\n0033300\n0000000\n0000000",
        "000000000000\n000000000000\n001111111100\n000001100000\n000000000000",
    )
}


def _key(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _counts(rng, seed, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("Invalid pair count")
    if num_train is None and num_test is None:
        return _COUNT_OPTIONS[int(seed) % len(_COUNT_OPTIONS)]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _runs(row):
    xs = np.flatnonzero(row)
    return 0 if not len(xs) else 1 + int(np.count_nonzero(np.diff(xs) > 1))


def _connected(mask):
    cells = np.argwhere(mask)
    if not len(cells):
        return False
    start = tuple(int(v) for v in cells[0])
    visited, pending = {start}, [start]
    h, w = mask.shape
    while pending:
        r, c = pending.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            point = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and point not in visited:
                visited.add(point)
                pending.append(point)
    return len(visited) == len(cells)


def _shape(rng, family, plain=False):
    if family == "rectangle":
        w, h = int(rng.integers(2, 15)), int(rng.integers(2, 11))
        spans = [(0, w)] * h
    elif family == "bar":
        w, h = int(rng.integers(1, 5)), int(rng.integers(8, 21))
        spans = [(0, w)] * h
    elif family == "cap":
        w = int(rng.integers(4, 18))
        stem = int(rng.integers(1, min(7, w - 1)))
        cap_height = int(rng.integers(1, 4))
        stem_height = int(rng.integers(1, 10))
        offset = int(rng.integers(0, w - stem + 1))
        spans = [(0, w)] * cap_height + [(offset, offset + stem)] * stem_height
    elif family == "widen":
        left, right = 0, int(rng.integers(2, 7))
        spans = [(left, right)]
        for _ in range(int(rng.integers(2, 8))):
            dl, dr = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            if dl + dr == 0:
                dr = 1
            left, right = left - dl, right + dr
            if right - left > 18:
                break
            spans.append((left, right))
        spans += [spans[-1]] * int(rng.integers(0, 3))
    elif family == "fork":
        arm1, gap, arm2 = (int(rng.integers(2, 6)), int(rng.integers(1, 5)),
                           int(rng.integers(2, 6)))
        width = arm1 + gap + arm2
        spans = [(0, width)] * int(rng.integers(2, 6))
        if int(rng.integers(0, 2)):
            stem = int(rng.integers(2, width + 1))
            offset = int(rng.integers(0, width - stem + 1))
            spans += [(offset, offset + stem)] * int(rng.integers(1, 5))
    elif family == "irregular":
        left, right = 0, int(rng.integers(4, 11))
        spans = [(left, right)]
        for _ in range(int(rng.integers(3, 12))):
            new_left = min(left + int(rng.integers(-2, 3)), right - 1)
            new_right = max(right + int(rng.integers(-2, 3)), left + 1)
            if new_right - new_left < 2:
                new_right = new_left + 2
            if new_right - new_left > 17:
                new_right = new_left + 17
            left, right = new_left, new_right
            spans.append((left, right))
    else:
        raise ValueError("Unknown shape family")

    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    mask = np.zeros((len(spans), hi - lo), dtype=bool)
    for r, (a, b) in enumerate(spans):
        mask[r, a - lo:b - lo] = True
    if family == "fork":
        mask[0, arm1:arm1 + gap] = False

    if not plain and mask.shape[0] >= 3 and int(rng.integers(0, 4)) == 0:
        rows = [r for r in range(1, mask.shape[0] - 1) if int(mask[r].sum()) >= 5]
        if rows:
            r = int(rng.choice(rows))
            xs = np.flatnonzero(mask[r])
            mask[r, int(xs[0] if int(rng.integers(0, 2)) == 0 else xs[-1])] = False
    if not plain and mask.shape[0] >= 3 and int(rng.integers(0, 4)) == 0:
        candidates = [
            (r, c) for r in range(1, mask.shape[0] - 1)
            for c in range(1, mask.shape[1] - 1)
            if mask[r, c] and mask[r - 1, c] and mask[r + 1, c]
            and mask[r, c - 1] and mask[r, c + 1]
        ]
        if candidates:
            r, c = candidates[int(rng.integers(len(candidates)))]
            mask[r, c] = False
    return mask


def _make_pair(rng, family, highlight, background, foreground, plain=False):
    for _ in range(80):
        shape = _shape(rng, family, plain)
        if not _connected(shape):
            continue
        h, w = shape.shape
        if h > 23 or w > 23:
            continue
        top = int(rng.integers(0, min(9, 30 - h) + 1))
        bottom = int(rng.integers(0, min(9, 30 - h - top) + 1))
        left = int(rng.integers(0, min(9, 30 - w) + 1))
        right = int(rng.integers(0, min(9, 30 - w - left) + 1))
        if (top + h + bottom) * (left + w + right) <= 2 * int(shape.sum()):
            continue
        grid = np.full((top + h + bottom, left + w + right), background, dtype=np.int8)
        grid[top:top + h, left:left + w][shape] = foreground
        output = grid.copy()
        output[top, grid[top] == foreground] = highlight
        return {"input": grid, "output": output}
    raise RuntimeError("Could not place a connected foreground on a clear background")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    highlight = int(rng.integers(1, 10))
    families = ("rectangle", "bar", "cap", "widen", "fork", "irregular")
    basic = ("rectangle", "bar", "cap", "widen", "irregular")
    test_families = [str(rng.choice(families)) for _ in range(test_count)]
    rng.shuffle(test_families)
    familiar = "fork" not in test_families and int(rng.integers(0, 4)) == 0
    if familiar:
        train_families = ["rectangle", "widen"]
        train_families += [str(rng.choice(("rectangle", "widen")))
                           for _ in range(train_count - 2)]
    elif "fork" in test_families or int(rng.integers(0, 5)) < 3:
        train_families = ["fork", str(rng.choice(basic))]
        train_families += [str(rng.choice(families)) for _ in range(train_count - 2)]
    elif int(rng.integers(0, 3)) == 0:
        train_families = ["rectangle", "widen"]
        train_families += [str(rng.choice(families)) for _ in range(train_count - 2)]
    else:
        first, second = rng.choice(basic, size=2, replace=False)
        train_families = [str(first), str(second)]
        train_families += [str(rng.choice(families)) for _ in range(train_count - 2)]
    rng.shuffle(train_families)

    puzzle = {"train": [], "test": []}
    seen, seen_shapes, train_backgrounds, train_foregrounds = set(), set(), set(), set()
    for split, family_list in (("train", train_families), ("test", test_families)):
        for index, family in enumerate(family_list):
            for _ in range(80):
                backgrounds = [c for c in range(10) if c != highlight]
                if split == "train" and index == 1:
                    backgrounds = [c for c in backgrounds if c not in train_backgrounds]
                background = int(rng.choice(backgrounds))
                foregrounds = [c for c in range(10) if c not in (background, highlight)]
                if split == "train" and index == 1:
                    foregrounds = [c for c in foregrounds if c not in train_foregrounds]
                foreground = int(rng.choice(foregrounds))
                pair = _make_pair(rng, family, highlight, background, foreground,
                                  plain=split == "train" and familiar)
                key = _key(pair["input"])
                shape_key = _inspect_pair(pair)["shape"]
                if key not in seen and key not in _SOURCE_INPUTS and shape_key not in seen_shapes:
                    seen.add(key)
                    seen_shapes.add(shape_key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_backgrounds.add(background)
                        train_foregrounds.add(foreground)
                    break
            else:
                raise RuntimeError("Could not make distinct pair")
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = pair["input"], pair["output"]
    for array in (grid, output):
        assert isinstance(array, np.ndarray) and array.ndim == 2
        assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
        assert np.issubdtype(array.dtype, np.integer)
        assert np.all((0 <= array) & (array <= 9))
    assert grid.shape == output.shape
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    bg = int(values[np.argmax(counts)])
    assert int(max(counts)) > grid.size // 2
    fg = int(values[0] if int(values[1]) == bg else values[1])
    mask = grid == fg
    assert _connected(mask)
    rows = np.flatnonzero(mask.any(axis=1))
    assert len(rows) >= 2 and len(rows) == int(rows[-1] - rows[0] + 1)
    top = int(rows[0])
    expected = np.zeros(grid.shape, dtype=bool)
    expected[top] = mask[top]
    changed = grid != output
    assert np.array_equal(changed, expected)
    colors = np.unique(output[changed])
    assert len(colors) == 1
    highlight = int(colors[0])
    assert highlight not in (bg, fg)
    assert np.all(output[~changed] == grid[~changed])
    assert set(int(c) for c in np.unique(output)) == {bg, fg, highlight}
    cs = np.flatnonzero(mask.any(axis=0))
    cropped = mask[top:int(rows[-1]) + 1, int(cs[0]):int(cs[-1]) + 1]
    shape = tuple(tuple(int(v) for v in row) for row in cropped)
    return {"bg": bg, "fg": fg, "highlight": highlight,
            "split_top": _runs(mask[top]) > 1, "shape": shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    inputs, pairs, train_infos, test_infos = set(), set(), [], []
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            pair_key = (key, _key(pair["output"]))
            assert key not in inputs and pair_key not in pairs and key not in _SOURCE_INPUTS
            inputs.add(key)
            pairs.add(pair_key)
            infos.append(info)
    learned = {x["highlight"] for x in train_infos}
    assert len(learned) == 1
    assert all(x["highlight"] in learned for x in test_infos)
    assert any(not x["split_top"] for x in train_infos)
    assert not any(x["split_top"] for x in test_infos) or any(x["split_top"] for x in train_infos)
    assert len({x["bg"] for x in train_infos}) >= 2
    assert len({x["fg"] for x in train_infos}) >= 2
    train_shapes = {x["shape"] for x in train_infos}
    assert all(x["shape"] not in train_shapes for x in test_infos)


def _same_puzzle(a, b):
    return all(
        len(a[s]) == len(b[s]) and all(
            np.array_equal(x[k], y[k])
            for x, y in zip(a[s], b[s]) for k in ("input", "output")
        ) for s in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, highlights = set(), set(), set()
    shapes, dimensions, backgrounds, foregrounds = set(), set(), set(), set()
    split_train = split_test = top_touch = bottom_touch = 0
    holes = noninterval_rows = 0
    rectangles_train = rectangles_test = tall_train = cap_train = 0
    small_caps = offset_caps = asymmetric_widen = one_sided_widen = notches = 0
    familiar_puzzles = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        learned = _inspect_pair(puzzle["train"][0])["highlight"]
        highlights.add(learned)
        learned_from_grids = {
            int(v) for pair in puzzle["train"]
            for v in np.unique(pair["output"][pair["input"] != pair["output"]])
        }
        assert learned_from_grids == {learned}
        for pair in puzzle["test"]:
            test_input = pair["input"]
            values, counts = np.unique(test_input, return_counts=True)
            background = int(values[np.argmax(counts)])
            object_mask = test_input != background
            top_row = int(np.flatnonzero(object_mask.any(axis=1))[0])
            prediction = test_input.copy()
            prediction[top_row, object_mask[top_row]] = learned
            assert np.array_equal(prediction, pair["output"])
        signature = []
        familiar_types = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                grid = pair["input"]
                mask = grid == info["fg"]
                rs, cs = np.where(mask)
                cropped = mask[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
                assert info["highlight"] == learned
                backgrounds.add(info["bg"])
                foregrounds.add(info["fg"])
                shapes.add(info["shape"])
                dimensions.add(grid.shape)
                top_touch += rs.min() == 0
                bottom_touch += rs.max() == grid.shape[0] - 1
                noninterval_rows += sum(_runs(row) > 1 for row in cropped)
                holes += int(any(
                    not cropped[r, c] and cropped[r - 1, c] and cropped[r + 1, c]
                    and cropped[r, c - 1] and cropped[r, c + 1]
                    for r in range(1, cropped.shape[0] - 1)
                    for c in range(1, cropped.shape[1] - 1)
                ))
                rectangle = bool(cropped.all())
                widths = [int(row.sum()) for row in cropped]
                lefts = [int(np.flatnonzero(row)[0]) for row in cropped]
                rights = [int(np.flatnonzero(row)[-1]) for row in cropped]
                small_caps += 4 <= widths[0] <= 6 and widths[-1] <= widths[0] - 2
                offset_caps += (
                    widths[0] >= 7 and widths[-1] <= 5
                    and abs(lefts[-1] - (cropped.shape[1] - widths[-1]) / 2) > 1
                )
                asymmetric_widen += (
                    widths[-1] - widths[0] >= 3
                    and abs((lefts[0] - lefts[-1]) - (rights[-1] - rights[0])) >= 2
                )
                one_sided_widen += (
                    widths[-1] - widths[0] >= 3
                    and (max(lefts) == min(lefts) or max(rights) == min(rights))
                )
                notches += any(
                    (lefts[i] > lefts[i - 1] and lefts[i] > lefts[i + 1])
                    or (rights[i] < rights[i - 1] and rights[i] < rights[i + 1])
                    for i in range(1, len(lefts) - 1)
                )
                if split == "train":
                    split_train += info["split_top"]
                    rectangles_train += rectangle
                    tall_train += cropped.shape[0] >= 8 and cropped.shape[1] <= 4
                    cap_train += cropped[0].sum() >= 7 and cropped[-1].sum() <= 5
                    familiar_types.append(
                        "rectangle" if rectangle else
                        "widen" if widths[-1] - widths[0] >= 3 else "other"
                    )
                else:
                    split_test += info["split_top"]
                    rectangles_test += rectangle
                signature.append((_key(grid), _key(pair["output"])))
        signatures.add(tuple(signature))
        familiar_puzzles += (
            "rectangle" in familiar_types and "widen" in familiar_types
            and all(t in ("rectangle", "widen") for t in familiar_types)
        )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(highlights) >= 7 and len(backgrounds) >= 8 and len(foregrounds) >= 8
    assert len(shapes) >= 120 and len(dimensions) >= 80 and len(signatures) == 42
    assert 20 <= split_train and split_test >= 4
    assert rectangles_train >= 3 and rectangles_test >= 2
    assert tall_train >= 2 and cap_train >= 2
    assert top_touch >= 8 and bottom_touch >= 8
    assert holes >= 5 and noninterval_rows >= 45
    assert small_caps >= 5 and offset_caps >= 5 and notches >= 10
    assert asymmetric_widen >= 5 and one_sided_widen >= 3
    assert familiar_puzzles >= 2

    for nt, ne in ((2, 1), (10, 1), (10, 4),
                   (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(105, nt, ne)
        validate_generated_puzzle(p)
        assert _same_puzzle(p, generate_puzzle(105, nt, ne))
        assert nt is None or len(p["train"]) == nt
        assert ne is None or len(p["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
