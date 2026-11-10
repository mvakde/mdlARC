"""Generate framed-rectangle crop puzzles."""

import numpy as np


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


_SOURCE_INPUTS = tuple(
    np.array([[int(c) for c in row] for row in rows.split()], dtype=np.int8)
    for rows in (
        "0000000000 0000033300 0000031300 0000033300 0000000000 "
        "0000007770 0444007770 0444007770 0444007770 0000000000",
        "0000000000 0000444400 0000400400 0000400400 4440444400 "
        "4440000000 4440066600 4440066600 0000066600 0000000000",
        "0000000000 0000033300 0000033300 0888033300 0858033300 "
        "0888033300 0000000000 0000099900 0000099900 0000099900",
        "0000008880 0222008880 0222008880 0222008880 0222008880 "
        "0222008880 0000000000 0555005555 0555005335 0555005555",
        "000000000 011103330 011103330 011103330 000000000 "
        "003333333 003222223 003222223 003333333",
        "333333333 344444443 344444443 344444443 344444443 "
        "344444443 344444443 344444443 333333333",
    )
)
_SOURCE_KEYS = {_key(x) for x in _SOURCE_INPUTS}


def _apart(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah + 1 <= br or br + bh + 1 <= ar
            or ac + aw + 1 <= bc or bc + bw + 1 <= ac)


def _components(grid, background):
    occupied = grid != background
    visited = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    boxes = []
    for r in range(height):
        for c in range(width):
            if not occupied[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < height and 0 <= nc < width
                                and occupied[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            r0 = min(p[0] for p in cells)
            c0 = min(p[1] for p in cells)
            r1 = max(p[0] for p in cells) + 1
            c1 = max(p[1] for p in cells) + 1
            boxes.append((r0, c0, r1 - r0, c1 - c0))
    return boxes


def _interpret(grid, background):
    frames, solids = [], []
    for r, c, h, w in _components(grid, background):
        tile = grid[r:r + h, c:c + w]
        if h >= 3 and w >= 3:
            border = int(tile[0, 0])
            inside = tile[1:-1, 1:-1]
            if (border != background
                    and np.all(tile[0] == border)
                    and np.all(tile[-1] == border)
                    and np.all(tile[:, 0] == border)
                    and np.all(tile[:, -1] == border)
                    and np.all(inside == inside[0, 0])
                    and int(inside[0, 0]) != border):
                frames.append(((r, c, h, w), border, inside.copy()))
                continue
        if (h >= 2 and w >= 2 and int(tile[0, 0]) != background
                and np.all(tile == tile[0, 0])):
            solids.append(((r, c, h, w), int(tile[0, 0])))
        else:
            return None
    return (frames[0], solids) if len(frames) == 1 else None


def _parse(grid):
    interpretations = []
    for background in np.unique(grid):
        result = _interpret(grid, int(background))
        if result is not None:
            interpretations.append((int(background), *result))
    assert len(interpretations) == 1, "Expected one unambiguous frame"
    return interpretations[0]


def _check_count(value, low, high, name):
    if (value is not None and (isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high)):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _counts(rng, num_train, num_test):
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11)
                 for b in range(1, min(4, a - 1) + 1)]
        num_train, num_test = legal[int(rng.integers(len(legal)))]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _shape(rng, low, high):
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _solid_shapes(rng, frame, count, large):
    cap = 4 if count == 3 else 7 if count == 2 else 11
    shapes = [_shape(rng, 2, cap) for _ in range(count)]
    if large:
        fh, fw = frame
        choices = [(h, w) for h in range(2, 13) for w in range(2, 13)
                   if h * w >= fh * fw and h + fh + 1 <= 30
                   and w + fw + 1 <= 30]
        if not choices:
            return None
        shapes[0] = choices[int(rng.integers(len(choices)))]
    return shapes


def _layout(rng, frame, solids, whole):
    fh, fw = frame
    if whole:
        return (fh, fw), [(0, 0, fh, fw)]
    shapes = [frame] + solids
    if len(shapes) == 1:
        height = min(30, fh + int(rng.integers(0, 9)))
        width = min(30, fw + int(rng.integers(0, 9)))
        r = int(rng.integers(height - fh + 1))
        c = int(rng.integers(width - fw + 1))
        return (height, width), [(r, c, fh, fw)]
    axes = [axis for axis in (0, 1)
            if sum(s[axis] for s in shapes) + len(shapes) - 1 <= 30]
    if not axes:
        return None
    axis = axes[int(rng.integers(len(axes)))]
    gaps = [int(rng.integers(1, 4)) for _ in solids]
    total = sum(s[axis] for s in shapes) + sum(gaps)
    while total > 30:
        j = int(np.argmax(gaps))
        gaps[j] -= 1
        total -= 1
    along = min(30, total + int(rng.integers(0, 6)))
    across = min(30, max(s[1 - axis] for s in shapes)
                 + int(rng.integers(0, 9)))
    canvas = (along, across) if axis == 0 else (across, along)
    for _ in range(70):
        boxes = []
        for i, (h, w) in enumerate(shapes):
            found = None
            for _ in range(70):
                r = int(rng.integers(canvas[0] - h + 1))
                c = int(rng.integers(canvas[1] - w + 1))
                if i == 0 and rng.random() < .20:
                    if rng.random() < .5:
                        r = 0 if rng.random() < .5 else canvas[0] - h
                    else:
                        c = 0 if rng.random() < .5 else canvas[1] - w
                candidate = (r, c, h, w)
                if all(_apart(candidate, old) for old in boxes):
                    found = candidate
                    break
            if found is None:
                break
            boxes.append(found)
        if len(boxes) == len(shapes):
            return canvas, boxes
    order = list(map(int, rng.permutation(len(shapes))))
    cursor = int(rng.integers(along - total + 1))
    boxes = [None] * len(shapes)
    for j, i in enumerate(order):
        h, w = shapes[i]
        offset = int(rng.integers(canvas[1 - axis] - shapes[i][1 - axis] + 1))
        boxes[i] = (cursor, offset, h, w) if axis == 0 else (offset, cursor, h, w)
        cursor += shapes[i][axis]
        if j < len(gaps):
            cursor += gaps[j]
    return canvas, boxes


def _spec(inside, fill_bg, solids, large=False, share=False,
          whole=False, novel=False):
    return dict(inside=inside, fill_bg=fill_bg, solids=solids,
                large=large, share=share, whole=whole, novel=novel)


def _make_pair(rng, spec, seen_colors):
    ih, iw = spec["inside"]
    frame = (ih + 2, iw + 2)
    count = spec["solids"]
    solids = _solid_shapes(rng, frame, count, spec["large"])
    if solids is None:
        return None
    placed = _layout(rng, frame, solids, spec["whole"])
    if placed is None:
        return None
    canvas, boxes = placed
    background = 0 if rng.random() < .70 else int(rng.integers(1, 10))
    border = int(rng.choice([v for v in range(10) if v != background]))
    if spec["fill_bg"]:
        fill = background
    else:
        choices = [v for v in range(10) if v not in (background, border)]
        if spec["novel"]:
            unseen = [v for v in choices if v not in seen_colors]
            if unseen:
                choices = unseen
        fill = int(rng.choice(choices))
    colors = [border if i == 0 and spec["share"] else
              int(rng.choice([v for v in range(10) if v != background]))
              for i in range(count)]
    grid = np.full(canvas, background, dtype=np.int8)
    r, c, h, w = boxes[0]
    grid[r:r + h, c:c + w] = border
    grid[r + 1:r + h - 1, c + 1:c + w - 1] = fill
    for (r, c, h, w), color in zip(boxes[1:], colors):
        grid[r:r + h, c:c + w] = color
    return {"input": grid, "output": np.full((ih, iw), fill, dtype=np.int8)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # No hidden learned mapping: every pair follows the same visible frame rule.
    source_like_shift = rng.random() < .38
    cap = 2 if source_like_shift else 4
    colored = _spec(_shape(rng, 1, min(3, cap)), False,
                    1 + int(rng.random() < .5), large=True, share=True)
    blank = _spec(_shape(rng, 1, cap), True,
                  0 if rng.random() < .25 else int(rng.integers(1, 3)))
    train_specs = [colored, blank]
    for _ in range(num_train - 2):
        inside = _shape(rng, 1, 2 if source_like_shift else 8)
        count = int(rng.choice([0, 1, 2, 3], p=[.18, .34, .34, .14]))
        train_specs.append(_spec(
            inside, rng.random() < .28, count,
            large=(count == 1 and max(inside) <= 5 and rng.random() < .4),
            share=count > 0 and rng.random() < .42,
            whole=count == 0 and rng.random() < .18))
    if len({spec["inside"] for spec in train_specs}) == 1:
        h, w = train_specs[-1]["inside"]
        train_specs[-1]["inside"] = (h, 2 if w == 1 else 1)
    rng.shuffle(train_specs)
    test_specs = []
    for j in range(num_test):
        if not source_like_shift and j > 0 and rng.random() < .52:
            inside = train_specs[int(rng.integers(len(train_specs)))]["inside"]
        elif rng.random() < .17:
            side = int(rng.integers(3 if source_like_shift else 1, 10))
            inside = (side, side)
        else:
            inside = _shape(rng, 3, 9) if source_like_shift else _shape(rng, 1, 9)
        if j == 0 and inside in [s["inside"] for s in train_specs]:
            inside = (inside[0], 9 if inside[1] != 9 else 8)
        count = int(rng.choice([0, 1, 2, 3], p=[.22, .30, .27, .21]))
        test_specs.append(_spec(
            inside, rng.random() < .30, count,
            large=(count == 1 and max(inside) <= 6 and rng.random() < .38),
            share=count > 0 and rng.random() < .38,
            whole=count == 0 and rng.random() < .60,
            novel=rng.random() < .57))
    puzzle = {"train": [], "test": []}
    seen, seen_colors = set(), set()
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for spec in specs:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(120):
                pair = _make_pair(pair_rng, spec, seen_colors)
                if pair is None:
                    continue
                grid = pair["input"]
                key = _key(grid)
                if key in seen or key in _SOURCE_KEYS:
                    continue
                try:
                    _parse(grid)
                except AssertionError:
                    continue
                break
            else:
                raise RuntimeError("Could not make a unique, unambiguous pair")
            seen.add(key)
            seen_colors.update(map(int, np.unique(grid)))
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, training = set(), set(), []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((array >= 0) & (array <= 9))
            key = _key(grid)
            pair_key = (key, _key(output))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert key not in _SOURCE_KEYS
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            background, (box, border, inside), solids = _parse(grid)
            assert background != border and np.array_equal(output, inside)
            assert output.shape == (box[2] - 2, box[3] - 2)
            assert not np.array_equal(grid, output) and len(solids) <= 3
            for solid_box, color in solids:
                assert color != background and _apart(box, solid_box)
            if section == "train":
                training.append((background, box, border, inside, solids))
    assert any(int(inside[0, 0]) == bg for bg, _, _, inside, _ in training)
    assert any(int(inside[0, 0]) != bg for bg, _, _, inside, _ in training)
    assert any(any(s[2] * s[3] >= box[2] * box[3] for s, _ in solids)
               for _, box, _, _, solids in training)
    assert any(any(color == border for _, color in solids)
               for _, _, border, _, solids in training)


def _same_puzzle(a, b):
    return all(len(a[section]) == len(b[section])
               and all(np.array_equal(pa[key], pb[key])
                       for pa, pb in zip(a[section], b[section])
                       for key in ("input", "output"))
               for section in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    inputs, structures, canvases, crops = set(), set(), set(), set()
    bgs, borders, fills, counts = set(), set(), set(), set()
    large = whole = blank_test = three_test = novel_test = same_size = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = {int(v) for pair in puzzle["train"]
                        for v in np.unique(pair["input"])}
        train_size_roles = {}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid, output = pair["input"], pair["output"]
                bg, (box, border, inside), solids = _parse(grid)
                assert np.array_equal(output, inside)
                inputs.add(_key(grid))
                canvases.add(grid.shape)
                crops.add(output.shape)
                bgs.add(bg)
                borders.add(border)
                fills.add(int(output[0, 0]))
                counts.add(len(solids))
                structures.add((grid.shape, output.shape, box[:2],
                                tuple(sorted(s[0][2:] for s in solids))))
                whole += int(box == (0, 0, *grid.shape))
                large += int(any(s[0][2] * s[0][3] >= box[2] * box[3]
                                 for s in solids))
                if section == "train":
                    train_size_roles.setdefault(output.shape, set()).add(
                        int(output[0, 0]) == bg)
                else:
                    blank_test += int(int(output[0, 0]) == bg)
                    three_test += int(len(solids) == 3)
                    novel_test += int(bool(set(map(int, np.unique(grid)))
                                           - train_colors))
        same_size += int(any(len(roles) == 2 for roles in train_size_roles.values()))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(inputs) >= 120 and len(structures) >= 100
    assert len(canvases) >= 20 and len(crops) >= 20
    assert len(bgs) >= 4 and len(borders) >= 7 and len(fills) >= 7
    assert counts == {0, 1, 2, 3}
    assert min(large, whole, blank_test, three_test, novel_test, same_size) > 0
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1),
                          (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, ntrain, ntest)
        assert _same_puzzle(puzzle, generate_puzzle(12345, ntrain, ntest))
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(1, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
