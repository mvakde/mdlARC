from __future__ import annotations

import numpy as np


_SOURCE_INPUTS = (
    "333333333333/333333333333/333333000033/333333033033/333333033033/333333000033/333333333333/333333333333/333333333333/333333333333",
    "44444444444/44444444444/44044444444/40404444444/04440444444/40404400004/44044404404/44444404404/44444404404/44444400004/44444444444",
    "66666666666/66600006666/66066660666/60666666066/66066660666/66600006666/66666666666/66666666666",
    "11111111111/11101111111/11010111111/11010100001/11101101101/00011101101/01011100001/01011111111/00011111111",
    "555055555555/550505555555/505550555555/505550555555/505550555555/505550555555/505550555555/550005555555/555555555555/555555555555",
)

_SOURCE_BOXES = (
    ((2, 5, 6, 9),),
    ((2, 6, 0, 4), (5, 9, 6, 9)),
    ((1, 5, 1, 8),),
    ((5, 8, 0, 2), (1, 4, 2, 4), (3, 6, 6, 9)),
    ((0, 7, 1, 5),),
)

_SOURCE_OUTPUTS = (
    "33/33", "04044/44444/04044", "066660/666666/066660",
    "1111/1111", "050/555/555/555/555/555",
)


def _signature(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _regions(grid, color):
    h, w = grid.shape
    seen = np.zeros(grid.shape, bool)
    regions = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or grid[y, x] != color:
                continue
            todo = [(y, x)]
            seen[y, x] = True
            cells = []
            edge = False
            while todo:
                yy, xx = todo.pop()
                cells.append((yy, xx))
                edge |= yy in (0, h - 1) or xx in (0, w - 1)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = yy + dy, xx + dx
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and grid[ny, nx] == color):
                        seen[ny, nx] = True
                        todo.append((ny, nx))
            regions.append((cells, edge))
    return regions


def _analyze(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    field = int(colors[np.argmax(counts)])
    outline = int(colors[np.argmin(counts)])
    h, w = grid.shape
    figures, owned = [], set()
    for hole, edge in _regions(grid, field):
        if edge:
            continue
        border = set()
        for y, x in hole:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and grid[yy, xx] == outline:
                        border.add((yy, xx))
        assert border and not border & owned
        owned.update(border)
        ys, xs = zip(*border)
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        assert y1 - y0 >= 2 and x1 - x0 >= 2
        figures.append((y0, y1, x0, x1, border))
    assert figures and owned == set(zip(*np.where(grid == outline)))
    figures.sort(key=lambda f: (f[2], f[0]))
    assert len({f[1] - f[0] for f in figures}) == 1
    assert all(a[2] < b[2] and a[3] - b[2] <= 2
               for a, b in zip(figures, figures[1:]))
    crops = [grid[y0 + 1:y1, x0 + 1:x1] for y0, y1, x0, x1, _ in figures]
    return field, outline, figures, crops


def _solve(grid):
    return np.concatenate(_analyze(grid)[3], axis=1).astype(np.int8)


def _outline(rng, h, w, style):
    if style == "rectangle":
        left = right = np.zeros(h, dtype=int)
    elif style == "waist":
        dl = int(rng.integers(1, min(3, (w - 1) // 2) + 1))
        dr = int(rng.integers(1, min(3, w - dl - 2) + 1))
        yl = int(rng.integers(1, h - 1))
        yr = int(rng.integers(1, h - 1))
        left = np.maximum(0, dl - np.abs(np.arange(h) - yl))
        right = np.maximum(0, dr - np.abs(np.arange(h) - yr))
    else:
        capmax = min(w - 1, h - 2)
        if style == "point":
            top = min(w - 1, 2 * capmax)
            bottom = int(rng.integers(0, min(3, capmax) + 1))
        elif style == "diamond":
            top = bottom = min(w - 1, 2 * capmax)
        elif style in ("taper", "irregular"):
            top = int(rng.integers(1 if style == "taper" else 0,
                                   min(w, 2 * capmax + 1)))
            bottom = int(rng.integers(0, min(w, 2 * capmax + 1)))
        else:
            raise ValueError("unknown contour style")
        tl = int(rng.integers(max(0, top - capmax), min(capmax, top) + 1))
        tr = top - tl
        bl = int(rng.integers(max(0, bottom - capmax), min(capmax, bottom) + 1))
        br = bottom - bl
        rows = np.arange(h)
        left = np.maximum.reduce((np.zeros(h, int), tl - rows, bl - (h - 1 - rows)))
        right = np.maximum.reduce((np.zeros(h, int), tr - rows, br - (h - 1 - rows)))
        if style == "irregular" and h >= 6:
            for a in (left, right):
                a[int(rng.integers(2, h - 2))] += int(rng.integers(0, 2))
    mask = np.zeros((h, w), bool)
    for y in range(h):
        lo, hi = int(left[y]), w - 1 - int(right[y])
        if lo > hi or (0 < y < h - 1 and hi - lo < 2):
            raise ValueError("invalid contour")
        if y in (0, h - 1):
            mask[y, lo:hi + 1] = True
        else:
            mask[y, lo] = mask[y, hi] = True
    if not mask[:, 0].any() or not mask[:, -1].any():
        raise ValueError("contour lacks full width")
    return mask


def _compose(rng, masks, field, outline, challenge):
    h = masks[0].shape[0]
    widths = [m.shape[1] for m in masks]
    if len(masks) > 1 and challenge and rng.integers(0, 5) == 0:
        # A vertical stair permits neighboring boxes to overlap in one
        # column while their contours remain separate, as in source test 2.
        offsets = [i * h for i in range(len(masks))]
        if rng.integers(0, 2):
            offsets.reverse()
        gaps = [-1 for _ in range(len(masks) - 1)]
    else:
        offsets = [0]
        for _ in range(len(masks) - 1):
            offsets.append(int(rng.integers(-4 if challenge else -2,
                                            5 if challenge else 3)))
        gaps = [int(rng.integers(-2 if challenge else 0, 7))
                for _ in range(len(masks) - 1)]
    offsets = [v - min(offsets) for v in offsets]
    xs = [0]
    for width, gap in zip(widths, gaps):
        xs.append(xs[-1] + width + gap)
    span_w = max(x + width for x, width in zip(xs, widths))
    span_h = max(offsets) + h
    if span_w > 30 or span_h > 30:
        raise ValueError("figures exceed canvas")
    ml = int(rng.integers(0, min(6, 31 - span_w)))
    mr = int(rng.integers(0, min(6, 31 - span_w - ml)))
    mt = int(rng.integers(0, min(6, 31 - span_h)))
    mb = int(rng.integers(0, min(6, 31 - span_h - mt)))
    grid = np.full((span_h + mt + mb, span_w + ml + mr), field, np.int8)
    boxes = []
    for mask, x, offset in zip(masks, xs, offsets):
        y, x = mt + offset, ml + x
        region = grid[y:y + h, x:x + mask.shape[1]]
        if np.any(region[mask] != field):
            raise ValueError("figures overlap")
        region[mask] = outline
        boxes.append((y, y + h - 1, x, x + mask.shape[1] - 1))
    return grid, boxes


def _make_pair(rng, field, outline, n, h, styles, challenge, need_outline=False):
    for _ in range(80):
        try:
            max_w = 12 if n == 1 else (10 if n == 2 else 8)
            masks = [_outline(rng, h, int(rng.integers(3, max_w + 1)), s)
                     for s in styles]
            grid, boxes = _compose(rng, masks, field, outline, challenge)
            boxes.sort(key=lambda b: b[2])
            expected = np.concatenate(
                [grid[y0 + 1:y1, x0 + 1:x1] for y0, y1, x0, x1 in boxes],
                axis=1).astype(np.int8)
            if need_outline and not np.any(expected == outline):
                continue
            _, _, figures, crops = _analyze(grid)
            assert [f[:4] for f in figures] == boxes
            assert np.array_equal(expected, np.concatenate(crops, axis=1))
            return {"input": grid, "output": expected}
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("could not construct a distinguishable pair")


def _check_count(value, low, high, name):
    if value is not None and (isinstance(value, bool) or
                              not isinstance(value, (int, np.integer)) or
                              not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_test is None:
        upper = min(4, int(num_train) - 1) if num_train is not None else 4
        num_test = int(rng.integers(1, upper + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    outline = int(rng.integers(0, 10))
    palette = [c for c in range(10) if c != outline]
    rng.shuffle(palette)
    fields = [int(palette[i % len(palette)]) for i in range(num_train + num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    demo_counts = [1, 2]
    rng.shuffle(demo_counts)
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            field = fields[index if split == "train" else num_train + index]
            for _ in range(80):
                prng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if split == "train" and index < 2:
                    n = demo_counts[index]
                    h = int(prng.integers(4, 9))
                    if n == 1:
                        styles = ["rectangle" if demo_counts[0] == 1 else
                                  "point" if h >= 6 else "waist"]
                    else:
                        styles = ["rectangle", "diamond" if h >= 5 else "waist"]
                        prng.shuffle(styles)
                else:
                    n = int(prng.choice((1, 2, 3), p=(.38, .43, .19)))
                    h = int(prng.integers(4, 11))
                    styles = [str(prng.choice(
                        ("rectangle", "waist", "taper", "point", "diamond", "irregular"),
                        p=(.27, .17, .2, .13, .1, .13))) for _ in range(n)]
                if split == "test" and index == 0 and all(s == "rectangle" for s in styles):
                    styles[int(prng.integers(0, n))] = "point" if h >= 6 else "waist"
                challenge = split == "test" or bool(prng.integers(0, 3) == 0)
                try:
                    need_outline = (split == "train" and index < 2 and
                                    (index == 1 if demo_counts[0] == 1 else index == 0))
                    pair = _make_pair(prng, field, outline, n, h, styles,
                                      challenge, need_outline)
                except RuntimeError:
                    continue
                sig = (pair["input"].shape, pair["input"].tobytes())
                if sig in seen or _signature(pair["input"]) in _SOURCE_INPUTS:
                    continue
                seen.add(sig)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, roles, train_counts = set(), set(), set()
    train_outline, train_field = False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            sig = (inp.shape, inp.tobytes())
            assert sig not in seen and _signature(inp) not in _SOURCE_INPUTS
            seen.add(sig)
            field, outline, figures, crops = _analyze(inp)
            roles.add(outline)
            assert 1 <= len(figures) <= 3
            assert np.array_equal(out, np.concatenate(crops, axis=1))
            assert not np.array_equal(inp, out)
            if split == "train":
                train_counts.add(len(figures))
                train_outline |= any(np.any(c == outline) for c in crops)
                train_field |= any(np.all(c == field) for c in crops)
    assert len(roles) == 1 and {1, 2}.issubset(train_counts)
    assert train_outline and train_field
    assert len({_analyze(p["input"])[0] for p in puzzle["train"]}) >= 2


def _fingerprint(puzzle):
    return tuple((split, p["input"].shape, p["input"].tobytes(),
                  p["output"].shape, p["output"].tobytes())
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    for encoded, boxes, expected in zip(_SOURCE_INPUTS, _SOURCE_BOXES,
                                        _SOURCE_OUTPUTS):
        inp = np.array([[int(c) for c in row] for row in encoded.split("/")],
                       dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in expected.split("/")],
                       dtype=np.int8)
        assert [f[:4] for f in _analyze(inp)[2]] == list(boxes)
        direct = np.concatenate(
            [inp[y0 + 1:y1, x0 + 1:x1] for y0, y1, x0, x1 in boxes],
            axis=1)
        assert np.array_equal(direct, out)

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, counts, heights, outline_crops, field_crops = set(), set(), set(), 0, 0
    review = dict(overlap=0, edge=0, point=0, stagger=0,
                  small_test=0, field_test=0, varying_test=0)
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                field, outline, figures, crops = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                counts.add(len(figures))
                heights.add(pair["output"].shape[0])
                outline_crops += any(np.any(c == outline) for c in crops)
                field_crops += all(np.all(c == field) for c in crops)
                review["overlap"] += any(a[3] >= b[2]
                                         for a, b in zip(figures, figures[1:]))
                review["edge"] += any(y0 == 0 or x0 == 0 or
                                      y1 == pair["input"].shape[0] - 1 or
                                      x1 == pair["input"].shape[1] - 1
                                      for y0, y1, x0, x1, _ in figures)
                review["point"] += any(
                    sum(y == f[0] for y, x in f[4]) == 1 or
                    sum(y == f[1] for y, x in f[4]) == 1 for f in figures)
                review["stagger"] += len({f[0] for f in figures}) > 1
                if split == "test":
                    review["small_test"] += pair["output"].shape[0] in (2, 3)
                    review["field_test"] += np.all(pair["output"] == field)
                    review["varying_test"] += len({tuple(r) for r in pair["output"]}) > 1
                assert np.array_equal(pair["output"], _solve(pair["input"]))
    assert len(shapes) >= 35 and counts == {1, 2, 3}
    assert len(heights) >= 6 and outline_crops > 0 and field_crops > 0
    assert all(v > 0 for v in review.values())
    for seed in (0, 7, 41):
        assert _fingerprint(puzzles[seed]) == _fingerprint(generate_puzzle(seed))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        p = generate_puzzle(901 + (train or 0) * 7 + (test or 0), train, test)
        validate_generated_puzzle(p)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
