"""Generate puzzles that complete the tight rectangles of nearby marks."""

import numpy as np


_SOURCE_PAIRS = (
    (("080080000000", "088080080000", "088888800000", "880088800000",
      "000000000800", "000000000800", "000000008888", "000000000800"),
     ("282282220000", "288282280000", "288888820000", "882288820000",
      "000000002822", "000000002822", "000000008888", "000000002822")),
    (("000008000000", "000088000000", "000088880000", "000088880000",
      "000000008880", "000000008880", "000000008808", "000000008800"),
     ("000028220000", "000088220000", "000088880000", "000088880000",
      "000000008882", "000000008882", "000000008828", "000000008822")),
    (("0000000", "0800800", "0888800", "0008000", "0080000",
      "0000000", "0000000"),
     ("0000000", "0822800", "0888800", "0228200", "0282200",
      "0000000", "0000000")),
    (("000000000088", "000000000800", "000000008000", "000000008888",
      "880880000000", "080880880000", "008080080000", "000880080000"),
     ("000000002288", "000000002822", "000000008222", "000000008888",
      "882882220000", "282882880000", "228282280000", "222882280000")),
)


def _components(mask: np.ndarray, radius: int = 1, diagonal: bool = False) -> list:
    """Components under orthogonal adjacency, or square-radius proximity."""
    cells = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        component = [first]
        stack = [first]
        while stack:
            r, c = stack.pop()
            neighbors = []
            for rr, cc in cells:
                dr, dc = abs(rr - r), abs(cc - c)
                if (max(dr, dc) <= radius if diagonal else dr + dc == 1):
                    neighbors.append((rr, cc))
            for cell in neighbors:
                cells.remove(cell)
                stack.append(cell)
                component.append(cell)
        result.append(component)
    return result


def _pattern(rng: np.random.Generator, height: int, width: int,
             mode: str) -> np.ndarray:
    """Sample varied incomplete marks with a recoverable proximity grouping."""
    area = height * width
    for _ in range(600):
        density = float(rng.uniform(.58, .80) if mode == "connected"
                        else rng.uniform(.37, .63))
        mask = rng.random((height, width)) < density
        if bool(rng.integers(3) == 0):
            r = int(rng.integers(1, height - 1))
            c = int(rng.integers(1, width - 1))
            mask[r:min(height - 1, r + int(rng.integers(1, 5))),
                 c:min(width - 1, c + int(rng.integers(1, 5)))] = False
        for side in range(4):
            if side == 0:
                mask[0, int(rng.integers(width))] = True
            elif side == 1:
                mask[-1, int(rng.integers(width))] = True
            elif side == 2:
                mask[int(rng.integers(height)), 0] = True
            else:
                mask[int(rng.integers(height)), -1] = True
        count = int(mask.sum())
        if not (4 <= count <= area - 2):
            continue
        if not all(mask.any(axis=0)) or not all(mask.any(axis=1)):
            continue
        if len(_components(mask, 2, True)) != 1:
            continue
        parts = _components(mask)
        n = len(parts)
        if mode == "connected" and n != 1:
            continue
        if mode == "fragmented" and not 2 <= n <= 7:
            continue
        if mode == "satellite" and not (2 <= n <= 7 and
                                        any(len(part) == 1 for part in parts)):
            continue
        if mode == "multi" and not 3 <= n <= 8:
            continue
        if mode == "diagonal":
            if not 2 <= n <= 7:
                continue
            if not any(abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1
                       for i, part in enumerate(parts) for other in parts[i+1:]
                       for a in part for b in other):
                continue
        if mode != "connected" and max(map(len, parts)) < max(2, count // 4):
            continue
        return mask
    # The fallback is deterministic and preserves all extrema and grouping.
    if mode == "connected":
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 2, :] = True
        mask[:, width // 2] = True
    else:
        row, col = np.indices((height, width))
        mask = (row + col) % 2 == 0
    assert mask.sum() <= area - 2
    return mask


def _separate(a: tuple, b: tuple) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah + 2 <= br or br + bh + 2 <= ar or
            ac + aw + 2 <= bc or bc + bw + 2 <= ac)


def _layout(rng: np.random.Generator, count: int) -> tuple:
    """Place one to three rectangles with independently sampled dimensions."""
    for _ in range(300):
        limit = 7 if count == 3 else 9
        sizes = [(int(rng.integers(3, limit + 1)),
                  int(rng.integers(3, limit + 1))) for _ in range(count)]
        if count == 1:
            boxes = [(0, 0, *sizes[0])]
        elif count == 2:
            h1, w1 = sizes[0]
            h2, w2 = sizes[1]
            gap = int(rng.integers(2, 6))
            arrangement = int(rng.integers(4))
            if arrangement == 0:
                boxes = [(0, 0, h1, w1),
                         (int(rng.integers(-h2 + 1, h1)), w1 + gap, h2, w2)]
            elif arrangement == 1:
                boxes = [(0, 0, h1, w1),
                         (h1 + gap, int(rng.integers(-w2 + 1, w1)), h2, w2)]
            elif arrangement == 2:
                boxes = [(0, 0, h1, w1), (h1 + gap, w1 + gap, h2, w2)]
            else:
                boxes = [(0, w2 + gap, h1, w1), (h1 + gap, 0, h2, w2)]
            if bool(rng.integers(2)):
                boxes.reverse()
        else:
            boxes = [(int(rng.integers(19)), int(rng.integers(19)), h, w)
                     for h, w in sizes]
            if not all(_separate(a, b) for i, a in enumerate(boxes)
                       for b in boxes[i+1:]):
                continue
        min_r = min(b[0] for b in boxes)
        min_c = min(b[1] for b in boxes)
        boxes = [(r - min_r, c - min_c, h, w) for r, c, h, w in boxes]
        span_h = max(r + h for r, c, h, w in boxes)
        span_w = max(c + w for r, c, h, w in boxes)
        if span_h > 24 or span_w > 24:
            continue
        margins = [int(rng.integers(2, 4)) for _ in range(4)]
        rows = margins[0] + span_h + margins[1]
        cols = margins[2] + span_w + margins[3]
        if rows > 30 or cols > 30:
            continue
        boxes = [(r + margins[0], c + margins[2], h, w)
                 for r, c, h, w in boxes]
        assert all(_separate(a, b) for i, a in enumerate(boxes)
                   for b in boxes[i+1:])
        return (rows, cols), boxes
    raise RuntimeError("could not place separated rectangles")


def _pair(rng: np.random.Generator, section: str, index: int,
          palette: tuple, allow_repeated_colors: bool) -> dict:
    backgrounds, foregrounds, completion = palette
    if section == "train" and index == 0:
        count = 1
        background = backgrounds[0]
        colors = [foregrounds[0]]
        modes = ["connected"]
    elif section == "train" and index == 1:
        count = 2
        background = backgrounds[1]
        colors = [foregrounds[1], foregrounds[2]]
        modes = ["satellite", "connected"]
        if bool(rng.integers(2)):
            modes.reverse()
    elif section == "train" and index == 2:
        count = 2
        background = int(rng.choice(backgrounds))
        color = int(rng.choice(foregrounds))
        colors = [color, color]
        modes = ["connected", str(rng.choice(("fragmented", "satellite",
                                               "multi", "diagonal")))]
    else:
        count = int(rng.integers(1, 4))
        background = int(rng.choice(backgrounds))
        if count > 1 and allow_repeated_colors and bool(rng.integers(2)):
            color = int(rng.choice(foregrounds))
            colors = [color] * count
        else:
            colors = [int(x) for x in rng.choice(foregrounds, size=count,
                                                 replace=False)]
        probabilities = (.35, .23, .18, .12, .12) if section == "train" else (
            .22, .25, .20, .16, .17)
        modes = [str(rng.choice(("connected", "fragmented", "satellite",
                                 "multi", "diagonal"), p=probabilities))
                 for _ in range(count)]
    shape, boxes = _layout(rng, count)
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for (r, c, h, w), color, mode in zip(boxes, colors, modes):
        pattern = _pattern(rng, h, w, mode)
        inp[r:r+h, c:c+w][pattern] = color
        out[r:r+h, c:c+w] = completion
        out[r:r+h, c:c+w][pattern] = color
    if int(np.sum(inp == background)) <= inp.size // 2:
        raise RuntimeError("background must remain visually dominant")
    return {"input": inp, "output": out}


def _validate_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def _text(grid: np.ndarray) -> tuple:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _analyze(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    changed = inp != out
    assert changed.any()
    backgrounds = set(map(int, np.unique(inp[changed])))
    completions = set(map(int, np.unique(out[changed])))
    assert len(backgrounds) == len(completions) == 1
    background = backgrounds.pop()
    completion = completions.pop()
    assert background != completion
    assert completion not in inp
    assert np.sum(inp == background) > inp.size // 2
    assert np.array_equal(inp[~changed], out[~changed])
    groups = _components(inp != background, 2, True)
    assert 1 <= len(groups) <= 3
    boxes = []
    rebuilt = inp.copy()
    for group in groups:
        rr = [cell[0] for cell in group]
        cc = [cell[1] for cell in group]
        r, c = min(rr), min(cc)
        h, w = max(rr) - r + 1, max(cc) - c + 1
        mask = inp[r:r+h, c:c+w] != background
        assert int(mask.sum()) == len(group)
        assert h >= 3 and w >= 3
        assert mask[0].any() and mask[-1].any()
        assert mask[:, 0].any() and mask[:, -1].any()
        assert int((~mask).sum()) >= 2
        colors = set(map(int, np.unique(inp[r:r+h, c:c+w][mask])))
        assert len(colors) == 1
        color = colors.pop()
        assert color != completion
        assert np.all(out[r:r+h, c:c+w] != background)
        rebuilt[r:r+h, c:c+w][~mask] = completion
        parts = _components(mask)
        boxes.append((r, c, h, w, len(parts), color,
                      any(len(part) == 1 for part in parts)))
    assert all(_separate(a[:4], b[:4]) for i, a in enumerate(boxes)
               for b in boxes[i+1:])
    assert np.array_equal(rebuilt, out)
    return {"background": background, "completion": completion,
            "boxes": tuple(sorted(boxes))}


def _predict(train: list, test_input: np.ndarray) -> np.ndarray:
    """Use only demonstrated completion colors and the visible test input."""
    learned = {int(x) for pair in train
               for x in np.unique(pair["output"][pair["input"] != pair["output"]])}
    assert len(learned) == 1
    completion = learned.pop()
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    assert completion not in test_input
    answer = test_input.copy()
    for group in _components(test_input != background, 2, True):
        rr = [cell[0] for cell in group]
        cc = [cell[1] for cell in group]
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        region = answer[r0:r1+1, c0:c1+1]
        region[region == background] = completion
    return answer


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    colors = [int(x) for x in rng.choice(10, size=6, replace=False)]
    palette = (tuple(colors[:2]), tuple(colors[2:5]), colors[5])
    puzzle = {"train": [], "test": []}
    used = set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(100):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _pair(pair_rng, section, index, palette,
                                 num_train >= 3)
                except RuntimeError:
                    continue
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[x], list) for x in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = _key(pair["input"])
            pkey = (ikey, _key(pair["output"]))
            assert ikey not in inputs and pkey not in pairs
            inputs.add(ikey)
            pairs.add(pkey)
            assert (_text(pair["input"]), _text(pair["output"])) not in _SOURCE_PAIRS
            analyses[section].append(_analyze(pair))
    assert len({x["completion"] for xs in analyses.values() for x in xs}) == 1
    assert len({x["background"] for x in analyses["train"]}) >= 2
    train_colors = {box[5] for a in analyses["train"] for box in a["boxes"]}
    assert len(train_colors) >= 3
    assert any(len(a["boxes"]) == 1 for a in analyses["train"])
    assert any(len(a["boxes"]) >= 2 for a in analyses["train"])
    assert any(box[4] == 1 for a in analyses["train"] for box in a["boxes"])
    assert any(box[4] >= 2 and box[6] for a in analyses["train"]
               for box in a["boxes"])
    if len(puzzle["train"]) >= 3:
        assert any(len(a["boxes"]) >= 2 and
                   len({box[5] for box in a["boxes"]}) == 1
                   for a in analyses["train"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]),
                              pair["output"])


def _equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    analyses = [(_analyze(pair), section) for p in puzzles
                for section in ("train", "test") for pair in p[section]]
    boxes = [box for a, _ in analyses for box in a["boxes"]]
    assert len({(box[2], box[3]) for box in boxes}) >= 20
    assert any(box[2] > box[3] for box in boxes)
    assert any(box[3] > box[2] for box in boxes)
    assert any(box[4] >= 3 for box in boxes)
    assert any(box[4] >= 2 and box[2] == box[3] for box in boxes)
    assert len({len(a["boxes"]) for a, _ in analyses}) == 3
    assert any(len(a["boxes"]) >= 2 and
               len({box[5] for box in a["boxes"]}) == 1
               for a, _ in analyses)
    assert any(len(a["boxes"]) >= 2 and
               len({box[5] for box in a["boxes"]}) > 1
               for a, _ in analyses)
    assert len({p["train"][0]["input"].shape for p in puzzles}) >= 12
    assert len({(_analyze(p["train"][0])["background"],
                 _analyze(p["train"][0])["completion"]) for p in puzzles}) >= 20
    assert _equal(generate_puzzle(8128, 7, 3), generate_puzzle(8128, 7, 3))
    assert _equal(generate_puzzle(2718), generate_puzzle(2718))
    assert not _equal(generate_puzzle(1001), generate_puzzle(1002))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4), (2, None),
                                  (10, None))):
        p = generate_puzzle(9000 + i, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts did not raise ValueError")
