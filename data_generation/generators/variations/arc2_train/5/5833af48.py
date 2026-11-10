"""Synthetic examples of the stamp-by-placement-mask ARC rule."""

import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "ea7b6a2f462b7e8ecc020723a472c4001d521cded1b852f423272e69a6ed1795",
    "bb607123b3dbaf7dc5a5f68644bd259a5fa38984bbacf033ad5a559b3458b9c9",
    "bbcb9e26f3bf52a40664917520336cc21944a80c398b25a4d121e0a8d7d6917e",
    "c1e3b7eb564c1215105dae9727a9a79b1c2f64609ed639c37c74eb2196ebdcc0",
}

_STAMP_SHAPES = (
    (2, 3), (3, 2), (2, 4), (4, 2),
    (3, 3), (3, 4), (4, 3), (4, 4),
)


def _validate_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _fits(sh, sw, mh, mw, scale):
    ph, pw = sh * mh, sw * mw
    dh, dw = scale * mh, scale * mw
    top_band = max(sh, dh) + 1 + ph <= 28 and max(sw + 1 + dw, pw) <= 28
    side_band = max(sw, dw) + 1 + pw <= 28 and max(sh + 1 + dh, ph) <= 28
    return top_band or side_band


_SPECS = tuple(
    (sh, sw, mh, mw, scale)
    for sh, sw in _STAMP_SHAPES
    for mh in range(2, 7)
    for mw in range(2, 7)
    for scale in (1, 2)
    if mh * mw >= 6 and _fits(sh, sw, mh, mw, scale)
)
_TRAIN_SPECS = tuple(s for s in _SPECS if s[2] <= 5 and s[3] <= 5)
_CHALLENGE_SPECS = tuple(
    s for s in _SPECS
    if s[2] * s[3] >= 12 or s[2] == 6 or s[3] == 6
)


def _stamp(rng, shape):
    h, w = shape
    mode = _choice(rng, ("random", "random", "cross", "diagonal",
                         "line", "border", "checker", "sparse", "dense"))
    if mode == "random":
        count = int(rng.integers(2, h * w - 1))
        result = np.zeros(h * w, dtype=bool)
        result[rng.choice(h * w, size=count, replace=False)] = True
        result = result.reshape(shape)
    elif mode == "cross":
        result = np.zeros(shape, dtype=bool)
        result[h // 2, :] = True
        result[:, w // 2] = True
    elif mode == "diagonal":
        result = np.fromfunction(lambda y, x: (y * w // h) == x, shape, dtype=int)
        if float(rng.random()) < 0.5:
            result = np.fliplr(result)
    elif mode == "line":
        result = np.zeros(shape, dtype=bool)
        if float(rng.random()) < 0.5:
            result[int(rng.integers(h)), :] = True
        else:
            result[:, int(rng.integers(w))] = True
    elif mode == "border":
        result = np.ones(shape, dtype=bool)
        result[1:-1, 1:-1] = False
        if np.all(result):
            result[h // 2, w // 2] = False
    elif mode == "checker":
        yy, xx = np.indices(shape)
        result = (yy + xx + int(rng.integers(2))) % 2 == 0
    else:
        count = int(rng.integers(1, max(2, h * w // 3) + 1))
        result = np.zeros(h * w, dtype=bool)
        result[rng.choice(h * w, size=count, replace=False)] = True
        result = result.reshape(shape)
        if mode == "dense":
            result = ~result
    assert 0 < int(result.sum()) < h * w
    return result


def _mask(rng, shape, category):
    h, w = shape
    total = h * w
    small_max = max(2, total // 4)
    if category == "sparse":
        count = int(rng.integers(1, small_max + 1))
    elif category == "dense":
        count = total - int(rng.integers(1, small_max + 1))
    else:
        lo = small_max + 1
        hi = total - small_max - 1
        count = int(rng.integers(lo, hi + 1))
    mode = _choice(rng, ("random", "random", "block", "row", "column", "dispersed"))
    cells = [(y, x) for y in range(h) for x in range(w)]
    rng.shuffle(cells)
    if mode == "block":
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        cells.sort(key=lambda p: abs(p[0] - cy) + abs(p[1] - cx))
    elif mode == "row":
        row = int(rng.integers(h))
        cells.sort(key=lambda p: abs(p[0] - row))
    elif mode == "column":
        col = int(rng.integers(w))
        cells.sort(key=lambda p: abs(p[1] - col))
    elif mode == "dispersed":
        parity = int(rng.integers(2))
        cells.sort(key=lambda p: (p[0] + p[1] + parity) % 2)
    result = np.zeros(shape, dtype=bool)
    for y, x in cells[:count]:
        result[y, x] = True
    return result


def _palette(rng, mode, fixed, previous_base=None):
    if mode == "constant":
        return fixed
    if mode == "base":
        bg, ink, placeholder, _ = fixed
        choices = [c for c in range(10) if c not in (bg, ink, placeholder)]
        if previous_base is not None:
            choices = [c for c in choices if c != previous_base]
        return bg, ink, placeholder, _choice(rng, choices)
    choices = [c for c in range(10) if c != previous_base]
    colors = [int(c) for c in rng.choice(choices, size=4, replace=False)]
    return tuple(colors)


def _alignment(rng, slack):
    return int(rng.integers(slack + 1))


def _layout(rng, stamp, displayed_mask, panel, background):
    arrays = {"stamp": stamp, "mask": displayed_mask, "panel": panel}
    candidates = []
    for gap in (1, 2, 3):
        for first, second in (("stamp", "mask"), ("mask", "stamp")):
            a, b = arrays[first], arrays[second]
            ph, pw = panel.shape
            gh, gw = max(a.shape[0], b.shape[0]), a.shape[1] + gap + b.shape[1]
            ch, cw = gh + gap + ph, max(gw, pw)
            if ch <= 28 and cw <= 28:
                ax = _alignment(rng, cw - gw)
                px = _alignment(rng, cw - pw)
                ay = _alignment(rng, gh - a.shape[0])
                by = _alignment(rng, gh - b.shape[0])
                upper = [(first, ay, ax), (second, by, ax + a.shape[1] + gap)]
                candidates.append((upper + [("panel", gh + gap, px)], ch, cw))
                lower = [(name, y + ph + gap, x) for name, y, x in upper]
                candidates.append(([("panel", 0, px)] + lower, ch, cw))
            gh, gw = a.shape[0] + gap + b.shape[0], max(a.shape[1], b.shape[1])
            ch, cw = max(gh, ph), gw + gap + pw
            if ch <= 28 and cw <= 28:
                gy = _alignment(rng, ch - gh)
                py = _alignment(rng, ch - ph)
                ax = _alignment(rng, gw - a.shape[1])
                bx = _alignment(rng, gw - b.shape[1])
                left = [(first, gy, ax), (second, gy + a.shape[0] + gap, bx)]
                candidates.append((left + [("panel", py, gw + gap)], ch, cw))
                right = [(name, y, x + pw + gap) for name, y, x in left]
                candidates.append(([("panel", py, 0)] + right, ch, cw))
    assert candidates
    positions, ch, cw = _choice(rng, candidates)
    margins = [1, 1, 1, 1]  # top, bottom, left, right
    if float(rng.random()) < 0.42:
        side = int(rng.integers(4))
        margins[side] = 0
        margins[side ^ 1] = 2
    extra_h = int(rng.integers(min(3, 30 - ch - 2) + 1))
    extra_w = int(rng.integers(min(3, 30 - cw - 2) + 1))
    if margins[0] == 0:
        margins[1] += extra_h
    elif margins[1] == 0:
        margins[0] += extra_h
    else:
        add = int(rng.integers(extra_h + 1))
        margins[0] += add
        margins[1] += extra_h - add
    if margins[2] == 0:
        margins[3] += extra_w
    elif margins[3] == 0:
        margins[2] += extra_w
    else:
        add = int(rng.integers(extra_w + 1))
        margins[2] += add
        margins[3] += extra_w - add
    canvas = np.full((ch + margins[0] + margins[1],
                      cw + margins[2] + margins[3]), background, dtype=np.int8)
    for name, y, x in positions:
        piece = arrays[name]
        y += margins[0]
        x += margins[2]
        region = canvas[y:y + piece.shape[0], x:x + piece.shape[1]]
        assert np.all(region == background)
        region[:] = piece
    return canvas


def _make_pair(rng, spec, category, palette):
    sh, sw, mh, mw, scale = spec
    background, ink, placeholder, base = palette
    binary_stamp = _stamp(rng, (sh, sw))
    logical_mask = _mask(rng, (mh, mw), category)
    stamp = np.where(binary_stamp, ink, placeholder).astype(np.int8)
    enlarged = np.kron(logical_mask, np.ones((scale, scale), dtype=bool))
    displayed_mask = np.where(enlarged, ink, base).astype(np.int8)
    foreground = np.kron(logical_mask, binary_stamp)
    output = np.where(foreground, ink, base).astype(np.int8)
    panel = np.full(output.shape, base, dtype=np.int8)
    return {"input": _layout(rng, stamp, displayed_mask, panel, background),
            "output": output}


def _components(grid, background):
    occupied = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    height, width = grid.shape
    for sy in range(height):
        for sx in range(width):
            if not occupied[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and occupied[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            y0, y1 = min(p[0] for p in cells), max(p[0] for p in cells) + 1
            x0, x1 = min(p[1] for p in cells), max(p[1] for p in cells) + 1
            block = grid[y0:y1, x0:x1]
            assert len(cells) == block.size and np.all(block != background)
            components.append((block, (y0, y1, x0, x1),
                               {int(v) for v in np.unique(block)}))
    return components


def _analyze_pair(input_grid, output_grid):
    corners = (int(input_grid[0, 0]), int(input_grid[0, -1]),
               int(input_grid[-1, 0]), int(input_grid[-1, -1]))
    assert len(set(corners)) == 1
    background = corners[0]
    components = _components(input_grid, background)
    assert len(components) == 3
    solids = [c for c in components if len(c[2]) == 1]
    assert len(solids) == 1
    panel = solids[0]
    base = next(iter(panel[2]))
    patterned = [c for c in components if c is not panel]
    masks = [c for c in patterned if base in c[2]]
    assert len(masks) == 1
    mask = masks[0]
    stamp = next(c for c in patterned if c is not mask)
    assert len(mask[2]) == len(stamp[2]) == 2
    ink_set = mask[2] & stamp[2]
    assert len(ink_set) == 1
    ink = next(iter(ink_set))
    placeholder = next(iter(stamp[2] - {ink}))
    assert len({background, ink, placeholder, base}) == 4
    assert mask[2] == {base, ink}
    assert set(int(v) for v in np.unique(input_grid)) == {background, ink, placeholder, base}
    sh, sw = stamp[0].shape
    ph, pw = panel[0].shape
    assert sh >= 2 and sw >= 2 and (sh, sw) in _STAMP_SHAPES
    assert ph % sh == 0 and pw % sw == 0
    mh, mw = ph // sh, pw // sw
    assert mh >= 2 and mw >= 2 and mh <= 6 and mw <= 6 and mh * mw >= 6
    dh, dw = mask[0].shape
    assert dh % mh == 0 and dw % mw == 0
    scale = dh // mh
    assert scale == dw // mw and scale in (1, 2)
    logical_mask = np.zeros((mh, mw), dtype=bool)
    for y in range(mh):
        for x in range(mw):
            block = mask[0][y * scale:(y + 1) * scale,
                            x * scale:(x + 1) * scale]
            assert np.all(block == block[0, 0])
            logical_mask[y, x] = int(block[0, 0]) == ink
    binary_stamp = stamp[0] == ink
    assert 0 < int(binary_stamp.sum()) < sh * sw
    assert 0 < int(logical_mask.sum()) < mh * mw
    expected = np.where(np.kron(logical_mask, binary_stamp), ink, base)
    assert output_grid.shape == panel[0].shape
    assert np.array_equal(output_grid, expected)
    assert set(int(v) for v in np.unique(output_grid)) == {ink, base}
    assert not (input_grid.shape == output_grid.shape
                and np.array_equal(input_grid, output_grid))
    edge = any(y0 == 0 or x0 == 0 or y1 == input_grid.shape[0]
               or x1 == input_grid.shape[1] for _, (y0, y1, x0, x1), _ in components)
    sy0, sy1, sx0, sx1 = stamp[1]
    my0, my1, mx0, mx1 = mask[1]
    py0, py1, px0, px1 = panel[1]
    if py1 < min(sy0, my0):
        layout = "panel_above"
    elif py0 > max(sy1, my1):
        layout = "panel_below"
    elif px1 < min(sx0, mx0):
        layout = "panel_left"
    else:
        assert px0 > max(sx1, mx1)
        layout = "panel_right"
    return {"roles": (background, ink, placeholder, base),
            "stamp_shape": (sh, sw), "logical_shape": (mh, mw),
            "stamp": binary_stamp, "mask": logical_mask,
            "scale": scale, "active": int(logical_mask.sum()),
            "bboxes": tuple(c[1] for c in components), "edge": edge,
            "layout": layout}


def _signature(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _source_digest(pair):
    payload = b"".join(bytes(pair[key].shape) + pair[key].astype(np.uint8).tobytes()
                       for key in ("input", "output"))
    return hashlib.sha256(payload).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            signature = _signature(pair["input"])
            pair_signature = signature, _signature(pair["output"])
            assert signature not in inputs and pair_signature not in pairs
            inputs.add(signature)
            pairs.add(pair_signature)
            assert _source_digest(pair) not in _SOURCE_PAIR_DIGESTS
            info = _analyze_pair(pair["input"], pair["output"])
            spec = (*info["stamp_shape"], *info["logical_shape"], info["scale"])
            assert spec in _SPECS
            if split == "train":
                assert spec in _TRAIN_SPECS
            analyses[split].append(info)
    train = analyses["train"]
    test = analyses["test"]
    # A doubled test mask must be demonstrated in the training examples.
    if any(info["scale"] == 2 for info in test):
        assert any(info["scale"] == 2 for info in train)
    # Sparse and dense test masks are demonstrated with varying occupancy.
    for info in test:
        area = info["mask"].size
        count = info["active"]
        if count <= max(2, area // 4):
            assert any(t["active"] <= max(2, t["mask"].size // 4) for t in train)
        if count >= area - max(2, area // 4):
            assert any(t["active"] >= t["mask"].size - max(2, t["mask"].size // 4)
                       for t in train)
    assert len({t["active"] for t in train}) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")

    # The relation between colors, rather than any literal color, is the rule.
    # Palette mode controls puzzle difficulty; all output-relevant roles are local.
    palette_mode = _choice(rng, ("constant", "base", "full", "full"))
    fixed_palette = tuple(int(v) for v in rng.choice(10, size=4, replace=False))
    test_specs = []
    test_categories = []
    for _ in range(test_count):
        pool = _CHALLENGE_SPECS if float(rng.random()) < 0.68 else _TRAIN_SPECS
        test_specs.append(_choice(rng, pool))
        test_categories.append(_choice(rng, ("sparse", "balanced", "dense")))

    needed_categories = {c for c in test_categories if c != "balanced"}
    train_categories = sorted(needed_categories)
    while len(train_categories) < train_count:
        train_categories.append(_choice(rng, ("sparse", "balanced", "dense")))
    if len(set(train_categories[:2])) == 1:
        train_categories[1] = "dense" if train_categories[0] != "dense" else "sparse"
    rng.shuffle(train_categories)

    train_specs = []
    for index in range(train_count):
        pool = _TRAIN_SPECS
        if index == 0 and any(s[-1] == 2 for s in test_specs):
            pool = tuple(s for s in pool if s[-1] == 2)
        if index == 1 and train_count == 2:
            alternate = tuple(s for s in pool if s[:2] != train_specs[0][:2])
            if alternate:
                pool = alternate
        train_specs.append(_choice(rng, pool))
    if any(s[-1] == 2 for s in test_specs) and not any(s[-1] == 2 for s in train_specs):
        train_specs[0] = _choice(rng, tuple(s for s in _TRAIN_SPECS if s[-1] == 2))

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_base = None
    for split, specs, categories in (
        ("train", train_specs, train_categories),
        ("test", test_specs, test_categories),
    ):
        for spec, category in zip(specs, categories):
            for attempt in range(12):
                child = np.random.default_rng(int(rng.integers(0, 2**63)))
                force_new_base = previous_base if split == "train" and len(puzzle["train"]) == 1 else None
                palette = _palette(child, palette_mode, fixed_palette, force_new_base)
                pair = _make_pair(child, spec, category, palette)
                signature = _signature(pair["input"])
                if signature not in used_inputs and _source_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                    used_inputs.add(signature)
                    puzzle[split].append(pair)
                    previous_base = palette[3]
                    break
            else:
                raise ValueError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_signature(p) for p in automatic}) >= 40
    for seed, tr, te in ((1907, 2, 1), (1908, 10, 1), (1909, 10, 4),
                         (1910, None, 1), (1911, None, 4),
                         (1912, 2, None), (1913, 10, None)):
        p = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(p)
        if tr is not None:
            assert len(p["train"]) == tr
        if te is not None:
            assert len(p["test"]) == te
    assert _puzzle_signature(generate_puzzle(1234, 6, 2)) == _puzzle_signature(
        generate_puzzle(1234, 6, 2))
    assert _puzzle_signature(generate_puzzle(1234, 6, 2)) != _puzzle_signature(
        generate_puzzle(1235, 6, 2))
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (True, 1)):
        try:
            generate_puzzle(55, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    info = [(p, split, _analyze_pair(pair["input"], pair["output"]))
            for p in automatic for split in ("train", "test") for pair in p[split]]
    assert len({v["roles"] for _, _, v in info}) >= 40
    assert len({v["stamp_shape"] for _, _, v in info}) == len(_STAMP_SHAPES)
    assert len({v["logical_shape"] for _, _, v in info}) >= 15
    assert {v["scale"] for _, _, v in info} == {1, 2}
    assert len({v["active"] for _, _, v in info}) >= 12
    assert any(v["roles"][0] == 0 for _, _, v in info)
    assert any(v["edge"] for _, _, v in info)
    assert any(v["logical_shape"][0] > v["logical_shape"][1] for _, _, v in info)
    assert any(v["stamp_shape"][0] != v["stamp_shape"][1] for _, _, v in info)
    assert any(np.array_equal(v["stamp"], np.fliplr(v["stamp"])) for _, _, v in info)
    assert any(split == "test" and v["scale"] == 2 for _, split, v in info)
    assert {v["layout"] for _, _, v in info} == {
        "panel_above", "panel_below", "panel_left", "panel_right"}
    assert any(v["stamp"].sum() <= max(2, v["stamp"].size // 4)
               for _, _, v in info)
    assert any(v["stamp"].sum() >= v["stamp"].size - max(2, v["stamp"].size // 4)
               for _, _, v in info)
    assert any(np.any(np.all(v["mask"], axis=1)) for _, _, v in info)
    assert any(np.any(np.all(v["mask"], axis=0)) for _, _, v in info)
    assert any(v["mask"].shape[0] >= 2 and v["mask"].shape[1] >= 2
               and any(np.all(v["mask"][y:y + 2, x:x + 2])
                       for y in range(v["mask"].shape[0] - 1)
                       for x in range(v["mask"].shape[1] - 1))
               for _, _, v in info)
    assert any(v["active"] >= 2
               and not np.any(v["mask"][1:, :] & v["mask"][:-1, :])
               and not np.any(v["mask"][:, 1:] & v["mask"][:, :-1])
               for _, _, v in info)
    assert any(len({x["roles"][3] for s in ("train", "test")
                    for x in (_analyze_pair(q["input"], q["output"]) for q in p[s])}) > 1
               for p in automatic)
    assert any(len({x["roles"] for s in ("train", "test")
                    for x in (_analyze_pair(q["input"], q["output"]) for q in p[s])}) == 1
               for p in automatic)
