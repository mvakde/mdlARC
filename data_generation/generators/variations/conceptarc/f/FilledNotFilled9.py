"""Generate rowwise completion puzzles with one outlined shape."""

import hashlib

import numpy as np


SOURCE_PAIR_DIGESTS = frozenset({
    "de11ce893c87be269ce6228320b37b1b064bc3a77428b90392c8255f29619035",
    "f9abaa0a710542c17376b87ac807662855c92ae82cfc65f99d6f9c20f168fdd5",
    "d73dc3e97aa9aa9584e22d1021f10ee8290f910847bfbb76f908dfcfaaaf4bb2",
    "a38415380e9169fd5beed867283f940709fcddf0025e14ee406690b9e2a4a8d0",
    "57a650c1a0441fbca320b0d53fc9a35daa68d76bb95e82fe99852d862b91f6e9",
    "071e2ee0cbcd9233131dfee10f7184ae171ff4f3a3fa3eb645c51829f8d4ea2e",
    "f4e4e5e93646e39b2a8cfbd3b0033b5e961608f6be5b85611346f574c8bb0452",
})


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _pair_digest(inp, out):
    h, w = inp.shape
    raw = bytes((h, w)) + inp.astype(np.uint8).tobytes()
    raw += out.astype(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _sides(rng, height, style, maximum_wall, partial=False):
    """Walk both sides with continuous outline strips."""
    minimum_start = 4 if partial else 3
    left, right = 0, int(rng.integers(minimum_start, 22)) - 1
    tl, tr = _wall_widths(rng, right - left + 1, maximum_wall,
                          minimum_interior=2 if partial else 1)
    bounds = [(left, right, tl, tr)]
    lo, hi = left, right
    turn = int(rng.integers(1, height - 1))
    for row in range(1, height - 1):
        if style == "straight":
            dl = dr = 0
            if rng.random() < 0.18:
                dl = dr = int(rng.choice((-1, 1)))
        elif style == "taper":
            dl = int(rng.choice((0, 1, 2), p=(0.45, 0.42, 0.13)))
            dr = -int(rng.choice((0, 1, 2), p=(0.45, 0.42, 0.13)))
            if rng.random() < 0.12:
                dl -= 1
        elif style == "step":
            dl = int(rng.choice((-2, -1, 0, 0, 0, 1, 2)))
            dr = int(rng.choice((-2, -1, 0, 0, 0, 1, 2)))
        elif style == "bulge":
            direction = -1 if row < turn else 1
            dl = direction * int(rng.choice((0, 1, 2)))
            dr = -direction * int(rng.choice((0, 1, 2)))
        else:
            moves = (-3, -2, -1, 0, 1, 2, 3)
            weights = (0.04, 0.10, 0.20, 0.32, 0.20, 0.10, 0.04)
            dl = int(rng.choice(moves, p=weights))
            dr = int(rng.choice(moves, p=weights))
        for move_l, move_r in ((dl, dr), (dl, 0), (0, dr), (0, 0)):
            a, b = left + move_l, right + move_r
            min_tl, min_tr = max(1, left - a), max(1, b - right)
            if not (3 <= b - a + 1 <= 24 and
                    max(hi, b) - min(lo, a) + 1 <= 25 and
                    a <= right and b >= left and
                    a <= left + tl and b >= right - tr and
                    min_tl <= maximum_wall and min_tr <= maximum_wall and
                    min_tl + min_tr < b - a + 1):
                continue
            new_tl, new_tr = _wall_widths(
                rng, b - a + 1, maximum_wall, min_tl, min_tr)
            left, right, tl, tr = a, b, new_tl, new_tr
            lo, hi = min(lo, a), max(hi, b)
            break
        bounds.append((left, right, tl, tr))
    return bounds


def _wall_widths(rng, width, maximum_wall, min_left=1, min_right=1,
                 minimum_interior=1):
    weights = (0.48, 0.30, 0.12, 0.07, 0.03)
    def draw(minimum, maximum):
        choices = np.arange(minimum, maximum + 1)
        probabilities = np.array([weights[k - 1] for k in choices])
        probabilities /= probabilities.sum()
        return int(rng.choice(choices, p=probabilities))
    left = draw(min_left, min(maximum_wall, width - min_right - minimum_interior))
    right = draw(min_right, min(maximum_wall, width - left - minimum_interior))
    return left, right


def _cap(rng, left, right, tl, tr):
    if left + tl == right - tr and rng.random() < 0.65:
        return left + tl, left + tl
    cap_left = int(rng.integers(left, min(left + tl, right) + 1))
    cap_right = int(rng.integers(max(right - tr, cap_left), right + 1))
    return cap_left, cap_right


def _make_pair(rng, colors, regime, mask, depth_mode="any"):
    bg, outline, fill = colors
    minimum = 7 if depth_mode == "long" else 5 if mask == "partial" else 4
    if regime == "source_train":
        height = int(rng.integers(minimum, 19))
        weights = (0.22, 0.24, 0.20, 0.14, 0.20)
    else:
        height = int(rng.integers(max(minimum, 5), 21))
        weights = (0.13, 0.19, 0.23, 0.16, 0.29)
    style = str(rng.choice(("straight", "taper", "step", "bulge", "irregular"),
                           p=weights))
    maximum_wall = int(rng.choice((2, 3, 5), p=(0.46, 0.32, 0.22)))
    sides = _sides(rng, height, style, maximum_wall, mask == "partial")
    cap = _cap(rng, *sides[-1])
    bounds = [(a, b) for a, b, _, _ in sides] + [cap]
    lo = min(a for a, _ in bounds)
    hi = max(b for _, b in bounds)
    span = hi - lo + 1
    extra = 30 - span
    ml = int(rng.integers(0, min(4, extra) + 1))
    mr = int(rng.integers(0, min(4, extra - ml) + 1))
    top, bottom = int(rng.integers(0, 5)), int(rng.integers(0, 5))
    if top == 0 and ml == 0:
        top = 1
    output = np.full((top + height + bottom, ml + span + mr), bg,
                     dtype=np.int8)
    interior = []
    for row, (left, right, tl, tr) in enumerate(sides):
        x0, x1 = ml + left - lo, ml + right - lo
        y = top + row
        output[y, x0:x0 + tl] = outline
        output[y, x0 + tl:x1 - tr + 1] = fill
        output[y, x1 - tr + 1:x1 + 1] = outline
        interior.append((y, x0 + tl, x1 - tr + 1))
    x0, x1 = ml + cap[0] - lo, ml + cap[1] - lo
    output[top + height - 1, x0:x1 + 1] = outline
    inp = output.copy()
    body = len(interior)
    if mask == "top":
        if depth_mode == "long":
            depth = int(rng.integers(4, body - 1))
        elif depth_mode == "short":
            depth = int(rng.integers(1, min(2, body - 2) + 1))
        else:
            depth = int(rng.integers(1, body - 1))
        selected = list(range(depth))
    elif mask == "middle":
        depth = int(rng.integers(1, body - 1))
        start = int(rng.integers(1, body - depth))
        selected = list(range(start, start + depth))
    elif mask == "partial":
        eligible = [i for i, (_, a, b) in enumerate(interior) if b - a >= 2]
        depth = int(rng.integers(1, min(len(eligible), body - 2, 5) + 1))
        first = int(rng.integers(0, len(eligible) - depth + 1))
        selected = eligible[first:first + depth]
    else:
        selected = []
    for row in selected:
        y, a, b = interior[row]
        if mask == "partial":
            gap_width = int(rng.integers(1, b - a))
            start = int(rng.integers(a, b - gap_width + 1))
            inp[y, start:start + gap_width] = bg
        else:
            inp[y, a:b] = bg
    return {"input": inp, "output": output}


def _complete(inp, bg, outline, fill):
    result = inp.copy()
    for y, row in enumerate(inp):
        occupied = np.flatnonzero(row != bg)
        if len(occupied) < 2:
            continue
        left, right = int(occupied[0]), int(occupied[-1])
        a, b = left, right
        while a <= right and row[a] == outline:
            a += 1
        while b >= left and row[b] == outline:
            b -= 1
        if a <= b:
            segment = result[y, a:b + 1]
            segment[segment == bg] = fill
    return result


def _predict_from_visible(train, inp):
    """Use demonstrations and input pixels only; no generation metadata."""
    assert any(not np.array_equal(p["input"], p["output"]) for p in train)
    bg = int(inp[0, 0])
    active = np.flatnonzero(np.any(inp != bg, axis=1))
    cap = inp[int(active[-1])]
    cap_colors = set(int(x) for x in cap if x != bg)
    assert len(cap_colors) == 1
    outline = cap_colors.pop()
    fills = set(int(x) for x in np.unique(inp)) - {bg, outline}
    assert len(fills) == 1
    return _complete(inp, bg, outline, fills.pop())


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    shared_palette = _palette(rng)
    varied_palette = bool(rng.random() < 0.28)
    classic_shift = bool(rng.random() < 0.22)
    extension = ("none" if classic_shift else
                 str(rng.choice(("none", "middle", "partial", "both"),
                                p=(0.48, 0.20, 0.20, 0.12))))
    if extension == "both" and ntrain < 3:
        extension = str(rng.choice(("middle", "partial")))
    demonstrated = []
    if extension in ("middle", "both"):
        demonstrated.append("middle")
    if extension in ("partial", "both"):
        demonstrated.append("partial")
    train_masks = ["top"] + demonstrated
    while len(train_masks) < ntrain:
        train_masks.append(str(rng.choice(["top", "top", "none"] + demonstrated)))
    if varied_palette and ntrain == 2 and train_masks[1] == "none":
        train_masks[1] = "top"
    rng.shuffle(train_masks)
    test_masks = []
    for _ in range(ntest):
        test_masks.append(str(rng.choice(["top", "top", "top", "none", "none"] +
                                         demonstrated * 2)))
    rng.shuffle(test_masks)
    long_test_index = int(rng.integers(0, ntest)) if classic_shift else -1
    if classic_shift:
        test_masks[long_test_index] = "top"
    seen = set()
    puzzle = {"train": [], "test": []}
    train_palettes = []
    for regime, masks in (("source_train", train_masks),
                          ("target_test", test_masks)):
        destination = "train" if regime == "source_train" else "test"
        for i, mask in enumerate(masks):
            depth_mode = ("short" if classic_shift and destination == "train"
                          else "long" if i == long_test_index and destination == "test"
                          else "any")
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                colors = _palette(pair_rng) if varied_palette else shared_palette
                if varied_palette and destination == "train" and i == 1 and colors == train_palettes[0]:
                    continue
                pair = _make_pair(pair_rng, colors, regime, mask, depth_mode)
                inp, out = pair["input"], pair["output"]
                key = (inp.shape, inp.tobytes())
                if key in seen or _pair_digest(inp, out) in SOURCE_PAIR_DIGESTS:
                    continue
                seen.add(key)
                puzzle[destination].append(pair)
                if destination == "train":
                    train_palettes.append(colors)
                break
            else:
                raise RuntimeError("could not produce a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= size <= 30 for size in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    assert _pair_digest(inp, out) not in SOURCE_PAIR_DIGESTS
    bg = int(out[0, 0])
    assert int(inp[0, 0]) == bg
    active = np.flatnonzero(np.any(out != bg, axis=1))
    assert len(active) >= 4
    assert np.array_equal(active, np.arange(active[0], active[-1] + 1))
    first, last = int(active[0]), int(active[-1])
    assert np.array_equal(inp[:first], out[:first])
    assert np.array_equal(inp[last + 1:], out[last + 1:])
    assert np.array_equal(inp[last], out[last])
    cap_cells = np.flatnonzero(out[last] != bg)
    assert len(cap_cells) >= 1
    assert np.array_equal(cap_cells, np.arange(cap_cells[0], cap_cells[-1] + 1))
    cap_colors = set(int(v) for v in out[last, cap_cells])
    assert len(cap_colors) == 1
    outline = cap_colors.pop()
    fills = set(int(v) for v in np.unique(out)) - {bg, outline}
    assert len(fills) == 1
    fill = fills.pop()
    assert len({bg, outline, fill}) == 3
    assert set(int(v) for v in np.unique(inp)) == {bg, outline, fill}
    assert np.array_equal(_complete(inp, bg, outline, fill), out)
    full_rows, partial_rows, intact_rows = [], [], []
    minimum_interior = 31
    thick_left = thick_right = 0
    outer_bounds = []
    left_strips, right_strips = [], []
    for y in range(first, last):
        row = out[y]
        occupied = np.flatnonzero(row != bg)
        assert len(occupied) >= 3
        left, right = int(occupied[0]), int(occupied[-1])
        assert np.array_equal(occupied, np.arange(left, right + 1))
        fills_here = np.flatnonzero(row == fill)
        assert len(fills_here) >= 1
        a, b = int(fills_here[0]), int(fills_here[-1])
        assert a > left and b < right
        assert np.all(row[left:a] == outline)
        assert np.all(row[a:b + 1] == fill)
        assert np.all(row[b + 1:right + 1] == outline)
        assert np.array_equal(inp[y, :a], row[:a])
        assert np.array_equal(inp[y, b + 1:], row[b + 1:])
        assert set(int(v) for v in inp[y, a:b + 1]).issubset({bg, fill})
        gap = int(np.count_nonzero(inp[y, a:b + 1] == bg))
        if gap == b - a + 1:
            full_rows.append(y - first)
        elif gap:
            partial_rows.append(y - first)
        else:
            intact_rows.append(y - first)
        minimum_interior = min(minimum_interior, b - a + 1)
        thick_left = max(thick_left, a - left)
        thick_right = max(thick_right, right - b)
        outer_bounds.append((left, right))
        left_strips.append((left, a - 1))
        right_strips.append((b + 1, right))
    assert len(intact_rows) >= 2
    assert not (full_rows and partial_rows)
    if full_rows:
        assert full_rows == list(range(full_rows[0], full_rows[-1] + 1))
        assert full_rows[-1] < len(outer_bounds) - 1
        mask = "top" if full_rows[0] == 0 else "middle"
    elif partial_rows:
        mask = "partial"
    else:
        mask = "none"
    assert all(a[0] <= b[1] and b[0] <= a[1]
               for a, b in zip(outer_bounds, outer_bounds[1:]))
    for strips in (left_strips, right_strips):
        assert all(a[0] <= b[1] + 1 and b[0] <= a[1] + 1
                   for a, b in zip(strips, strips[1:]))
    assert int(cap_cells[0]) >= outer_bounds[-1][0]
    assert int(cap_cells[-1]) <= outer_bounds[-1][1]
    assert int(cap_cells[0]) <= left_strips[-1][1] + 1
    assert int(cap_cells[-1]) >= right_strips[-1][0] - 1
    return {
        "colors": (bg, outline, fill), "mask": mask,
        "depth": len(full_rows) if full_rows else len(partial_rows),
        "height": len(active),
        "width": max(b for _, b in outer_bounds) - min(a for a, _ in outer_bounds) + 1,
        "minimum_interior": minimum_interior, "cap_width": len(cap_cells),
        "thick_left": thick_left, "thick_right": thick_right,
        "edge_touch": bool(first == 0 or last == out.shape[0] - 1 or
                           any(a == 0 or b == out.shape[1] - 1
                               for a, b in outer_bounds)),
        "sharp_move": max((abs(a[0] - b[0]) + abs(a[1] - b[1])
                           for a, b in zip(outer_bounds, outer_bounds[1:])),
                          default=0),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_info, test_info = [], []
    for section, infos in ((train, train_info), (test, test_info)):
        for pair in section:
            info = _analyze(pair)
            infos.append(info)
            inp, out = pair["input"], pair["output"]
            inp_key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = (inp_key, out.astype(np.uint8).tobytes())
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
    assert any(info["mask"] == "top" for info in train_info)
    train_palettes = {info["colors"] for info in train_info}
    if len(train_palettes) == 1:
        assert all(info["colors"] in train_palettes for info in test_info)
    for info in test_info:
        if info["mask"] in ("middle", "partial"):
            assert any(t["mask"] == info["mask"] for t in train_info)
    for pair in test:
        assert np.array_equal(_predict_from_visible(train, pair["input"]),
                              pair["output"])


def _same_puzzle(left, right):
    for section in ("train", "test"):
        if len(left[section]) != len(right[section]):
            return False
        for a, b in zip(left[section], right[section]):
            if any(not np.array_equal(a[key], b[key])
                   for key in ("input", "output")):
                return False
    return True


def self_test() -> None:
    train_counts, test_counts, palettes = set(), set(), set()
    sizes, shapes, masks, depths, cap_widths = set(), set(), set(), set(), set()
    narrow = edge = thick = sharp = long_train = short_test = identity_test = 0
    combined = varied_puzzles = 0
    classic_shifts = varied_shapes = varied_masks = varied_walls = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        all_info = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze(pair)
                all_info.append(info)
                inp, out = pair["input"], pair["output"]
                assert _pair_digest(inp, out) not in SOURCE_PAIR_DIGESTS
                assert np.array_equal(_predict_from_visible(puzzle["train"], inp), out)
                palettes.add(info["colors"])
                sizes.add(inp.shape)
                masks.add(info["mask"])
                depths.add(info["depth"])
                cap_widths.add(info["cap_width"])
                shapes.add((info["height"], info["width"], info["cap_width"],
                            info["minimum_interior"], info["thick_left"],
                            info["thick_right"]))
                narrow += info["minimum_interior"] <= 2
                edge += info["edge_touch"]
                thick += max(info["thick_left"], info["thick_right"]) >= 3
                sharp += info["sharp_move"] >= 3
                combined += (info["minimum_interior"] <= 2 and
                             info["edge_touch"] and info["depth"] >= 1)
                if section == "train":
                    long_train += info["depth"] >= 4
                else:
                    short_test += 0 < info["depth"] <= 2
                    identity_test += info["mask"] == "none"
        varied_puzzles += len({x["colors"] for x in all_info}) > 1
        train_info = all_info[:len(puzzle["train"])]
        test_info = all_info[len(puzzle["train"]):]
        classic_shifts += (max(x["depth"] for x in train_info) <= 2 and
                           max(x["depth"] for x in test_info) >= 4 and
                           all(x["mask"] in ("top", "none") for x in train_info))
        varied_shapes += len({p["input"].shape for section in ("train", "test")
                              for p in puzzle[section]}) >= 2
        varied_masks += len({x["mask"] for x in all_info}) >= 2
        varied_walls += (any(max(x["thick_left"], x["thick_right"]) >= 3
                             for x in all_info) and
                         any(max(x["thick_left"], x["thick_right"]) <= 2
                             for x in all_info))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 35 and len(sizes) >= 45 and len(shapes) >= 100
    assert {"none", "top", "middle", "partial"}.issubset(masks)
    assert {1, 2, 3, 4, 7}.issubset(depths)
    assert min(cap_widths) == 1 and max(cap_widths) >= 9
    assert min(narrow, edge, thick, sharp, long_train, short_test,
               identity_test, combined, varied_puzzles, classic_shifts) > 0
    assert varied_shapes >= 35 and varied_masks >= 25 and varied_walls >= 20
    count_cases = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None))
    for i, counts in enumerate(count_cases):
        puzzle = generate_puzzle(1000 + i, *counts)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(1000 + i, *counts))
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, None)):
        try:
            generate_puzzle(99, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    a = generate_puzzle(72, 2, 1)
    assert not _same_puzzle(a, generate_puzzle(73, 2, 1))
    changed = next(p for p in a["train"] if not np.array_equal(p["input"], p["output"]))
    y, x = np.argwhere(changed["output"] != changed["input"])[0]
    changed["output"][y, x] = changed["input"][y, x]
    try:
        validate_generated_puzzle(a)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output was accepted")
