"""Corner-anchored periodic completion of separated rectangular fields."""

import hashlib
import numpy as np


CORNERS = ("TL", "TR", "BL", "BR")
SHAPES = ((2, 2), (3, 3), (2, 3), (3, 2), (2, 4), (4, 2),
          (3, 4), (4, 3), (3, 5), (5, 3), (4, 4))
SOURCE_DIGESTS = {
    "39e11a907008bbff8e88fe7ff6c7ce252ca7f4766762bc6a5e32257178e8bd95",
    "e171d986deb6163453b3cee6be321d366e419f163b76e27536c856a5a1cb1508",
    "848dd5fb81c17aeeefd604472108d6430c932356a4c1a9fce0399665239b9362",
    "92d231bbd9908b9231d56ba8fd4b9a05266a100f9ce5e9e094dc009e6d771708",
}


def _counts(rng, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _shuffle(rng, values):
    return [values[int(i)] for i in rng.permutation(len(values))]


def _primitive(mask):
    h, w = mask.shape
    for period in range(1, h):
        if h % period == 0 and np.array_equal(mask, np.tile(mask[:period], (h // period, 1))):
            return False
    for period in range(1, w):
        if w % period == 0 and np.array_equal(mask, np.tile(mask[:, :period], (1, w // period))):
            return False
    return True


def _motif(rng, shape, blank, accent, style):
    h, w = shape
    for _ in range(500):
        if style == 0:  # A sparse drawing with just enough boundary evidence.
            mask = rng.random(shape) < float(rng.uniform(0.16, 0.34))
        elif style == 1:  # Dense drawing with repeated blank holes.
            mask = rng.random(shape) < float(rng.uniform(0.62, 0.84))
        elif style == 2:  # Checker, stagger, diagonal, and stripe variations.
            r, c = np.indices(shape)
            mode = int(rng.integers(4))
            if mode == 0:
                mask = (r + c + int(rng.integers(2))) % 2 == 0
            elif mode == 1:
                mask = (r + 2 * c + int(rng.integers(3))) % 3 != 0
            elif mode == 2:
                mask = (2 * r + c + int(rng.integers(3))) % 3 != 0
            else:
                mask = ((r + c) % 3 == 0) | ((r - c) % 3 == 0)
            if rng.random() < 0.45:
                mask[int(rng.integers(h)), int(rng.integers(w))] ^= True
        else:
            mask = rng.random(shape) < float(rng.uniform(0.32, 0.64))
        # The accent's tight bounding box must expose the whole period.
        for axis, edge in ((0, 0), (0, h - 1), (1, 0), (1, w - 1)):
            if axis == 0 and not mask[edge].any():
                mask[edge, int(rng.integers(w))] = True
            if axis == 1 and not mask[:, edge].any():
                mask[int(rng.integers(h)), edge] = True
        if mask.all():
            mask[int(rng.integers(h)), int(rng.integers(w))] = False
        if mask.any() and not mask.all() and _primitive(mask):
            result = np.full(shape, blank, dtype=np.int8)
            result[mask] = accent
            return result
    raise RuntimeError("motif construction failed")


def _spread(rng, total, n):
    parts = [0] * n
    for _ in range(total):
        parts[int(rng.integers(n))] += 1
    return parts


def _slots(rng, shapes, mode):
    n = len(shapes)
    minimum = [(2 * h, 2 * w) for h, w in shapes]
    if mode == "row":
        min_h = max(h for h, _ in minimum) + 2
        min_w = sum(w for _, w in minimum) + n + 1
        if min_h > 30 or min_w > 30:
            return None
        H = int(rng.integers(min_h, 31))
        W = int(rng.integers(min_w, 31))
        extras = _spread(rng, W - min_w, n)
        widths = [minimum[i][1] + extras[i] for i in range(n)]
        slots, left = [], 1
        for width in widths:
            slots.append((1, left, H - 2, width))
            left += width + 1
        return H, W, slots
    if mode == "column":
        min_h = sum(h for h, _ in minimum) + n + 1
        min_w = max(w for _, w in minimum) + 2
        if min_h > 30 or min_w > 30:
            return None
        H = int(rng.integers(min_h, 31))
        W = int(rng.integers(min_w, 31))
        extras = _spread(rng, H - min_h, n)
        heights = [minimum[i][0] + extras[i] for i in range(n)]
        slots, top = [], 1
        for height in heights:
            slots.append((top, 1, height, W - 2))
            top += height + 1
        return H, W, slots
    if mode == "grid":
        cells = _shuffle(rng, [(0, 0), (0, 1), (1, 0), (1, 1)])[:n]
        row_min = [max((minimum[i][0] for i in range(n) if cells[i][0] == row), default=4)
                   for row in (0, 1)]
        col_min = [max((minimum[i][1] for i in range(n) if cells[i][1] == col), default=4)
                   for col in (0, 1)]
        min_h, min_w = sum(row_min) + 3, sum(col_min) + 3
        if min_h > 30 or min_w > 30:
            return None
        H, W = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        hs = [row_min[i] + x for i, x in enumerate(_spread(rng, H - min_h, 2))]
        ws = [col_min[i] + x for i, x in enumerate(_spread(rng, W - min_w, 2))]
        slots = [(1 if r == 0 else hs[0] + 2,
                  1 if c == 0 else ws[0] + 2, hs[r], ws[c]) for r, c in cells]
        return H, W, slots
    return None


def _field_dimension(rng, period, cap, exact_only):
    choices = []
    for copies in range(2, 6):
        if copies * period <= cap:
            choices.append(copies)
    if not choices:
        raise ValueError("field does not fit slot")
    weights = np.array([5, 3, 2, 2][:len(choices)], dtype=float)
    weights /= weights.sum()
    copies = int(rng.choice(choices, p=weights))
    base = copies * period
    if exact_only or base == cap or rng.random() < 0.52:
        return base
    return base + int(rng.integers(1, min(period, cap - base + 1)))


def _build_fields(rng, specs, caps, blank, exact_only):
    fields = []
    for spec, (cap_h, cap_w) in zip(specs, caps):
        h, w = spec["shape"]
        fh = _field_dimension(rng, h, cap_h, exact_only)
        fw = _field_dimension(rng, w, cap_w, exact_only)
        fields.append((fh, fw, _motif(rng, (h, w), blank, spec["accent"], spec["style"]), spec["anchor"]))
    return fields


def _free_positions(rng, fields, H, W):
    positions = []
    for fh, fw, _, _ in fields:
        candidates = [(r, c) for r in range(1, H - fh) for c in range(1, W - fw)
                      if all(r + fh + 1 <= rr or rr + hh + 1 <= r or
                             c + fw + 1 <= cc or cc + ww + 1 <= c
                             for (rr, cc), (hh, ww, _, _) in zip(positions, fields))]
        if not candidates:
            return None
        positions.append(_choice(rng, candidates))
    return positions


def _tile(motif, fh, fw, origin_r, origin_c):
    h, w = motif.shape
    rows = (np.arange(fh) - origin_r) % h
    cols = (np.arange(fw) - origin_c) % w
    return motif[rows[:, None], cols[None, :]].copy()


def _pair(rng, specs, background, blank, exact_only):
    shapes = [spec["shape"] for spec in specs]
    n = len(shapes)
    modes = ["row", "column", "grid"] if n > 1 else ["row"]
    if n > 1 and all(2 * h <= 12 and 2 * w <= 12 for h, w in shapes):
        modes += ["free", "free"]
    for mode in _shuffle(rng, modes):
        if mode == "free":
            H, W = int(rng.integers(23, 31)), int(rng.integers(23, 31))
            fields = _build_fields(rng, specs, [(12, 12)] * n, blank, exact_only)
            positions = _free_positions(rng, fields, H, W)
            if positions is None:
                continue
        else:
            layout = _slots(rng, shapes, mode)
            if layout is None:
                continue
            H, W, slots = layout
            fields = _build_fields(rng, specs, [(s[2], s[3]) for s in slots], blank, exact_only)
            positions = [(top + int(rng.integers(cap_h - fh + 1)),
                          left + int(rng.integers(cap_w - fw + 1)))
                         for (top, left, cap_h, cap_w), (fh, fw, _, _) in zip(slots, fields)]
        inp = np.full((H, W), background, dtype=np.int8)
        out = inp.copy()
        for (fh, fw, motif, anchor), (top, left) in zip(fields, positions):
            h, w = motif.shape
            origin_r = 0 if anchor[0] == "T" else fh - h
            origin_c = 0 if anchor[1] == "L" else fw - w
            block = np.full((fh, fw), blank, dtype=np.int8)
            block[origin_r:origin_r + h, origin_c:origin_c + w] = motif
            inp[top:top + fh, left:left + fw] = block
            out[top:top + fh, left:left + fw] = _tile(motif, fh, fw, origin_r, origin_c)
        return {"input": inp, "output": out}
    raise RuntimeError("no valid layout")


def _signature(array):
    return array.shape, array.tobytes()


def _digest(pair):
    sha = hashlib.sha256()
    for key in ("input", "output"):
        a = pair[key]
        sha.update(bytes(a.shape))
        sha.update(a.astype(np.uint8).tobytes())
    return sha.hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    colors = [int(v) for v in rng.permutation(10)]
    base_background, base_blank = colors[:2]
    palette = colors[2:2 + int(rng.integers(2, 5))]
    role_swap = bool(rng.random() < 0.38)
    source_like_exact = bool(rng.random() < 0.20)

    train_counts = [2, 2] if nt == 2 else [1 + int(rng.random() < 0.55) for _ in range(nt)]
    while sum(train_counts) < max(4, len(palette)):
        i = int(rng.integers(nt))
        train_counts[i] = 2
    total_fields = sum(train_counts)
    anchors = _shuffle(rng, list(CORNERS))
    anchors += [_choice(rng, CORNERS) for _ in range(total_fields - 4)]
    accents = _shuffle(rng, palette)
    accents += [_choice(rng, palette) for _ in range(total_fields - len(accents))]
    # The shape pool is taught explicitly; tests only draw from this pool.
    non_square = [s for s in SHAPES if s[0] != s[1]]
    shape_pool = [(2, 2), _choice(rng, non_square)]
    if rng.random() < 0.55:
        shape_pool.append((3, 5) if rng.random() < 0.7 else (5, 3))
    if rng.random() < 0.65 and (3, 3) not in shape_pool:
        shape_pool.append((3, 3))
    while len(shape_pool) < min(total_fields, int(rng.integers(3, 6))):
        candidate = _choice(rng, SHAPES)
        if candidate not in shape_pool:
            shape_pool.append(candidate)
    shapes = _shuffle(rng, shape_pool)
    shapes += [_choice(rng, shape_pool) for _ in range(total_fields - len(shapes))]
    styles = _shuffle(rng, [0, 1, 2])
    styles += [int(rng.integers(4)) for _ in range(total_fields - len(styles))]
    roles = [(base_background, base_blank)] * nt
    if role_swap:
        roles = _shuffle(rng, [roles[0], (base_blank, base_background)] + roles[2:])
        for i in range(nt):
            if i >= 2 and rng.random() < 0.5:
                roles[i] = (base_blank, base_background)
    train_specs, offset = [], 0
    for count in train_counts:
        train_specs.append([{"anchor": anchors[offset + i], "accent": accents[offset + i],
                             "shape": shapes[offset + i], "style": styles[offset + i]}
                            for i in range(count)])
        offset += count
    demonstrated_shapes = list(dict.fromkeys(shapes))
    demonstrated_accents = list(dict.fromkeys(accents))
    test_counts = [int(_choice(rng, [1, 2, 3, 3, 3, 4])) for _ in range(ns)]
    test_specs = []
    for count in test_counts:
        pair_specs = []
        for _ in range(count):
            pair_specs.append({"anchor": _choice(rng, CORNERS),
                               "accent": _choice(rng, demonstrated_accents),
                               "shape": _choice(rng, demonstrated_shapes),
                               "style": int(rng.integers(4))})
        test_specs.append(pair_specs)
    seen = set()
    result = {"train": [], "test": []}
    for split, groups in (("train", train_specs), ("test", test_specs)):
        for index, specs in enumerate(groups):
            background, blank = roles[index] if split == "train" else _choice(rng, roles)
            # Exact-only pairs occur even in puzzles that contain clipped fields.
            exact = source_like_exact or bool(rng.random() < 0.25)
            for _ in range(20):
                pair = _pair(rng, specs, background, blank, exact)
                signature = _signature(pair["input"])
                if signature not in seen and _digest(pair) not in SOURCE_DIGESTS:
                    seen.add(signature)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("unique pair construction failed")
    validate_generated_puzzle(result)
    return result


def _rectangles(grid, background):
    mask = grid != background
    seen = np.zeros(mask.shape, dtype=bool)
    rectangles = []
    for row, col in zip(*np.where(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        rs, cs = zip(*cells)
        top, bottom, left, right = min(rs), max(rs) + 1, min(cs), max(cs) + 1
        assert len(cells) == (bottom - top) * (right - left)
        rectangles.append((top, bottom, left, right))
    return rectangles


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    background = int(inp[0, 0])
    assert np.all(inp[0] == background) and np.all(inp[-1] == background)
    assert np.all(inp[:, 0] == background) and np.all(inp[:, -1] == background)
    assert np.array_equal(inp == background, out == background)
    fields = _rectangles(inp, background)
    assert fields
    info = []
    reconstruction = inp.copy()
    blanks = set()
    for top, bottom, left, right in fields:
        block = inp[top:bottom, left:right]
        h, w = block.shape
        values, counts = np.unique(block, return_counts=True)
        assert len(values) == 2
        blank = int(values[int(np.argmax(counts))])
        accent = int(values[int(np.argmin(counts))])
        blanks.add(blank)
        ar, ac = np.where(block == accent)
        r0, r1, c0, c1 = int(ar.min()), int(ar.max()) + 1, int(ac.min()), int(ac.max()) + 1
        ph, pw = r1 - r0, c1 - c0
        assert 2 <= ph <= 5 and 2 <= pw <= 5
        assert (r0 == 0) != (r1 == h) and (c0 == 0) != (c1 == w)
        anchor = ("T" if r0 == 0 else "B") + ("L" if c0 == 0 else "R")
        assert h >= 2 * ph and w >= 2 * pw
        motif = block[r0:r1, c0:c1]
        assert np.any(motif == blank) and _primitive(motif == accent)
        expected_input = np.full((h, w), blank, dtype=block.dtype)
        expected_input[r0:r1, c0:c1] = motif
        assert np.array_equal(block, expected_input)
        reconstruction[top:bottom, left:right] = _tile(motif, h, w, r0, c0)
        info.append({"anchor": anchor, "shape": (ph, pw), "field_shape": (h, w),
                     "accent": accent, "blank": blank,
                     "exact": h % ph == 0 and w % pw == 0,
                     "partial_axes": int(h % ph != 0) + int(w % pw != 0),
                     "copies": (h // ph, w // pw)})
    assert len(blanks) == 1 and np.array_equal(reconstruction, out)
    return {"background": background, "blank": blanks.pop(), "fields": info}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            si = _signature(pair["input"])
            sp = (si, _signature(pair["output"]))
            assert si not in seen_inputs and sp not in seen_pairs
            assert _digest(pair) not in SOURCE_DIGESTS
            seen_inputs.add(si)
            seen_pairs.add(sp)
            analysis = _analyze(pair)
            assert 1 <= len(analysis["fields"]) <= (2 if split == "train" else 4)
            analyses[split].append(analysis)
    train_fields = [f for a in analyses["train"] for f in a["fields"]]
    test_fields = [f for a in analyses["test"] for f in a["fields"]]
    assert {f["anchor"] for f in train_fields} == set(CORNERS)
    assert {f["shape"] for f in test_fields} <= {f["shape"] for f in train_fields}
    assert {f["accent"] for f in test_fields} <= {f["accent"] for f in train_fields}
    train_roles = {(a["background"], a["blank"]) for a in analyses["train"]}
    assert {(a["background"], a["blank"]) for a in analyses["test"]} <= train_roles
    assert len({c for pair in train_fields for c in (pair["accent"],)}) in (2, 3, 4)
    assert all(a["background"] != a["blank"] for a in analyses["train"] + analyses["test"])
    assert all(f["accent"] not in (a["background"], a["blank"])
               for split in ("train", "test") for a in analyses[split] for f in a["fields"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p["test"][0]["input"]) for p in puzzles}) >= 40
    all_info = {s: [_analyze(pair) for p in puzzles for pair in p[s]] for s in ("train", "test")}
    fields = [f for infos in all_info.values() for a in infos for f in a["fields"]]
    assert len({f["shape"] for f in fields}) >= 8
    assert (3, 5) in {f["shape"] for f in fields}
    assert {f["partial_axes"] for f in fields} == {0, 1, 2}
    assert any(max(f["copies"]) >= 5 for f in fields)
    assert any(len(a["fields"]) == 1 for a in all_info["train"])
    assert {len(a["fields"]) for a in all_info["test"]} == {1, 2, 3, 4}
    assert any(any(f["anchor"] == "TL" for f in a["fields"]) for a in all_info["test"])
    assert any(len({f["accent"] for f in a["fields"]}) < len(a["fields"])
               for a in all_info["test"] if len(a["fields"]) > 1)
    assert any(all(f["exact"] for a in [_analyze(x) for s in ("train", "test")
                                 for x in p[s]] for f in a["fields"]) for p in puzzles)
    assert any(len({(_analyze(x)["background"], _analyze(x)["blank"])
                    for s in ("train", "test") for x in p[s]}) > 1 for p in puzzles)
    palette_sizes = set()
    layouts = set()
    density_bins = set()
    patterned = 0
    training_only_color = 0
    same_color_different_tile = 0
    reused_test_anchor = 0
    repeated_test_shape = 0
    small_test = 0
    large_train = 0
    for puzzle in puzzles:
        train_analyses = [_analyze(pair) for pair in puzzle["train"]]
        test_analyses = [_analyze(pair) for pair in puzzle["test"]]
        train_colors = {f["accent"] for a in train_analyses for f in a["fields"]}
        test_colors = {f["accent"] for a in test_analyses for f in a["fields"]}
        palette_sizes.add(len(train_colors))
        training_only_color += bool(train_colors - test_colors)
        small_test += any(min(pair["input"].shape) < 22 for pair in puzzle["test"])
        large_train += any(min(pair["input"].shape) >= 22 for pair in puzzle["train"])
        reused_test_anchor += any(len({f["anchor"] for f in a["fields"]}) < len(a["fields"])
                                  for a in test_analyses)
        repeated_test_shape += any(len({f["shape"] for f in a["fields"]}) < len(a["fields"])
                                   for a in test_analyses)
        for split, analyses in (("train", train_analyses), ("test", test_analyses)):
            for pair, analysis in zip(puzzle[split], analyses):
                boxes = _rectangles(pair["input"], analysis["background"])
                if len(boxes) > 1:
                    if all(a[3] < b[2] or b[3] < a[2]
                           for i, a in enumerate(boxes) for b in boxes[i + 1:]):
                        layouts.add((split, "row"))
                    elif all(a[1] < b[0] or b[1] < a[0]
                             for i, a in enumerate(boxes) for b in boxes[i + 1:]):
                        layouts.add((split, "column"))
                    else:
                        layouts.add((split, "staggered"))
                motifs = []
                for (top, bottom, left, right), field in zip(boxes, analysis["fields"]):
                    block = pair["input"][top:bottom, left:right]
                    rows, cols = np.where(block == field["accent"])
                    motif = block[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
                    mask = motif == field["accent"]
                    motifs.append((field["accent"], motif.shape, mask.tobytes()))
                    density = float(mask.mean())
                    density_bins.add("sparse" if density <= 0.4 else
                                     "dense" if density >= 0.6 else "medium")
                    r, c = np.indices(mask.shape)
                    if any(np.array_equal(mask, (a * r + b * c + phase) % period == 0)
                           or np.array_equal(mask, (a * r + b * c + phase) % period != 0)
                           for period in (2, 3) for a, b in ((1, 1), (1, 2), (2, 1))
                           for phase in range(period)):
                        patterned += 1
                if split == "test" and any(x[0] == y[0] and x[2] != y[2]
                                           for i, x in enumerate(motifs) for y in motifs[i + 1:]):
                    same_color_different_tile += 1
    assert palette_sizes == {2, 3, 4}
    assert {("test", x) for x in ("row", "column", "staggered")} <= layouts
    assert {("train", x) for x in ("row", "column")} <= layouts
    assert density_bins == {"sparse", "medium", "dense"} and patterned >= 20
    assert training_only_color >= 1 and same_color_different_tile >= 1
    assert reused_test_anchor >= 1 and repeated_test_shape >= 1
    assert small_test >= 1 and large_train >= 1
    assert any(f["shape"] in ((2, 3), (3, 2)) for a in all_info["test"] for f in a["fields"])
    assert _equal(generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3))
    assert _equal(generate_puzzle(24680), generate_puzzle(24680))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                  (None, 4), (2, None), (10, None))):
        p = generate_puzzle(30000 + i, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for i, (nt, ns) in enumerate(((1, None), (11, None), (None, 0),
                                  (None, 5), (2, 2), (3, 4), (True, 1))):
        try:
            generate_puzzle(40000 + i, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
