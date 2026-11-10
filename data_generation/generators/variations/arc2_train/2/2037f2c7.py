"""ARC copies with one-way erasures and a full-width row-interval mask."""

import hashlib
import numpy as np


SOURCE_DIGESTS = {
    "7eaa333f82abb8f0f4dd9ad8bd45ba9c1f18955335d33bf3a0cca5df7661df50",
    "53a6658fb62e3ddaf696840996d8be55f5856478b6b8af9e287627cd4b337a02",
    "0bbf62f4bb94e59f4ee1c0813b73c739704ae61af528321fc3eebe0706bd5123",
    "3ebe1df288334aeed37fe75ebdc6a10d1ee8178969bf153e98520e3d705b5de9",
}


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} out of range")
    return value


def _key(grid):
    return grid.shape, grid.tobytes()


def _connected(mask):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(map(int, points[0]))
    seen, stack = {start}, [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(points)


def _motif(rng, h, w, bg, palette):
    for _ in range(100):
        symmetric = rng.random() < 0.43
        mask = np.zeros((h, w), dtype=bool)
        mid_l, mid_r = (w - 1) // 2, w // 2
        for r in range(h):
            left = int(rng.integers(0, mid_l + 1))
            right = w - 1 - left if symmetric else int(rng.integers(mid_r, w))
            mask[r, left:right + 1] = True
        bridge = int(rng.integers(1, h - 1)) if rng.random() < 0.55 else None
        if bridge is not None:
            mask[bridge, :] = True
        for r in range(h):
            if r != bridge and rng.random() < 0.42:
                c = int(rng.integers(1, mid_l + 1 if symmetric else w - 1))
                mask[r, c] = False
                if symmetric:
                    mask[r, w - 1 - c] = False
                elif c + 1 < w - 1 and rng.random() < 0.3:
                    mask[r, c + 1] = False
        if bridge is None:
            for r in range(1, h - 1):
                if mask[r].all():
                    if symmetric:
                        mask[r, 0] = mask[r, -1] = False
                    else:
                        mask[r, 0 if rng.random() < 0.5 else -1] = False
        if rng.random() < 0.28:
            k = max(2, w // 3)
            mask[0, :] = False
            mask[0, :k] = True
            mask[0, w - k:] = True
            mask[1, :] = True
        if not mask[0].any() or not mask[-1].any() or not mask.any(axis=0).all() or not _connected(mask):
            continue
        values = np.full((h, w), bg, dtype=np.int8)
        mode = int(rng.integers(0, 4))
        coarse = rng.integers(0, len(palette), size=(h // 2 + 2, w // 2 + 2))
        row_colors = rng.integers(0, len(palette), size=h)
        col_colors = rng.integers(0, len(palette), size=w)
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                cc = min(c, w - 1 - c) if symmetric else c
                if mode == 0:
                    index = (int(row_colors[r]) + cc // 2) % len(palette)
                elif mode == 1:
                    index = (int(col_colors[cc]) + r // 3) % len(palette)
                elif mode == 2:
                    index = int(coarse[r // 2, cc // 2])
                else:
                    index = (int(coarse[r // 2, cc // 2]) + int(row_colors[r])) % len(palette)
                values[r, c] = palette[index]
        return values
    raise ValueError("motif construction failed")


def _damage(rng, motif, bg, kind):
    h, w = motif.shape
    erased = np.zeros((h, w), dtype=bool)
    if kind == "single":
        cells = np.argwhere(motif != bg)
        r, c = cells[int(rng.integers(len(cells)))]
        erased[r, c] = True
    elif kind == "full_row":
        rows = [r for r in range(1, h - 1) if np.all(motif[r] != bg)]
        if not rows:
            raise ValueError("no complete row")
        erased[int(rng.choice(rows)), :] = True
    else:
        if kind == "gap":
            span = int(rng.integers(3, min(6, h) + 1))
            start = int(rng.integers(0, h - span + 1))
            rows = [start, start + span - 1]
            if span >= 5 and rng.random() < 0.5:
                rows.append(start + span - 2)
        elif kind == "edge":
            rows = [int(rng.choice([0, h - 1]))]
            if rng.random() < 0.4:
                rows.append(1 if rows[0] == 0 else h - 2)
        else:
            span = int(rng.integers(1, min(6, h) + 1))
            start = int(rng.integers(0, h - span + 1))
            rows = list(range(start, start + span))
        center = int(rng.integers(0, w))
        radius = int(rng.integers(1, max(2, w // 2 + 1)))
        density = float(rng.uniform(0.13, 0.82))
        for r in rows:
            columns = np.flatnonzero(motif[r] != bg)
            if kind == "witness":
                columns = columns[(columns > 0) & (columns < w - 1)]
            if kind == "cluster":
                columns = columns[(columns >= center - radius) & (columns <= center + radius)]
            if not len(columns):
                raise ValueError("no eligible cells")
            amount = 1 if kind == "witness" else max(1, min(len(columns), int(np.ceil(len(columns) * density))))
            erased[r, rng.choice(columns, size=amount, replace=False)] = True
    if kind == "gap":
        affected = np.flatnonzero(erased.any(axis=1))
        if affected[-1] - affected[0] + 1 == len(affected):
            raise ValueError("no blank intervening row")
    damaged = motif.copy()
    damaged[erased] = bg
    original, survivor = np.argwhere(motif != bg), np.argwhere(damaged != bg)
    if not len(survivor) or not np.array_equal(original.min(axis=0), survivor.min(axis=0)) or not np.array_equal(original.max(axis=0), survivor.max(axis=0)):
        raise ValueError("motif extent lost")
    return damaged, erased


def _place(rng, intact, damaged, bg):
    h, w = intact.shape
    layout = int(rng.integers(0, 3))
    if layout == 0:
        dr, dc = int(rng.integers(-h + 1, h)), w + int(rng.integers(1, 5))
        if rng.random() < 0.24:
            dr = 0
    elif layout == 1:
        dr, dc = h + int(rng.integers(1, 5)), int(rng.integers(-w + 1, w))
        if rng.random() < 0.24:
            dc = 0
    else:
        dr, dc = h + int(rng.integers(1, 4)), w + int(rng.integers(1, 4))
    if rng.random() < 0.5:
        dr = -dr
    if rng.random() < 0.5:
        dc = -dc
    good_r, good_c = max(0, -dr), max(0, -dc)
    bad_r, bad_c = good_r + dr, good_c + dc
    span_h, span_w = max(good_r, bad_r) + h, max(good_c, bad_c) + w
    if span_h > 30 or span_w > 30:
        raise ValueError("canvas overflow")
    extra_h = 30 - span_h if rng.random() < 0.12 else int(rng.integers(0, min(6, 30 - span_h) + 1))
    extra_w = 30 - span_w if rng.random() < 0.12 else int(rng.integers(0, min(6, 30 - span_w) + 1))
    top, left = int(rng.integers(0, extra_h + 1)), int(rng.integers(0, extra_w + 1))
    grid = np.full((span_h + extra_h, span_w + extra_w), bg, dtype=np.int8)
    grid[top + good_r:top + good_r + h, left + good_c:left + good_c + w] = intact
    grid[top + bad_r:top + bad_r + h, left + bad_c:left + bad_c + w] = damaged
    return grid


def _runs(projection):
    active = np.flatnonzero(projection)
    if not len(active):
        return []
    runs = []
    start = previous = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value > previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _bbox(mask):
    cells = np.argwhere(mask)
    if not len(cells):
        return None
    low, high = cells.min(axis=0), cells.max(axis=0)
    return int(low[0]), int(high[0]), int(low[1]), int(high[1])


def _recover(grid, bg):
    active = grid != bg
    candidates = {}
    row_coordinates = np.arange(grid.shape[0])[:, None]
    col_coordinates = np.arange(grid.shape[1])[None, :]
    for axis in (0, 1):
        runs = _runs(active.any(axis=1 - axis))
        for boundary in range(1, len(runs)):
            cut = runs[boundary][0]
            first = active & ((row_coordinates < cut) if axis == 0 else (col_coordinates < cut))
            boxes = [_bbox(first), _bbox(active & ~first)]
            if None in boxes:
                continue
            a, b = boxes
            if (a[1] - a[0], a[3] - a[2]) != (b[1] - b[0], b[3] - b[2]):
                continue
            patches = [grid[x[0]:x[1] + 1, x[2]:x[3] + 1] for x in boxes]
            diff = patches[0] != patches[1]
            if not diff.any():
                continue
            erased0 = (patches[0] == bg) & (patches[1] != bg)
            erased1 = (patches[1] == bg) & (patches[0] != bg)
            if np.array_equal(diff, erased0):
                erased, bad = erased0, 0
            elif np.array_equal(diff, erased1):
                erased, bad = erased1, 1
            else:
                continue
            rows = np.flatnonzero(erased.any(axis=1))
            key = tuple(sorted(tuple(box) for box in boxes))
            candidates[key] = (erased[rows[0]:rows[-1] + 1], boxes, bad, patches)
    if len(candidates) != 1:
        raise AssertionError("ambiguous coordinate frames")
    return next(iter(candidates.values()))


def _make_pair(rng, highlight, bg, w, h, kind):
    colors = [c for c in range(10) if c not in (bg, highlight)]
    palette = [int(x) for x in rng.choice(colors, size=int(rng.integers(1, 8)), replace=False)]
    for _ in range(180):
        quarter_turns = int(rng.integers(0, 4))
        base_h, base_w = (w, h) if quarter_turns % 2 else (h, w)
        motif = np.rot90(_motif(rng, base_h, base_w, bg, palette), quarter_turns)
        if rng.random() < 0.5:
            motif = motif[:, ::-1].copy()
        try:
            damaged, erased = _damage(rng, motif, bg, kind)
            inp = _place(rng, motif, damaged, bg)
        except ValueError:
            continue
        if np.count_nonzero(inp == bg) <= max(np.count_nonzero(inp == c) for c in colors + [highlight]):
            continue
        try:
            recovered, _, _, _ = _recover(inp, bg)
        except AssertionError:
            continue
        rows = np.flatnonzero(erased.any(axis=1))
        expected = erased[rows[0]:rows[-1] + 1]
        if not np.array_equal(recovered, expected):
            continue
        out = np.full(expected.shape, bg, dtype=np.int8)
        out[expected] = highlight
        return {"input": inp, "output": out}
    raise ValueError("unambiguous pair construction failed")


def _analyze(pair, highlight):
    inp, out = pair["input"], pair["output"]
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    assert bg != highlight and highlight not in colors
    assert set(map(int, np.unique(out))) <= {bg, highlight}
    mask, boxes, bad, patches = _recover(inp, bg)
    assert mask.any() and out.shape == mask.shape
    expected = np.full(mask.shape, bg, dtype=np.int8)
    expected[mask] = highlight
    assert np.array_equal(out, expected)
    assert np.all((patches[bad] == bg) | (patches[bad] == patches[1 - bad]))
    intact = patches[1 - bad]
    assert _connected(intact != bg) and intact.shape[0] >= 6 and intact.shape[1] >= 5
    return {
        "background": bg, "width": out.shape[1], "band": out.shape[0],
        "blank_row": bool(np.any(~mask.any(axis=1))),
        "horizontal_margin": bool(not mask[:, 0].any() and not mask[:, -1].any()),
        "all_highlight": bool(mask.all()), "boxes": boxes,
        "damaged_index": bad, "intact": intact,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            ikey = _key(pair["input"])
            pkey = (ikey, _key(pair["output"]))
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            digest = hashlib.sha256(pair["input"].astype(np.int8).tobytes() + pair["output"].astype(np.int8).tobytes()).hexdigest()
            assert digest not in SOURCE_DIGESTS
    first = puzzle["train"][0]
    values, counts = np.unique(first["input"], return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    highlights = set(map(int, np.unique(first["output"]))) - {bg}
    assert len(highlights) == 1
    highlight = highlights.pop()
    infos = {s: [_analyze(pair, highlight) for pair in puzzle[s]] for s in ("train", "test")}
    assert infos["train"][0]["horizontal_margin"]
    assert infos["train"][1]["blank_row"]
    assert infos["train"][0]["background"] != infos["train"][1]["background"]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _count("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("invalid counts")
    highlight = int(rng.integers(0, 10))
    backgrounds = [c for c in range(10) if c != highlight]
    rng.shuffle(backgrounds)
    heldout = rng.random() < 0.3
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            bg = backgrounds[index] if split == "train" and index < 2 else int(rng.choice(backgrounds))
            kind = "witness" if split == "train" and index == 0 else "gap" if split == "train" and index == 1 else str(rng.choice(["single", "sparse", "scatter", "cluster", "dense", "edge", "gap", "full_row"]))
            for _ in range(100):
                w = int(rng.integers(5, 9)) if heldout and split == "train" else int(rng.integers(9, 12)) if heldout and split == "test" else int(rng.integers(5, 12))
                h = int(rng.integers(6, 16))
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _make_pair(child, highlight, bg, w, h, kind)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key in used:
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    highlights, backgrounds, widths, bands, shapes, positions = set(), set(), set(), set(), set(), set()
    blank_rows = full_masks = margin_masks = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first = puzzle["train"][0]
        colors, counts = np.unique(first["input"], return_counts=True)
        bg = int(colors[int(np.argmax(counts))])
        highlight = (set(map(int, np.unique(first["output"]))) - {bg}).pop()
        highlights.add(highlight)
        local_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair, highlight)
                local_backgrounds.add(info["background"])
                backgrounds.add(info["background"])
                widths.add(info["width"])
                bands.add(info["band"])
                shapes.add(pair["input"].shape)
                boxes = info["boxes"]
                bad = info["damaged_index"]
                positions.add((int(np.sign(boxes[bad][0] - boxes[1 - bad][0])), int(np.sign(boxes[bad][2] - boxes[1 - bad][2]))))
                blank_rows += info["blank_row"]
                full_masks += info["all_highlight"]
                margin_masks += info["horizontal_margin"]
        assert len(local_backgrounds) >= 2
    assert len(highlights) >= 8 and len(backgrounds) >= 8
    assert len(widths) >= 7 and len(bands) >= 5 and len(shapes) >= 40
    assert len(positions) >= 8 and blank_rows >= 42 and margin_masks >= 42 and full_masks > 0
    assert _equal(generate_puzzle(12345, 6, 3), generate_puzzle(12345, 6, 3))
    assert _equal(generate_puzzle(54321), generate_puzzle(54321))
    assert not _equal(generate_puzzle(777, 5, 2), generate_puzzle(778, 5, 2))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(60000 + i, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (2, 4)):
        try:
            generate_puzzle(62000, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
