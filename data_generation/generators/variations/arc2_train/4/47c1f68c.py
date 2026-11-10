"""Solved puzzles for the removable-cross, two-axis reflection family."""

import numpy as np


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(value) for value in grid.flat))


def _source_key(rows):
    return ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))


_SOURCE_INPUT_KEYS = frozenset(
    _source_key(rows)
    for rows in (
        (
            "303080000", "330080000", "300080000", "000080000",
            "888888888", "000080000", "000080000", "000080000",
            "000080000",
        ),
        (
            "00000200000", "01000200000", "11000200000", "01100200000",
            "00000200000", "22222222222", "00000200000", "00000200000",
            "00000200000", "00000200000", "00000200000",
        ),
        (
            "2004000", "0224000", "0204000", "4444444",
            "0004000", "0004000", "0004000",
        ),
        (
            "0080003000000", "0800003000000", "8080003000000",
            "0080003000000", "0088003000000", "0000003000000",
            "3333333333333", "0000003000000", "0000003000000",
            "0000003000000", "0000003000000", "0000003000000",
            "0000003000000",
        ),
    )
)


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train")
    test = _checked_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _parameters(rng):
    colors = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    return {
        "size_mode": int(rng.integers(0, 5)),
        "quadrant_mode": int(rng.integers(0, 3)),
        "palette_mode": int(rng.integers(0, 3)),
        "complexity": int(rng.integers(0, 3)),
        "anchor_quadrant": int(rng.integers(0, 4)),
        "colors": colors,
    }


def _dimensions(rng, parameters, section, index, train_dimensions):
    mode = parameters["size_mode"]
    if mode == 0:
        return train_dimensions[0]
    if mode == 1:
        low, high = (3, 7) if section == "train" else (8, 14)
    elif mode == 2:
        low, high = (8, 14) if section == "train" else (2, 7)
    elif mode == 3 and section == "test" and index == 0:
        return train_dimensions[int(rng.integers(0, len(train_dimensions)))]
    else:
        low, high = 2, 14
    height = int(rng.integers(low, high + 1))
    width = height if rng.random() < 0.3 else int(rng.integers(low, high + 1))
    return height, width


def _quadrant(rng, parameters, section, index):
    anchor = parameters["anchor_quadrant"]
    mode = parameters["quadrant_mode"]
    if mode == 0:
        return anchor
    if mode == 2 and section == "test":
        return anchor
    if mode == 2:
        choices = [q for q in range(4) if q != anchor]
        if index < 2:
            return choices[index]
        return choices[int(rng.integers(0, 3))]
    if section == "train" and index < 2:
        return (anchor + index) % 4
    return int(rng.integers(0, 4))


def _palette(rng, parameters, section, index, train_palettes):
    mode = parameters["palette_mode"]
    base = parameters["colors"]
    if mode == 0:
        return base
    if mode == 1:
        if section == "train" and index == 0:
            return base
        if section == "train" and index == 1:
            return (base[0], base[2], base[1])
        if section == "test" and index == 0:
            bg, src, cross = train_palettes[0]
            return (bg, cross, src)
        return base if rng.random() < 0.5 else (base[0], base[2], base[1])
    if section == "train" and index == 0:
        return base
    roll = rng.random()
    if roll < 0.25:
        return train_palettes[int(rng.integers(0, len(train_palettes)))] if train_palettes else base
    if roll < 0.48:
        bg, src, cross = train_palettes[int(rng.integers(0, len(train_palettes)))] if train_palettes else base
        return (bg, cross, src)
    return tuple(int(c) for c in rng.choice(10, size=3, replace=False))


def _informative(mask):
    if mask.shape[0] != mask.shape[1] or mask.shape[0] < 3:
        return False
    count = int(mask.sum())
    if not 3 <= count <= mask.size - 2:
        return False
    return all(
        not np.array_equal(mask, transform)
        for transform in (
            mask[::-1, :], mask[:, ::-1], mask.T, mask[::-1, ::-1].T
        )
    )


def _place(rng, local, height, width):
    rows, cols = np.nonzero(local)
    if len(rows) == 0:
        local = local.copy()
        local[0, 0] = True
        rows, cols = np.nonzero(local)
    local = local[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    bh, bw = local.shape
    if bh <= height - 2 and rng.random() < 0.38:
        top = int(rng.integers(1, height - bh))
    else:
        top = int(rng.integers(0, height - bh + 1))
    if bw <= width - 2 and rng.random() < 0.38:
        left = int(rng.integers(1, width - bw))
    else:
        left = int(rng.integers(0, width - bw + 1))
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + bh, left:left + bw] = local
    return mask


def _sample_mask(rng, height, width, complexity, informative=False):
    if informative:
        for _ in range(200):
            probability = float(rng.uniform(0.18, 0.62))
            mask = rng.random((height, width)) < probability
            if _informative(mask):
                return mask
        raise RuntimeError("unable to sample an informative motif")

    if complexity == 0:
        style = int(rng.choice((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), p=(.13, .07, .08, .16, .13, .12, .11, .08, .08, .04)))
    elif complexity == 1:
        style = int(rng.integers(0, 10))
    else:
        style = int(rng.choice((3, 4, 5, 6, 7, 8, 9)))

    if style == 0:
        return _place(rng, np.ones((1, 1), dtype=bool), height, width)
    if style == 1:
        return np.ones((height, width), dtype=bool)
    if style == 2:
        mask = np.ones((height, width), dtype=bool)
        mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = False
        return mask
    if style == 3:
        horizontal = bool(rng.integers(0, 2))
        length = int(rng.integers(2, (width if horizontal else height) + 1))
        local = np.ones((1, length) if horizontal else (length, 1), dtype=bool)
    elif style == 4:
        bh = int(rng.integers(2, min(height, 7) + 1))
        bw = int(rng.integers(2, min(width, 7) + 1))
        local = np.zeros((bh, bw), dtype=bool)
        local[:, 0] = True
        local[-1, :] = True
        if rng.random() < 0.5:
            local = local[::-1, :]
        if rng.random() < 0.5:
            local = local[:, ::-1]
    elif style == 5 and min(height, width) >= 3:
        bh = int(rng.integers(3, min(height, 9) + 1))
        bw = int(rng.integers(3, min(width, 9) + 1))
        local = np.zeros((bh, bw), dtype=bool)
        local[0, :] = local[-1, :] = True
        local[:, 0] = local[:, -1] = True
        if bh > 3 and bw > 3 and rng.random() < 0.5:
            local[int(rng.integers(1, bh - 1)), int(rng.integers(1, bw - 1))] = True
    elif style == 6:
        bh = int(rng.integers(2, min(height, 9) + 1))
        bw = int(rng.integers(2, min(width, 9) + 1))
        local = np.zeros((bh, bw), dtype=bool)
        for _ in range(int(rng.integers(2, 6))):
            r = int(rng.integers(0, bh))
            c = int(rng.integers(0, bw))
            local[r:min(r + int(rng.integers(1, 3)), bh), c:min(c + int(rng.integers(1, 3)), bw)] = True
    elif style == 7:
        bh = int(rng.integers(2, min(height, 9) + 1))
        bw = int(rng.integers(2, min(width, 9) + 1))
        local = np.zeros((bh, bw), dtype=bool)
        start = (int(rng.integers(0, bh)), int(rng.integers(0, bw)))
        local[start] = True
        cells = [start]
        target = int(rng.integers(2, bh * bw + 1))
        for _ in range(10 * bh * bw):
            if len(cells) >= target:
                break
            r, c = cells[int(rng.integers(0, len(cells)))]
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(0, 4))]
            nr, nc = r + dr, c + dc
            if 0 <= nr < bh and 0 <= nc < bw and not local[nr, nc]:
                local[nr, nc] = True
                cells.append((nr, nc))
    else:
        probability = float(rng.uniform(.08, .35) if style == 8 else rng.uniform(.45, .8))
        local = rng.random((height, width)) < probability
    return _place(rng, local, height, width)


def _make_pair(height, width, quadrant, palette, motif):
    background, source, cross = palette
    inp = np.full((2 * height + 1, 2 * width + 1), background, dtype=np.int8)
    inp[height, :] = cross
    inp[:, width] = cross
    r0 = 0 if quadrant < 2 else height + 1
    c0 = 0 if quadrant % 2 == 0 else width + 1
    inp[r0:r0 + height, c0:c0 + width][motif] = source

    out = np.full((2 * height, 2 * width), background, dtype=np.int8)
    rr, cc = np.nonzero(motif)
    rr = rr + (height if quadrant >= 2 else 0)
    cc = cc + (width if quadrant % 2 else 0)
    out[rr, cc] = cross
    out[2 * height - 1 - rr, cc] = cross
    out[rr, 2 * width - 1 - cc] = cross
    out[2 * height - 1 - rr, 2 * width - 1 - cc] = cross
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = _parameters(rng)
    puzzle = {"train": [], "test": []}
    train_dimensions = []
    train_palettes = []
    seen = set()

    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(400):
                if section == "train" and index == 0:
                    mode = parameters["size_mode"]
                    low, high = (3, 7) if mode == 1 else ((8, 14) if mode == 2 else (3, 14))
                    side = int(rng.integers(low, high + 1))
                    height, width = side, side
                else:
                    height, width = _dimensions(rng, parameters, section, index, train_dimensions)
                quadrant = _quadrant(rng, parameters, section, index)
                palette = _palette(rng, parameters, section, index, train_palettes)
                motif = _sample_mask(rng, height, width, parameters["complexity"], informative=(section == "train" and index == 0))
                pair = _make_pair(height, width, quadrant, palette, motif)
                key = _grid_key(pair["input"])
                if key not in seen and key not in _SOURCE_INPUT_KEYS:
                    break
            else:
                raise RuntimeError("unable to sample a distinct input")
            seen.add(key)
            puzzle[section].append(pair)
            if section == "train":
                train_dimensions.append((height, width))
                train_palettes.append(palette)

    # The asymmetric teaching example need not occupy a fixed list position.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    rows, cols = inp.shape
    assert rows >= 5 and cols >= 5 and rows % 2 == 1 and cols % 2 == 1
    h, w = rows // 2, cols // 2
    assert out.shape == (2 * h, 2 * w)
    cross = int(inp[h, w])
    assert np.all(inp[h, :] == cross) and np.all(inp[:, w] == cross)
    quadrants = (
        inp[:h, :w], inp[:h, w + 1:],
        inp[h + 1:, :w], inp[h + 1:, w + 1:],
    )
    values, counts = np.unique(np.concatenate([q.ravel() for q in quadrants]), return_counts=True)
    assert len(values) == 2
    background = int(values[int(np.argmax(counts))])
    source = int(values[int(np.argmin(counts))])
    assert len({background, source, cross}) == 3
    occupied = [i for i, q in enumerate(quadrants) if np.any(q == source)]
    assert len(occupied) == 1
    quadrant = occupied[0]
    assert all(np.all(q == background) for i, q in enumerate(quadrants) if i != quadrant)
    motif = quadrants[quadrant] == source
    assert np.all(np.isin(quadrants[quadrant], (background, source)))
    assert int(motif.sum()) >= 1
    expected = _make_pair(h, w, quadrant, (background, source, cross), motif)["output"]
    assert np.array_equal(out, expected)
    expected_colors = {cross} if np.all(motif) else {background, cross}
    assert set(int(v) for v in np.unique(out)) == expected_colors
    rr, cc = np.nonzero(motif)
    margins = (int(rr.min()), int(h - 1 - rr.max()), int(cc.min()), int(w - 1 - cc.max()))
    return {"height": h, "width": w, "quadrant": quadrant,
            "palette": (background, source, cross), "mask": motif,
            "count": int(motif.sum()), "margins": margins,
            "informative": _informative(motif)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in _SOURCE_INPUT_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info[section].append(_analyze_pair(pair))
    assert any(item["informative"] for item in info["train"])
    train_quadrants = {item["quadrant"] for item in info["train"]}
    test_quadrants = {item["quadrant"] for item in info["test"]}
    if not test_quadrants <= train_quadrants:
        assert len(train_quadrants) >= 2


def _component_count(mask):
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        todo = [remaining.pop()]
        while todo:
            r, c = todo.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    todo.append(neighbor)
    return count


def self_test() -> None:
    first = generate_puzzle(123456)
    second = generate_puzzle(123456)
    assert all(_grid_key(a[k]) == _grid_key(b[k]) for s in ("train", "test")
               for a, b in zip(first[s], second[s]) for k in ("input", "output"))
    assert _grid_key(first["train"][0]["input"]) != _grid_key(generate_puzzle(123457)["train"][0]["input"])
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(20000 + 100 * train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(30000 + i, train, test)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(40000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    train_counts, test_counts = set(), set()
    quadrants, sizes, palettes, layouts, styles = set(), set(), set(), set(), set()
    boxes, margins_seen = set(), set()
    size_styles = {}
    size_relations, quadrant_relations = set(), set()
    inset = rectangles = singletons = full = almost_full = connected = disconnected = 0
    lines = hollows = bents = compact_clusters = broad_scatter = 0
    symmetric = divider_touching = 0
    changing_backgrounds = repeated_palettes = role_swaps = fresh_test_palettes = 0
    combined = min_dimension = max_dimension = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(p) for p in puzzle["train"]]
        test_info = [_analyze_pair(p) for p in puzzle["test"]]
        all_info = train_info + test_info
        all_palettes = [i["palette"] for i in all_info]
        changing_backgrounds += len({p[0] for p in all_palettes}) > 1
        repeated_palettes += len(set(all_palettes)) < len(all_palettes)
        role_swaps += any(
            a[0] == b[0] and a[1:] == b[:0:-1]
            for j, a in enumerate(all_palettes) for b in all_palettes[j + 1:]
        )
        fresh_test_palettes += any(
            palette not in all_palettes[:len(train_info)]
            and not any(palette[0] == old[0] and palette[1:] == old[:0:-1]
                        for old in all_palettes[:len(train_info)])
            for palette in all_palettes[len(train_info):]
        )
        train_areas = [i["height"] * i["width"] for i in train_info]
        test_areas = [i["height"] * i["width"] for i in test_info]
        if min(test_areas) > max(train_areas):
            size_relations.add("larger")
        elif max(test_areas) < min(train_areas):
            size_relations.add("smaller")
        elif set((i["height"], i["width"]) for i in test_info) & set((i["height"], i["width"]) for i in train_info):
            size_relations.add("shared")
        else:
            size_relations.add("overlap")
        tq = {i["quadrant"] for i in train_info}
        sq = {i["quadrant"] for i in test_info}
        quadrant_relations.add("held_out" if not sq <= tq else ("mixed" if len(tq | sq) > 1 else "constant"))
        combined += (any(i["height"] != i["width"] for i in all_info)
                     and len(tq | sq) > 1 and len(set(all_palettes)) > 1)
        copy_distinguished = turn_distinguished = False
        for pair, item in zip(puzzle["train"], train_info):
            h, w = item["height"], item["width"]
            if h != w:
                continue
            inp = pair["input"]
            cropped = np.delete(np.delete(inp, h, axis=0), w, axis=1)
            source_mask = cropped != item["palette"][0]
            q = item["quadrant"]
            tile = (source_mask[:h, :w], source_mask[:h, w:],
                    source_mask[h:, :w], source_mask[h:, w:])[q]
            copied = np.tile(tile, (2, 2))
            turned = source_mask.copy()
            for quarter_turn in (1, 2, 3):
                turned |= np.rot90(source_mask, quarter_turn)
            actual = pair["output"] == item["palette"][2]
            copy_distinguished |= not np.array_equal(copied, actual)
            turn_distinguished |= not np.array_equal(turned, actual)
        assert copy_distinguished and turn_distinguished
        for pair, item in zip(puzzle["train"] + puzzle["test"], all_info):
            h, w, mask = item["height"], item["width"], item["mask"]
            inp = pair["input"]
            cropped = np.delete(np.delete(inp, h, axis=0), w, axis=1)
            background, _, cross = item["palette"]
            supplied = cropped != background
            predicted_mask = supplied | supplied[::-1, :] | supplied[:, ::-1] | supplied[::-1, ::-1]
            assert np.array_equal(pair["output"], np.where(predicted_mask, cross, background))
            quadrants.add(item["quadrant"])
            sizes.add((h, w))
            palettes.add(item["palette"])
            layouts.add((h, w, tuple(bool(v) for v in mask.flat)))
            count = item["count"]
            styles.add(count)
            size_styles.setdefault((h, w), set()).add((count, _component_count(mask)))
            margins_seen.add(item["margins"])
            inset += all(m > 0 for m in item["margins"])
            rectangles += h != w
            singletons += count == 1
            full += count == h * w
            almost_full += count == h * w - 1
            connected += _component_count(mask) == 1
            disconnected += _component_count(mask) > 1
            symmetric += np.array_equal(mask, mask[::-1, :]) or np.array_equal(mask, mask[:, ::-1])
            inner_row = 1 if item["quadrant"] < 2 else 0
            inner_col = 3 if item["quadrant"] % 2 == 0 else 2
            divider_touching += item["margins"][inner_row] == 0 or item["margins"][inner_col] == 0
            rr, cc = np.nonzero(mask)
            box = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            boxes.add(tuple(box.shape))
            lines += box.shape[0] == 1 or box.shape[1] == 1
            compact_clusters += (_component_count(mask) > 1 and box.shape[0] < h
                                 and box.shape[1] < w)
            broad_scatter += (_component_count(mask) >= 5 and box.shape[0] >= max(2, h - 1)
                              and box.shape[1] >= max(2, w - 1))
            if min(box.shape) >= 2:
                for edge_row in (0, -1):
                    for edge_col in (0, -1):
                        bent = np.zeros_like(box)
                        bent[edge_row, :] = True
                        bent[:, edge_col] = True
                        if np.array_equal(box, bent):
                            bents += 1
                            break
                    else:
                        continue
                    break
            if min(box.shape) >= 3:
                perimeter = np.zeros_like(box)
                perimeter[0, :] = perimeter[-1, :] = True
                perimeter[:, 0] = perimeter[:, -1] = True
                hollows += np.array_equal(box, perimeter)
            min_dimension += min(2 * h + 1, 2 * w + 1) == 5
            max_dimension += max(2 * h + 1, 2 * w + 1) == 29
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert quadrants == {0, 1, 2, 3} and len(sizes) >= 60
    assert len(palettes) >= 40 and len(layouts) >= 200 and len(styles) >= 25
    assert len(boxes) >= 40 and len(margins_seen) >= 100
    assert sum(len(features) >= 3 for features in size_styles.values()) >= 10
    assert {"larger", "smaller", "shared"} <= size_relations
    assert {"held_out", "mixed", "constant"} <= quadrant_relations
    assert min(inset, rectangles, singletons, full, almost_full, connected, disconnected) > 0
    assert min(lines, hollows, bents, compact_clusters, broad_scatter,
               symmetric, divider_touching, min_dimension, max_dimension) > 0
    assert min(changing_backgrounds, repeated_palettes, role_swaps, fresh_test_palettes, combined) > 0
    for seed in (267, 470, 1114):
        validate_generated_puzzle(generate_puzzle(seed, 2, 1))
