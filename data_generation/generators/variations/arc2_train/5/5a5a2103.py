"""Solved lattice-stencil ARC puzzles."""

import numpy as np


def _validate_optional_count(name, value, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(rng, num_train, num_test):
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)
    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _lattice(rows, cols, interior, background, separator):
    step = interior + 1
    grid = np.full((rows * step - 1, cols * step - 1), background, dtype=np.int8)
    grid[interior::step, :] = separator
    grid[:, interior::step] = separator
    return grid


def _paint(grid, row, col, interior, coords, color):
    step = interior + 1
    for local_row, local_col in coords:
        grid[row * step + local_row, col * step + local_col] = color


def _key_shape(rng, interior, avoid=None):
    k = interior
    shapes = [
        ((k - 2) // 2, (k - 2) // 2),
        (0, 0),
        (k - 2, k - 2),
    ]
    choices = [
        tuple((r + dr, c + dc) for dr in (0, 1) for dc in (0, 1))
        for r, c in shapes
    ]
    for r in range(k):
        for c in range(k):
            choices.append(((r, c),))
            if c + 1 < k:
                choices.append(((r, c), (r, c + 1)))
            if r + 1 < k:
                choices.append(((r, c), (r + 1, c)))
            if r + 1 < k and c + 1 < k:
                choices.append(((r, c), (r + 1, c), (r + 1, c + 1)))
    choices = list(dict.fromkeys(tuple(sorted(x)) for x in choices))
    if avoid is not None:
        choices = [x for x in choices if x != avoid]
    return choices[int(rng.integers(len(choices)))]


def _motif(rng, interior, prior=(), force_new=False):
    k = interior
    mode = int(rng.integers(5))
    if prior and not force_new and rng.random() < 0.16:
        reusable = [x for x in prior if all(r < k and c < k for r, c in x)]
        if reusable:
            return reusable[int(rng.integers(len(reusable)))]
    cells = [(r, c) for r in range(k) for c in range(k)]
    if mode == 0:  # connected stroke, with turns and optional branches
        count = int(rng.integers(3, min(k * k, 2 * k + 3) + 1))
        occupied = {cells[int(rng.integers(len(cells)))]}
        while len(occupied) < count:
            frontier = {
                (r + dr, c + dc)
                for r, c in occupied
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if 0 <= r + dr < k and 0 <= c + dc < k
            } - occupied
            ordered = sorted(frontier)
            occupied.add(ordered[int(rng.integers(len(ordered)))])
    elif mode == 1:  # compact block, sometimes with a cutout
        h = int(rng.integers(2, k + 1))
        w = int(rng.integers(2, k + 1))
        r0 = int(rng.integers(k - h + 1))
        c0 = int(rng.integers(k - w + 1))
        occupied = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
        if len(occupied) > 4 and rng.random() < 0.45:
            ordered = sorted(occupied)
            occupied.remove(ordered[int(rng.integers(len(ordered)))])
    elif mode == 2:  # visible bilateral symmetry
        occupied = set()
        for r in range(k):
            for c in range((k + 1) // 2):
                if rng.random() < 0.5:
                    occupied.add((r, c))
                    occupied.add((r, k - 1 - c))
    elif mode == 3:  # sparse, possibly disconnected
        count = int(rng.integers(1, min(k * k, k + 4) + 1))
        flat = rng.choice(k * k, size=count, replace=False)
        occupied = {(int(x) // k, int(x) % k) for x in flat}
    else:  # unrestricted local stencil
        count = int(rng.integers(3, min(k * k, 2 * k + 4) + 1))
        flat = rng.choice(k * k, size=count, replace=False)
        occupied = {(int(x) // k, int(x) % k) for x in flat}
    if not occupied:
        occupied.add(cells[int(rng.integers(len(cells)))])
    motif = tuple(sorted(occupied))
    if force_new and motif in prior:
        return _motif(rng, interior, prior, True)
    return motif


def _payloads(rng, count, excluded):
    colors = [x for x in range(10) if x not in excluded]
    if rng.random() < 0.23:
        selected = [int(x) for x in rng.choice(colors, size=count, replace=True)]
        if len(set(selected)) < 2:
            selected[-1] = colors[(colors.index(selected[0]) + 1) % len(colors)]
    else:
        selected = [int(x) for x in rng.choice(colors, size=count, replace=False)]
    return tuple(selected)


def _dimensions(rng, regime, index, growth):
    if growth:
        if regime == "source_train" and index < 2:
            rows = cols = 3 + index
            interior = 4
            return rows, cols, interior
        if regime == "target_test":
            return 5, 5, 4
    layouts = ((3, 3), (4, 4), (5, 5), (3, 5), (5, 3),
               (3, 4), (4, 3), (4, 5), (5, 4))
    rows, cols = layouts[int(rng.integers(len(layouts)))]
    interior = int(rng.choice((3, 4, 4, 4, 5)))
    return rows, cols, interior


def _make_pair(rng, regime, index, params, prior):
    growth = params["growth"]
    rows, cols, interior = _dimensions(rng, regime, index, growth)
    if regime == "source_train" and index == 1:
        previous_shape = prior[0]["shape"]
        while (rows * (interior + 1) - 1, cols * (interior + 1) - 1) == previous_shape:
            rows, cols, interior = _dimensions(rng, regime, index, False)
    if params["palette_mode"] == "stable":
        background, separator, marker = params["palette"][:3]
    elif params["palette_mode"] == "background":
        background = params["palette"][0]
        available = [x for x in range(10) if x != background]
        picked = rng.choice(available, size=2, replace=False)
        separator, marker = (int(x) for x in picked)
    else:
        background, separator, marker = (int(x) for x in rng.choice(10, size=3, replace=False))
    axis = params["axis"]
    strip_count = rows if axis == "horizontal" else cols
    cross_count = cols if axis == "horizontal" else rows
    payloads = _payloads(rng, strip_count, {background, separator, marker})
    key_cross = int(rng.integers(cross_count))
    marker_cross_options = [x for x in range(cross_count) if x != key_cross]
    marker_cross = marker_cross_options[int(rng.integers(len(marker_cross_options)))]
    marker_strip = int(rng.integers(strip_count))
    avoid = prior[0]["key"] if regime == "source_train" and index == 1 else None
    key = _key_shape(rng, interior, avoid)
    motifs = [x["motif"] for x in prior]
    motif = _motif(rng, interior, motifs, regime == "source_train" and index == 1)
    inp = _lattice(rows, cols, interior, background, separator)
    out = _lattice(rows, cols, interior, background, separator)
    for strip, payload in enumerate(payloads):
        key_row, key_col = (strip, key_cross) if axis == "horizontal" else (key_cross, strip)
        _paint(inp, key_row, key_col, interior, key, payload)
        for cross in range(cross_count):
            row, col = (strip, cross) if axis == "horizontal" else (cross, strip)
            _paint(out, row, col, interior, motif, payload)
    marker_row, marker_col = ((marker_strip, marker_cross) if axis == "horizontal"
                              else (marker_cross, marker_strip))
    _paint(inp, marker_row, marker_col, interior, motif, marker)
    return {"input": inp, "output": out}, {
        "shape": inp.shape, "key": key, "motif": motif,
    }


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    params = {
        "axis": "vertical" if rng.random() < 0.27 else "horizontal",
        "growth": bool(rng.random() < 0.33),
        "palette_mode": ("stable", "background", "free")[int(rng.choice(3, p=(0.30, 0.30, 0.40)))],
        "palette": tuple(int(x) for x in rng.permutation(10)),
    }
    seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    prior = []
    for split, regime, count, offset in (
        ("train", "source_train", train_count, 0),
        ("test", "target_test", test_count, train_count),
    ):
        for index in range(count):
            pair_rng = np.random.default_rng(int(seeds[offset + index]))
            pair, info = _make_pair(pair_rng, regime, index, params, prior)
            puzzle[split].append(pair)
            prior.append(info)
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_signature(pair):
    return (_grid_signature(pair["input"]), _grid_signature(pair["output"]))


def _source_signatures():
    key = ((1, 1), (1, 2), (2, 1), (2, 2))
    source = (
        (4, 3, 6, (4, 2, 8, 1), 3, 1,
         ((1, 0), (1, 1), (1, 3), (2, 1), (2, 2), (3, 2))),
        (3, 8, 5, (2, 3, 1), 0, 1,
         ((0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (3, 1))),
        (5, 5, 1, (2, 3, 4, 6, 7), 1, 2,
         ((0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 2))),
    )
    result = set()
    for n, separator, marker, payloads, mr, mc, motif in source:
        inp = _lattice(n, n, 4, 0, separator)
        out = _lattice(n, n, 4, 0, separator)
        for row, payload in enumerate(payloads):
            _paint(inp, row, 0, 4, key, payload)
            for col in range(n):
                _paint(out, row, col, 4, motif, payload)
        _paint(inp, mr, mc, 4, motif, marker)
        result.add(_pair_signature({"input": inp, "output": out}))
    return result


_SOURCE_SIGNATURES = _source_signatures()


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    matches = []
    for k in (3, 4, 5):
        step = k + 1
        height, width = inp.shape
        if (height + 1) % step or (width + 1) % step:
            continue
        rows, cols = (height + 1) // step, (width + 1) // step
        if not (3 <= rows <= 5 and 3 <= cols <= 5):
            continue
        sep_rows = list(range(k, height, step))
        sep_cols = list(range(k, width, step))
        separator = int(inp[k, 0])
        if not all(np.all(grid[sep_rows, :] == separator) and
                   np.all(grid[:, sep_cols] == separator) for grid in (inp, out)):
            continue
        interior_mask = np.ones(inp.shape, dtype=bool)
        interior_mask[sep_rows, :] = False
        interior_mask[:, sep_cols] = False
        if np.any(inp[interior_mask] == separator) or np.any(out[interior_mask] == separator):
            continue
        counts = np.bincount(inp[interior_mask].astype(int), minlength=10)
        background = int(np.argmax(counts))
        if int(np.sum(counts == counts.max())) != 1 or background == separator:
            continue
        blocks = {}
        for row in range(rows):
            for col in range(cols):
                block = inp[row * step:row * step + k, col * step:col * step + k]
                coords = tuple(map(tuple, np.argwhere(block != background).tolist()))
                if coords:
                    colors = {int(block[r, c]) for r, c in coords}
                    if len(colors) != 1:
                        blocks = None
                        break
                    blocks[row, col] = (coords, colors.pop())
            if blocks is None:
                break
        if blocks is None:
            continue
        for axis in ("horizontal", "vertical"):
            strip_count = rows if axis == "horizontal" else cols
            cross_count = cols if axis == "horizontal" else rows
            for key_cross in range(cross_count):
                keys = [
                    blocks.get((strip, key_cross) if axis == "horizontal" else (key_cross, strip))
                    for strip in range(strip_count)
                ]
                if any(x is None for x in keys) or len({x[0] for x in keys}) != 1:
                    continue
                key_positions = {
                    (strip, key_cross) if axis == "horizontal" else (key_cross, strip)
                    for strip in range(strip_count)
                }
                extras = [(pos, val) for pos, val in blocks.items() if pos not in key_positions]
                if len(extras) != 1:
                    continue
                marker_position, (motif, marker) = extras[0]
                payloads = tuple(x[1] for x in keys)
                if len(set(payloads)) < 2 or marker in payloads:
                    continue
                if any(x in (background, separator) for x in (*payloads, marker)):
                    continue
                expected = _lattice(rows, cols, k, background, separator)
                for strip, payload in enumerate(payloads):
                    for cross in range(cross_count):
                        r, c = (strip, cross) if axis == "horizontal" else (cross, strip)
                        _paint(expected, r, c, k, motif, payload)
                if np.array_equal(out, expected):
                    matches.append({
                        "axis": axis, "rows": rows, "cols": cols, "interior": k,
                        "background": background, "separator": separator,
                        "marker": marker, "payloads": payloads,
                        "key_cross": key_cross, "key": keys[0][0],
                        "marker_position": marker_position, "motif": motif,
                        "shape": inp.shape,
                    })
    assert len(matches) == 1
    return matches[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, infos = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            signature = _pair_signature(pair)
            assert signature[0] not in inputs and signature not in pairs
            assert signature not in _SOURCE_SIGNATURES
            assert not np.array_equal(pair["input"], pair["output"])
            inputs.add(signature[0])
            pairs.add(signature)
            infos[split].append(_inspect_pair(pair))
    axes = {x["axis"] for x in infos["train"] + infos["test"]}
    assert len(axes) == 1
    assert infos["train"][0]["shape"] != infos["train"][1]["shape"]
    assert infos["train"][0]["motif"] != infos["train"][1]["motif"]
    assert infos["train"][0]["key"] != infos["train"][1]["key"]


def self_test() -> None:
    automatic = generate_puzzle(123456)
    assert _same_puzzle(automatic, generate_puzzle(123456))
    assert not _same_puzzle(automatic, generate_puzzle(123457))
    for seed, train, test in ((654321, 7, 3), (70001, 2, 1),
                              (70002, 10, 1), (70003, 10, 4),
                              (71001, None, 1), (71002, None, 4),
                              (71003, 2, None), (71004, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
        assert _same_puzzle(p, generate_puzzle(seed, train, test))
    for offset, args in enumerate(((1, None), (11, None), (None, 0),
                                   (None, 5), (2, 2), (3, 4), (4, 4),
                                   (True, 1), (3.0, 1))):
        try:
            generate_puzzle(72000 + offset, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, shapes, interiors = set(), set(), set(), set()
    orientations, key_shapes, motifs, layouts, palette_roles = set(), set(), set(), set(), set()
    palette_modes, connected, repeated, smaller, same, larger = set(), 0, 0, 0, 0, 0
    rectangular = non_four = later_marker = symmetric = repeated_motif_puzzles = 0
    role_reuse_puzzles = stable_palette_puzzles = varied_palette_puzzles = 0
    edge_keys = interior_keys = source_like_growth = combined = 0
    key_kinds = set()
    marker_positions = {"horizontal": {"train": set(), "test": set()},
                        "vertical": {"train": set(), "test": set()}}
    motif_sizes = {"train": set(), "test": set()}
    puzzle_signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzle_signatures.add(tuple(tuple(_pair_signature(x) for x in p[s]) for s in ("train", "test")))
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        info = {s: [_inspect_pair(x) for x in p[s]] for s in ("train", "test")}
        all_info = info["train"] + info["test"]
        orientations.add(all_info[0]["axis"])
        palette_roles.add(tuple((x["background"], x["separator"], x["marker"]) for x in all_info))
        role_count = len({(x["background"], x["separator"], x["marker"]) for x in all_info})
        palette_modes.add(role_count)
        stable_palette_puzzles += role_count == 1
        varied_palette_puzzles += role_count > 1
        role_reuse_puzzles += any(
            x["marker"] in y["payloads"] or x["separator"] in y["payloads"]
            for i, x in enumerate(all_info) for j, y in enumerate(all_info) if i != j)
        repeated_motif_puzzles += len({x["motif"] for x in all_info}) < len(all_info)
        source_like_growth += (
            all_info[0]["axis"] == "horizontal"
            and {(x["rows"], x["cols"], x["interior"]) for x in info["train"][:2]}
                == {(3, 3, 4), (4, 4, 4)}
            and all((x["rows"], x["cols"], x["interior"]) == (5, 5, 4)
                    for x in info["test"]))
        assert len({x["shape"] for x in info["train"]}) >= 2
        base_area = max(x["rows"] * x["cols"] for x in info["train"])
        for x in info["test"]:
            area = x["rows"] * x["cols"]
            smaller += area < base_area
            same += area == base_area
            larger += area > base_area
        for split in ("train", "test"):
            for x in info[split]:
                motif_sizes[split].add(len(x["motif"]))
                marker_cross = x["marker_position"][1 if x["axis"] == "horizontal" else 0]
                marker_positions[x["axis"]][split].add(marker_cross)
        for x in all_info:
            shapes.add(x["shape"])
            interiors.add(x["interior"])
            key_shapes.add(x["key"])
            motifs.add(x["motif"])
            layouts.add((x["axis"], x["rows"], x["cols"], x["interior"], x["key_cross"], x["marker_position"]))
            repeated += len(set(x["payloads"])) < len(x["payloads"])
            connected += _is_connected(x["motif"])
            rectangular += x["rows"] != x["cols"]
            non_four += x["interior"] != 4
            marker_cross = x["marker_position"][1 if x["axis"] == "horizontal" else 0]
            later_marker += marker_cross >= 2
            cross_count = x["cols"] if x["axis"] == "horizontal" else x["rows"]
            edge_keys += x["key_cross"] in (0, cross_count - 1)
            interior_keys += x["key_cross"] not in (0, cross_count - 1)
            k = x["interior"]
            motif = set(x["motif"])
            symmetric += (motif == {(r, k - 1 - c) for r, c in motif}
                          or motif == {(k - 1 - r, c) for r, c in motif})
            key_kinds.add(len(x["key"]))
            combined += (x["rows"] != x["cols"] and x["interior"] != 4
                         and marker_cross >= 2 and len(x["motif"]) in (6, 8))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40 and len(layouts) >= 100
    assert len(shapes) >= 15 and interiors == {3, 4, 5}
    assert orientations == {"horizontal", "vertical"}
    assert len(key_shapes) >= 20 and len(motifs) >= 100
    assert len(palette_roles) >= 40 and len(palette_modes) >= 2
    assert connected >= 30 and repeated >= 10
    assert smaller and same and larger
    assert rectangular >= 20 and non_four >= 20 and later_marker >= 20
    assert symmetric >= 15 and repeated_motif_puzzles >= 5
    assert role_reuse_puzzles >= 5 and stable_palette_puzzles >= 2
    assert varied_palette_puzzles >= 5 and source_like_growth >= 1
    assert edge_keys >= 20 and interior_keys >= 20
    assert {1, 2, 3, 4} <= key_kinds
    assert {6, 8} <= motif_sizes["train"] & motif_sizes["test"]
    assert combined >= 1
    assert all(marker_positions[axis][split] == {0, 1, 2, 3, 4}
               for axis in ("horizontal", "vertical") for split in ("train", "test"))


def _is_connected(coords):
    unseen = set(coords)
    stack = [unseen.pop()]
    while stack:
        r, c = stack.pop()
        neighbors = {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)} & unseen
        unseen -= neighbors
        stack.extend(neighbors)
    return not unseen


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[part], y[part]) for x, y in zip(a[s], b[s])
                   for part in ("input", "output")) for s in ("train", "test"))
