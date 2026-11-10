"""Dense-panel ARC puzzle generator: crop, then complete marker rows and columns."""

import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = {
    "3498a437556c4c5b0dc8c7cb642e761894523ea6ec34a75a8e3636b5b972f8ae",
    "707acb9891beb2034932a0198172066cc633265b743a166557e8de357b24bae7",
    "00810e4e14692fa57508d639cff1b8105bd0b413d6b7f246890b707dcf27248c",
    "d44ccbf0324af244f20943ddf93d30cde5be07f53c85bafa88807d4013ad1320",
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _source_pair_hash(pair):
    data = (bytes(pair["input"].shape)
            + np.ascontiguousarray(pair["input"], dtype=np.int8).tobytes()
            + bytes(pair["output"].shape)
            + np.ascontiguousarray(pair["output"], dtype=np.int8).tobytes())
    return hashlib.sha256(data).hexdigest()


def _panel_candidates(grid):
    """Find complete, two-color, sparse-marker rectangles without a noise moat."""
    height, width = grid.shape
    indicators = (np.arange(10, dtype=np.int16)[:, None, None]
                  == grid[None, :, :]).astype(np.int16)
    prefix = np.zeros((10, height + 1, width + 1), dtype=np.int16)
    prefix[:, 1:, 1:] = indicators.cumsum(1).cumsum(2)
    sizes = sorted(((h * w, h, w)
                    for h in range(4, min(18, height) + 1)
                    for w in range(4, min(18, width) + 1)
                    if h * w >= 32), reverse=True)
    best_area = 0
    best = []
    for area, h, w in sizes:
        if area < best_area:
            break
        max_markers = min(8, int(area * 0.13), (h - 2) * (w - 2))
        counts = (prefix[:, h:, w:] - prefix[:, :-h, w:]
                  - prefix[:, h:, :-w] + prefix[:, :-h, :-w])
        possible = (counts >= area - max_markers) & (counts <= area - 1)
        for fill, top, left in np.argwhere(possible):
            fill, top, left = int(fill), int(top), int(left)
            block = grid[top:top + h, left:left + w]
            if (not np.all(block[0] == fill)
                    or not np.all(block[-1] == fill)
                    or not np.all(block[:, 0] == fill)
                    or not np.all(block[:, -1] == fill)):
                continue
            other = np.unique(block[block != fill])
            if other.size != 1:
                continue
            markers = np.argwhere(block == other[0])
            if not 1 <= len(markers) <= max_markers:
                continue
            candidate = (top, left, h, w, fill, int(other[0]))
            if area > best_area:
                best_area, best = area, [candidate]
            elif candidate not in best:
                best.append(candidate)
    return best


def _analyze_pair(input_grid, output_grid):
    candidates = _panel_candidates(input_grid)
    _require(len(candidates) == 1, "one exact dense panel must be identifiable")
    top, left, h, w, fill, marker = candidates[0]
    block = input_grid[top:top + h, left:left + w]
    positions = np.argwhere(block == marker)
    rows = np.unique(positions[:, 0])
    cols = np.unique(positions[:, 1])
    _require(output_grid.shape == (h, w), "output must have the exact crop size")
    expected = np.full((h, w), fill, dtype=np.int8)
    expected[rows, :] = marker
    expected[:, cols] = marker
    _require(np.array_equal(output_grid, expected), "wrong row/column union")
    _require(int(np.count_nonzero(output_grid == marker))
             == len(rows) * w + len(cols) * h - len(rows) * len(cols),
             "wrong count of activated cells")
    return {"top": top, "left": left, "height": h, "width": w,
            "fill": fill, "marker": marker, "marker_count": len(positions),
            "rows": tuple(int(x) for x in rows),
            "cols": tuple(int(x) for x in cols),
            "positions": tuple(map(tuple, positions.tolist()))}


def _count_arg(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _select_counts(seed, rng, num_train, num_test):
    num_train = _count_arg(num_train, "num_train")
    num_test = _count_arg(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)]
        return legal[seed % len(legal)]
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _sample_dimensions(rng, regime):
    mode = rng.choice(3, p=(0.57, 0.27, 0.16) if regime == "source_train"
                      else (0.37, 0.40, 0.23))
    if mode == 0:
        h, w = int(rng.integers(6, 13)), int(rng.integers(6, 13))
    elif mode == 1:
        short, long = int(rng.integers(4, 8)), int(rng.integers(11, 19))
        h, w = (short, long) if rng.random() < 0.5 else (long, short)
    else:
        h, w = int(rng.integers(12, 19)), int(rng.integers(12, 19))
    return h, w


def _marker_cap(h, w):
    return min(8, int(h * w * 0.11), (h - 2) * (w - 2))


def _sample_specs(rng, train_count, test_count):
    """All semantic roles are local; only the crop-and-union rule is shared."""
    specs = []
    for index in range(train_count + test_count):
        regime = "source_train" if index < train_count else "target_test"
        h, w = _sample_dimensions(rng, regime)
        if index >= train_count and rng.random() < 0.30:
            h, w = specs[int(rng.integers(0, train_count))]["dims"]
        if index == 1:
            while (h, w) == specs[0]["dims"]:
                h, w = _sample_dimensions(rng, regime)
        cap = _marker_cap(h, w)
        if index == 0:
            count = 1
        elif index == 1:
            count = min(cap, int(rng.integers(2, 4)))
        else:
            count = int(rng.integers(1, cap + 1))
        if index >= train_count and rng.random() < 0.3:
            count = min(cap, max(s["count"] for s in specs[:train_count]) + 1)
        if index > 0 and rng.random() < 0.25:
            old = specs[int(rng.integers(0, min(index, train_count)))]
            fill, marker = old["marker"], old["fill"]
        elif index > 0 and rng.random() < 0.17:
            old = specs[int(rng.integers(0, min(index, train_count)))]
            fill, marker = old["fill"], old["marker"]
        else:
            colors = rng.choice(10, size=2, replace=False)
            fill, marker = int(colors[0]), int(colors[1])
        specs.append({"dims": (h, w), "count": count,
                      "fill": fill, "marker": marker})
    return specs[:train_count], specs[train_count:]


def _marker_positions(rng, h, w, count):
    interior_rows = np.arange(1, h - 1)
    interior_cols = np.arange(1, w - 1)
    mode = "single" if count == 1 else str(rng.choice(
        ["distinct", "row", "col", "both", "scatter"],
        p=[0.30, 0.20, 0.20, 0.12, 0.18]))
    if mode == "distinct" and count > min(len(interior_rows), len(interior_cols)):
        mode = "scatter"
    if mode == "both" and count < 3:
        mode = "scatter"
    if mode == "distinct":
        rows = rng.choice(interior_rows, size=count, replace=False)
        cols = rng.choice(interior_cols, size=count, replace=False)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    if mode == "row":
        r = int(rng.choice(interior_rows))
        cols = rng.choice(interior_cols, size=min(count, len(interior_cols)), replace=False)
        positions = [(r, int(c)) for c in cols]
    elif mode == "col":
        c = int(rng.choice(interior_cols))
        rows = rng.choice(interior_rows, size=min(count, len(interior_rows)), replace=False)
        positions = [(int(r), c) for r in rows]
    elif mode == "both":
        rows = rng.choice(interior_rows, size=2, replace=False)
        cols = rng.choice(interior_cols, size=2, replace=False)
        positions = [(int(rows[0]), int(cols[0])),
                     (int(rows[0]), int(cols[1])),
                     (int(rows[1]), int(cols[1]))]
    else:
        positions = []
    available = [(r, c) for r in interior_rows for c in interior_cols
                 if (r, c) not in positions]
    if len(positions) < count:
        selected = rng.choice(len(available), size=count - len(positions), replace=False)
        positions.extend(available[int(i)] for i in selected)
    return positions


def _make_noise(rng, shape, fill, marker):
    palette_size = int(rng.integers(4, 11))
    palette = [int(x) for x in rng.choice(10, size=palette_size, replace=False)]
    if rng.random() < 0.7:
        palette[0] = fill
    if rng.random() < 0.7 and marker not in palette:
        palette[-1] = marker
    palette = np.array(sorted(set(palette)), dtype=np.int8)
    mode = int(rng.integers(0, 4))
    if mode == 0:
        noise = rng.choice(palette, size=shape)
    else:
        weights = rng.dirichlet(np.full(len(palette), 2.0))
        noise = rng.choice(palette, size=shape, p=weights)
        if mode == 2:
            for _ in range(int(rng.integers(2, 6))):
                row = int(rng.integers(0, shape[0]))
                start = int(rng.integers(0, shape[1]))
                stop = min(shape[1], start + int(rng.integers(2, 7)))
                noise[row, start:stop] = rng.choice(palette)
        elif mode == 3:
            for _ in range(int(rng.integers(2, 5))):
                row = int(rng.integers(0, shape[0]))
                col = int(rng.integers(0, shape[1]))
                noise[row:min(shape[0], row + 3), col:min(shape[1], col + 4)] = rng.choice(palette)
    return noise.astype(np.int8)


def _add_distraction(rng, canvas, top, left, h, w):
    if rng.random() >= 0.37:
        return
    dh, dw = int(rng.integers(3, 6)), int(rng.integers(3, 7))
    if dh > canvas.shape[0] or dw > canvas.shape[1]:
        return
    places = [(r, c) for r in range(canvas.shape[0] - dh + 1)
              for c in range(canvas.shape[1] - dw + 1)
              if r + dh <= top or r >= top + h or c + dw <= left or c >= left + w]
    if not places:
        return
    r, c = places[int(rng.integers(0, len(places)))]
    main = int(rng.integers(0, 10))
    mask = rng.random((dh, dw)) < 0.62
    canvas[r:r + dh, c:c + dw][mask] = main
    # Two forced holes make this an incomplete/multicolor distraction.
    canvas[r, c] = (main + 1) % 10
    canvas[r + dh - 1, c + dw - 1] = (main + 2) % 10


def _generate_pair(rng, regime, spec):
    h, w = spec["dims"]
    fill, marker = spec["fill"], spec["marker"]
    for _ in range(60):
        extra_h = int(rng.integers(2, min(12, 30 - h) + 1))
        extra_w = int(rng.integers(2, min(12, 30 - w) + 1))
        shape = (h + extra_h, w + extra_w)
        top = (int(rng.choice([0, extra_h])) if rng.random() < 0.13
               else int(rng.integers(1, extra_h)))
        left = (int(rng.choice([0, extra_w])) if rng.random() < 0.13
                else int(rng.integers(1, extra_w)))
        canvas = _make_noise(rng, shape, fill, marker)
        _add_distraction(rng, canvas, top, left, h, w)
        canvas[top:top + h, left:left + w] = fill
        positions = _marker_positions(rng, h, w, spec["count"])
        rows, cols = set(), set()
        for row, col in positions:
            canvas[top + row, left + col] = marker
            rows.add(row)
            cols.add(col)
        output = np.full((h, w), fill, dtype=np.int8)
        output[list(rows), :] = marker
        output[:, list(cols)] = marker
        candidates = _panel_candidates(canvas)
        if candidates == [(top, left, h, w, fill, marker)]:
            return {"input": canvas, "output": output}
    raise ValueError("could not build a uniquely identifiable panel")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys must be train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys must be input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(all(1 <= dim <= 30 for dim in grid.shape), "invalid grid shape")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(_source_pair_hash(pair) not in _SOURCE_PAIR_HASHES,
                     "exact source pair copied")
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    _require(analyses["train"][0]["marker_count"] == 1,
             "first training pair must show one complete cross")
    _require(analyses["train"][1]["marker_count"] >= 2,
             "second training pair must show union of multiple crosses")
    _require(len({(a["height"], a["width"]) for a in analyses["train"]}) >= 2,
             "training panels must vary in size")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _select_counts(seed, rng, num_train, num_test)
    train_specs, test_specs = _sample_specs(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, specs in (("train", "source_train", train_specs),
                                 ("test", "target_test", test_specs)):
        for spec in specs:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(12):
                pair = _generate_pair(pair_rng, regime, spec)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not build unique inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    trains, tests, shapes, roles, locations, layouts = set(), set(), set(), set(), set(), set()
    shared_row = shared_col = both = edge = narrow = large = novel_role = 0
    palette_sizes, near_fill, near_marker = set(), 0, 0
    edge_shared_ring = large_many_shared = 0
    noise_signatures = set()
    within_shapes = within_counts = within_roles = within_noise = 0
    familiar_test_size = new_test_size = new_test_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        trains.add(len(puzzle["train"]))
        tests.add(len(puzzle["test"]))
        train_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        train_fills = {x["fill"] for x in train_info}
        train_markers = {x["marker"] for x in train_info}
        train_shapes = {(x["height"], x["width"]) for x in train_info}
        train_counts = {x["marker_count"] for x in train_info}
        familiar_test_size += any((x["height"], x["width"]) in train_shapes for x in test_info)
        new_test_size += any((x["height"], x["width"]) not in train_shapes for x in test_info)
        new_test_count += any(x["marker_count"] not in train_counts for x in test_info)
        puzzle_shapes, puzzle_counts, puzzle_roles, puzzle_noise = set(), set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair["input"], pair["output"])
                h, w = info["height"], info["width"]
                shapes.add((h, w))
                roles.add((info["fill"], info["marker"]))
                puzzle_shapes.add((h, w))
                puzzle_counts.add(info["marker_count"])
                puzzle_roles.add((info["fill"], info["marker"]))
                locations.add((pair["input"].shape, info["top"], info["left"]))
                layouts.add((h, w, info["positions"]))
                shared_row += info["marker_count"] > len(info["rows"])
                shared_col += info["marker_count"] > len(info["cols"])
                both += (info["marker_count"] > len(info["rows"])
                         and info["marker_count"] > len(info["cols"]))
                edge += (info["top"] == 0 or info["left"] == 0
                         or info["top"] + h == pair["input"].shape[0]
                         or info["left"] + w == pair["input"].shape[1])
                narrow += max(h, w) / min(h, w) >= 2
                large += h * w >= 180
                palette_sizes.add(len(np.unique(pair["input"])))
                outside = np.ones(pair["input"].shape, dtype=bool)
                outside[info["top"]:info["top"] + h,
                        info["left"]:info["left"] + w] = False
                puzzle_noise.add(len(np.unique(pair["input"][outside])))
                noise_signatures.add(tuple(np.bincount(pair["input"][outside], minlength=10)))
                ring = np.zeros(pair["input"].shape, dtype=bool)
                r0, c0 = max(0, info["top"] - 1), max(0, info["left"] - 1)
                ring[r0:min(pair["input"].shape[0], info["top"] + h + 1),
                     c0:min(pair["input"].shape[1], info["left"] + w + 1)] = True
                ring &= outside
                near_fill += bool(np.any(pair["input"][ring] == info["fill"]))
                near_marker += bool(np.any(pair["input"][ring] == info["marker"]))
                edge_shared_ring += bool(
                    (info["top"] == 0 or info["left"] == 0
                     or info["top"] + h == pair["input"].shape[0]
                     or info["left"] + w == pair["input"].shape[1])
                    and info["marker_count"] > len(info["rows"])
                    and np.any(pair["input"][ring] == info["fill"]))
                large_many_shared += bool(
                    h * w >= 180 and info["marker_count"] >= 4
                    and (info["marker_count"] > len(info["rows"])
                         or info["marker_count"] > len(info["cols"])))
                if split == "test":
                    novel_role += (info["fill"] not in train_fills
                                   or info["marker"] not in train_markers)
        within_shapes += len(puzzle_shapes) >= 2
        within_counts += len(puzzle_counts) >= 2
        within_roles += len(puzzle_roles) >= 2
        within_noise += len(puzzle_noise) >= 2
    _require(len(trains) >= 4 and len(tests) >= 3, "automatic counts lack diversity")
    _require(len(shapes) >= 35 and len(layouts) >= 90 and len(locations) >= 90,
             "geometric diversity is too narrow")
    _require(len(roles) >= 35 and len(noise_signatures) >= 90,
             "color or noise diversity is too narrow")
    _require(shared_row >= 8 and shared_col >= 8 and both >= 2,
             "shared marker-line layouts are missing")
    _require(edge >= 10 and narrow >= 8 and large >= 8,
             "edge, narrow, or large panels are missing")
    _require(near_fill >= 20 and near_marker >= 20 and len(palette_sizes) >= 4,
             "incidental panel colors or palette variation are missing")
    _require(novel_role >= 8, "test colors do not generalize local roles")
    _require(min(within_shapes, within_counts, within_roles, within_noise) >= 40,
             "within-puzzle variation is missing")
    _require(familiar_test_size >= 8 and new_test_size >= 8 and new_test_count >= 8,
             "test regimes lack familiar and extended configurations")
    _require(edge_shared_ring >= 4 and large_many_shared >= 4,
             "compatible review axes do not combine")
    a, b = generate_puzzle(987654, 6, 3), generate_puzzle(987654, 6, 3)
    _require(_puzzles_equal(a, b), "seeded puzzle is not deterministic")
    _require(not _puzzles_equal(a, generate_puzzle(987655, 6, 3)),
             "different seeds generated the same puzzle")
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, ntrain, ntest)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest),
                 "explicit counts not honored")
    for index, (ntrain, ntest) in enumerate(((None, 1), (None, 4),
                                             (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, ntrain, ntest)
        _require((ntrain is None or len(puzzle["train"]) == ntrain)
                 and (ntest is None or len(puzzle["test"]) == ntest),
                 "partial counts not honored")
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
