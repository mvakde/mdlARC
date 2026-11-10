"""Generate lattice puzzles completed by local four-way reflection."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "8f47dc4b3d3cad318a4341e0c4212fc24896d66d0286ea13092b7af7bad96192",
    "56e549254d3862c10075af46f91b52a733a207f6af353538cc2d62367a83a0fe",
    "0a97eeb8e6cca8bdaf110bf1930f5679afe917d0de74fc13d3012ec146897234",
}


def _count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count_value("num_train", num_train, 2, 10)
    num_test = _count_value("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _slices(lines, total):
    starts = [0] + [line + 1 for line in lines]
    ends = list(lines) + [total]
    return [slice(a, b) for a, b in zip(starts, ends)]


def _orient(template, quadrant):
    if quadrant == 0:
        return template
    if quadrant == 1:
        return np.fliplr(template)
    if quadrant == 2:
        return np.flipud(template)
    if quadrant == 3:
        return np.flip(template, axis=(0, 1))
    raise AssertionError("invalid quadrant")


def _quadrants(row_slices, col_slices, row, col):
    return ((row_slices[row], col_slices[col]),
            (row_slices[row], col_slices[col + 1]),
            (row_slices[row + 1], col_slices[col]),
            (row_slices[row + 1], col_slices[col + 1]))


def _footprint(row, col):
    return {(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)}


def _infer_lattice(grid, background=None):
    candidates = []
    for color in np.unique(grid):
        rows = np.flatnonzero(np.all(grid == color, axis=1))
        cols = np.flatnonzero(np.all(grid == color, axis=0))
        if len(rows) and len(cols):
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1
    structural, rows, cols = candidates[0]
    assert rows[0] > 0 and rows[-1] < grid.shape[0] - 1
    assert cols[0] > 0 and cols[-1] < grid.shape[1] - 1
    assert np.all(np.diff(rows) > 1) and np.all(np.diff(cols) > 1)
    line_mask = np.zeros(grid.shape, dtype=bool)
    line_mask[rows, :] = True
    line_mask[:, cols] = True
    assert np.array_equal(grid == structural, line_mask)
    if background is None:
        values, counts = np.unique(grid[~line_mask], return_counts=True)
        assert len(values) >= 2 and np.count_nonzero(counts == max(counts)) == 1
        background = int(values[int(np.argmax(counts))])
    assert background != structural
    return structural, background, tuple(int(x) for x in rows), tuple(int(x) for x in cols)


def _crossings(grid, background, rows, cols):
    row_slices = _slices(rows, grid.shape[0])
    col_slices = _slices(cols, grid.shape[1])
    incomplete, complete = {}, {}
    for row in range(len(rows)):
        for col in range(len(cols)):
            regions = _quadrants(row_slices, col_slices, row, col)
            cells = [grid[part] for part in regions]
            if len({cell.shape for cell in cells}) != 1:
                continue
            visible = [i for i, cell in enumerate(cells) if np.any(cell != background)]
            if len(visible) not in (3, 4):
                continue
            canonical = _orient(cells[visible[0]], visible[0])
            if all(np.array_equal(_orient(cells[i], i), canonical) for i in visible[1:]):
                if len(visible) == 3:
                    missing = next(i for i in range(4) if i not in visible)
                    incomplete[(row, col)] = (missing, canonical.copy())
                else:
                    complete[(row, col)] = canonical.copy()
    return incomplete, complete, row_slices, col_slices


def _analyze_pair_arrays(input_grid, output_grid, regime=None):
    assert input_grid.shape == output_grid.shape
    structural, background, rows, cols = _infer_lattice(input_grid)
    assert _infer_lattice(output_grid, background) == (structural, background, rows, cols)
    incomplete, unchanged, row_slices, col_slices = _crossings(input_grid, background, rows, cols)
    output_incomplete, output_complete, _, _ = _crossings(output_grid, background, rows, cols)
    assert incomplete and not output_incomplete
    assert set(output_complete) == set(incomplete) | set(unchanged)
    assert set(incomplete).isdisjoint(unchanged)
    allowed = np.zeros(input_grid.shape, dtype=bool)
    expected = input_grid.copy()
    occupied = set()
    missing_positions, templates = [], []
    for crossing in list(incomplete) + list(unchanged):
        row, col = crossing
        footprint = _footprint(row, col)
        assert occupied.isdisjoint(footprint)
        occupied.update(footprint)
        parts = _quadrants(row_slices, col_slices, row, col)
        for part in parts:
            allowed[part] = True
        template = incomplete[crossing][1] if crossing in incomplete else unchanged[crossing]
        assert np.array_equal(output_complete[crossing], template)
        assert np.any(template != background) and not np.any(template == structural)
        if crossing in incomplete:
            missing = incomplete[crossing][0]
            assert np.all(input_grid[parts[missing]] == background)
            expected[parts[missing]] = _orient(template, missing)
            missing_positions.append(missing)
        templates.append(template)
    assert np.array_equal(expected, output_grid)
    assert np.any(input_grid != output_grid)
    for grid in (input_grid, output_grid):
        accents = (grid != background) & (grid != structural)
        assert not np.any(accents & ~allowed)
    input_colors = set(int(c) for c in np.unique(input_grid))
    output_colors = set(int(c) for c in np.unique(output_grid))
    assert input_colors == output_colors
    row_lengths = tuple(part.stop - part.start for part in row_slices)
    col_lengths = tuple(part.stop - part.start for part in col_slices)
    return {"background": background, "structural": structural,
            "rows": rows, "cols": cols, "row_lengths": row_lengths,
            "col_lengths": col_lengths, "incomplete": tuple(incomplete),
            "complete": tuple(unchanged), "missing": tuple(missing_positions),
            "templates": tuple(templates),
            "accents": frozenset(output_colors - {background, structural})}


def _choose_crossings(rng, comp_rows, comp_cols, requested):
    candidates = [(r, c) for r in range(1, comp_rows) for c in range(1, comp_cols)]
    chosen, occupied = [], set()
    for index in rng.permutation(len(candidates)):
        crossing = candidates[int(index)]
        block = _footprint(crossing[0] - 1, crossing[1] - 1)
        if occupied.isdisjoint(block):
            chosen.append(crossing)
            occupied.update(block)
            if len(chosen) == requested:
                break
    return chosen


def _band_lengths(rng, count, crossing_indices):
    parent = list(range(count))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    involved = set()
    for index in crossing_indices:
        parent[root(index)] = root(index - 1)
        involved.update((index - 1, index))
    for _ in range(60):
        values = {}
        for i in range(count):
            group = root(i)
            if group not in values:
                lower = 2 if any(root(j) == group for j in involved) else 1
                values[group] = int(rng.integers(lower, 7))
        lengths = tuple(values[root(i)] for i in range(count))
        if sum(lengths) + count - 1 <= 30:
            return lengths
    return tuple(3 for _ in range(count))


def _shape_mask(rng, height, width, asymmetric=False, minimum=1):
    limit = max(2, (height * width) // 2)
    for _ in range(80):
        mode = int(rng.choice(7, p=[.08, .18, .23, .17, .19, .09, .06]))
        mask = np.zeros((height, width), dtype=bool)
        if mode == 6:
            rr = [height // 2] if height % 2 else [height // 2 - 1, height // 2]
            cc = [width // 2] if width % 2 else [width // 2 - 1, width // 2]
            for r in rr:
                for c in cc:
                    mask[r, c] = True
        elif mode == 5 and height >= 3 and width >= 3 and limit >= 8:
            mask[:3, :3] = True
            mask[1, 1] = False
        else:
            if mode == 0:
                count = int(rng.integers(1, min(2, limit) + 1))
            else:
                count = int(rng.integers(min(3, limit), limit + 1))
            if mode == 1:
                mask.flat[rng.choice(height * width, size=count, replace=False)] = True
            else:
                start = (int(rng.integers(height)), int(rng.integers(width)))
                cells = [start]
                mask[start] = True
                while len(cells) < count:
                    anchor = cells[-1] if mode == 2 and rng.random() < .72 else cells[int(rng.integers(len(cells)))]
                    neighbors = [(anchor[0] + dr, anchor[1] + dc)
                                 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                 if 0 <= anchor[0] + dr < height
                                 and 0 <= anchor[1] + dc < width
                                 and not mask[anchor[0] + dr, anchor[1] + dc]]
                    if not neighbors:
                        frontier = [(r, c) for r, c in cells
                                    if any(0 <= r + dr < height and 0 <= c + dc < width
                                           and not mask[r + dr, c + dc]
                                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
                        if not frontier:
                            break
                        anchor = frontier[int(rng.integers(len(frontier)))]
                        neighbors = [(anchor[0] + dr, anchor[1] + dc)
                                     for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                     if 0 <= anchor[0] + dr < height
                                     and 0 <= anchor[1] + dc < width
                                     and not mask[anchor[0] + dr, anchor[1] + dc]]
                    if mode in (3, 4):
                        contacts = [sum(0 <= r + dr < height and
                                        0 <= c + dc < width and mask[r + dr, c + dc]
                                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                                    for r, c in neighbors]
                        target = min(contacts) if mode == 3 else max(contacts)
                        neighbors = [cell for cell, score in zip(neighbors, contacts)
                                     if score == target]
                    cell = neighbors[int(rng.integers(len(neighbors)))]
                    mask[cell] = True
                    cells.append(cell)
            if mode >= 2:
                coords = np.argwhere(mask)
                top, left = coords.min(axis=0)
                bottom, right = coords.max(axis=0)
                patch = mask[top:bottom + 1, left:right + 1].copy()
                mask[:] = False
                y = height - patch.shape[0] if rng.random() < .48 else int(rng.integers(height - patch.shape[0] + 1))
                x = width - patch.shape[1] if rng.random() < .48 else int(rng.integers(width - patch.shape[1] + 1))
                mask[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
        if np.count_nonzero(mask) < minimum or np.count_nonzero(mask) > limit:
            continue
        if asymmetric and (np.array_equal(mask, np.fliplr(mask)) or
                           np.array_equal(mask, np.flipud(mask))):
            continue
        return mask
    raise RuntimeError("failed to make a motif")


def _template(rng, height, width, background, accents, multicolor, asymmetric=False):
    mask = _shape_mask(rng, height, width, asymmetric, 2 if multicolor else 1)
    cell_count = int(np.count_nonzero(mask))
    color_count = min(len(accents), cell_count, int(rng.integers(2, 4))) if multicolor else 1
    colors = [int(x) for x in rng.choice(accents, size=color_count, replace=False)]
    template = np.full((height, width), background, dtype=np.int8)
    values = [colors[i % color_count] for i in range(cell_count)]
    rng.shuffle(values)
    template[mask] = values
    return template


def _make_pair(rng, regime, multicolor=False, demonstrate=False,
               missing_pool=(0, 1, 2, 3)):
    for _ in range(100):
        if regime == "target_test" and rng.random() < .48:
            comp_rows, comp_cols = int(rng.integers(4, 6)), int(rng.integers(4, 6))
        else:
            comp_rows, comp_cols = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        selected = _choose_crossings(rng, comp_rows, comp_cols, int(rng.integers(1, 5)))
        if not selected:
            continue
        complete_count = min(len(selected) - 1, int(rng.choice(3, p=[.55, .32, .13])))
        incomplete_count = len(selected) - complete_count
        row_lengths = _band_lengths(rng, comp_rows, [r for r, _ in selected])
        col_lengths = _band_lengths(rng, comp_cols, [c for _, c in selected])
        height = sum(row_lengths) + comp_rows - 1
        width = sum(col_lengths) + comp_cols - 1
        row_lines = tuple(np.cumsum(row_lengths)[:-1] + np.arange(comp_rows - 1))
        col_lines = tuple(np.cumsum(col_lengths)[:-1] + np.arange(comp_cols - 1))
        colors = [int(x) for x in rng.permutation(10)]
        background, structural = colors[:2]
        accent_count = int(rng.integers(2 if multicolor else 1, 5))
        accents = colors[2:2 + accent_count]
        output = np.full((height, width), background, dtype=np.int8)
        output[list(row_lines), :] = structural
        output[:, list(col_lines)] = structural
        rs, cs = _slices(row_lines, height), _slices(col_lines, width)
        missing = []
        try:
            for i, (r, c) in enumerate(selected):
                h, w = row_lengths[r - 1], col_lengths[c - 1]
                colored = multicolor and (i == 0 or rng.random() < .19)
                template = _template(rng, h, w, background, accents, colored,
                                     asymmetric=demonstrate and i == 0)
                for quadrant, part in enumerate(_quadrants(rs, cs, r - 1, c - 1)):
                    output[part] = _orient(template, quadrant)
                if i < incomplete_count:
                    missing.append(int(rng.choice(missing_pool)))
            input_grid = output.copy()
            for (r, c), quadrant in zip(selected[:incomplete_count], missing):
                input_grid[_quadrants(rs, cs, r - 1, c - 1)[quadrant]] = background
            metadata = _analyze_pair_arrays(input_grid, output, regime)
            assert len(metadata["incomplete"]) == incomplete_count
            assert len(metadata["complete"]) == complete_count
            if multicolor:
                assert any(len(np.unique(t[t != background])) >= 2 for t in metadata["templates"])
            return {"input": input_grid, "output": output}
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid pair")


def _grid_key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _pair_digest(pair):
    digest = hashlib.sha256()
    for role in ("input", "output"):
        grid = pair[role].astype(np.int8, copy=False)
        digest.update(bytes(grid.shape))
        digest.update(grid.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Reflection is the sole puzzle-level convention; palettes have local roles.
    plans = [bool(rng.random() < .36) for _ in range(num_train + num_test)]
    if any(plans[num_train:]) and not any(plans[:num_train]):
        plans[int(rng.integers(num_train))] = True
    # Some puzzles deliberately repeat just one or two omission directions.
    missing_pool = (tuple(int(x) for x in rng.choice(4, size=int(rng.integers(1, 3)),
                                                       replace=False))
                    if rng.random() < .24 else (0, 1, 2, 3))
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(num_train + num_test):
        split = "train" if index < num_train else "test"
        for _ in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                              multicolor=plans[index], demonstrate=index == 0,
                              missing_pool=missing_pool)
            key = _grid_key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("failed to find a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_multi, test_multi, train_asym = False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for role in ("input", "output"):
                grid = pair[role]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
            info = _analyze_pair_arrays(pair["input"], pair["output"], split)
            for template in info["templates"]:
                mask = template != info["background"]
                multi = len(np.unique(template[mask])) >= 2
                if split == "train":
                    train_multi |= multi
                    train_asym |= (not np.array_equal(mask, np.fliplr(mask)) and
                                   not np.array_equal(mask, np.flipud(mask)))
                else:
                    test_multi |= multi
    assert train_asym
    assert not test_multi or train_multi


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) for split in ("train", "test")) and all(
        np.array_equal(a[role], b[role])
        for split in ("train", "test")
        for a, b in zip(first[split], second[split])
        for role in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_pairs = [pair for p in puzzles for split in ("train", "test") for pair in p[split]]
    infos = [_analyze_pair_arrays(p["input"], p["output"]) for p in all_pairs]
    assert len({_grid_key(p["input"]) for p in all_pairs}) == len(all_pairs)
    assert len({_pair_digest(p) for p in all_pairs}) == len(all_pairs)
    assert {q for info in infos for q in info["missing"]} == {0, 1, 2, 3}
    assert any(info["complete"] for info in infos)
    assert any(len(set(info["row_lengths"])) > 1 for info in infos)
    assert any(len(set(info["col_lengths"])) > 1 for info in infos)
    assert any(info["row_lengths"][0] != info["col_lengths"][0] for info in infos)
    assert any(len(np.unique(t[t != info["background"]])) >= 2
               for info in infos for t in info["templates"])
    assert any(len(info["incomplete"]) >= 3 for info in infos)
    assert any(len({t.shape for t in info["templates"]}) > 1 for info in infos)
    assert any(any(t.shape[0] != t.shape[1] for t in info["templates"])
               for info in infos)
    assert any(any(int(np.count_nonzero(t != info["background"])) <= 2
                   for t in info["templates"]) for info in infos)
    assert any(any(np.array_equal(t != info["background"],
                                  np.fliplr(t != info["background"]))
                   for t in info["templates"]) for info in infos)
    assert any(any(bool(set(int(c) for c in np.unique(a) if c != info["background"]) &
                        set(int(c) for c in np.unique(b) if c != info["background"]))
                   for i, a in enumerate(info["templates"]) for b in info["templates"][i + 1:])
               for info in infos)
    assert any(len(set(info["missing"])) < len(info["missing"])
               for info in infos if len(info["missing"]) > 1)
    assert any(len({q for split in ("train", "test") for pair in puzzle[split]
                    for q in _analyze_pair_arrays(pair["input"], pair["output"])["missing"]}) <= 2
               for puzzle in puzzles)
    assert any(len({(_analyze_pair_arrays(pair["input"], pair["output"])["background"],
                     _analyze_pair_arrays(pair["input"], pair["output"])["structural"])
                    for split in ("train", "test") for pair in puzzle[split]}) > 1
               for puzzle in puzzles)
    assert len({p["input"].shape for p in all_pairs}) >= 30
    assert _puzzles_equal(generate_puzzle(7001, 6, 3), generate_puzzle(7001, 6, 3))
    assert not _puzzles_equal(generate_puzzle(7001, 6, 3), generate_puzzle(7002, 6, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + train * 10 + test, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
    validate_generated_puzzle(generate_puzzle(78947394767170271, 4, 3))
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(10000 + i, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(11000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
