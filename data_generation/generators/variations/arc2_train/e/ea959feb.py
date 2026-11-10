"""Generate cyclic wallpapers damaged by same-color rectangular overwrites."""

import hashlib

import numpy as np


_SOURCE_SIGNATURES = {
    "e18bcec2ef58f355c4daefc2be6c951152254ab330dff4ba752a6fe9ab2f905f",
    "9985adf1b1ed920e91a386aed7630924859ba0f45b17ba97c0ca6394ed04160a",
    "437679c87a14dab0529ff6a7b96b0a922276053865f19efd19cf5f2e2bf9a1b6",
    "282c681092b78f243f7ea52be80303140b6d00a0d436593d03b32850126b34a0",
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _checked_count("num_train", num_train, 2, 10)
    ns = _checked_count("num_test", num_test, 1, 4)
    if nt is not None and ns is not None:
        if nt <= ns:
            raise ValueError("num_train must exceed num_test")
    elif nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    else:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    return nt, ns


def _grid_key(grid):
    a = grid.astype(np.uint8, copy=False)
    return a.shape, a.tobytes()


def _source_signature(pair):
    digest = hashlib.sha256()
    for name in ("input", "output"):
        a = pair[name].astype(np.uint8, copy=False)
        digest.update(bytes(a.shape))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _minimal_period(word):
    for period in range(1, len(word) + 1):
        if all(word[i] == word[i % period] for i in range(len(word))):
            return period
    raise AssertionError("missing period")


def _sample_word(rng, size, period):
    for _ in range(80):
        word = rng.integers(0, size, size=period)
        if len(np.unique(word)) >= min(3, period) and _minimal_period(word) == period:
            return word
    word = np.arange(period) % size
    assert _minimal_period(word) == period
    return word


def _sample_size(rng, period, minimum):
    choices = [n for n in range(max(minimum, 2 * period), 31)]
    return int(rng.choice(choices))


def _analyze_wallpaper(grid):
    colors = tuple(int(c) for c in np.unique(grid))
    assert len(colors) >= 2
    successor = {}
    for upper, lower in zip(grid[:-1].flat, grid[1:].flat):
        upper, lower = int(upper), int(lower)
        if upper in successor:
            assert successor[upper] == lower
        else:
            successor[upper] = lower
    assert set(successor) == set(colors)
    assert set(successor.values()) == set(colors)
    cycle = [colors[0]]
    for _ in range(len(colors) - 1):
        cycle.append(successor[cycle[-1]])
    assert len(set(cycle)) == len(colors)
    assert successor[cycle[-1]] == cycle[0]
    width = grid.shape[1]
    period = next(
        p for p in range(1, width + 1)
        if p == width or np.array_equal(grid[:, :-p], grid[:, p:])
    )
    return {"colors": colors, "successor": successor, "period": period}


def _cycle_from_mask(analysis, mask):
    cycle = [int(mask)]
    for _ in range(len(analysis["colors"]) - 1):
        cycle.append(analysis["successor"][cycle[-1]])
    assert len(set(cycle)) == len(cycle)
    assert analysis["successor"][cycle[-1]] == mask
    return tuple(cycle)


def _candidate_for_period(grid, mask, size, period):
    height, width = grid.shape
    tile = np.full((size, period), mask, dtype=np.int8)
    for r in range(size):
        for c in range(period):
            samples = grid[r::size, c::period].ravel()
            nonmask = np.unique(samples[samples != mask])
            if len(nonmask) > 1:
                return None
            if len(nonmask) == 1:
                tile[r, c] = nonmask[0]
    candidate = tile[np.arange(height)[:, None] % size,
                     np.arange(width)[None, :] % period]
    difference = grid != candidate
    if not np.any(difference) or np.any(grid[difference] != mask):
        return None
    if not np.array_equal(np.unique(candidate), np.unique(grid)):
        return None
    try:
        analysis = _analyze_wallpaper(candidate)
    except AssertionError:
        return None
    if len(analysis["colors"]) != size:
        return None
    return candidate


def _recover_wallpaper(grid, mask):
    size = len(np.unique(grid))
    solutions = {}
    for period in range(1, grid.shape[1] // 2 + 1):
        candidate = _candidate_for_period(grid, mask, size, period)
        if candidate is not None:
            solutions[_grid_key(candidate)] = candidate
    assert len(solutions) == 1
    return next(iter(solutions.values())).copy()


def _rectangles(rng, height, width, count, distributed):
    boxes = []
    for index in range(count):
        # Sizes and positions are sampled independently of palette and period.
        rh = int(rng.integers(2, min(12, height - 1) + 1))
        rw = int(rng.integers(2, min(13, width - 1) + 1))
        if distributed and index < 3:
            anchor = int((index + 0.5) * height / 3)
            r0 = anchor - rh // 2 + int(rng.integers(-2, 3))
        elif boxes and rng.random() < 0.32:
            previous = boxes[int(rng.integers(len(boxes)))]
            r0 = int(previous[0] + rng.integers(-rh + 1, previous[1] - previous[0] + 2))
        else:
            r0 = int(rng.integers(-rh // 3, height - 1))
        if boxes and rng.random() < 0.32:
            previous = boxes[int(rng.integers(len(boxes)))]
            c0 = int(previous[2] + rng.integers(-rw + 1, previous[3] - previous[2] + 2))
        else:
            c0 = int(rng.integers(-rw // 3, width - 1))
        r0 = max(0, min(height - 2, r0))
        c0 = max(0, min(width - 2, c0))
        boxes.append((r0, min(height, r0 + rh), c0, min(width, c0 + rw)))
    return boxes


def _apply_damage(rng, clean, mask, count, distributed, required_edge=None):
    height, width = clean.shape
    for _ in range(90):
        boxes = _rectangles(rng, height, width, count, distributed)
        damaged = clean.copy()
        covered = np.zeros(clean.shape, dtype=bool)
        for r0, r1, c0, c1 in boxes:
            damaged[r0:r1, c0:c1] = mask
            covered[r0:r1, c0:c1] = True
        changed = damaged != clean
        if np.count_nonzero(changed) < max(4, count * 2):
            continue
        if np.count_nonzero(changed) > clean.size * 0.4:
            continue
        if not np.any(covered & (clean == mask)):
            continue
        if required_edge is not None:
            upper, lower = required_edge
            if not np.any((damaged[:-1] == upper) & (damaged[1:] == lower)):
                continue
        if distributed:
            intact_rows = ~np.any(changed, axis=1)
            size = len(np.unique(clean))
            if any(np.all(intact_rows[i:i + size]) for i in range(height - size + 1)):
                continue
        try:
            if np.array_equal(_recover_wallpaper(damaged, mask), clean):
                return damaged
        except AssertionError:
            pass
    raise RuntimeError("failed to place unambiguous damage")


def _generate_pair(rng, cycle, mask, size, regime):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    period = int(rng.integers(3, 11))
    height = _sample_size(rng, size, 12)
    width = _sample_size(rng, period, 12)
    word = _sample_word(rng, size, period)
    row_phase = int(rng.integers(size))
    col_phase = int(rng.integers(period))
    indices = (row_phase + np.arange(height)[:, None]
               + word[(np.arange(width) + col_phase) % period][None, :]) % size
    clean = cycle[:size][indices].astype(np.int8)
    if regime == "source_train":
        count = int(rng.integers(1, 7))
    else:
        count = int(rng.integers(1, 8))
    distributed = count >= 3 and rng.random() < 0.55
    required_edge = (int(cycle[7]), int(cycle[8])) if regime == "target_test" and size == 9 else None
    damaged = _apply_damage(rng, clean, mask, count, distributed, required_edge)
    return {"input": damaged, "output": clean}


def _mask_rectangle_cover(input_grid, output_grid, mask):
    """Check that every change lies in a broad all-mask patch."""
    changed = input_grid != output_grid
    height, width = input_grid.shape
    covered = np.zeros(changed.shape, dtype=bool)
    for r in range(height - 1):
        for c in range(width - 1):
            if np.all(input_grid[r:r + 2, c:c + 2] == mask):
                covered[r:r + 2, c:c + 2] = True
    return np.all(~changed | covered)


def _components4(binary):
    seen = np.zeros(binary.shape, dtype=bool)
    components = []
    for row, col in np.argwhere(binary):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        cells = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if (0 <= nr < binary.shape[0] and 0 <= nc < binary.shape[1]
                        and binary[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(cells)
    return components


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[section], list) for section in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    masks = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            changed = input_grid != output_grid
            assert np.any(changed)
            changed_colors = np.unique(input_grid[changed])
            assert len(changed_colors) == 1
            mask = int(changed_colors[0])
            masks.add(mask)
            assert np.array_equal(np.unique(input_grid), np.unique(output_grid))
            assert np.any((input_grid == mask) & (output_grid == mask))
            assert _mask_rectangle_cover(input_grid, output_grid, mask)
            assert _source_signature(pair) not in _SOURCE_SIGNATURES
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _analyze_wallpaper(output_grid)
            assert mask in analysis["colors"]
            assert 2 * len(analysis["colors"]) <= output_grid.shape[0]
            assert 2 * analysis["period"] <= output_grid.shape[1]
            assert np.array_equal(_recover_wallpaper(input_grid, mask), output_grid)
            analyses[section].append(analysis)
    assert len(masks) == 1
    mask = next(iter(masks))
    train = analyses["train"]
    test = analyses["test"]
    assert all(len(a["colors"]) in (6, 7, 8) for a in train)
    assert any(len(a["colors"]) == 8 for a in train)
    assert all(len(a["colors"]) in (6, 7, 8, 9) for a in test)
    prefixes = {_cycle_from_mask(a, mask) for a in train if len(a["colors"]) == 8}
    assert len(prefixes) == 1
    prefix = next(iter(prefixes))
    for a in train + test:
        order = _cycle_from_mask(a, mask)
        assert order[:min(len(order), 8)] == prefix[:min(len(order), 8)]
    extensions = {_cycle_from_mask(a, mask)[8] for a in test if len(a["colors"]) == 9}
    assert len(extensions) <= 1
    for pair, analysis in zip(puzzle["test"], test):
        if len(analysis["colors"]) == 9:
            extra = _cycle_from_mask(analysis, mask)[8]
            grid = pair["input"]
            assert np.any((grid[:-1] == prefix[7]) & (grid[1:] == extra))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    mask = int(rng.integers(10))
    remainder = np.array([c for c in range(10) if c != mask], dtype=np.int8)
    rng.shuffle(remainder)
    cycle = np.concatenate((np.array([mask], dtype=np.int8), remainder))
    train_sizes = [8, int(rng.choice([6, 7]))]
    train_sizes.extend(int(x) for x in rng.choice([6, 7, 8], size=nt - 2))
    rng.shuffle(train_sizes)
    test_sizes = [int(x) for x in rng.choice([6, 7, 8, 9], size=ns)]
    puzzle = {"train": [], "test": []}
    used = set()
    for section, sizes in (("train", train_sizes), ("test", test_sizes)):
        regime = "source_train" if section == "train" else "target_test"
        for size in sizes:
            for _ in range(20):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                pair = _generate_pair(pair_rng, cycle, mask, size, regime)
                key = _grid_key(pair["input"])
                if key not in used and _source_signature(pair) not in _SOURCE_SIGNATURES:
                    used.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("failed to generate unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts, masks, shapes, periods, palettes = (set() for _ in range(6))
    combinations = set()
    measures = {name: 0 for name in (
        "no_zero", "no_clean_cycle", "clean_cycle", "nonrect_component",
        "singleton_component", "many_components", "edge_change",
        "whole_vertical", "partial_vertical", "whole_horizontal",
        "partial_horizontal", "sparse", "broad", "test_nine",
        "test_demonstrated", "test_nine_short", "multi_shape",
        "multi_period", "multi_palette")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first = puzzle["train"][0]
        train_diff = first["input"] != first["output"]
        learned_mask = int(first["input"][train_diff][0])
        pair_shapes, pair_periods, pair_palettes = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                diff = inp != out
                mask = int(np.unique(inp[diff])[0])
                assert mask == learned_mask
                masks.add(mask)
                shapes.add(out.shape)
                analysis = _analyze_wallpaper(out)
                size = len(analysis["colors"])
                period = analysis["period"]
                periods.add(period)
                palettes.add(size)
                combinations.add((section, size, period))
                pair_shapes.add(out.shape)
                pair_periods.add(period)
                pair_palettes.add(size)
                measures["no_zero"] += 0 not in analysis["colors"]
                intact = ~np.any(diff, axis=1)
                clean_cycle = any(np.all(intact[i:i + size])
                                  for i in range(out.shape[0] - size + 1))
                measures["clean_cycle"] += clean_cycle
                measures["no_clean_cycle"] += not clean_cycle
                measures["whole_vertical"] += out.shape[0] % size == 0
                measures["partial_vertical"] += out.shape[0] % size != 0
                measures["whole_horizontal"] += out.shape[1] % period == 0
                measures["partial_horizontal"] += out.shape[1] % period != 0
                measures["sparse"] += np.mean(diff) < .05
                measures["broad"] += np.mean(diff) > .2
                measures["edge_change"] += (np.any(diff[0]) or np.any(diff[-1])
                                            or np.any(diff[:, 0]) or np.any(diff[:, -1]))
                components = _components4(diff)
                measures["singleton_component"] += any(len(cells) == 1 for cells in components)
                measures["many_components"] += len(components) >= 5
                for cells in components:
                    rows = [cell[0] for cell in cells]
                    cols = [cell[1] for cell in cells]
                    if np.any(inp[min(rows):max(rows) + 1, min(cols):max(cols) + 1] != mask):
                        measures["nonrect_component"] += 1
                        break
                if section == "test":
                    measures["test_nine"] += size == 9
                    measures["test_demonstrated"] += size <= 8
                    measures["test_nine_short"] += size == 9 and period < 9
                assert np.array_equal(_recover_wallpaper(inp.copy(), learned_mask), out)
        measures["multi_shape"] += len(pair_shapes) > 1
        measures["multi_period"] += len(pair_periods) > 1
        measures["multi_palette"] += len(pair_palettes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(masks) >= 7 and len(shapes) >= 40
    assert len(periods) >= 7 and len(palettes) >= 4
    assert len(combinations) >= 35
    assert measures["multi_shape"] >= 35
    assert measures["multi_period"] >= 35
    assert measures["multi_palette"] >= 35
    assert all(value > 0 for value in measures.values())
    a = generate_puzzle(123456, 7, 3)
    b = generate_puzzle(123456, 7, 3)
    assert all(np.array_equal(x[name], y[name])
               for section in ("train", "test")
               for x, y in zip(a[section], b[section])
               for name in ("input", "output"))
    assert _grid_key(generate_puzzle(123457, 7, 3)["train"][0]["input"]) != _grid_key(a["train"][0]["input"])
    for seed, nt, ns in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4),
                         (5004, None, 1), (5005, None, 4),
                         (5006, 2, None), (5007, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
