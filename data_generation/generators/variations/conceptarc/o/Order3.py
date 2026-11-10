"""Generate ARC puzzles that sort equal panels by their accent counts."""

import numpy as np


_SOURCE_INPUTS = {
    ("0330330330440", "0340330330330", "0330430340430", "0330340330340", "0430340330330", "0330330330330"),
    ("0000000000000", "0777077707770", "0666077707770", "0777066707770", "0777077707670", "0777077707770"),
    ("00000000000", "00000000000", "51011011015", "11011011051", "51051011011", "11011011051"),
    ("0000000000000", "4444440444444", "8888880444444", "4444440444444", "4444440888884", "4444440444444"),
    ("0000000000000", "2202202202200", "2202202200200", "0202202200200", "2002002200200", "2202202202200"),
}


def _signature(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _sample_panel(rng, height, width, fill, accent, count, hole):
    panel = np.full((height, width), fill, dtype=np.int8)
    if count == 0:
        return panel
    # A visible fill row keeps every column of a hole panel identifiable.
    protected_row = int(rng.integers(height)) if hole else -1
    available = {(y, x) for y in range(height) for x in range(width)
                 if y != protected_row}
    selected = set()
    mode = int(rng.integers(5))
    if mode == 1:  # Horizontal run, sometimes a complete bar.
        y = int(rng.choice([r for r in range(height) if r != protected_row]))
        length = min(count, width if count >= width and rng.random() < .7
                     else int(rng.integers(2, width + 1)))
        x = int(rng.integers(width - length + 1))
        selected.update((y, xx) for xx in range(x, x + length))
    elif mode == 2:  # Vertical streak.
        x = int(rng.integers(width))
        rows = [y for y in range(height) if y != protected_row]
        length = min(count, len(rows))
        if length > 1:
            start = int(rng.integers(len(rows) - length + 1))
            selected.update((y, x) for y in rows[start:start + length])
    elif mode == 3:  # Connected cluster.
        candidates = tuple(available)
        start = candidates[int(rng.integers(len(candidates)))]
        selected.add(start)
        target = min(count, max(2, int(count * rng.uniform(.55, .9))))
        while len(selected) < target:
            frontier = {(y + dy, x + dx) for y, x in selected
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if (y + dy, x + dx) in available
                        and (y + dy, x + dx) not in selected}
            if not frontier:
                break
            candidates = tuple(frontier)
            selected.add(candidates[int(rng.integers(len(candidates)))])
    elif mode == 4:  # Two local runs.
        for _ in range(2):
            y = int(rng.choice([r for r in range(height) if r != protected_row]))
            x = int(rng.integers(width))
            selected.update((y, xx) for xx in range(x, min(width, x + 2)))
        if len(selected) > count:
            candidates = tuple(selected)
            selected = {candidates[int(i)] for i in
                        rng.choice(len(candidates), size=count, replace=False)}
    if len(selected) < count:
        candidates = tuple(available - selected)
        selected.update(candidates[int(i)] for i in
                        rng.choice(len(candidates), size=count - len(selected),
                                   replace=False))
    for y, x in selected:
        panel[y, x] = accent
    return panel


def _sample_counts(rng, n, height, width, hole):
    area = height * width
    maximum = min(int(area * .68), (height - 1) * width if hole else area - 1)
    assert maximum >= n
    # Mix source-like sparse markings with panels carrying much larger counts.
    if rng.random() < .35:
        maximum = min(maximum, max(n, 8))
    for _ in range(30):
        counts = rng.choice(np.arange(maximum + 1), size=n, replace=False)
        if rng.random() < .4:
            counts[np.argmin(counts)] = 0
        if len(set(counts)) == n and sum(counts) < .46 * n * area:
            break
    else:
        counts = np.arange(n)
    rng.shuffle(counts)
    if all(a < b for a, b in zip(counts[:-1], counts[1:])):
        counts[0], counts[-1] = counts[-1], counts[0]
    if n >= 3 and all(a > b for a, b in zip(counts[:-1], counts[1:])):
        counts[0], counts[1] = counts[1], counts[0]
    return tuple(map(int, counts))


def _make_pair(rng, regime, force_multi=False, force_visible=False):
    if regime == "source_train":
        weights = (0, 4, 4, 2, 1) if force_multi else (2, 4, 4, 2, 1)
    elif regime == "target_test":
        weights = (3, 3, 4, 3, 1)
    else:
        raise ValueError("Unknown pair regime")
    n = int(rng.choice(np.arange(2, 7), p=np.array(weights) / sum(weights)))
    left, right = map(int, rng.integers(0, 4, size=2))
    width_limit = min(10, (30 - left - right - n + 1) // n)
    width = int(rng.integers(2, width_limit + 1))
    top, bottom = map(int, rng.integers(0, 4, size=2))
    if top == bottom == 0:
        top = 1
    min_height = max(4, int(np.ceil((n + 1) / (width * .68))))
    height = int(rng.integers(min_height, min(14, 30 - top - bottom) + 1))
    background = int(rng.integers(10))
    fill = int(rng.choice([c for c in range(10) if c != background]))
    hole = not force_visible and rng.random() < (
        .28 if regime == "source_train" else .38)
    accent = background if hole else int(rng.choice(
        [c for c in range(10) if c not in (background, fill)]))
    counts = _sample_counts(rng, n, height, width, hole)
    for _ in range(30):
        panels = [_sample_panel(rng, height, width, fill, accent, count, hole)
                  for count in counts]
        if all(any(np.any(panel[y] != background) for panel in panels)
               for y in range(height)):
            break
    else:
        raise RuntimeError("Could not keep panel rows visible")
    shape = (top + height + bottom, left + n * width + n - 1 + right)
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    order = np.argsort(counts)
    for slot in range(n):
        x = left + slot * (width + 1)
        inp[top:top + height, x:x + width] = panels[slot]
        out[top:top + height, x:x + width] = panels[int(order[slot])]
    return inp, out


def _validate_counts(num_train, num_test):
    if num_train is not None and (type(num_train) is not int or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 through 10")
    if num_test is not None and (type(num_test) is not int or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    _validate_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(3, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(3, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_order = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(2**63)))
                inp, out = _make_pair(
                    pair_rng, "source_train" if split == "train" else "target_test",
                    force_multi=split == "train" and index < 2,
                    force_visible=split == "train" and index == 0)
                sig = _signature(inp)
                if sig in seen_inputs or sig in _SOURCE_INPUTS:
                    continue
                order = tuple(map(int, np.argsort(_inspect(inp)["counts"])))
                if split == "train" and index == 1 and order == first_order:
                    continue
                break
            else:
                raise RuntimeError("Could not generate a distinct pair")
            seen_inputs.add(sig)
            if split == "train" and index == 0:
                first_order = order
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert grid.dtype.kind in ("i", "u")
    assert np.all((grid >= 0) & (grid <= 9))


def _inspect(grid):
    _check_grid(grid)
    uniform_rows = [y for y in range(grid.shape[0])
                    if np.all(grid[y] == grid[y, 0])]
    assert uniform_rows
    background = int(grid[uniform_rows[0], 0])
    active_rows = np.flatnonzero(np.any(grid != background, axis=1))
    assert len(active_rows) >= 4
    top, end = int(active_rows[0]), int(active_rows[-1]) + 1
    assert np.array_equal(active_rows, np.arange(top, end))
    assert top + grid.shape[0] - end >= 1
    height = end - top
    active_columns = np.flatnonzero(np.any(grid != background, axis=0))
    assert len(active_columns)
    starts = [int(active_columns[0])]
    ends = []
    for a, b in zip(active_columns[:-1], active_columns[1:]):
        if b != a + 1:
            ends.append(int(a) + 1)
            starts.append(int(b))
    ends.append(int(active_columns[-1]) + 1)
    segments = tuple(zip(starts, ends))
    n = len(segments)
    widths = [b - a for a, b in segments]
    assert 2 <= n <= 6 and len(set(widths)) == 1
    assert 2 <= widths[0] <= 10 and 4 <= height <= 14
    assert all(segments[i + 1][0] - segments[i][1] == 1 for i in range(n - 1))
    assert 0 <= starts[0] <= 3 and 0 <= grid.shape[1] - ends[-1] <= 3
    panels = [grid[top:end, a:b] for a, b in segments]
    outside = np.ones_like(grid, dtype=bool)
    for a, b in segments:
        outside[top:end, a:b] = False
    assert np.all(grid[outside] == background)
    colors, amounts = np.unique(grid[grid != background], return_counts=True)
    assert 1 <= len(colors) <= 2
    fill = int(colors[int(np.argmax(amounts))])
    accent = background if len(colors) == 1 else int(next(c for c in colors if c != fill))
    if accent == background:
        assert all(np.all(np.any(panel == fill, axis=0)) for panel in panels)
    counts = tuple(int(np.count_nonzero(panel != fill)) for panel in panels)
    assert len(set(counts)) == n
    return {"background": background, "fill": fill, "accent": accent,
            "n": n, "w": widths[0], "height": height, "top": top,
            "bottom": grid.shape[0] - end, "left": starts[0],
            "right": grid.shape[1] - ends[-1], "segments": segments,
            "counts": counts}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen_inputs and signature not in _SOURCE_INPUTS
            seen_inputs.add(signature)
            info, result = _inspect(inp), _inspect(out)
            assert not all(a < b for a, b in
                           zip(info["counts"][:-1], info["counts"][1:]))
            for key in ("background", "fill", "accent", "n", "w", "height",
                        "top", "bottom", "left", "right", "segments"):
                assert info[key] == result[key]
            assert result["counts"] == tuple(sorted(info["counts"]))
            expected = np.full_like(inp, info["background"])
            y0, y1 = info["top"], info["top"] + info["height"]
            for destination, source in enumerate(np.argsort(info["counts"])):
                dx0, dx1 = info["segments"][destination]
                sx0, sx1 = info["segments"][int(source)]
                expected[y0:y1, dx0:dx1] = inp[y0:y1, sx0:sx1]
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            if split == "train":
                training.append(info)
    assert sum(info["n"] >= 3 for info in training) >= 2
    assert len({tuple(map(int, np.argsort(info["counts"])))
                for info in training[:2]}) == 2
    assert any(info["accent"] != info["background"] for info in training)


def _fingerprint(puzzle):
    return tuple((split, tuple((_signature(p["input"]), _signature(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    fingerprints, shapes, layouts = set(), set(), set()
    train_counts, test_counts = set(), set()
    train_holes = test_holes = full_rows = dense = zero = 0
    multi_tests = wide_train = margins = varied_background = 0
    n_values, width_values, height_values, margin_pairs, zero_slots = (
        set(), set(), set(), set(), set())
    sparse = source_like_tests = hole_zero_multi = first_combined = 0
    varied_layouts = varied_hole_roles = horizontal = vertical = clusters = singletons = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        backgrounds = set()
        puzzle_layouts, puzzle_holes = set(), set()
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                info = _inspect(pair["input"])
                backgrounds.add(info["background"])
                puzzle_layouts.add((info["n"], info["w"]))
                puzzle_holes.add(info["accent"] == info["background"])
                n_values.add(info["n"])
                width_values.add(info["w"])
                height_values.add(info["height"])
                margin_pairs.add((info["left"], info["right"]))
                if 0 in info["counts"]:
                    zero_slots.add(info["counts"].index(0))
                shapes.add(pair["input"].shape)
                layouts.add((info["n"], info["w"], info["height"], info["left"],
                             info["right"], info["top"], info["bottom"]))
                if info["accent"] == info["background"]:
                    if split == "train":
                        train_holes += 1
                    else:
                        test_holes += 1
                for a, b in info["segments"]:
                    panel = pair["input"][info["top"]:info["top"] + info["height"], a:b]
                    mask = panel != info["fill"]
                    full_rows += int(np.any(np.all(mask, axis=1)))
                    horizontal += int(np.any(mask[:, :-1] & mask[:, 1:]))
                    vertical += int(np.any(mask[:-1, :] & mask[1:, :]))
                    clusters += int(np.any(mask[:-1, :-1] & mask[1:, :-1]
                                           & mask[:-1, 1:] & mask[1:, 1:]))
                    neighbors = np.zeros_like(mask)
                    neighbors[1:] |= mask[:-1]
                    neighbors[:-1] |= mask[1:]
                    neighbors[:, 1:] |= mask[:, :-1]
                    neighbors[:, :-1] |= mask[:, 1:]
                    singletons += int(np.any(mask & ~neighbors))
                dense += int(max(info["counts"]) > 8)
                sparse += int(max(info["counts"]) <= 8)
                zero += int(0 in info["counts"])
                margins += int(info["right"] >= 2 or info["left"] >= 2)
                if split == "test" and info["n"] >= 3:
                    multi_tests += 1
                if split == "test":
                    source_like_tests += int(info["n"] in (3, 4) and info["w"] in (2, 3))
                    hole_zero_multi += int(info["accent"] == info["background"]
                                           and 0 in info["counts"] and info["n"] >= 3)
                    if index == 0:
                        first_combined += int(info["n"] == 2 and info["w"] >= 4
                                              and info["accent"] == info["background"]
                                              and 0 in info["counts"])
                if split == "train" and info["n"] == 2 and info["w"] >= 4:
                    wide_train += 1
        varied_background += int(len(backgrounds) > 1)
        varied_layouts += int(len(puzzle_layouts) > 1)
        varied_hole_roles += int(len(puzzle_holes) > 1)
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(layouts) >= 100
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert train_holes > 0 and test_holes > 0 and full_rows > 0
    assert dense > 0 and zero > 0 and multi_tests > 0 and wide_train > 0
    assert margins > 0 and varied_background > 0
    assert n_values == {2, 3, 4, 5, 6}
    assert width_values == set(range(2, 11))
    assert height_values == set(range(4, 15))
    assert len(margin_pairs) >= 12 and len(zero_slots) >= 4
    assert sparse > 0 and source_like_tests > 0 and hole_zero_multi > 0
    assert first_combined < 42 and varied_layouts >= 35 and varied_hole_roles >= 20
    assert horizontal > 0 and vertical > 0 and clusters > 0 and singletons > 0
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(891, nt, ns)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(891, nt, ns))
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1), (2, 1.5)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts accepted")
