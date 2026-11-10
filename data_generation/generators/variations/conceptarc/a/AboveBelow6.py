"""Compositional examples of downward, background-bounded marker propagation."""

import numpy as np


SOURCE_INPUTS = {
    tuple(s.split()) for s in (
        """0000000000 0444444440 0444444440 0444444440
        0222222220 0444444440 0444444440 0000000000""",
        """00000000000000 00000000555550 55555550555550
        55555550555550 55555550555550 52252250555550
        55555550555550 55555550225220 00000000555550""",
        """0000000000 0000000000 0000333333 0220333333
        0330222222 0330000000 0330000000 0330000000""",
        """0000000000000 0000000000000 0000000000000
        8888888888888 8888888822222 2222888888888
        8888888888888 0000000000000 0000000000000
        0000000000000""",
        """0000000 0000000 0000000 2200021 1100011
        1100011 1100011 1100011 1100011""",
    )
}


def _integer(rng, low, high):
    return int(rng.integers(low, high + 1))


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError("illegal pair count")
    if num_train is None and num_test is None:
        num_train = _integer(rng, 2, 10)
        num_test = _integer(rng, 1, min(4, num_train - 1))
    elif num_train is None:
        num_train = _integer(rng, max(2, num_test + 1), 10)
    elif num_test is None:
        num_test = _integer(rng, 1, min(4, num_train - 1))
    if num_train <= num_test:
        raise ValueError("training count must exceed test count")
    return int(num_train), int(num_test)


def _propagate(grid, fill, marker):
    """Construct the answer from the original input, one marker ray at a time."""
    out = grid.copy()
    height = grid.shape[0]
    for row, col in np.argwhere(grid == marker):
        row = int(row) + 1
        col = int(col)
        while row < height and grid[row, col] == fill:
            out[row, col] = marker
            row += 1
    return out


def _shape(rng, height, width, regime, demo, force_gap, allow_gap):
    """Make horizontally separated regions with independent vertical footprints."""
    two_tall = (not demo and not force_gap and width >= 14 and
                rng.random() < 0.11)
    complexity = 0.25 if regime == "source_train" else 0.55
    regions = int(rng.choice((1, 2, 3), p=(0.46 - complexity / 4,
                                            0.38, 0.16 + complexity / 4)))
    if demo:
        regions = _integer(rng, 1, 2)
    if force_gap:
        regions = _integer(rng, 1, 2)
    if two_tall:
        regions = 2
        widths = [_integer(rng, 2, 3), _integer(rng, 2, 3)]
        gap_widths = [_integer(rng, 1, 4)]
        span = sum(widths) + gap_widths[0]
    else:
        minimums = [5 if (demo or force_gap) and i == 0 else 2
                    for i in range(regions)]
        gap_widths = [_integer(rng, 1, 3) for _ in range(regions - 1)]
        minimum_span = sum(minimums) + sum(gap_widths)
        if minimum_span > width:
            gap_widths = [1] * (regions - 1)
            minimum_span = sum(minimums) + sum(gap_widths)
        cap = max(minimum_span, int(width * 0.78))
        span = _integer(rng, minimum_span, min(width, cap))
        extra = span - minimum_span
        additions = rng.multinomial(extra, np.ones(regions) / regions)
        widths = [minimums[i] + int(additions[i]) for i in range(regions)]
    x = _integer(rng, 0, width - span)
    mask = np.zeros((height, width), dtype=bool)
    boxes = []
    hole = None
    for i, region_width in enumerate(widths):
        min_height = 6 if force_gap and i == 0 else (5 if demo and i == 0 else 3)
        max_height = height if region_width <= 4 else max(min_height, int(height * 0.67))
        region_height = (_integer(rng, 6, min(height, 12)) if two_tall else
                         _integer(rng, min_height, max_height))
        top = (height - region_height if two_tall else
               _integer(rng, 0, height - region_height))
        right = x + region_width
        bottom = top + region_height
        mask[top:bottom, x:right] = True
        if force_gap and i == 0:
            hole_y = _integer(rng, top + 2, bottom - 2)
            hole_x = _integer(rng, x + 1, right - 2)
            mask[hole_y, hole_x] = False
            hole = (hole_y, hole_x, top, bottom)
        elif not demo or i > 0:
            shape = int(rng.choice(4, p=((0.46, 0.22, 0.22, 0.10)
                                         if allow_gap else
                                         (0.56, 0.22, 0.22, 0.0))))
            if shape in (1, 2) and region_width >= 3 and region_height >= 4:
                cut_width = _integer(rng, 1, region_width - 1)
                cut_height = _integer(rng, 1, max(1, region_height // 2))
                side = _integer(rng, 0, 1)
                xa, xb = ((x, x + cut_width) if side else
                          (right - cut_width, right))
                ya, yb = ((top, top + cut_height) if shape == 1 else
                          (bottom - cut_height, bottom))
                mask[ya:yb, xa:xb] = False
            elif shape == 3 and region_width >= 3 and region_height >= 5:
                hole_y = _integer(rng, top + 1, bottom - 2)
                hole_x = _integer(rng, x + 1, right - 2)
                mask[hole_y, hole_x] = False
                if hole is None:
                    hole = (hole_y, hole_x, top, bottom)
        boxes.append((top, bottom, x, right))
        if i < regions - 1:
            x = right + gap_widths[i]
    return mask, boxes, hole


def _pattern(rng, columns):
    columns = list(columns)
    if len(columns) <= 2:
        size = _integer(rng, 1, len(columns))
        return [int(c) for c in rng.choice(columns, size=size, replace=False)]
    style = _integer(rng, 0, 3)
    if style == 0:
        return columns
    if style == 1:
        return [int(rng.choice(columns))]
    if style == 2:
        start = _integer(rng, 0, len(columns) - 2)
        end = _integer(rng, start + 1, len(columns) - 1)
        return columns[start:end + 1]
    size = _integer(rng, 2, len(columns) - 1)
    return [int(c) for c in rng.choice(columns, size=size, replace=False)]


def _make_pair(rng, background, marker, fill, regime, demo=False,
               force_gap=False, allow_gap=False):
    height = _integer(rng, 8, 22)
    width = _integer(rng, 10, 26)
    mask, boxes, hole = _shape(rng, height, width, regime, demo,
                               force_gap, allow_gap)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[mask] = fill
    for i, (top, bottom, left, right) in enumerate(boxes):
        if force_gap and i == 0:
            hole_y, hole_x, _, _ = hole
            grid[hole_y - 2, hole_x] = marker
            other = [col for col in range(left, right)
                     if col != hole_x and mask[hole_y - 2, col] and
                     mask[hole_y - 1, col]]
            grid[hole_y - 2, int(rng.choice(other))] = marker
            continue
        if demo and i == 0:
            row = _integer(rng, top + 1, bottom - 2)
            columns = list(range(left, right))
            count = _integer(rng, 1, len(columns) - 1)
            chosen = rng.choice(columns, size=count, replace=False)
            grid[row, chosen] = marker
            continue
        candidates = [r for r in range(top, bottom)
                      if np.any(mask[r, left:right])]
        row_count = _integer(rng, 1, min(3, len(candidates)))
        rows = rng.choice(candidates, size=row_count, replace=False)
        for row in rows:
            row = int(row)
            columns = [c for c in range(left, right) if mask[row, c]]
            for col in _pattern(rng, columns):
                if hole is None or (row <= hole[0] or col != hole[1]):
                    grid[row, col] = marker
    # A bottom marker can be inert, but every pair also has an active one.
    if not demo and rng.random() < 0.26:
        terminal = [(r, c) for r, c in np.argwhere(grid == fill)
                    if r == height - 1 or not mask[r + 1, c]]
        if hole is not None:
            terminal = [(r, c) for r, c in terminal
                        if not (c == hole[1] and r > hole[0])]
        if terminal:
            row, col = terminal[_integer(rng, 0, len(terminal) - 1)]
            grid[row, col] = marker
    answer = _propagate(grid, fill, marker)
    if np.array_equal(answer, grid):
        active = [(r, c) for r, c in np.argwhere(mask)
                  if r + 1 < height and mask[r + 1, c] and
                  (hole is None or c != hole[1])]
        row, col = active[_integer(rng, 0, len(active) - 1)]
        grid[row, col] = marker
        grid[row + 1, col] = fill
        answer = _propagate(grid, fill, marker)
    if fill not in answer:
        # Keep one unaffected fill column visible in the answer.
        removable = [int(c) for c in np.unique(np.argwhere(grid == marker)[:, 1])
                     if not (force_gap and c == hole[1])]
        col = removable[_integer(rng, 0, len(removable) - 1)]
        grid[grid[:, col] == marker, col] = fill
        answer = _propagate(grid, fill, marker)
        if np.array_equal(answer, grid):
            active = [(r, c) for r, c in np.argwhere(mask)
                      if c != col and r + 1 < height and mask[r + 1, c] and
                      (hole is None or c != hole[1])]
            row, active_col = active[_integer(rng, 0, len(active) - 1)]
            grid[row, active_col] = marker
            grid[row + 1, active_col] = fill
            answer = _propagate(grid, fill, marker)
    return {"input": grid, "output": answer}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    background, marker = (int(c) for c in rng.choice(10, size=2, replace=False))
    available = [c for c in range(10) if c not in (background, marker)]
    train_colors = [int(c) for c in rng.choice(available, size=2, replace=False)]
    gap_family = bool(rng.random() < 0.38)
    train = []
    test = []
    seen = set()
    for split, count in (("source_train", num_train), ("target_test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            if split == "source_train":
                fill = train_colors[index] if index < 2 else int(pair_rng.choice(available))
                demo = index == 0
                force_gap = gap_family and index == 1
            else:
                fill = (int(pair_rng.choice(train_colors)) if pair_rng.random() < 0.52
                        else int(pair_rng.choice(available)))
                demo = False
                force_gap = gap_family and index == 0 and pair_rng.random() < 0.75
            for _ in range(32):
                pair = _make_pair(pair_rng, background, marker, fill, split,
                                  demo, force_gap, gap_family)
                grid = pair["input"]
                rows = tuple("".join(map(str, row)) for row in grid)
                key = _key(grid)
                if (key not in seen and rows not in SOURCE_INPUTS and
                        np.count_nonzero(grid == background) > grid.size // 2):
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            seen.add(key)
            (train if split == "source_train" else test).append(pair)
    return {"train": train, "test": test}


def _scan_prediction(grid, fill, marker, background):
    """Independent top-to-bottom state scan used only by validation."""
    result = grid.copy()
    for col in range(grid.shape[1]):
        active = False
        for row in range(grid.shape[0]):
            value = int(grid[row, col])
            if value == background:
                active = False
            elif value == marker:
                active = True
            elif active and value == fill:
                result[row, col] = marker
    return result


def _gap_evidence(grid, fill, marker, background):
    for col in range(grid.shape[1]):
        marked = False
        interrupted = False
        for row in range(grid.shape[0]):
            value = int(grid[row, col])
            if value == marker:
                marked = True
            elif value == background and marked:
                interrupted = True
            elif value == fill and interrupted:
                return True
    return False


def _direction_evidence(pair, fill, marker):
    grid = pair["input"]
    out = pair["output"]
    for row in range(1, grid.shape[0] - 1):
        for col in range(grid.shape[1]):
            if (grid[row, col] == marker and grid[row - 1, col] == fill and
                    grid[row + 1, col] == fill and out[row - 1, col] == fill and
                    out[row + 1, col] == marker and
                    np.any((grid[row] == fill) & (out[row] == fill))):
                return True
    return False


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    found = []
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            row, col = stack.pop()
            for point in ((row - 1, col), (row + 1, col),
                          (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    stack.append(point)
        found.append(component)
    return found


def _visible_features(grid, background, marker):
    parts = _components(grid != background)
    boxes = []
    patterns = set()
    nonrectangular = False
    multilevel = False
    overlap = False
    for part in parts:
        rows = [r for r, _ in part]
        cols = [c for _, c in part]
        top, bottom = min(rows), max(rows) + 1
        left, right = min(cols), max(cols) + 1
        boxes.append((top, bottom, left, right))
        nonrectangular |= len(part) != (bottom - top) * (right - left)
        marked_rows = {}
        for row, col in part:
            if grid[row, col] == marker:
                marked_rows.setdefault(row, set()).add(col)
        multilevel |= len(marked_rows) >= 2
        groups = list(marked_rows.values())
        overlap |= any(a & b for i, a in enumerate(groups)
                       for b in groups[i + 1:])
        for row, marked in marked_rows.items():
            available = {c for r, c in part if r == row}
            if len(marked) == len(available):
                patterns.add("full")
            elif len(marked) == 1:
                patterns.add("single")
            elif max(marked) - min(marked) + 1 == len(marked):
                patterns.add("short_run")
            else:
                patterns.add("irregular_gap")
    terminal = any(row == grid.shape[0] - 1 or
                   grid[row + 1, col] == background
                   for row, col in np.argwhere(grid == marker))
    two_long = (len(boxes) == 2 and all(
        bottom == grid.shape[0] and bottom - top >= 6 and right - left <= 4
        for top, bottom, left, right in boxes))
    return (len(parts), nonrectangular, multilevel, overlap,
            terminal, two_long, patterns)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen = set()
    pairs = train + test
    changes = set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        grid, output = pair["input"], pair["output"]
        for item in (grid, output):
            assert isinstance(item, np.ndarray) and item.ndim == 2
            assert 1 <= item.shape[0] <= 30 and 1 <= item.shape[1] <= 30
            assert np.issubdtype(item.dtype, np.integer)
            assert np.all((item >= 0) & (item <= 9))
        assert grid.shape == output.shape
        key = _key(grid)
        assert key not in seen
        seen.add(key)
        assert tuple("".join(map(str, row)) for row in grid) not in SOURCE_INPUTS
        before, after = np.unique(grid, return_counts=True), np.unique(output)
        assert len(before[0]) == 3 and set(before[0]) == set(after)
        assert np.any(grid != output)
        changes.update(int(v) for v in np.unique(output[grid != output]))
    assert len(changes) == 1
    marker = changes.pop()
    shared = set(np.unique(train[0]["input"])) & set(np.unique(train[1]["input"]))
    assert marker in shared and len(shared) == 2
    background = next(int(v) for v in shared if v != marker)
    train_fills = set()
    has_gap_train = False
    has_gap_test = False
    for split, group in (("source_train", train), ("target_test", test)):
        for pair in group:
            grid, output = pair["input"], pair["output"]
            colors = set(int(v) for v in np.unique(grid))
            assert background in colors and marker in colors
            fill = next(iter(colors - {background, marker}))
            assert np.array_equal(output, _scan_prediction(grid, fill, marker, background))
            assert np.all(grid[grid == background] == output[grid == background])
            assert np.all(grid[grid == marker] == output[grid == marker])
            assert np.all(grid[grid != output] == fill)
            assert 1 <= len(_components(grid != background)) <= 3
            if split == "source_train":
                train_fills.add(fill)
                has_gap_train |= _gap_evidence(grid, fill, marker, background)
            else:
                has_gap_test |= _gap_evidence(grid, fill, marker, background)
    assert len(train_fills) >= 2
    first_fill = next(iter(set(np.unique(train[0]["input"])) - {background, marker}))
    assert _direction_evidence(train[0], first_fill, marker)
    assert not has_gap_test or has_gap_train


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, structures, roles = set(), set(), set()
    counts_by_split = {"train": set(), "test": set()}
    patterns = {"train": set(), "test": set()}
    nonrect = {"train": 0, "test": 0}
    two_long = {"train": 0, "test": 0}
    gaps, terminal, multilevel, overlap = 0, 0, 0, 0
    reused, new_fill, varied_regions, combined = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        marker = int(np.unique(puzzle["train"][0]["output"][
            puzzle["train"][0]["input"] != puzzle["train"][0]["output"]])[0])
        shared = (set(np.unique(puzzle["train"][0]["input"])) &
                  set(np.unique(puzzle["train"][1]["input"])))
        background = next(int(c) for c in shared if c != marker)
        roles.add((background, marker))
        train_fills = {next(iter(set(np.unique(p["input"])) -
                                 {background, marker})) for p in puzzle["train"]}
        region_counts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                shapes.add(grid.shape)
                fill = next(iter(set(np.unique(grid)) - {background, marker}))
                mask = grid != background
                ys, xs = np.nonzero(mask)
                crop = grid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                normalized = tuple(0 if v == background else
                                   1 if v == fill else 2 for v in crop.flat)
                structures.add((crop.shape, normalized))
                number, irregular, levels, overlapping, inert, long, kinds = (
                    _visible_features(grid, background, marker))
                counts_by_split[split].add(number)
                region_counts.add(number)
                patterns[split].update(kinds)
                nonrect[split] += irregular
                two_long[split] += long
                gaps += _gap_evidence(grid, fill, marker, background)
                terminal += inert
                multilevel += levels
                overlap += overlapping
                combined += irregular and levels
                if split == "test":
                    reused += fill in train_fills
                    new_fill += fill not in train_fills
        varied_regions += len(region_counts) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(structures) >= 100
    assert len(roles) >= 20 and len({a for a, _ in roles}) >= 6
    assert len({b for _, b in roles}) >= 6
    assert counts_by_split["train"] == counts_by_split["test"] == {1, 2, 3}
    assert patterns["train"] == patterns["test"] == {
        "full", "single", "short_run", "irregular_gap"}
    assert min(nonrect.values()) >= 10 and min(two_long.values()) >= 2
    assert gaps >= 10 and terminal >= 10 and multilevel >= 20
    assert overlap >= 10 and combined >= 10 and varied_regions >= 20
    assert reused >= 5 and new_fill >= 5
    for seed, a, b in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                       (103, None, 1), (104, None, 4), (105, 2, None),
                       (106, 10, None), (107, 6, 3)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed, a, b))
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(123, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
