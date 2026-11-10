"""ARC generator: select the filled rectangle with most signal cells."""

import hashlib

import numpy as np


_SOURCE_HASHES = {
    "b202c94aeb53a19226ad0a7f721af5c3b58ef8e56fc2b866b52d45dd72616238",
    "67496ee4350d71deab0a8619c627dc380a5a301bd9100118b389fbd70e16615b",
    "9d418563abe37b596c685038ac9a8b52b6c6c7bd2b77ddcb597",
    "d323df52c8244d71515edbd84f4666dee926307d5a0d2a361819698fdd6804dc",
}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _count("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + (seed // 3) % (10 - num_test)
    elif num_test is None:
        num_test = 1 + (seed // 3) % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _shape(rect):
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _area(rect):
    h, w = _shape(rect)
    return h * w


def _separated(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 2 <= br0 or br1 + 2 <= ar0 or ac1 + 2 <= bc0 or bc1 + 2 <= ac0


def _layout(rng, height, width, count):
    """Recursively partition a canvas with a one-cell background gap."""
    inset = int(rng.random() < 0.22)
    boxes = [(inset, inset, height - 1 - inset, width - 1 - inset)]
    while len(boxes) < count:
        options = []
        for index, box in enumerate(boxes):
            h, w = _shape(box)
            if h >= 7:
                options.append((index, 0))
            if w >= 7:
                options.append((index, 1))
        if not options:
            return None
        index, axis = options[int(rng.integers(len(options)))]
        r0, c0, r1, c1 = boxes.pop(index)
        if axis == 0:
            gap = int(rng.integers(r0 + 3, r1 - 2))
            boxes.extend(((r0, c0, gap - 1, c1), (gap + 1, c0, r1, c1)))
        else:
            gap = int(rng.integers(c0 + 3, c1 - 2))
            boxes.extend(((r0, c0, r1, gap - 1), (r0, gap + 1, r1, c1)))
    rects = []
    for r0, c0, r1, c1 in boxes:
        shrink = []
        for size in (r1 - r0 + 1, c1 - c0 + 1):
            budget = min(4, size - 3)
            total = int(rng.integers(budget + 1)) if rng.random() < 0.7 else 0
            first = int(rng.integers(total + 1))
            shrink.append((first, total - first))
        rects.append((r0 + shrink[0][0], c0 + shrink[1][0],
                      r1 - shrink[0][1], c1 - shrink[1][1]))
    return rects


def _equalize_pair(rng, rects):
    options = []
    for a in range(len(rects)):
        ah, aw = _shape(rects[a])
        for b in range(a):
            bh, bw = _shape(rects[b])
            by_area = {}
            for h in range(3, ah + 1):
                for w in range(3, aw + 1):
                    by_area.setdefault(h * w, []).append((h, w))
            for h in range(3, bh + 1):
                for w in range(3, bw + 1):
                    area = h * w
                    if area >= 18 and area in by_area:
                        for first in by_area[area]:
                            if first != (h, w):
                                options.append((a, b, first, (h, w)))
    if options:
        # Favor the source's equal-area, different-aspect comparison.
        a, b, shape_a, shape_b = options[int(rng.integers(len(options)))]
    else:
        a, b = (int(x) for x in rng.choice(len(rects), size=2, replace=False))
        ah, aw = _shape(rects[a])
        bh, bw = _shape(rects[b])
        shape_a = shape_b = (min(ah, bh), min(aw, bw))
    for index, (h, w) in ((a, shape_a), (b, shape_b)):
        r0, c0, r1, c1 = rects[index]
        top = int(rng.integers(r0, r1 - h + 2))
        left = int(rng.integers(c0, c1 - w + 2))
        rects[index] = (top, left, top + h - 1, left + w - 1)
    return a, b


def _count_plan(rng, rects, mode, equal_pair=None):
    areas = [_area(rect) for rect in rects]
    caps = [min(22, (area - 1) // 2) for area in areas]
    n = len(rects)
    candidates = []
    if mode in ("area", "quiet"):
        # Density agrees in both cases; area agrees only for the quiet case.
        for win in range(n):
            if mode == "area" and areas[win] >= max(areas):
                continue
            if mode == "quiet" and areas[win] <= max(areas[j] for j in range(n) if j != win):
                continue
            for k in range(4, caps[win] + 1):
                limits = [min(k - 1, caps[j], (k * areas[j] - 1) // areas[win])
                          for j in range(n)]
                if all(limits[j] >= 1 for j in range(n) if j != win):
                    candidates.append((win, k, limits, None))
    elif mode == "density":
        # A smaller foil has higher density but fewer absolute signal cells.
        for win in range(n):
            for foil in range(n):
                if areas[foil] >= areas[win]:
                    continue
                for k in range(4, caps[win] + 1):
                    lo = max(1, k * areas[foil] // areas[win] + 1)
                    hi = min(k - 1, caps[foil])
                    if lo <= hi:
                        candidates.append((win, k, foil, (lo, hi)))
    elif mode in ("equal", "close"):
        pairs = ((equal_pair[0], equal_pair[1]), (equal_pair[1], equal_pair[0])) if mode == "equal" else (
            (i, j) for i in range(n) for j in range(n) if i != j)
        for win, foil in pairs:
            for k in range(4, caps[win] + 1):
                if caps[foil] >= k - 1:
                    candidates.append((win, k, foil, None))
    else:
        for win in range(n):
            for k in range(4, caps[win] + 1):
                candidates.append((win, k, None, None))
    if not candidates:
        return None
    win, k, special, extra = candidates[int(rng.integers(len(candidates)))]
    counts = [0] * n
    counts[win] = k
    for j in range(n):
        if j == win:
            continue
        upper = special[j] if mode in ("area", "quiet") else min(k - 1, caps[j])
        if mode == "density" and j == special:
            counts[j] = int(rng.integers(extra[0], extra[1] + 1))
        elif mode in ("equal", "close") and j == special:
            counts[j] = k - 1
        else:
            counts[j] = int(rng.integers(1, upper + 1))
    return win, counts


def _signal_cells(rng, height, width, count, texture):
    rows, cols = np.indices((height, width))
    if texture == "scattered":
        score = rng.random((height, width))
    elif texture == "cluster":
        cr, cc = int(rng.integers(height)), int(rng.integers(width))
        score = abs(rows - cr) + abs(cols - cc) + rng.random((height, width)) * 0.5
    elif texture == "bar":
        if rng.random() < 0.5:
            line = int(rng.integers(height))
            score = abs(rows - line) * 4 + rng.random((height, width)) * 3
        else:
            line = int(rng.integers(width))
            score = abs(cols - line) * 4 + rng.random((height, width)) * 3
    elif texture == "edge":
        score = np.minimum.reduce((rows, cols, height - 1 - rows, width - 1 - cols))
        score = score + rng.random((height, width)) * 0.6
    else:
        period = int(rng.integers(2, 5))
        rp, cp = int(rng.integers(period)), int(rng.integers(period))
        score = ((rows - rp) % period + (cols - cp) % period).astype(float)
        score += rng.random((height, width)) * 0.6
    selected = np.argpartition(score.ravel(), count - 1)[:count]
    return selected // width, selected % width


def _render(rng, height, width, rects, counts, colors):
    background, carrier, signal = colors
    grid = np.full((height, width), background, dtype=np.int8)
    textures = ("scattered", "cluster", "bar", "edge", "spaced")
    for rect, count in zip(rects, counts):
        r0, c0, r1, c1 = rect
        ph, pw = _shape(rect)
        grid[r0:r1 + 1, c0:c1 + 1] = carrier
        rows, cols = _signal_cells(rng, ph, pw, count,
                                   textures[int(rng.integers(len(textures)))])
        grid[r0 + rows, c0 + cols] = signal
    return grid


def _region(rect, height, width):
    r0, c0, r1, c1 = rect
    row = (r0 + r1 + 1) / (2 * height)
    col = (c0 + c1 + 1) / (2 * width)
    return min(2, int(row * 3)), min(2, int(col * 3))


def _new_pair(rng, colors, mode, forbidden_region=None, forbidden_count=None):
    for _ in range(240):
        choices = [n for n in (3, 4, 5) if n != forbidden_count]
        count = choices[int(rng.integers(len(choices)))]
        lower = {3: 15, 4: 18, 5: 21}[count]
        if rng.random() < 0.18 and lower <= 20:
            height = width = 20
        else:
            height = int(rng.integers(lower, 31))
            width = int(rng.integers(lower, 31))
        rects = _layout(rng, height, width, count)
        if rects is None:
            continue
        equal_pair = _equalize_pair(rng, rects) if mode == "equal" else None
        plan = _count_plan(rng, rects, mode, equal_pair)
        if plan is None:
            continue
        if sum(_area(rect) for rect in rects) > 0.70 * height * width:
            continue
        winner, counts = plan
        if _region(rects[winner], height, width) == forbidden_region:
            continue
        grid = _render(rng, height, width, rects, counts, colors)
        r0, c0, r1, c1 = rects[winner]
        return {"input": grid, "output": grid[r0:r1 + 1, c0:c1 + 1].copy()}
    raise RuntimeError(f"could not construct {mode} pair")


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _is_source_input(grid):
    return grid.shape == (20, 20) and hashlib.sha256(
        grid.astype(np.int8).tobytes()).hexdigest() in _SOURCE_HASHES


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_palette = bool(rng.integers(2))
    shared_colors = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    puzzle = {"train": [], "test": []}
    used = set()
    first_region = None
    previous_colors = None
    train_modes = ["area", "density"] + [
        "quiet", "equal", "close", "free", "area", "density", "quiet", "equal"][:ntrain - 2]
    target_modes = ("equal", "close", "density", "free", "quiet")
    test_modes = [target_modes[int(rng.integers(len(target_modes)))]
                  for _ in range(ntest)]
    for section, modes in (("train", train_modes), ("test", test_modes)):
        for index, mode in enumerate(modes):
            for _ in range(30):
                if fixed_palette:
                    colors = shared_colors
                else:
                    colors = tuple(int(x) for x in rng.choice(10, 3, replace=False))
                    if colors == previous_colors:
                        continue
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                forbidden = first_region if section == "train" and index == 1 else None
                forbidden_count = (len(_infer_pair(puzzle["train"][0]["input"])["rects"])
                                   if section == "train" and index == 1 else None)
                pair = _new_pair(pair_rng, colors, mode, forbidden, forbidden_count)
                key = _grid_key(pair["input"])
                if key not in used and not _is_source_input(pair["input"]):
                    used.add(key)
                    puzzle[section].append(pair)
                    previous_colors = colors
                    if section == "train" and index == 0:
                        analysis = _infer_pair(pair["input"])
                        first_region = _region(analysis["rects"][analysis["winner"]],
                                               *pair["input"].shape)
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width and not seen[nr, nc]
                            and int(grid[nr, nc]) != background):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _infer_pair(grid):
    colors = [int(x) for x in np.unique(grid)]
    assert len(colors) == 3
    interpretations = []
    for background in colors:
        components = _components(grid, background)
        if not 3 <= len(components) <= 5:
            continue
        rects = []
        for cells in components:
            rows, cols = zip(*cells)
            rect = min(rows), min(cols), max(rows), max(cols)
            r0, c0, r1, c1 = rect
            crop = grid[r0:r1 + 1, c0:c1 + 1]
            if len(cells) != crop.size or background in crop:
                break
            rects.append(rect)
        if len(rects) != len(components):
            continue
        if any(not _separated(rects[i], rects[j])
               for i in range(len(rects)) for j in range(i)):
            continue
        roles = [x for x in colors if x != background]
        for signal in roles:
            carrier = next(x for x in roles if x != signal)
            counts = []
            for r0, c0, r1, c1 in rects:
                crop = grid[r0:r1 + 1, c0:c1 + 1]
                signal_count = int(np.count_nonzero(crop == signal))
                carrier_count = int(np.count_nonzero(crop == carrier))
                if not (1 <= signal_count < carrier_count):
                    break
                counts.append(signal_count)
            if len(counts) == len(rects) and counts.count(max(counts)) == 1:
                interpretations.append({"background": background, "carrier": carrier,
                                        "signal": signal, "rects": rects, "counts": counts,
                                        "areas": [_area(rect) for rect in rects],
                                        "winner": counts.index(max(counts))})
    assert len(interpretations) == 1, "input must reveal unique color roles and panels"
    return interpretations[0]


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            _check_grid(grid)
            _check_grid(output)
            key = _grid_key(grid)
            assert key not in used
            used.add(key)
            assert not _is_source_input(grid)
            analysis = _infer_pair(grid)
            assert np.count_nonzero(grid == analysis["background"]) >= 0.30 * grid.size
            assert set(int(x) for x in np.unique(output)) == {
                analysis["carrier"], analysis["signal"]}
            r0, c0, r1, c1 = analysis["rects"][analysis["winner"]]
            assert np.array_equal(output, grid[r0:r1 + 1, c0:c1 + 1])
            assert grid.shape != output.shape
            analyses[section].append(analysis)
    area_refutation = False
    density_refutation = False
    regions = set()
    for pair, analysis in zip(puzzle["train"], analyses["train"]):
        win = analysis["winner"]
        wa, wc = analysis["areas"][win], analysis["counts"][win]
        for j, (area, count) in enumerate(zip(analysis["areas"], analysis["counts"])):
            if j != win:
                area_refutation |= area > wa
                density_refutation |= count * wa > wc * area
        regions.add(_region(analysis["rects"][win], *pair["input"].shape))
    assert area_refutation and density_refutation
    assert len(regions) >= 2
    assert len({len(analysis["rects"]) for analysis in analyses["train"]}) >= 2


def _puzzle_key(puzzle):
    return tuple((section, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    pairs = [pair for puzzle in puzzles for section in ("train", "test")
             for pair in puzzle[section]]
    assert len({pair["input"].shape for pair in pairs}) >= 25
    assert len({pair["output"].shape for pair in pairs}) >= 25
    assert any(pair["input"].shape == (20, 20) for pair in pairs)
    assert any(any(_infer_pair(pair["input"])["background"] != int(x)
                   for x in pair["input"][0]) for pair in pairs)
    assert len({_infer_pair(pair["input"])["signal"] for pair in pairs}) >= 8
    assert {len(_infer_pair(pair["input"])["rects"]) for pair in pairs} == {3, 4, 5}
    palette_counts = [
        len({tuple(_infer_pair(pair["input"])[role]
                   for role in ("background", "carrier", "signal"))
             for section in ("train", "test") for pair in puzzle[section]})
        for puzzle in puzzles]
    assert sum(count == 1 for count in palette_counts) >= 8
    assert sum(count > 1 for count in palette_counts) >= 8
    observed = {"train_regions": set(), "test_regions": set(),
                "train_counts": set(), "test_counts": set(), "margins": set(),
                "borders": set(), "test_shapes": set(), "tall_sides": set(),
                "thin_sides": set()}
    totals = {key: 0 for key in ("interior_pairs", "larger_train", "denser_train", "area_agrees",
                                   "density_agrees", "equal_test", "hard_test",
                                   "quiet_test", "adjacent", "edge", "line", "spaced")}
    for puzzle in puzzles:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                a = _infer_pair(grid)
                win = a["winner"]
                wr = a["rects"][win]
                wa, wc = a["areas"][win], a["counts"][win]
                observed[section + "_regions"].add(_region(wr, *grid.shape))
                observed[section + "_counts"].add(len(a["rects"]))
                observed["margins"].add(wc - max(c for j, c in enumerate(a["counts"]) if j != win))
                totals["interior_pairs"] += all(
                    r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1
                    for r0, c0, r1, c1 in a["rects"])
                if section == "test":
                    observed["test_shapes"].add(pair["output"].shape)
                for rect in a["rects"]:
                    r0, c0, r1, c1 = rect
                    if r0 == 0:
                        observed["borders"].add("top")
                    if c0 == 0:
                        observed["borders"].add("left")
                    if r1 == grid.shape[0] - 1:
                        observed["borders"].add("bottom")
                    if c1 == grid.shape[1] - 1:
                        observed["borders"].add("right")
                    crop = grid[r0:r1 + 1, c0:c1 + 1]
                    mask = crop == a["signal"]
                    adjacent = int(np.sum(mask[:, 1:] & mask[:, :-1]) +
                                   np.sum(mask[1:, :] & mask[:-1, :]))
                    signal_count = int(mask.sum())
                    edge_count = int(mask[0].sum() + mask[-1].sum() +
                                     mask[:, 0].sum() + mask[:, -1].sum() -
                                     mask[0, 0] - mask[0, -1] -
                                     mask[-1, 0] - mask[-1, -1])
                    totals["adjacent"] += adjacent >= max(2, signal_count // 3)
                    totals["edge"] += signal_count >= 4 and edge_count / signal_count >= 0.75
                    totals["line"] += signal_count >= 4 and max(
                        mask.sum(axis=0).max(), mask.sum(axis=1).max()) / signal_count >= 0.6
                    totals["spaced"] += signal_count >= 4 and adjacent == 0
                larger = any(j != win and area > wa for j, area in enumerate(a["areas"]))
                denser = any(j != win and count * wa > wc * a["areas"][j]
                             for j, count in enumerate(a["counts"]))
                if section == "train":
                    totals["larger_train"] += larger
                    totals["denser_train"] += denser
                    totals["area_agrees"] += not larger
                    totals["density_agrees"] += not denser
                else:
                    totals["hard_test"] += larger and denser
                    totals["quiet_test"] += not larger and not denser
                    totals["equal_test"] += any(j != win and area == wa and
                        _shape(a["rects"][j]) != _shape(wr)
                        for j, area in enumerate(a["areas"]))
                    wy, wx = wr[0] + wr[2], wr[1] + wr[3]
                    for j, rect in enumerate(a["rects"]):
                        if j == win:
                            continue
                        ph, pw = _shape(rect)
                        kind = "tall" if ph >= 2 * pw else "thin" if pw >= 2 * ph else None
                        if kind is None:
                            continue
                        dy, dx = rect[0] + rect[2] - wy, rect[1] + rect[3] - wx
                        side = ("below" if dy > 0 else "above") if abs(dy) >= abs(dx) else (
                            "right" if dx > 0 else "left")
                        observed[kind + "_sides"].add(side)
    assert len(observed["train_regions"]) == len(observed["test_regions"]) == 9
    assert observed["train_counts"] == observed["test_counts"] == {3, 4, 5}
    assert observed["borders"] == {"top", "bottom", "left", "right"}
    assert len(observed["margins"]) >= 8
    assert any(h == w for h, w in observed["test_shapes"])
    assert any(w >= 3 * h for h, w in observed["test_shapes"])
    assert any(h >= 3 * w for h, w in observed["test_shapes"])
    assert observed["tall_sides"] == observed["thin_sides"] == {
        "above", "below", "left", "right"}
    assert all(totals[key] >= 5 for key in ("interior_pairs", "larger_train", "denser_train",
                                            "area_agrees", "density_agrees", "equal_test",
                                            "hard_test", "quiet_test"))
    assert all(totals[key] >= 50 for key in ("adjacent", "edge", "line", "spaced"))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(100 + ntrain * 10 + ntest, ntrain, ntest)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
    for ntrain, ntest in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(233, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    assert _puzzle_key(generate_puzzle(777, 7, 3)) == _puzzle_key(generate_puzzle(777, 7, 3))
    assert _puzzle_key(generate_puzzle(778, 7, 3)) != _puzzle_key(generate_puzzle(777, 7, 3))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
