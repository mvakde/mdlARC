"""Connected-glyph extraction puzzles with dispersed, sometimes clustered noise."""

import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "9b4fd4dac47528fe85b15fc8c9b7cdbf858059565a7f5054779d458ea5426d57",
    "d04ed4fce780147e76bfa503c625cd94c82439cbc92aec53431a4b0a824b1c76",
    "84b0417b813ded0a375f3cbb7e8cae52ce8163b846ddebc6477dd4e8d9ce9ddc",
}
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count_arg("num_train", num_train, 2, 10)
    test = _count_arg("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + seed % 9)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape
    for rr, cc in np.argwhere(mask):
        row, col = int(rr), int(cc)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in _NEIGHBORS:
                y, x = r + dr, c + dc
                if 0 <= y < height and 0 <= x < width and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        result.append(component)
    return result


def _holes(mask):
    """Background cells enclosed by a glyph, in local coordinates."""
    open_cells = ~mask
    exterior = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    stack = []
    for row in range(height):
        for col in (0, width - 1):
            if open_cells[row, col] and not exterior[row, col]:
                exterior[row, col] = True
                stack.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if open_cells[row, col] and not exterior[row, col]:
                exterior[row, col] = True
                stack.append((row, col))
    while stack:
        row, col = stack.pop()
        for dr, dc in _NEIGHBORS:
            y, x = row + dr, col + dc
            if 0 <= y < height and 0 <= x < width and open_cells[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                stack.append((y, x))
    return open_cells & ~exterior


def _make_mask(height, width, mode, rng):
    if mode == "solid":
        return np.ones((height, width), dtype=bool)
    if mode == "ring":
        mask = np.ones((height, width), dtype=bool)
        candidates = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
        rng.shuffle(candidates)
        chosen = []
        target_holes = int(rng.integers(1, min(4, len(candidates) + 1)))
        for cell in candidates:
            if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == target_holes:
                    break
        for r, c in chosen:
            mask[r, c] = False
        return mask
    area = height * width
    for _ in range(240):
        density = float(rng.uniform(0.42, 0.83))
        desired = min(area - 1, max(6, int(round(area * density))))
        mask = np.zeros((height, width), dtype=bool)
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
        while int(mask.sum()) < desired:
            frontier = np.zeros(mask.shape, dtype=bool)
            frontier[1:] |= mask[:-1]
            frontier[:-1] |= mask[1:]
            frontier[:, 1:] |= mask[:, :-1]
            frontier[:, :-1] |= mask[:, 1:]
            options = np.argwhere(frontier & ~mask)
            chosen = options[int(rng.integers(len(options)))]
            mask[int(chosen[0]), int(chosen[1])] = True
        if (mask[0].any() and mask[-1].any() and mask[:, 0].any()
                and mask[:, -1].any() and not _holes(mask).any()):
            return mask
    # Bounded fallback: a connected border with a concavity.
    mask = np.ones((height, width), dtype=bool)
    mask[0, 0] = False
    return mask


def _canvas(rng):
    if rng.random() < 0.12:
        return 30, 30
    kind = int(rng.integers(3))
    if kind == 0:
        return int(rng.integers(14, 31)), int(rng.integers(14, 31))
    if kind == 1:
        return int(rng.integers(23, 31)), int(rng.integers(14, 22))
    return int(rng.integers(14, 22)), int(rng.integers(23, 31))


def _shape(rng, area, regime, mode):
    choices = []
    for height in range(2, 8):
        for width in range(2, 8):
            if height * width < 6 or height * width > min(42, int(area * 0.18)):
                continue
            if mode == "growth" and height * width == 6:
                continue
            if mode == "ring" and (height < 3 or width < 3):
                continue
            if max(height, width) / min(height, width) > 3.5:
                continue
            weight = 1 + (height * width >= 18) * (regime == "target_test")
            choices.extend([(height, width)] * weight)
    return choices[int(rng.integers(len(choices)))]


def _sample_plan(rng, regime, shared_canvas):
    canvas = shared_canvas if shared_canvas is not None else _canvas(rng)
    mode = str(rng.choice(("growth", "ring", "solid"), p=(0.65, 0.28, 0.07)))
    shape = _shape(rng, canvas[0] * canvas[1], regime, mode)
    if mode == "solid":
        internal = "clean"
    elif mode == "ring":
        internal = str(rng.choice(("clean", "hole"), p=(0.48, 0.52)))
    else:
        internal = str(rng.choice(("clean", "indent"), p=(0.59, 0.41)))
    if regime == "source_train":
        noise_colors = int(rng.choice((1, 2, 3, 4), p=(0.34, 0.36, 0.24, 0.06)))
        density = int(rng.choice((0, 1, 2), p=(0.32, 0.47, 0.21)))
    else:
        noise_colors = int(rng.choice((1, 2, 3, 4), p=(0.19, 0.31, 0.32, 0.18)))
        density = int(rng.choice((0, 1, 2), p=(0.16, 0.43, 0.41)))
    return {"canvas": canvas, "shape": shape, "mode": mode, "internal": internal,
            "noise_colors": noise_colors, "density": density}


def _plans(rng, train_count, test_count, shared_canvas):
    train = [_sample_plan(rng, "source_train", shared_canvas) for _ in range(train_count)]
    test = [_sample_plan(rng, "target_test", shared_canvas) for _ in range(test_count)]
    if all(plan["mode"] == "solid" for plan in train):
        train[0]["mode"] = "growth"
        train[0]["shape"] = _shape(rng, np.prod(train[0]["canvas"]),
                                   "source_train", "growth")
    if any(plan["mode"] == "ring" for plan in test) and not any(p["mode"] == "ring" for p in train):
        index = int(rng.integers(train_count))
        train[index]["mode"] = "ring"
        train[index]["shape"] = _shape(rng, np.prod(train[index]["canvas"]), "source_train", "ring")
        train[index]["internal"] = str(rng.choice(("clean", "hole")))
    if any(plan["internal"] != "clean" for plan in test) and not any(p["internal"] != "clean" for p in train):
        index = int(rng.integers(train_count))
        if train[index]["mode"] == "solid":
            train[index]["mode"] = "growth"
            train[index]["shape"] = _shape(rng, np.prod(train[index]["canvas"]),
                                           "source_train", "growth")
        train[index]["internal"] = "hole" if train[index]["mode"] == "ring" else "indent"
    return train, test


def _pick_background(rng, palette, stable):
    if stable is not None:
        return stable
    if rng.random() < 0.46:
        return 0
    return int(rng.choice([color for color in palette if color != 0]))


def _pick_target(rng, palette, background, history, split, index):
    allowed = [c for c in palette if c != background]
    if split == "train" and index == 1:
        allowed = [c for c in allowed if c != history[0]["target"]]
    previous_targets = [x["target"] for x in history if x["target"] in allowed]
    previous_noise = [c for x in history for c in x["noise"] if c in allowed]
    unseen = [c for c in allowed if all(c != x["target"] and c not in x["noise"] for x in history)]
    route = int(rng.choice(4, p=(0.25, 0.25, 0.35, 0.15)))
    group = (previous_targets, previous_noise, unseen, allowed)[route]
    if not group:
        group = allowed
    if 0 in group and rng.random() < 0.22:
        return 0
    return int(rng.choice(group))


def _noise_palette(rng, palette, background, target, count, history):
    allowed = [c for c in palette if c not in (background, target)]
    chosen = []
    previous_targets = [x["target"] for x in history]
    if previous_targets and rng.random() < 0.52:
        options = [c for c in previous_targets if c in allowed]
        if options:
            chosen.append(int(rng.choice(options)))
    if len(chosen) == count:
        return chosen
    rng.shuffle(allowed)
    for color in allowed:
        if color not in chosen:
            chosen.append(color)
        if len(chosen) == count:
            break
    return chosen


def _noise_counts(rng, area, target_size, count, level):
    ranges = ((0.005, 0.040), (0.065, 0.115), (0.13, 0.22))
    low, high = ranges[level]
    desired = int(round(area * float(rng.uniform(low, high))))
    ceiling = min(int(area * 0.30) - target_size, count * min(52, int(area * 0.10) + 8))
    total = max(3 * count, min(desired, ceiling))
    amounts = [3] * count
    cap = min(52, int(area * 0.10) + 8)
    for _ in range(total - 3 * count):
        options = [i for i in range(count) if amounts[i] < cap]
        if not options:
            break
        amounts[int(rng.choice(options))] += 1
    rng.shuffle(amounts)
    return amounts


def _adjacent_to_color(row, col, color_cells, height, width, cluster):
    for dr, dc in _NEIGHBORS:
        y, x = row + dr, col + dc
        if 0 <= y < height and 0 <= x < width and (y, x) in color_cells and (y, x) not in cluster:
            return True
    return False


def _place_noise(grid, background, color, count, rng, box, required):
    height, width = grid.shape
    r0, c0, gh, gw = box
    color_cells = set()

    def free(row, col, cluster):
        return (grid[row, col] == background and (row, col) not in color_cells
                and (row, col) not in cluster
                and not _adjacent_to_color(row, col, color_cells, height, width, cluster))

    for row, col in required:
        if not free(row, col, set()):
            raise RuntimeError("internal noise cell unavailable")
        color_cells.add((row, col))

    band_h, band_w = max(2, height // 4), max(2, width // 4)
    if int(rng.integers(2)):
        regions = ((0, band_h, 0, band_w), (height - band_h, height, width - band_w, width))
    else:
        regions = ((0, band_h, width - band_w, width), (height - band_h, height, 0, band_w))
    for rlo, rhi, clo, chi in regions:
        options = [(r, c) for r in range(rlo, rhi) for c in range(clo, chi)]
        rng.shuffle(options)
        anchor = next(((r, c) for r, c in options
                       if not (r0 <= r < r0 + gh and c0 <= c < c0 + gw)
                       and free(r, c, set())), None)
        if anchor is None:
            raise RuntimeError("noise anchor unavailable")
        color_cells.add(anchor)

    while len(color_cells) < count:
        left = count - len(color_cells)
        size = min(left, int(rng.choice((1, 2, 3, 4), p=(0.39, 0.31, 0.20, 0.10))))
        placed = False
        for trial_size in range(size, 0, -1):
            for _ in range(300):
                row, col = int(rng.integers(height)), int(rng.integers(width))
                if r0 <= row < r0 + gh and c0 <= col < c0 + gw:
                    continue
                if not free(row, col, set()):
                    continue
                cluster = {(row, col)}
                while len(cluster) < trial_size:
                    frontier = set()
                    for r, c in cluster:
                        for dr, dc in _NEIGHBORS:
                            y, x = r + dr, c + dc
                            if (0 <= y < height and 0 <= x < width
                                    and not (r0 <= y < r0 + gh and c0 <= x < c0 + gw)
                                    and free(y, x, cluster)):
                                frontier.add((y, x))
                    if not frontier:
                        break
                    cells = sorted(frontier)
                    cluster.add(cells[int(rng.integers(len(cells)))])
                if len(cluster) == trial_size:
                    color_cells.update(cluster)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError("noise placement exhausted")
    for row, col in color_cells:
        grid[row, col] = color


def _crop(grid, target, background):
    coords = np.argwhere(grid == target)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    crop = grid[r0:r1, c0:c1]
    output = np.full(crop.shape, background, dtype=np.int8)
    output[crop == target] = target
    return output


def _make_pair(rng, plan, background, target, noise_colors):
    height, width = plan["canvas"]
    gh, gw = plan["shape"]
    mask = _make_mask(gh, gw, plan["mode"], rng)
    r0 = int(rng.integers(0, height - gh + 1))
    c0 = int(rng.integers(0, width - gw + 1))
    grid = np.full((height, width), background, dtype=np.int8)
    region = grid[r0:r0 + gh, c0:c0 + gw]
    region[mask] = target
    internal = plan["internal"]
    required = []
    if internal != "clean":
        hole_cells = _holes(mask)
        blank = hole_cells if internal == "hole" else (~mask & ~hole_cells)
        candidates = np.argwhere(blank)
        if len(candidates) == 0:
            raise RuntimeError("requested internal clutter lacks a cell")
        cell = candidates[int(rng.integers(len(candidates)))]
        required = [(r0 + int(cell[0]), c0 + int(cell[1]))]
    counts = _noise_counts(rng, height * width, int(mask.sum()),
                           len(noise_colors), plan["density"])
    for index, (color, count) in enumerate(zip(noise_colors, counts)):
        _place_noise(grid, background, color, count, rng, (r0, c0, gh, gw),
                     required if index == 0 else [])
    return {"input": grid, "output": _crop(grid, target, background)}


def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.uint8).tobytes()


def _source_digest(inp, out):
    a = inp.astype(np.uint8)
    b = out.astype(np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + bytes(b.shape) + b.tobytes()).hexdigest()


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    values, counts = np.unique(inp, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be unique")
    background = int(values[np.argmax(counts)])
    _require(int(counts.max()) >= int(0.68 * inp.size), "background must dominate")
    colors = [int(c) for c in values if int(c) != background]
    _require(2 <= len(colors) <= 5, "need a target and one to four noise colors")
    components = {c: _components(inp == c) for c in colors}
    candidates = [c for c in colors if len(components[c]) == 1 and len(components[c][0]) >= 6]
    _require(len(candidates) == 1, "unique connected glyph absent")
    target = candidates[0]
    coords = np.argwhere(inp == target)
    r0, c0 = [int(v) for v in coords.min(axis=0)]
    r1, c1 = [int(v) + 1 for v in coords.max(axis=0)]
    gh, gw = r1 - r0, c1 - c0
    _require(2 <= gh <= 7 and 2 <= gw <= 7, "glyph dimensions invalid")
    _require(len(coords) >= 6 and len(coords) / (gh * gw) >= 0.38, "glyph too thin")
    expected = np.full((gh, gw), background, dtype=out.dtype)
    expected[inp[r0:r1, c0:c1] == target] = target
    _require(np.array_equal(out, expected), "output is not the exact target crop")
    _require(out.dtype == inp.dtype, "grid dtypes differ")
    for color in colors:
        if color == target:
            continue
        parts = components[color]
        _require(len(parts) >= 2 and max(map(len, parts)) <= 4,
                 "noise has a competing connected component")
        points = np.argwhere(inp == color)
        _require(np.ptp(points[:, 0]) >= inp.shape[0] * 0.38
                 and np.ptp(points[:, 1]) >= inp.shape[1] * 0.38,
                 "noise color is not dispersed")
    _require(_source_digest(inp, out) not in _SOURCE_PAIR_DIGESTS,
             "pair duplicates an original source pair")
    crop = inp[r0:r1, c0:c1]
    hole = _holes(out == target)
    internal = bool(np.any((crop != background) & (crop != target)))
    return {"background": background, "target": target, "noise": tuple(c for c in colors if c != target),
            "shape": (gh, gw), "canvas": inp.shape, "origin": (r0, c0),
            "hole": bool(hole.any()), "internal": internal,
            "solid": bool(np.all(out == target)),
            "target_count": len(coords), "noise_counts": tuple(int(np.count_nonzero(inp == c))
                                                               for c in colors if c != target),
            "noise_max": max(len(part) for c in colors if c != target for part in components[c])}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys invalid")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "pair counts invalid")
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys invalid")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid dimensions invalid")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype invalid")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid colors invalid")
            key = _grid_key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(pair["input"].shape != pair["output"].shape,
                     "extraction must reduce the canvas")
            analyses[split].append(_analyze(pair))
    _require(any(not a["solid"] for a in analyses["train"]),
             "training must show background restoration")
    if any(a["internal"] for a in analyses["test"]):
        _require(any(a["internal"] for a in analyses["train"]),
                 "training must demonstrate internal clutter removal")
    if any(a["hole"] for a in analyses["test"]):
        _require(any(a["hole"] for a in analyses["train"]),
                 "training must demonstrate hole preservation")
    _require(len({a["target"] for a in analyses["train"][:2]}) == 2,
             "first two training examples need distinct target colors")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    palette = list(dict.fromkeys([0] + palette[:int(rng.integers(8, 11))]))
    stable_background = None
    if rng.random() < 0.25:
        stable_background = _pick_background(rng, palette, None)
    shared_canvas = _canvas(rng) if rng.random() < 0.18 else None
    train_plans, test_plans = _plans(rng, train_count, test_count, shared_canvas)
    puzzle = {"train": [], "test": []}
    history = []
    seen = set()
    for split, plans in (("train", train_plans), ("test", test_plans)):
        for index, plan in enumerate(plans):
            background = _pick_background(rng, palette, stable_background)
            target = _pick_target(rng, palette, background, history, split, index)
            noise = _noise_palette(rng, palette, background, target, plan["noise_colors"], history)
            for _ in range(16):
                try:
                    pair = _make_pair(rng, plan, background, target, noise)
                    key = _grid_key(pair["input"])
                    if key in seen:
                        continue
                    _analyze(pair)
                    seen.add(key)
                    puzzle[split].append(pair)
                    history.append({"target": target, "noise": noise})
                    break
                except (RuntimeError, AssertionError):
                    continue
            else:
                raise RuntimeError(f"could not construct {split} pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[key], b[key])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for key in ("input", "output")) and all(
                   len(left[s]) == len(right[s]) for s in ("train", "test"))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(12345), generate_puzzle(12345)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(54321, 7, 3),
                            generate_puzzle(54321, 7, 3)),
             "explicit generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(54321, 7, 3),
                                generate_puzzle(54322, 7, 3)),
             "different seeds produced the same puzzle")
    for seed, nt, nv in ((201, 2, 1), (202, 10, 1), (203, 10, 4),
                         (301, None, 1), (302, None, 4),
                         (303, 2, None), (304, 10, None)):
        puzzle = generate_puzzle(seed, nt, nv)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (nv is None or len(puzzle["test"]) == nv),
                 "requested counts not honored")
        validate_generated_puzzle(puzzle)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(400, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {args}")
    train_counts, test_counts = set(), set()
    canvases, shapes, targets, backgrounds = set(), set(), set(), set()
    structures, global_inputs = set(), set()
    cluster_pairs = solid_pairs = hole_pairs = internal_pairs = clean_pairs = 0
    sparse_pairs = rich_pairs = large_pairs = small_pairs = 0
    test_target_larger = test_target_smaller = 0
    varied_canvas_puzzles = varied_background_puzzles = 0
    role_exchange_puzzles = 0
    constant_canvas_puzzles = constant_background_puzzles = 0
    new_test_target_puzzles = repeated_test_target_puzzles = 0
    different_multi_test_targets = combined_puzzles = 0
    internal_hole_pairs = internal_indent_pairs = several_hole_pairs = 0
    central_30_pairs = edge_pairs = zero_target_pairs = zero_noise_pairs = 0
    test_rarer_all = test_more_all = irregular_noise_clusters = 0
    noise_color_counts = {"train": set(), "test": set()}
    first_train_noise_totals = set()
    noise_component_sizes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_data, test_data = [], []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                info = _analyze(pair)
                (train_data if split == "train" else test_data).append(info)
                key = _grid_key(pair["input"])
                _require(key not in global_inputs, "cross-seed duplicate input")
                global_inputs.add(key)
                canvases.add(info["canvas"])
                shapes.add(info["shape"])
                targets.add(info["target"])
                backgrounds.add(info["background"])
                structures.add((info["canvas"], info["shape"],
                                tuple((pair["output"] == info["target"]).flat)))
                cluster_pairs += info["noise_max"] >= 2
                solid_pairs += info["solid"]
                hole_pairs += info["hole"]
                internal_pairs += info["internal"]
                clean_pairs += not info["internal"]
                noise_total = sum(info["noise_counts"])
                sparse_pairs += noise_total <= 8
                rich_pairs += noise_total >= 35
                large_pairs += min(info["canvas"]) >= 24
                small_pairs += min(info["canvas"]) <= 18
                noise_color_counts[split].add(len(info["noise"]))
                if split == "train" and index == 0:
                    first_train_noise_totals.add(noise_total)
                zero_target_pairs += info["target"] == 0
                zero_noise_pairs += 0 in info["noise"]
                edge_pairs += (
                    info["origin"][0] == 0 or info["origin"][1] == 0
                    or info["origin"][0] + info["shape"][0] == info["canvas"][0]
                    or info["origin"][1] + info["shape"][1] == info["canvas"][1])
                central_30_pairs += (info["canvas"] == (30, 30)
                                     and 8 <= info["origin"][0] <= 16
                                     and 9 <= info["origin"][1] <= 22)
                mask = pair["output"] == info["target"]
                holes = _holes(mask)
                several_hole_pairs += len(_components(holes)) >= 2
                for color in info["noise"]:
                    for part in _components(pair["input"] == color):
                        noise_component_sizes.add(len(part))
                        if len(part) >= 3:
                            rows = [r for r, _ in part]
                            cols = [c for _, c in part]
                            irregular_noise_clusters += (max(rows) > min(rows)
                                                         and max(cols) > min(cols))
                # Independently reconstruct the tight crop from the returned input.
                coords = np.argwhere(pair["input"] == info["target"])
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0) + 1
                crop = pair["input"][r0:r1, c0:c1]
                interior_noise = ((crop != info["background"])
                                  & (crop != info["target"]))
                internal_hole_pairs += bool(np.any(interior_noise & holes))
                internal_indent_pairs += bool(np.any(interior_noise & ~holes))
                expected = np.where(crop == info["target"], info["target"],
                                    info["background"])
                _require(np.array_equal(pair["output"], expected),
                         "independent crop check failed")
        varied_canvas_puzzles += len({a["canvas"] for a in train_data + test_data}) > 1
        varied_background_puzzles += len({a["background"] for a in train_data + test_data}) > 1
        constant_canvas_puzzles += len({a["canvas"] for a in train_data + test_data}) == 1
        constant_background_puzzles += len({a["background"] for a in train_data + test_data}) == 1
        role_exchange_puzzles += bool(
            {a["target"] for a in train_data}
            & {c for a in test_data for c in a["noise"]})
        test_target_larger += any(a["target_count"] > max(x["target_count"] for x in train_data)
                                  for a in test_data)
        test_target_smaller += any(a["target_count"] < min(x["target_count"] for x in train_data)
                                   for a in test_data)
        train_targets = {a["target"] for a in train_data}
        train_colors = train_targets | {c for a in train_data for c in a["noise"]}
        new_test_target_puzzles += any(a["target"] not in train_colors for a in test_data)
        repeated_test_target_puzzles += any(a["target"] in train_targets for a in test_data)
        different_multi_test_targets += len({a["target"] for a in test_data}) > 1
        combined_puzzles += (
            len({a["canvas"] for a in train_data + test_data}) > 1
            and len({a["background"] for a in train_data + test_data}) > 1
            and any(a["noise_max"] >= 3 for a in train_data + test_data)
            and any(a["internal"] for a in train_data + test_data)
            and len({a["shape"] for a in train_data + test_data}) > 1)
        for info in test_data:
            test_rarer_all += all(info["target_count"] < count for count in info["noise_counts"])
            test_more_all += all(info["target_count"] > count for count in info["noise_counts"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity weak")
    _require(len(canvases) >= 45 and len(shapes) >= 22, "dimension diversity weak")
    _require(len(targets) >= 8 and len(backgrounds) >= 7, "color diversity weak")
    _require(len(structures) >= 150, "structural diversity weak")
    _require(cluster_pairs >= 40 and solid_pairs >= 5 and hole_pairs >= 25,
             "glyph or cluster coverage weak")
    _require(internal_pairs >= 25 and clean_pairs >= 50,
             "crop contamination coverage weak")
    _require(sparse_pairs >= 15 and rich_pairs >= 15,
             "clutter density coverage weak")
    _require(large_pairs >= 10 and small_pairs >= 20,
             "canvas size coverage weak")
    _require(varied_canvas_puzzles >= 20 and varied_background_puzzles >= 20,
             "within-puzzle canvas/background variation weak")
    _require(role_exchange_puzzles >= 10 and test_target_larger >= 10
             and test_target_smaller >= 5, "role and train/test shift coverage weak")
    _require(noise_color_counts["train"] == {1, 2, 3, 4}
             and noise_color_counts["test"] == {1, 2, 3, 4},
             "both splits need the full noise palette range")
    _require(len(first_train_noise_totals) >= 10 and min(first_train_noise_totals) <= 8
             and max(first_train_noise_totals) >= 35,
             "first training clutter remains fixed or narrow")
    _require(noise_component_sizes == {1, 2, 3, 4}
             and irregular_noise_clusters >= 10, "noise cluster geometry weak")
    _require(several_hole_pairs >= 10 and internal_hole_pairs >= 10
             and internal_indent_pairs >= 10, "hole and indentation coverage weak")
    _require(central_30_pairs >= 2 and edge_pairs >= 20,
             "central or boundary placement coverage weak")
    _require(zero_target_pairs >= 5 and zero_noise_pairs >= 5,
             "zero is not changing semantic role")
    _require(test_rarer_all >= 10 and test_more_all >= 10,
             "test frequency rankings remain predictable")
    _require(constant_canvas_puzzles >= 2 and constant_background_puzzles >= 2,
             "consistent puzzle regimes absent")
    _require(new_test_target_puzzles >= 5 and repeated_test_target_puzzles >= 5
             and different_multi_test_targets >= 5,
             "target color schedules remain narrow")
    _require(combined_puzzles >= 10, "independent variation axes fail to combine")
