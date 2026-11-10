"""Compositional count-and-fill ARC puzzle generator."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "ab4ff181cf389c06c97bee3f30bbca9d88556f83e4315a248786b0c08ad595cf",
    "2e4dca69fc4f696ead52f949728732315cb9cdf3ee964e4834af11a3461cb28f",
    "68490e860a698ede43455e875e77591da13a33607f6a768093681bb36af02732",
    "17e67138d964b2a52c54989f623f81abb8636ee1570e4123ff44bb44a2decb9b",
})


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _capacity(shape):
    return (shape[0] * shape[1] + 1) // 2


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        legal = list(range(num_test + 1, 11))
        num_train = legal[(seed // 4 + seed) % len(legal)]
    elif num_train is None:
        legal = list(range(num_test + 1, 11))
        num_train = legal[(3 * seed + num_test) % len(legal)]
    elif num_test is None:
        legal = list(range(1, min(4, num_train - 1) + 1))
        num_test = legal[(5 * seed + num_train) % len(legal)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_digest(grid):
    canonical = grid.astype(np.int8, copy=False)
    return hashlib.sha256(bytes(grid.shape) + canonical.tobytes()).hexdigest()


def _components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    found = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        visited[row, col] = True
        stack = [(row, col)]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    stack.append((rr, cc))
        found.append(cells)
    return found


def _frame_bounds(cells):
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    height, width = bottom - top + 1, right - left + 1
    if height < 3 or width < 3 or len(cells) != 2 * height + 2 * width - 4:
        return None
    if not all(r in (top, bottom) or c in (left, right) for r, c in cells):
        return None
    return (top, left, height, width)


def _inside(box, row, col):
    r, c, h, w = box
    return r <= row < r + h and c <= col < c + w


def _overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _touching(a, b):
    if _overlap(a, b):
        return True
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    horizontal = (ac + aw == bc or bc + bw == ac) and ar < br + bh and br < ar + ah
    vertical = (ar + ah == br or br + bh == ar) and ac < bc + bw and bc < ac + aw
    return horizontal or vertical


def _box_gap(a, b):
    dr = max(a[0] - b[0] - b[2], b[0] - a[0] - a[2], 0)
    dc = max(a[1] - b[1] - b[3], b[1] - a[1] - a[3], 0)
    return dr + dc


def _draw_frame(grid, box, color):
    r, c, h, w = box
    grid[r, c:c + w] = color
    grid[r + h - 1, c:c + w] = color
    grid[r:r + h, c] = color
    grid[r:r + h, c + w - 1] = color


def _shape_pool(max_capacity, thin_only=False):
    pool = []
    for h in range(1, 10):
        for w in range(1, 10):
            if _capacity((h, w)) <= max_capacity and (not thin_only or h == 1 or w == 1):
                pool.append((h, w))
    return pool


def _sample_shapes(rng, count, profile, phase_demo, thin_only):
    max_capacity = {"simple": 9, "moderate": 16, "large": 27}[profile]
    if count >= 4:
        max_capacity = min(max_capacity, 16 if count == 4 else 12)
    pool = _shape_pool(max_capacity, thin_only)
    by_capacity = {}
    for shape in pool:
        by_capacity.setdefault(_capacity(shape), []).append(shape)
    capacities = rng.choice(list(by_capacity), size=count, replace=False)
    shapes = []
    for capacity in capacities:
        options = by_capacity[int(capacity)]
        # Thin and two-dimensional versions of the same count can both occur.
        shapes.append(options[int(rng.integers(len(options)))])
    if phase_demo and not any(h >= 2 and w >= 2 for h, w in shapes):
        alternatives = [(h, w) for h in range(2, 6) for w in range(2, 6)
                        if _capacity((h, w)) <= max_capacity
                        and _capacity((h, w)) not in set(int(x) for x in capacities[1:])]
        if alternatives:
            shapes[0] = alternatives[int(rng.integers(len(alternatives)))]
    if thin_only and count >= 2:
        # The two strip orientations are both visible in this one example.
        first_cap, second_cap = int(capacities[0]), int(capacities[1])
        shapes[0] = (1, 2 * first_cap - 1)
        shapes[1] = (2 * second_cap - 1, 1)
        if max(shapes[0] + shapes[1]) > 9:
            return None
    rng.shuffle(shapes)
    if len({_capacity(s) for s in shapes}) != count:
        return None
    return shapes


def _place_boxes(rng, height, width, shapes, style):
    boxes = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: (shapes[i][0] + 2) * (shapes[i][1] + 2), reverse=True)
    placed = []
    for index in order:
        h, w = shapes[index][0] + 2, shapes[index][1] + 2
        candidates = [(r, c, h, w) for r in range(height - h + 1)
                      for c in range(width - w + 1)
                      if all(not _overlap((r, c, h, w), old) for old in placed)]
        if not candidates:
            return None
        if style == "edge" and not placed:
            preferred = [b for b in candidates if b[0] == 0 or b[1] == 0
                         or b[0] + b[2] == height or b[1] + b[3] == width]
            if preferred:
                candidates = preferred
        if style == "tight" and placed:
            preferred = [b for b in candidates if min(_box_gap(b, old) for old in placed) <= 1]
            if preferred:
                candidates = preferred
        box = candidates[int(rng.integers(len(candidates)))]
        boxes[index] = box
        placed.append(box)
    return boxes


def _colors(rng, background, boxes, forbidden_payload=(), preferred_payload=None):
    available = [c for c in range(10) if c != background]
    payload_pool = [c for c in available if c not in forbidden_payload]
    if len(payload_pool) < len(boxes):
        return None
    payload = []
    if preferred_payload is not None and preferred_payload in payload_pool:
        payload.append(int(preferred_payload))
        payload_pool.remove(preferred_payload)
    need = len(boxes) - len(payload)
    payload.extend(int(c) for c in rng.choice(payload_pool, size=need, replace=False))
    rng.shuffle(payload)
    frame_pool = [c for c in available if c not in payload]
    if not frame_pool:
        return None
    frames = []
    for i, box in enumerate(boxes):
        options = [c for c in frame_pool if all(c != frames[j] or not _touching(box, boxes[j])
                                                  for j in range(i))]
        if not options:
            return None
        if i and rng.random() < 0.12:
            repeated = [c for c in options if c in frames]
            if repeated:
                options = repeated
        elif len(options) > len(boxes) - i:
            fresh = [c for c in options if c not in frames]
            if fresh:
                options = fresh
        frames.append(int(options[int(rng.integers(len(options)))]))
    return frames, payload


def _scatter(rng, frame_grid, boxes, shapes, payload, mode):
    background = int(frame_grid[0, 0]) if not any(_inside(b, 0, 0) for b in boxes) else None
    if background is None:
        # The caller supplies a uniform canvas before frames; use its dominant color.
        counts = np.bincount(frame_grid.ravel().astype(int), minlength=10)
        background = int(np.argmax(counts))
    available = frame_grid == background
    for r, c, h, w in boxes:
        available[r:r + h, c:c + w] = False
    counts = [_capacity(s) for s in shapes]
    if int(np.count_nonzero(available)) < sum(counts):
        return None
    assigned = {}
    remaining = counts.copy()
    near = []
    adjacent_pairs = []
    for r, c in np.argwhere(available):
        r, c = int(r), int(c)
        neighbors = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if any(0 <= rr < available.shape[0] and 0 <= cc < available.shape[1]
               and any(_inside(box, rr, cc) for box in boxes) for rr, cc in neighbors):
            near.append((r, c))
        for rr, cc in ((r + 1, c), (r, c + 1)):
            if 0 <= rr < available.shape[0] and 0 <= cc < available.shape[1] and available[rr, cc]:
                adjacent_pairs.append(((r, c), (rr, cc)))
    if mode == "isolated":
        near_set = set(near)
        safe = [tuple(int(x) for x in cell) for cell in np.argwhere(available)
                if tuple(int(x) for x in cell) not in near_set]
        chosen = []
        for candidate_index in rng.permutation(len(safe)):
            cell = safe[int(candidate_index)]
            if all(abs(cell[0] - old[0]) + abs(cell[1] - old[1]) > 1 for old in chosen):
                chosen.append(cell)
                if len(chosen) == sum(counts):
                    break
        if len(chosen) != sum(counts):
            return None
        slots = [i for i, amount in enumerate(counts) for _ in range(amount)]
        rng.shuffle(slots)
        if len({(r + c) % 2 for r, c in chosen}) < 2:
            return None
        result = frame_grid.copy()
        for (r, c), index in zip(chosen, slots):
            result[r, c] = payload[index]
        return result
    if mode in ("frame", "combined") and near:
        cell = near[int(rng.integers(len(near)))]
        index = int(rng.integers(len(payload)))
        assigned[cell] = index
        remaining[index] -= 1
    if mode in ("same", "different", "combined") and adjacent_pairs:
        pairs = [p for p in adjacent_pairs if p[0] not in assigned and p[1] not in assigned]
        if pairs:
            a, b = pairs[int(rng.integers(len(pairs)))]
            if mode == "same":
                choices = [i for i, n in enumerate(remaining) if n >= 2]
                if choices:
                    i = int(rng.choice(choices))
                    assigned[a] = assigned[b] = i
                    remaining[i] -= 2
            else:
                choices = [i for i, n in enumerate(remaining) if n >= 1]
                if len(choices) >= 2:
                    i, j = (int(x) for x in rng.choice(choices, size=2, replace=False))
                    assigned[a], assigned[b] = i, j
                    remaining[i] -= 1
                    remaining[j] -= 1
    free = [tuple(int(x) for x in cell) for cell in np.argwhere(available)
            if tuple(int(x) for x in cell) not in assigned]
    needed = [i for i, n in enumerate(remaining) for _ in range(n)]
    rng.shuffle(needed)
    chosen = rng.choice(len(free), size=len(needed), replace=False)
    for location, index in zip(chosen, needed):
        assigned[free[int(location)]] = index
    if len({(r + c) % 2 for r, c in assigned}) < 2:
        return None
    result = frame_grid.copy()
    for (r, c), index in assigned.items():
        result[r, c] = payload[index]
    return result


def _make_pair(rng, regime, background, count, profile, phase_demo=False,
               thin_only=False, forbidden_payload=(), preferred_payload=None):
    for _ in range(120):
        shapes = _sample_shapes(rng, count, profile, phase_demo, thin_only)
        if shapes is None:
            continue
        minimum = 12 if profile == "simple" else 16 if profile == "moderate" else 22
        minimum = max(minimum, 18 if count >= 4 else 12, 21 if count == 5 else 12)
        height = int(rng.integers(minimum, 31))
        width = int(rng.integers(minimum, 31))
        style = str(rng.choice(("free", "tight", "edge"), p=(0.35, 0.35, 0.30)))
        boxes = _place_boxes(rng, height, width, shapes, style)
        if boxes is None:
            continue
        colors = _colors(rng, background, boxes, forbidden_payload, preferred_payload)
        if colors is None:
            continue
        frame_colors, payload = colors
        frame_grid = np.full((height, width), background, dtype=np.int8)
        for box, color in zip(boxes, frame_colors):
            _draw_frame(frame_grid, box, color)
        mode = str(rng.choice(("uniform", "isolated", "frame", "same", "different", "combined"),
                              p=(0.15, 0.12, 0.17, 0.17, 0.17, 0.22)))
        input_grid = _scatter(rng, frame_grid, boxes, shapes, payload, mode)
        if input_grid is None:
            continue
        output_grid = frame_grid.copy()
        for (h, w), (r, c, _, _), color in zip(shapes, boxes, payload):
            patch = output_grid[r + 1:r + h + 1, c + 1:c + w + 1]
            parity = np.indices((h, w)).sum(axis=0) % 2 == 0
            patch[parity] = color
        pair = {"input": input_grid, "output": output_grid}
        try:
            _analyze_pair(input_grid, output_grid)
        except AssertionError:
            continue
        return pair
    raise RuntimeError(f"could not construct {regime} pair")


def _validate_grid(grid, label):
    _check(isinstance(grid, np.ndarray), f"{label}: not an ndarray")
    _check(grid.ndim == 2, f"{label}: not two-dimensional")
    _check(all(1 <= n <= 30 for n in grid.shape), f"{label}: bad shape")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label}: not integer")
    _check(bool(np.all((grid >= 0) & (grid <= 9))), f"{label}: illegal color")


def _analyze_pair(input_grid, output_grid):
    _check(input_grid.shape == output_grid.shape, "input/output shapes differ")
    frequencies = np.bincount(input_grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(frequencies))
    _check(np.count_nonzero(frequencies == frequencies[background]) == 1,
           "background is not uniquely dominant")
    frames = []
    payload = []
    for color in (int(c) for c in np.unique(input_grid) if int(c) != background):
        components = _components(input_grid == color)
        bounds = [_frame_bounds(comp) for comp in components]
        if any(bound is not None for bound in bounds):
            _check(all(bound is not None for bound in bounds), "frame hue has exterior cells")
            for box in bounds:
                frames.append((box, color))
        else:
            payload.append(color)
    _check(2 <= len(frames) <= 5 and len(payload) == len(frames),
           "frame/payload inventory count mismatch")
    for i, (box, _) in enumerate(frames):
        r, c, h, w = box
        _check(bool(np.all(input_grid[r + 1:r + h - 1, c + 1:c + w - 1] == background)),
               "frame interior is not hollow")
        for other, _ in frames[i + 1:]:
            _check(not _overlap(box, other), "overlapping frames")
    for color in payload:
        for r, c in np.argwhere(input_grid == color):
            _check(all(not _inside(box, int(r), int(c)) for box, _ in frames),
                   "payload occurs inside a frame")
    capacities = [_capacity((box[2] - 2, box[3] - 2)) for box, _ in frames]
    inventories = {color: int(np.count_nonzero(input_grid == color)) for color in payload}
    _check(len(set(capacities)) == len(capacities), "ambiguous frame capacities")
    _check(len(set(inventories.values())) == len(inventories), "ambiguous payload counts")
    _check(sorted(capacities) == sorted(inventories.values()), "count match fails")
    count_to_color = {count: color for color, count in inventories.items()}
    expected = input_grid.copy()
    for color in payload:
        expected[expected == color] = background
    for box, _ in frames:
        r, c, h, w = box
        patch = expected[r + 1:r + h - 1, c + 1:c + w - 1]
        patch[np.indices(patch.shape).sum(axis=0) % 2 == 0] = count_to_color[_capacity(patch.shape)]
    _check(np.array_equal(expected, output_grid), "incorrect count-and-fill output")
    _check(not np.array_equal(input_grid, output_grid), "unchanged pair")
    _check(_source_digest(input_grid) not in _SOURCE_INPUT_DIGESTS, "original source input copied")
    parities = {(int(r) + int(c)) % 2 for color in payload for r, c in np.argwhere(input_grid == color)}
    _check(len(parities) == 2, "scatter lies on one global parity")
    return {"background": background, "frames": frames, "payload": set(payload),
            "capacities": capacities, "shape": input_grid.shape,
            "thin_only": all(box[2] == 3 or box[3] == 3 for box, _ in frames),
            "phase": any(box[2] >= 4 and box[3] >= 4 for box, _ in frames)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "bad split schema")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
           "illegal pair counts")
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, "input")
            _validate_grid(out, "output")
            key = _grid_key(inp)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(_analyze_pair(inp, out))
    _check(any(info["phase"] for info in infos["train"]), "training does not teach 2D phase")
    _check(any(len(info["frames"]) >= 2 for info in infos["train"]), "training lacks count match")
    _check(len({len(info["frames"]) for info in infos["train"]}) >= 2,
           "training lacks frame-count diversity")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_background = bool(rng.integers(2))
    base_background = int(rng.integers(10))
    reserved = int(rng.choice([c for c in range(10) if c != base_background]))
    novel_mode = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    backgrounds = []
    used_inputs = set()
    training_payload = set()
    pair_total = num_train + num_test
    for index in range(pair_total):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        if stable_background or index == 0:
            background = base_background
        elif index == 1:
            background = int(rng.choice([c for c in range(10) if c != base_background and c != reserved]))
        else:
            background = int(rng.choice([c for c in range(10) if c != reserved]))
        backgrounds.append(background)
        if split == "train":
            count = 2 if local_index == 0 else 3 if local_index == 1 else int(rng.choice((2, 3, 4, 5), p=(0.35, 0.32, 0.25, 0.08)))
            thin_only = local_index == 1 and rng.random() < 0.43
            phase_demo = local_index == 0
            profile = "moderate" if local_index == 0 else str(rng.choice(("simple", "moderate", "large"), p=(0.42, 0.48, 0.10)))
        else:
            count = int(rng.choice((2, 3, 4, 5), p=(0.24, 0.37, 0.30, 0.09)))
            thin_only = False
            phase_demo = False
            profile = str(rng.choice(("simple", "moderate", "large"), p=(0.25, 0.35, 0.40)))
        forbidden = (reserved,) if split == "train" and novel_mode else ()
        preferred = None
        if split == "test" and local_index == 0:
            if novel_mode and reserved != background:
                preferred = reserved
            elif not novel_mode:
                familiar = sorted(training_payload - {background})
                if familiar:
                    preferred = int(rng.choice(familiar))
        for _ in range(12):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                              background, count, profile, phase_demo, thin_only, forbidden, preferred)
            key = _grid_key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                break
        else:
            raise RuntimeError("could not avoid duplicate input")
        puzzle[split].append(pair)
        if split == "train":
            training_payload.update(_analyze_pair(pair["input"], pair["output"])["payload"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "few automatic train counts")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "few automatic test counts")
    _check(len({_signature(p) for p in puzzles}) == 42, "duplicate seeded puzzle")
    shape_set, layout_set, interior_set, backgrounds = set(), set(), set(), set()
    frame_counts, scatter_parities = set(), set()
    thin_pairs = 0
    horizontal_thin = 0
    vertical_thin = 0
    close_pairs = 0
    edge_pairs = 0
    shared_frame_pairs = 0
    frame_contact_pairs = 0
    same_contact_pairs = 0
    different_contact_pairs = 0
    isolated_scatter_pairs = 0
    variable_background_puzzles = 0
    familiar_tests = 0
    novel_tests = 0
    smaller_tests = 0
    extrapolating_tests = 0
    mixed_frame_count_puzzles = 0
    for puzzle in puzzles:
        infos = {split: [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle[split]]
                 for split in ("train", "test")}
        variable_background_puzzles += len({x["background"] for split in infos for x in infos[split]}) > 1
        mixed_frame_count_puzzles += len({len(x["frames"]) for split in infos for x in infos[split]}) > 1
        familiar = set().union(*(x["payload"] for x in infos["train"]))
        largest_train_area = max(x["shape"][0] * x["shape"][1] for x in infos["train"])
        largest_train_capacity = max(max(x["capacities"]) for x in infos["train"])
        for info in infos["test"]:
            familiar_tests += bool(info["payload"] & familiar)
            novel_tests += bool(info["payload"] - familiar)
            smaller_tests += info["shape"][0] * info["shape"][1] < largest_train_area
            extrapolating_tests += max(info["capacities"]) > largest_train_capacity
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                shape_set.add(info["shape"])
                backgrounds.add(info["background"])
                frame_counts.add(len(info["frames"]))
                thin_pairs += info["thin_only"] and split == "train"
                boxes = [box for box, _ in info["frames"]]
                horizontal_thin += info["thin_only"] and any(box[2] == 3 for box in boxes)
                vertical_thin += info["thin_only"] and any(box[3] == 3 for box in boxes)
                interior_set.update((box[2] - 2, box[3] - 2) for box in boxes)
                layout_set.add((info["shape"], tuple(sorted(boxes))))
                close_pairs += any(_box_gap(a, b) <= 1 for i, a in enumerate(boxes) for b in boxes[i + 1:])
                shared_frame_pairs += len({color for _, color in info["frames"]}) < len(boxes)
                edge_pairs += any(b[0] == 0 or b[1] == 0 or b[0] + b[2] == info["shape"][0]
                                  or b[1] + b[3] == info["shape"][1] for b in boxes)
                inp = pair["input"]
                coords = [tuple(x) for color in info["payload"] for x in np.argwhere(inp == color)]
                scatter_parities.update((r + c) % 2 for r, c in coords)
                frame_cells = {(r, c) for box in boxes
                               for r in range(box[0], box[0] + box[2])
                               for c in range(box[1], box[1] + box[3])
                               if r in (box[0], box[0] + box[2] - 1)
                               or c in (box[1], box[1] + box[3] - 1)}
                frame_contact = same_contact = different_contact = False
                for r, c in coords:
                    color = int(inp[r, c])
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if (rr, cc) in frame_cells:
                            frame_contact = True
                        elif 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1] and int(inp[rr, cc]) in info["payload"]:
                            if int(inp[rr, cc]) == color:
                                same_contact = True
                            else:
                                different_contact = True
                frame_contact_pairs += frame_contact
                same_contact_pairs += same_contact
                different_contact_pairs += different_contact
                isolated_scatter_pairs += not (frame_contact or same_contact or different_contact)
    _check(len(shape_set) >= 35 and len(layout_set) >= 100 and len(interior_set) >= 20,
           "insufficient geometric diversity")
    _check(frame_counts >= {2, 3, 4}, "frame-count diversity missing")
    _check(thin_pairs >= 3 and horizontal_thin >= 3 and vertical_thin >= 3
           and close_pairs >= 8 and edge_pairs >= 8 and shared_frame_pairs >= 3,
           "thin/close/edge examples missing")
    _check(frame_contact_pairs >= 8 and same_contact_pairs >= 8
           and different_contact_pairs >= 8 and isolated_scatter_pairs >= 8,
           "scatter contact diversity missing")
    _check(variable_background_puzzles >= 8 and familiar_tests >= 8 and novel_tests >= 8,
           "background/payload color diversity missing")
    _check(smaller_tests >= 8 and extrapolating_tests >= 8 and mixed_frame_count_puzzles >= 20,
           "train/test shift diversity missing")
    _check(len(backgrounds) >= 8 and scatter_parities == {0, 1}, "color or scatter diversity missing")
    explicit = ((2, 1), (10, 1), (10, 4), (6, 2))
    for train, test in explicit:
        p = generate_puzzle(20000 + 100 * train + test, train, test)
        _check((len(p["train"]), len(p["test"])) == (train, test), "explicit counts ignored")
    for seed, train, test in ((30001, None, 1), (30002, None, 4),
                              (30003, 2, None), (30004, 10, None)):
        p = generate_puzzle(seed, train, test)
        _check(train is None or len(p["train"]) == train, "partial train count ignored")
        _check(test is None or len(p["test"]) == test, "partial test count ignored")
    _check(_signature(generate_puzzle(12345, 7, 3)) == _signature(generate_puzzle(12345, 7, 3)),
           "generation is not deterministic")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {arguments}")
