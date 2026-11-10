"""Generate rectangles divided by a one-cell cut and moved apart."""

import hashlib
import numpy as np


SOURCE_HASHES = {
    "2b18c2dc393a3b78ae953f71bfebb81e7d4af65fbf9f0d0d52e80920ecd5ac25",
    "eba5cf0257016378d2adbaeb03a2fef43c3e932922cc88b326b0e6b21c584c4a",
    "3b7dcc8fe1e2083b95f54cdcc532e07dc3ed3809598a285b43befd5f1e0c8de1",
    "a5beb71abf55a876d27ee20c72b6d234031f701a72953212234c20e904fd8ab7",
}


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def count_value(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def resolve_counts(rng, seed, num_train, num_test):
    if num_train is not None:
        num_train = count_value(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = count_value(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def new_color(rng, forbidden):
    available = [c for c in range(10) if c not in forbidden]
    return int(available[int(rng.integers(len(available)))])


def pair_palette(rng, base, flags, index):
    bg, cut, material = base
    if index and flags[0]:
        bg = new_color(rng, {bg})
    if index and flags[1]:
        cut = new_color(rng, {bg, cut})
    elif cut == bg:
        cut = new_color(rng, {bg})
    if index and flags[2]:
        material = new_color(rng, {bg, cut, material})
    elif material in (bg, cut):
        material = new_color(rng, {bg, cut})
    return bg, cut, material


def cut_path(rng, length, span, force_jagged):
    """Choose an interior path with one cell per cross section."""
    if span == 3:
        return tuple([1] * length)
    low, high = 1, span - 2
    mode = int(rng.integers(1 if force_jagged else 0, 4))
    current = int(rng.integers(low, high + 1))
    path = [current]
    direction = int(rng.choice((-1, 1)))
    next_turn = int(rng.integers(1, 4))
    for step in range(1, length):
        if mode == 0:
            move = 0
        elif mode == 1:
            # Two-position zigzags with unequal straight runs.
            if step >= next_turn:
                direction = -direction
                next_turn = step + int(rng.integers(1, 4))
            move = direction if step == 1 or step >= next_turn - 1 else 0
        elif mode == 2:
            move = int(rng.choice((-1, 0, 1)))
        else:
            if step >= next_turn or not low <= current + direction <= high:
                direction = -direction
                next_turn = step + int(rng.integers(2, 6))
            move = direction if rng.random() < 0.75 else 0
        current = max(low, min(high, current + move))
        path.append(current)
    if force_jagged and len(set(path)) == 1:
        at = min(length - 1, max(1, length // 2))
        path[at] = path[at] + (1 if path[at] < high else -1)
    return tuple(path)


def object_size(rng, orientation, shape, scale, force_jagged):
    rows, cols = shape
    length_max = rows if orientation == "V" else cols
    span_max = cols - 2 if orientation == "V" else rows - 2
    minimum_length, minimum_span = (3, 4) if force_jagged else (2, 3)
    if length_max < minimum_length or span_max < minimum_span:
        return None
    ranges = {"small": ((2, 6), (3, 6)),
              "medium": ((3, 10), (3, 9)),
              "large": ((7, 14), (4, 10))}
    (lo_l, hi_l), (lo_s, hi_s) = ranges[scale]
    lo_l = max(minimum_length, min(lo_l, length_max))
    lo_s = max(minimum_span, min(lo_s, span_max))
    hi_l = max(lo_l, min(hi_l, length_max))
    hi_s = max(lo_s, min(hi_s, span_max))
    length = int(rng.integers(lo_l, hi_l + 1))
    span = int(rng.integers(lo_s, hi_s + 1))
    return (length, span) if orientation == "V" else (span, length)


def candidate(rng, shape, orientation, size, material, cut, jagged):
    rows, cols = shape
    height, width = size
    if orientation == "V":
        if height > rows or width + 2 > cols:
            return None
        top = int(rng.integers(0, rows - height + 1))
        left = int(rng.integers(1, cols - width))
        path = cut_path(rng, height, width, jagged)
    else:
        if width > cols or height + 2 > rows:
            return None
        top = int(rng.integers(1, rows - height))
        left = int(rng.integers(0, cols - width + 1))
        path = cut_path(rng, width, height, jagged)
    block = np.full((height, width), material, dtype=np.int8)
    targets = []
    if orientation == "V":
        for r, split in enumerate(path):
            block[r, split] = cut
            for c in range(width):
                if c != split:
                    targets.append((top + r, left + c + (-1 if c < split else 1)))
    else:
        for c, split in enumerate(path):
            block[split, c] = cut
            for r in range(height):
                if r != split:
                    targets.append((top + r + (-1 if r < split else 1), left + c))
    return top, left, block, targets


def fits(item, occupied_input, occupied_output):
    top, left, block, targets = item
    height, width = block.shape
    rows, cols = occupied_input.shape
    if top == 0 and left == 0:
        return False
    # Orthogonal one-cell buffer makes every filled rectangle a distinct component.
    if occupied_input[max(0, top - 1):min(rows, top + height + 1), left:left + width].any():
        return False
    if occupied_input[top:top + height, max(0, left - 1):min(cols, left + width + 1)].any():
        return False
    return not any(occupied_output[r, c] for r, c in targets)


def make_pair(rng, regime, index, base, flags):
    dense = rng.random() < (0.30 if regime == "source_train" else 0.44)
    if dense:
        rows, cols = int(rng.integers(15, 24)), int(rng.integers(15, 24))
    elif regime == "source_train":
        rows, cols = int(rng.integers(8, 24)), int(rng.integers(8, 24))
    else:
        rows, cols = int(rng.integers(11, 31)), int(rng.integers(11, 31))
    shape = rows, cols
    bg, cut, primary = pair_palette(rng, base, flags, index)
    maximum = 6 if regime == "source_train" else 10
    if index == 0:
        desired = int(rng.integers(2, 6))
    elif dense:
        desired = int(rng.integers(4, maximum + 1))
    else:
        desired = int(rng.integers(1, maximum + 1))
    desired = min(desired, max(1, rows * cols // 35))
    if index == 0:
        desired = max(2, desired)
    materials = [primary]
    if flags[3] and desired >= 2:
        materials.append(new_color(rng, {bg, cut, primary}))
        if desired >= 4 and rng.random() < 0.3:
            materials.append(new_color(rng, {bg, cut, *materials}))
    input_grid = np.full(shape, bg, dtype=np.int8)
    output_grid = np.full(shape, bg, dtype=np.int8)
    occupied_input = np.zeros(shape, dtype=bool)
    occupied_output = np.zeros(shape, dtype=bool)
    favored_orientation = "V" if rng.random() < 0.5 else "H"
    for obj_index in range(desired):
        forced = regime == "source_train" and index in (0, 1) and obj_index == 0
        orientation = ("V" if index == 0 else "H") if forced else (
            favored_orientation if rng.random() < 0.52 else ("V" if rng.random() < 0.5 else "H"))
        material = materials[obj_index % len(materials)] if obj_index < len(materials) else int(rng.choice(materials))
        placed = False
        for attempt in range(220):
            if dense and obj_index < 2 and attempt < 100:
                scale = "large"
            else:
                scale = ("small", "medium", "large")[int(rng.choice(3, p=(0.49, 0.35, 0.16)))]
            if attempt >= 130:
                scale = "small"
            size = object_size(rng, orientation, shape, scale, forced)
            if size is None:
                continue
            item = candidate(rng, shape, orientation, size, material, cut, forced)
            if item is None or not fits(item, occupied_input, occupied_output):
                continue
            top, left, block, targets = item
            height, width = block.shape
            input_grid[top:top + height, left:left + width] = block
            occupied_input[top:top + height, left:left + width] = True
            for r, c in targets:
                output_grid[r, c] = material
                occupied_output[r, c] = True
            placed = True
            break
        if not placed and forced:
            raise AssertionError("mandatory training example could not be placed")
        if not placed:
            break
    return {"input": input_grid, "output": output_grid}


def components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for raw_r, raw_c in np.argwhere(mask):
        r, c = int(raw_r), int(raw_c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, group = [(r, c)], []
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(group)
    return result


def analyze(grid, background, cut):
    objects = []
    for cells in components(grid != background):
        top, left = min(r for r, _ in cells), min(c for _, c in cells)
        bottom, right = max(r for r, _ in cells), max(c for _, c in cells)
        height, width = bottom - top + 1, right - left + 1
        require(height >= 2 and width >= 2 and len(cells) == height * width, "invalid rectangle")
        block = grid[top:bottom + 1, left:right + 1]
        colors = {int(x) for x in np.unique(block)}
        require(cut in colors and len(colors) == 2, "rectangle needs one material and cut")
        material = next(iter(colors - {cut}))
        vertical = []
        for r in range(height):
            positions = np.flatnonzero(block[r] == cut)
            if len(positions) != 1:
                break
            vertical.append(int(positions[0]))
        vvalid = len(vertical) == height and all(1 <= p < width - 1 for p in vertical) and all(
            abs(a - b) <= 1 for a, b in zip(vertical, vertical[1:]))
        horizontal = []
        for c in range(width):
            positions = np.flatnonzero(block[:, c] == cut)
            if len(positions) != 1:
                break
            horizontal.append(int(positions[0]))
        hvalid = len(horizontal) == width and all(1 <= p < height - 1 for p in horizontal) and all(
            abs(a - b) <= 1 for a, b in zip(horizontal, horizontal[1:]))
        require(vvalid != hvalid, "cut orientation is ambiguous")
        objects.append({"top": top, "left": left, "height": height, "width": width,
                        "orientation": "V" if vvalid else "H", "cut": tuple(vertical if vvalid else horizontal),
                        "material": material})
    require(objects, "no objects")
    return sorted(objects, key=lambda o: (o["top"], o["left"]))


def reconstruct(grid, background, objects):
    output = np.full(grid.shape, background, dtype=np.int8)
    for obj in objects:
        top, left, height, width = (obj[k] for k in ("top", "left", "height", "width"))
        if obj["orientation"] == "V":
            for r, split in enumerate(obj["cut"]):
                for c in range(width):
                    if c == split:
                        continue
                    rr, cc = top + r, left + c + (-1 if c < split else 1)
                    require(0 <= cc < grid.shape[1] and output[rr, cc] == background,
                            "movement leaves canvas or collides")
                    output[rr, cc] = obj["material"]
        else:
            for c, split in enumerate(obj["cut"]):
                for r in range(height):
                    if r == split:
                        continue
                    rr, cc = top + r + (-1 if r < split else 1), left + c
                    require(0 <= rr < grid.shape[0] and output[rr, cc] == background,
                            "movement leaves canvas or collides")
                    output[rr, cc] = obj["material"]
    return output


def signature(grid):
    return (grid.shape, grid.tobytes())


def source_hash(grid):
    array = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(bytes(array.shape) + array.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = resolve_counts(rng, seed, num_train, num_test)
    bg = int(rng.integers(10))
    cut = new_color(rng, {bg})
    material = new_color(rng, {bg, cut})
    base = bg, cut, material
    flags = tuple(bool(x) for x in rng.random(4) < (0.42, 0.30, 0.62, 0.36))
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(train_count + test_count):
        regime = "source_train" if index < train_count else "target_test"
        for _ in range(12):
            pair = make_pair(rng, regime, index, base, flags)
            key = signature(pair["input"])
            if key not in seen and source_hash(pair["input"]) not in SOURCE_HASHES:
                break
        else:
            raise AssertionError("could not produce a distinct pair")
        seen.add(key)
        puzzle["train" if regime == "source_train" else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
            len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    orientations, jagged = set(), set()
    train_multicolor = test_multicolor = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
                require(all(1 <= d <= 30 for d in grid.shape) and np.all((0 <= grid) & (grid <= 9)),
                        "invalid grid shape or color")
            inp, out = pair["input"], pair["output"]
            require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid output dimensions or change")
            in_key, out_key = signature(inp), signature(out)
            require(in_key not in seen_inputs and (in_key, out_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(in_key)
            seen_pairs.add((in_key, out_key))
            require(source_hash(inp) not in SOURCE_HASHES, "source input copied")
            background = int(inp[0, 0])
            input_colors = {int(x) for x in np.unique(inp)}
            output_colors = {int(x) for x in np.unique(out)}
            missing = input_colors - output_colors
            require(len(missing) == 1 and output_colors == input_colors - missing and
                    background in output_colors and len(output_colors) >= 2, "invalid color roles")
            cut = next(iter(missing))
            objects = analyze(inp, background, cut)
            rows, cols = inp.shape
            if split == "train":
                require(8 <= rows <= 23 and 8 <= cols <= 23 and 1 <= len(objects) <= 6,
                        "source_train size or object count outside its regime")
            else:
                require(11 <= rows <= 30 and 11 <= cols <= 30 and 1 <= len(objects) <= 10,
                        "target_test size or object count outside its regime")
            require(np.array_equal(reconstruct(inp, background, objects), out), "incorrect split and movement")
            materials = {o["material"] for o in objects}
            require(materials == output_colors - {background}, "material not preserved")
            if split == "train":
                train_multicolor |= len(materials) > 1
                for obj in objects:
                    orientations.add(obj["orientation"])
                    if len(set(obj["cut"])) > 1:
                        jagged.add(obj["orientation"])
            else:
                test_multicolor |= len(materials) > 1
    require(orientations == {"V", "H"} and jagged == {"V", "H"},
            "training must show jagged cuts in both orientations")
    require(not test_multicolor or train_multicolor, "multiple materials unseen in training")


def equal_puzzles(first, second):
    return all(len(first[s]) == len(second[s]) for s in ("train", "test")) and all(
        np.array_equal(a[k], b[k]) for s in ("train", "test")
        for a, b in zip(first[s], second[s]) for k in ("input", "output"))


def self_test() -> None:
    require(equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3)),
            "generation is not deterministic")
    for seed, tr, te in ((301, 2, 1), (302, 10, 1), (303, 10, 4),
                         (401, None, 1), (402, None, 4), (403, 2, None), (404, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        require((tr is None or len(puzzle["train"]) == tr) and
                (te is None or len(puzzle["test"]) == te), "requested counts ignored")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(500, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    object_counts, dimensions, cuts, densities = set(), set(), set(), []
    thin = boundary = multicolor = wandering = varied_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background = int(grid[0, 0])
                cut = next(iter({int(x) for x in np.unique(grid)} -
                                {int(x) for x in np.unique(pair["output"])}))
                objects = analyze(grid, background, cut)
                palette = (background, cut, tuple(sorted({o["material"] for o in objects})))
                palettes.add(palette)
                local_palettes.add(palette)
                shapes.add(grid.shape)
                object_counts.add(len(objects))
                densities.append(np.count_nonzero(grid != background) / grid.size)
                multicolor += len(palette[2]) > 1
                layouts.add((grid.shape, tuple((o["top"], o["left"], o["height"], o["width"],
                                                o["orientation"], o["cut"]) for o in objects)))
                for obj in objects:
                    dimensions.add((obj["height"], obj["width"], obj["orientation"]))
                    cuts.add(obj["cut"])
                    thin += min(obj["height"], obj["width"]) == 2
                    boundary += (obj["top"] == 0 or obj["left"] == 0 or
                                 obj["top"] + obj["height"] == grid.shape[0] or
                                 obj["left"] + obj["width"] == grid.shape[1])
                    wandering += len(set(obj["cut"])) >= 3
        varied_palettes += len(local_palettes) >= 2
    require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    require(len(shapes) >= 55 and len(layouts) >= 200 and len(dimensions) >= 70,
            "grid and rectangle geometry lacks diversity")
    require(len(object_counts) >= 6 and len(cuts) >= 40 and len(palettes) >= 60,
            "object counts, paths, or palettes lack diversity")
    require(thin >= 10 and boundary >= 15 and multicolor >= 10 and wandering >= 10,
            "reviewed variations missing")
    require(sum(d >= 0.30 for d in densities) >= 10 and sum(d <= 0.15 for d in densities) >= 10,
            "packing densities lack range")
    require(varied_palettes >= 10, "within-puzzle palette variation absent")
