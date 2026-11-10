import numpy as np


# All fixed orientations of the connected objects in the source family.
SHAPES = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)), ((0, 0), (1, 0))),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
}
SHAPE_SETS = {size: set(forms) for size, forms in SHAPES.items()}

SOURCE_INPUTS = (
    ("5555500555", "0050555550", "5555505005", "5055505555", "5555555505",
     "5555055555", "0055050050", "5555555055", "0555550550", "0055555505"),
    ("5555055505", "0055555555", "0555550050", "5505555050", "5555005555",
     "0505555050", "0555005550", "5555555555", "5555555550", "0555555050"),
    ("0055055505", "5500555505", "5050505505", "5055055555", "5550055050",
     "5505555050", "5505505555", "5550555555", "5055505055", "5505555555"),
    ("0555550055", "5550550505", "5505555505", "5005555555", "0555550555",
     "0550550500", "5505555505", "5505555550", "0055550555", "5555050050"),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _resolve_counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _components(grid, foreground):
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if visited[r, c] or int(grid[r, c]) != foreground:
                continue
            visited[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < height and 0 <= nc < width and
                            not visited[nr, nc] and int(grid[nr, nc]) == foreground):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            result.append(tuple(sorted(cells)))
    return result


def _shape(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 2, "each input needs one background and one foreground")
    index = int(np.argmax(counts))
    _require(int(counts[index]) > int(counts[1 - index]), "background is not dominant")
    return int(colors[index]), int(colors[1 - index])


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _dimensions(rng, regime):
    # Size is local to a pair. Compact squares retain the crowded source scale.
    probabilities = (0.26, 0.31, 0.29, 0.14) if regime == "source_train" else (0.23, 0.26, 0.31, 0.20)
    band = int(rng.choice(4, p=probabilities))
    if band == 0:
        return 10, 10
    if band == 1:
        return int(rng.integers(9, 14)), int(rng.integers(9, 15))
    if band == 2:
        return int(rng.integers(13, 20)), int(rng.integers(12, 22))
    return int(rng.integers(19, 28)), int(rng.integers(16, 29))


def _counts(rng, height, width, regime, first_train):
    area = height * width
    if height == width == 10 and rng.random() < 0.65:
        # These profiles are observed in the original task, and either split may use them.
        profiles = ((8, 6, 2), (7, 6, 2), (8, 5, 3), (6, 5, 4))
        counts = list(profiles[int(rng.integers(len(profiles)))])
    else:
        if area < 145:
            density = float(rng.uniform(0.10, 0.25))
        elif area < 300:
            density = float(rng.uniform(0.075, 0.21))
        else:
            density = float(rng.uniform(0.055, 0.15))
        if regime == "target_test" and rng.random() < 0.42:
            density = min(0.28, density * float(rng.uniform(1.08, 1.35)))
        target_cells = max(8, int(round(area * density)))
        weights = rng.dirichlet((3.5, 3.0, 2.5))
        if not first_train and rng.random() < 0.17:
            weights[int(rng.integers(3))] = 0
            weights /= weights.sum()
        objects = max(4, int(round(target_cells / float(np.dot(weights, (1, 2, 3))))))
        counts = list(rng.multinomial(objects, weights))
    if first_train:
        counts = [max(2, counts[0]), max(2, counts[1]), max(2, counts[2])]
    if sum(counts) < 4:
        counts[0] += 4 - sum(counts)
    return counts


def _specs(rng, counts, first_train, unseen_form, reserved_form):
    specs = [(1, SHAPES[1][0]) for _ in range(counts[0])]
    for size in (2, 3):
        n = counts[size - 1]
        forms = [form for form in SHAPES[size] if not (size == 3 and reserved_form == form)]
        if first_train:
            chosen = [forms[int(i)] for i in rng.choice(len(forms), size=2, replace=False)]
        else:
            chosen = []
        if size == 3 and unseen_form is not None and n:
            if len(chosen) < n:
                chosen.append(unseen_form)
            else:
                chosen[0] = unseen_form
        while len(chosen) < n:
            chosen.append(forms[int(rng.integers(len(forms)))])
        specs.extend((size, form) for form in chosen)
    return specs


def _blocked(occupied, no_diagonal):
    height, width = occupied.shape
    blocked = occupied.copy()
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if no_diagonal:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    for dr, dc in offsets:
        src_r = slice(max(0, -dr), min(height, height - dr))
        src_c = slice(max(0, -dc), min(width, width - dc))
        dst_r = slice(max(0, dr), min(height, height + dr))
        dst_c = slice(max(0, dc), min(width, width + dc))
        blocked[dst_r, dst_c] |= occupied[src_r, src_c]
    return blocked


def _diagonal_mask(cells, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                mask[nr, nc] = True
    return mask


def _positions(form, blocked, diagonal_to):
    height, width = blocked.shape
    rows = height - max(r for r, _ in form)
    cols = width - max(c for _, c in form)
    allowed = np.ones((rows, cols), dtype=bool)
    for dr, dc in form:
        allowed &= ~blocked[dr:dr + rows, dc:dc + cols]
    if diagonal_to is not None:
        touching = np.zeros((rows, cols), dtype=bool)
        for dr, dc in form:
            touching |= diagonal_to[dr:dr + rows, dc:dc + cols]
        allowed &= touching
    return np.argwhere(allowed)


def _contact_graph(components):
    edges = []
    sets = [set(cells) for cells in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if any((r + dr, c + dc) in sets[j] for r, c in components[i]
                   for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))):
                edges.append((i, j))
    return edges


def _place(rng, height, width, specs, contact_mode):
    indices = list(range(len(specs)))
    anchors = []
    if contact_mode in ("same", "mixed", "chain"):
        if contact_mode == "same":
            options = [size for size in (1, 2, 3) if sum(s == size for s, _ in specs) >= 2]
            if options:
                size = options[int(rng.integers(len(options)))]
                anchors = list(rng.choice([i for i in indices if specs[i][0] == size], 2, replace=False))
        elif contact_mode == "mixed":
            sizes = [size for size in (1, 2, 3) if any(s == size for s, _ in specs)]
            if len(sizes) >= 2:
                pair = rng.choice(sizes, 2, replace=False)
                anchors = [int(rng.choice([i for i in indices if specs[i][0] == size])) for size in pair]
        elif len(indices) >= 3:
            anchors = list(rng.choice(indices, 3, replace=False))
    anchored = set(int(i) for i in anchors)
    rest = [i for i in indices if i not in anchored]
    # Large forms go in first; singleton gaps remain available late in construction.
    rest.sort(key=lambda i: (-specs[i][0], float(rng.random())))
    order = [int(i) for i in anchors] + rest
    for _ in range(55):
        occupied = np.zeros((height, width), dtype=bool)
        placed = []
        failed = False
        for step, index in enumerate(order):
            form = specs[index][1]
            target = None
            if 1 <= step < len(anchors):
                target = _diagonal_mask(placed[step - 1], height, width)
            candidates = _positions(form, _blocked(occupied, contact_mode == "none"), target)
            if not len(candidates):
                failed = True
                break
            top, left = candidates[int(rng.integers(len(candidates)))]
            cells = tuple((int(top) + r, int(left) + c) for r, c in form)
            for r, c in cells:
                occupied[r, c] = True
            placed.append(cells)
        if not failed:
            if contact_mode == "chain" and len(_contact_graph(placed)) < 2:
                continue
            return placed
    return None


def _make_pair(rng, regime, first_train, unseen_form, reserved_form,
               previous_shape, repeated_profile):
    for _ in range(16):
        height, width = (repeated_profile[0] if repeated_profile is not None
                         else _dimensions(rng, regime))
        if previous_shape is not None and (height, width) == previous_shape:
            continue
        counts = (list(repeated_profile[1]) if repeated_profile is not None
                  else _counts(rng, height, width, regime, first_train))
        if unseen_form is not None:
            counts[2] = max(1, counts[2])
        area = height * width
        density = sum((i + 1) * value for i, value in enumerate(counts)) / area
        modes = ("none", "same", "mixed", "chain", "free")
        mode_weights = (0.18, 0.22, 0.22, 0.16, 0.22) if density < 0.18 else (0, 0.25, 0.25, 0.20, 0.30)
        mode = str(rng.choice(modes, p=mode_weights))
        if first_train and mode == "none":
            mode = "mixed"
        for reduction in range(7):
            specs = _specs(rng, counts, first_train, unseen_form, reserved_form)
            placed = _place(rng, height, width, specs, mode)
            if placed is not None:
                return placed, (height, width)
            # A crowded random composition may not pack. Relax it a cell at a time.
            choices = [i for i in range(3) if counts[i] > (2 if first_train else 0)]
            if not choices:
                break
            largest = max(choices, key=lambda i: (i + 1) * counts[i])
            counts[largest] -= 1
    raise ValueError("could not pack a valid pair")


def _recolor(grid, foreground, labels):
    output = grid.copy()
    for cells in _components(grid, foreground):
        for r, c in cells:
            output[r, c] = labels[len(cells)]
    return output


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    labels = {size: int(color) for size, color in zip((1, 2, 3), rng.choice(10, 3, replace=False))}
    available = [color for color in range(10) if color not in labels.values()]
    reserved_form = SHAPES[3][int(rng.integers(6))] if rng.random() < 0.35 else None
    puzzle = {"train": [], "test": []}
    used = set()
    training_forms = set()
    training_profiles = []
    previous_shape = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            regime = "source_train" if split == "train" else "target_test"
            unseen = None
            if split == "test" and index == 0:
                if reserved_form is not None:
                    unseen = reserved_form
                elif rng.random() < 0.58:
                    choices = [form for form in SHAPES[3] if form not in training_forms]
                    if choices:
                        unseen = choices[int(rng.integers(len(choices)))]
            repeated_profile = None
            if split == "test" and index == 0 and rng.random() < 0.38:
                repeated_profile = training_profiles[int(rng.integers(len(training_profiles)))]
            for _ in range(8):
                placed, (height, width) = _make_pair(
                    rng, regime, split == "train" and index == 0,
                    unseen, reserved_form if split == "train" else None,
                    previous_shape if split == "train" and index == 1 else None,
                    repeated_profile,
                )
                background, foreground = [int(v) for v in rng.choice(available, 2, replace=False)]
                grid = np.full((height, width), background, dtype=np.int8)
                for cells in placed:
                    for r, c in cells:
                        grid[r, c] = foreground
                if _key(grid) not in used:
                    break
            else:
                raise ValueError("could not create a unique input")
            used.add(_key(grid))
            if split == "train":
                training_forms.update(_shape(cells) for cells in placed if len(cells) == 3)
                training_profiles.append((grid.shape, tuple(sum(len(cells) == size for cells in placed)
                                                         for size in (1, 2, 3))))
            output = _recolor(grid, foreground, labels)
            puzzle[split].append({"input": grid, "output": output})
            if split == "train" and index == 0:
                previous_shape = grid.shape
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "bad pair counts")
    records = []
    inputs = set()
    pairs = set()
    train_labels = {1: set(), 2: set(), 3: set()}
    all_labels = {1: set(), 2: set(), 3: set()}
    train_shapes = {2: set(), 3: set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair")
            grid, output = pair["input"], pair["output"]
            for matrix in (grid, output):
                _require(isinstance(matrix, np.ndarray) and matrix.ndim == 2, "grid must be a 2D array")
                _require(all(1 <= side <= 30 for side in matrix.shape), "illegal dimensions")
                _require(np.issubdtype(matrix.dtype, np.integer), "grid must contain integers")
                _require(int(matrix.min()) >= 0 and int(matrix.max()) <= 9, "illegal colors")
            _require(grid.shape == output.shape and not np.array_equal(grid, output), "bad output dimensions or unchanged grid")
            key = _key(grid)
            pair_key = key, _key(output)
            _require(key not in inputs and pair_key not in pairs, "duplicate pair")
            inputs.add(key)
            pairs.add(pair_key)
            rows = tuple("".join(str(int(v)) for v in row) for row in grid)
            _require(rows not in SOURCE_INPUTS, "source input copied")
            background, foreground = _roles(grid)
            _require(np.all(output[grid == background] == background), "background changed")
            cells = _components(grid, foreground)
            _require(bool(cells), "no objects")
            counts = {size: 0 for size in (1, 2, 3)}
            for component in cells:
                size = len(component)
                _require(size in SHAPES and _shape(component) in SHAPE_SETS[size], "invalid object")
                counts[size] += 1
                colors = {int(output[r, c]) for r, c in component}
                _require(len(colors) == 1, "object not recolored uniformly")
                color = next(iter(colors))
                _require(color not in (background, foreground), "size label overlaps pair input roles")
                all_labels[size].add(color)
                if split == "train":
                    train_labels[size].add(color)
                    if size in (2, 3):
                        train_shapes[size].add(_shape(component))
            _require(sum(size * counts[size] for size in counts) < grid.size / 2, "foreground dominates")
            records.append((split, grid, output, foreground, cells, counts))
    labels = {}
    for size in (1, 2, 3):
        _require(len(train_labels[size]) == 1, f"training does not reveal area {size}")
        labels[size] = next(iter(train_labels[size]))
        _require(all_labels[size] == {labels[size]}, f"area {size} label changes")
    _require(len(set(labels.values())) == 3, "area labels must differ")
    _require(len(train_shapes[2]) == 2 and len(train_shapes[3]) >= 2, "training lacks form variation")
    for split, grid, output, foreground, cells, counts in records:
        _require(np.array_equal(output, _recolor(grid, foreground, labels)), "wrong size recoloring")
    _require(puzzle["train"][0]["input"].shape != puzzle["train"][1]["input"].shape,
             "training dimensions do not vary")


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[field], b[field]) for a, b in zip(first[split], second[split])
                   for field in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(12345)
    _require(_equal(a, generate_puzzle(12345)), "automatic generation is not deterministic")
    b = generate_puzzle(9876, 7, 3)
    _require(_equal(b, generate_puzzle(9876, 7, 3)), "explicit generation is not deterministic")
    _require(not _equal(b, generate_puzzle(9877, 7, 3)), "different seeds agree")
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train or 0) * 10 + (test or 0), train, test)
        _require(train is None or len(puzzle["train"]) == train, "train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "test count ignored")
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(17, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, dimensions, palettes, profiles, layouts = (set() for _ in range(6))
    compact_crowded = empty_contact = same_contact = mixed_contact = unseen_test_form = omitted = 0
    chains = multiple_contacts = 0
    changed_roles = compact_and_large = repeated_counts = 0
    denser_tests = sparser_tests = overlapping_densities = broad_train_forms = 0
    pairs_seen = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palette = []
        training_tri = set()
        local_roles, local_shapes = set(), set()
        train_profiles, test_profiles = set(), set()
        train_density, test_density = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                bg, fg = _roles(grid)
                cells = _components(grid, fg)
                counts = tuple(sum(len(c) == size for c in cells) for size in (1, 2, 3))
                labels = tuple(next(iter({int(output[r, c]) for c in cells if len(c) == size for r, c in c}))
                               if counts[size - 1] else -1 for size in (1, 2, 3))
                if not palette:
                    palette = list(labels)
                pairs_seen += 1
                dimensions.add(grid.shape)
                local_shapes.add(grid.shape)
                local_roles.add((bg, fg))
                profiles.add((split, counts))
                layouts.add((grid.shape, (grid == fg).tobytes()))
                if grid.shape == (10, 10) and sum(size * counts[size - 1] for size in (1, 2, 3)) >= 25:
                    compact_crowded += 1
                edges = _contact_graph(cells)
                empty_contact += not edges
                multiple_contacts += len(edges) >= 2
                chains += any(sum(i in edge for edge in edges) >= 2 for i in range(len(cells)))
                same_contact += any(len(cells[i]) == len(cells[j]) for i, j in edges)
                mixed_contact += any(len(cells[i]) != len(cells[j]) for i, j in edges)
                omitted += any(value == 0 for value in counts)
                if split == "train":
                    train_profiles.add(counts)
                    train_density.append(sum(size * counts[size - 1] for size in (1, 2, 3)) / grid.size)
                    training_tri.update(_shape(c) for c in cells if len(c) == 3)
                else:
                    test_profiles.add(counts)
                    test_density.append(sum(size * counts[size - 1] for size in (1, 2, 3)) / grid.size)
                    unseen_test_form += any(_shape(c) not in training_tri for c in cells if len(c) == 3)
        palettes.add(tuple(palette))
        changed_roles += len(local_roles) >= 2
        compact_and_large += (10, 10) in local_shapes and any(h >= 20 or w >= 20 for h, w in local_shapes)
        repeated_counts += bool(train_profiles & test_profiles)
        denser_tests += any(value > max(train_density) for value in test_density)
        sparser_tests += any(value < min(train_density) for value in test_density)
        overlapping_densities += any(min(train_density) <= value <= max(train_density) for value in test_density)
        broad_train_forms += len(training_tri) >= 4
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic pair counts lack variety")
    _require(len(dimensions) >= 25 and len(palettes) >= 18, "size or label variety is too narrow")
    _require(len(profiles) >= 35 and len(layouts) >= int(0.95 * pairs_seen), "object structure lacks variety")
    _require(compact_crowded >= 4 and empty_contact >= 3 and same_contact >= 15 and mixed_contact >= 15,
             "source density or diagonal-contact modes lack coverage")
    _require(unseen_test_form >= 3 and omitted >= 4, "test-form or omitted-area coverage is missing")
    _require(changed_roles >= 30 and compact_and_large >= 5, "within-puzzle presentation variety is missing")
    _require(repeated_counts >= 4 and denser_tests >= 3 and sparser_tests >= 3 and overlapping_densities >= 15,
             "train/test composition regimes do not overlap and shift")
    _require(broad_train_forms >= 8 and multiple_contacts >= 15 and chains >= 10,
             "shape breadth or diagonal chains are missing")
