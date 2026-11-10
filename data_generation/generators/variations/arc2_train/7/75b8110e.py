"""Four registered transparent quadrants, composited in a learned depth order."""

from itertools import permutations

import numpy as np


_SOURCE_ORDER = (1, 2, 3, 0)  # upper right, lower left, lower right, upper left
_SOURCE_INPUTS = {
    "40040505/00405005/00440055/40005005/66609099/66600999/60069909/66069099",
    "40045500/00000055/44040500/40440555/00060909/00600900/60060909/00660009",
    "00045000/40000500/00040050/04040050/60000990/60000909/60609990/60600000",
    "04440555/00405505/00005000/40005000/66060099/00069090/00069099/66060909",
    "44000050/44000000/00400050/04005500/00600090/66600009/60669900/06609000",
    "04045000/04445055/44400555/00005000/60669990/00060900/06000099/60000900",
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = seed % 4 + 1
        low = test + 1
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = test + 1
        train = low + (seed // 4 + 3 * test) % (11 - low)
    elif test is None:
        test = 1 + (seed + 5 * train) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shape(rng, anchor=False):
    if anchor:
        return 4, 4
    kind = int(rng.choice(5, p=[0.15, 0.16, 0.32, 0.18, 0.19]))
    if kind == 0:
        return int(rng.integers(3, 6)), int(rng.integers(3, 6))
    if kind == 1:
        return int(rng.integers(9, 16)), int(rng.integers(9, 16))
    if kind == 2:
        return int(rng.integers(5, 10)), int(rng.integers(5, 10))
    short = int(rng.integers(3, 7))
    long = int(rng.integers(8, 16))
    return (short, long) if kind == 3 else (long, short)


def _mask(rng, height, width):
    """Draw one object's foreground, independently of the other quadrants."""
    size = height * width
    style = int(rng.integers(0, 7))
    mask = np.zeros((height, width), dtype=bool)
    if style == 0:  # scattered marks
        mask = rng.random((height, width)) < rng.uniform(0.10, 0.57)
    elif style == 1:  # islands grown around several centers
        for _ in range(int(rng.integers(2, 7))):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            for _ in range(int(rng.integers(2, max(4, size // 3)))):
                mask[row, col] = True
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                row = min(height - 1, max(0, row + dr))
                col = min(width - 1, max(0, col + dc))
    elif style == 2:  # short bars
        for _ in range(int(rng.integers(1, 5))):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            length = int(rng.integers(2, max(3, min(height, width) + 1)))
            if rng.random() < 0.5:
                mask[row, col:min(width, col + length)] = True
            else:
                mask[row:min(height, row + length), col] = True
    elif style == 3:  # connected patches, sometimes with holes
        for _ in range(int(rng.integers(1, 4))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = min(height, r0 + int(rng.integers(2, max(3, height // 2 + 2))))
            c1 = min(width, c0 + int(rng.integers(2, max(3, width // 2 + 2))))
            mask[r0:r1, c0:c1] = True
        mask &= rng.random((height, width)) >= rng.uniform(0.0, 0.25)
    elif style == 4:  # smooth lopsided region
        noise = rng.random((height, width))
        field = noise.copy()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            field += np.roll(noise, (dr, dc), axis=(0, 1))
        mask = field > rng.uniform(1.9, 3.1)
    elif style == 5:  # near empty
        count = int(rng.integers(1, max(3, size // 7 + 1)))
        mask.reshape(-1)[rng.choice(size, size=count, replace=False)] = True
    else:  # near filled, with visible holes
        mask[:] = True
        holes = int(rng.integers(1, max(3, size // 5 + 1)))
        mask.reshape(-1)[rng.choice(size, size=holes, replace=False)] = False
    if not np.any(mask):
        mask.reshape(-1)[int(rng.integers(size))] = True
    if np.all(mask):
        mask.reshape(-1)[int(rng.integers(size))] = False
    return mask


def _sample_masks(rng, shape, regime, allow_blank):
    height, width = shape
    masks = np.stack([_mask(rng, height, width) for _ in range(4)])
    relationship = int(rng.choice(3, p=([0.53, 0.32, 0.15] if regime == "source_train" else [0.37, 0.33, 0.30])))
    if relationship == 1:  # aligned regions and occlusion
        source, target = (int(x) for x in rng.choice(4, size=2, replace=False))
        shared = rng.random((height, width)) < rng.uniform(0.45, 0.85)
        masks[target, shared] = masks[source, shared]
    elif relationship == 2:  # mostly separated objects
        order = rng.permutation(4)
        occupied = np.zeros((height, width), dtype=bool)
        for quadrant in order:
            keep_overlap = rng.random((height, width)) < rng.uniform(0.04, 0.25)
            masks[quadrant] &= ~occupied | keep_overlap
            occupied |= masks[quadrant]
    if allow_blank and rng.random() < (0.11 if regime == "source_train" else 0.19):
        blank = int(rng.choice(4, p=[0.18, 0.42, 0.20, 0.20]))
        masks[blank] = False
    return masks


def _background_in_each_panel(rng, masks, locked=()):
    height, width = masks.shape[1:]
    available = [n for n in range(height * width) if n not in locked]
    for quadrant in range(4):
        if np.all(masks[quadrant]):
            position = available[int(rng.integers(len(available)))]
            masks[quadrant].reshape(-1)[position] = False


def _palette(rng, base, mode, index):
    if mode == 0:
        return base
    if mode == 1:
        colors = list(base[1:])
        if index:
            colors = list(rng.permutation(colors))
            if index == 1 and tuple(colors) == base[1:]:
                colors[0], colors[1] = colors[1], colors[0]
        return (base[0], *colors)
    if index == 0:
        return base
    values = [int(x) for x in rng.choice(10, size=5, replace=False)]
    if index == 1 and values[0] == base[0]:
        alternate = next(color for color in range(10) if color not in values)
        values[0] = alternate
    return tuple(values)


def _make_pair(masks, palette, order):
    background = palette[0]
    height, width = masks.shape[1:]
    panels = []
    for quadrant in range(4):
        panel = np.full((height, width), background, dtype=np.int8)
        panel[masks[quadrant]] = palette[quadrant + 1]
        panels.append(panel)
    input_grid = np.block([[panels[0], panels[1]], [panels[2], panels[3]]])
    output = np.full((height, width), background, dtype=np.int8)
    for quadrant in reversed(order):
        output[masks[quadrant]] = palette[quadrant + 1]
    return {"input": input_grid, "output": output}


def _array_key(array):
    return (array.shape, tuple(int(x) for x in array.flat))


def _compact(array):
    return "/".join("".join(str(int(x)) for x in row) for row in array)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _checked_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    order = _SOURCE_ORDER if rng.random() < 0.52 else tuple(int(x) for x in rng.permutation(4))
    base = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    mode = int(rng.choice(3, p=[0.26, 0.34, 0.40]))
    source_like = rng.random() < 0.10
    if source_like:
        order = _SOURCE_ORDER
        mode = 0
    total = train_count + test_count
    masks_by_pair = []
    palettes = []
    for index in range(total):
        regime = "source_train" if index < train_count else "target_test"
        anchor = source_like or (index == 0 and rng.random() < 0.40)
        shape = _shape(rng, anchor)
        if not source_like and index == 1 and shape == masks_by_pair[0].shape[1:] and rng.random() < 0.85:
            shape = (min(15, shape[0] + 1), shape[1])
        masks = _sample_masks(rng, shape, regime, allow_blank=index >= 2)
        masks_by_pair.append(masks)
        palettes.append(_palette(rng, base, mode, index))

    # Only eight demonstration cells are controlled, spread over two training
    # examples. The remaining cells keep independently sampled mask geometry.
    witnesses = (0, 1, 2, 4, 8,
                 (1 << order[0]) | (1 << order[1]),
                 (1 << order[1]) | (1 << order[2]),
                 (1 << order[2]) | (1 << order[3]))
    locked_by_pair = []
    for index in (0, 1):
        masks = masks_by_pair[index]
        count = 4
        positions = [int(x) for x in rng.choice(masks.shape[1] * masks.shape[2], size=count, replace=False)]
        locked_by_pair.append(positions)
        for position, signature in zip(positions, witnesses[index * 4:(index + 1) * 4]):
            for quadrant in range(4):
                masks[quadrant].reshape(-1)[position] = bool(signature & (1 << quadrant))
    for index, masks in enumerate(masks_by_pair):
        _background_in_each_panel(rng, masks, locked_by_pair[index] if index < 2 else ())

    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(total):
        regime = "train" if index < train_count else "test"
        pair = _make_pair(masks_by_pair[index], palettes[index], order)
        # Deterministic bounded repair for the extremely unlikely duplicate.
        for attempt in range(24):
            key = _array_key(pair["input"])
            if key not in seen and _compact(pair["input"]) not in _SOURCE_INPUTS:
                break
            if index < 2:
                locked = locked_by_pair[index]
                choices = [x for x in range(masks_by_pair[index].shape[1] * masks_by_pair[index].shape[2]) if x not in locked]
            else:
                choices = list(range(masks_by_pair[index].shape[1] * masks_by_pair[index].shape[2]))
            position = choices[int(rng.integers(len(choices)))]
            quadrant = int(rng.integers(4))
            masks_by_pair[index][quadrant].reshape(-1)[position] ^= True
            _background_in_each_panel(rng, masks_by_pair[index], locked_by_pair[index] if index < 2 else ())
            pair = _make_pair(masks_by_pair[index], palettes[index], order)
        else:
            raise RuntimeError("could not construct a unique pair")
        seen.add(_array_key(pair["input"]))
        puzzle[regime].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    input_grid, output = pair["input"], pair["output"]
    height, width = input_grid.shape
    _require(height % 2 == 0 and width % 2 == 0, "input must have four equal quadrants")
    height //= 2
    width //= 2
    _require(output.shape == (height, width), "output must have one panel's dimensions")
    panels = (input_grid[:height, :width], input_grid[:height, width:],
              input_grid[height:, :width], input_grid[height:, width:])
    common = set(int(x) for x in np.unique(panels[0]))
    for panel in panels[1:]:
        common &= set(int(x) for x in np.unique(panel))
    _require(len(common) == 1, "one shared transparent color is required")
    background = next(iter(common))
    colors = []
    for panel in panels:
        foreground = set(int(x) for x in np.unique(panel)) - {background}
        _require(len(foreground) <= 1, "each quadrant is a monochrome mask")
        _require(np.any(panel == background), "each quadrant must show background")
        colors.append(next(iter(foreground)) if foreground else None)
    present = [color for color in colors if color is not None]
    _require(len(present) == len(set(present)), "foreground colors must be distinct")
    signatures = np.zeros((height, width), dtype=np.uint8)
    for quadrant, panel in enumerate(panels):
        signatures[panel != background] |= np.uint8(1 << quadrant)
    return {"shape": (height, width), "background": background,
            "colors": tuple(colors), "panels": panels, "signatures": signatures}


def _composite(analysis, order):
    result = np.full(analysis["shape"], analysis["background"], dtype=np.int8)
    for quadrant in reversed(order):
        panel = analysis["panels"][quadrant]
        occupied = panel != analysis["background"]
        result[occupied] = panel[occupied]
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape), "grid dimensions")
                _require(np.all((grid >= 0) & (grid <= 9)), "grid colors")
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            _require(_compact(pair["input"]) not in _SOURCE_INPUTS, "source input copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair))

    # Infer depth exclusively from the returned training grids. Every competing
    # order is checked, so test outputs have a uniquely learned convention.
    candidate_orders = [
        order for order in permutations(range(4))
        if all(np.array_equal(_composite(analysis, order), pair["output"])
               for analysis, pair in zip(analyses["train"], puzzle["train"]))
    ]
    _require(len(candidate_orders) == 1, "training does not identify a unique depth order")
    order = candidate_orders[0]
    singletons = set()
    edges = set()
    uncovered = False
    for analysis, pair in zip(analyses["train"], puzzle["train"]):
        colors = analysis["colors"]
        for signature, output_color in zip(analysis["signatures"].flat, pair["output"].flat):
            signature = int(signature)
            if signature == 0:
                uncovered = True
            if signature in (1, 2, 4, 8):
                singletons.add(signature)
            if signature:
                winners = [q for q in range(4) if colors[q] == int(output_color)]
                _require(len(winners) == 1, "output does not copy a present layer")
                winner = winners[0]
                edges.update((winner, loser) for loser in range(4)
                             if loser != winner and signature & (1 << loser))
    _require(uncovered and singletons == {1, 2, 4, 8}, "training lacks transparency or exposed layers")
    _require(all((order[i], order[i + 1]) in edges for i in range(3)),
             "training lacks direct adjacent precedence comparisons")
    for analysis, pair in zip(analyses["test"], puzzle["test"]):
        _require(np.array_equal(_composite(analysis, order), pair["output"]),
                 "test output violates the learned order")


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and
               all(np.array_equal(pa[name], pb[name])
                   for pa, pb in zip(a[split], b[split]) for name in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    shapes, backgrounds, signatures, layouts = set(), set(), set(), set()
    changing_shapes = changing_palettes = blanks = anchors = 0
    wide = tall = large = small = 0
    changed_roles = changed_backgrounds = constant_roles = combined_changes = 0
    train_triples = test_triples = test_sparse_overlap = unseen_test_signatures = 0
    train_near_empty = train_near_filled = test_blank_front = 0
    order_set = set()
    visibility_bins = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        all_analyses = [_analyze_pair(pair) for split in ("train", "test") for pair in puzzle[split]]
        train_count = len(puzzle["train"])
        all_pairs = puzzle["train"] + puzzle["test"]
        shape_set = {a["shape"] for a in all_analyses}
        palette_set = {(a["background"], a["colors"]) for a in all_analyses}
        changing_shapes += len(shape_set) > 1
        changing_palettes += len(palette_set) > 1
        combined_changes += len(shape_set) > 1 and len(palette_set) > 1
        changed_backgrounds += len({a["background"] for a in all_analyses}) > 1
        role_sets = [{a["colors"][q] for a in all_analyses if a["colors"][q] is not None}
                     for q in range(4)]
        changed_roles += any(len(values) > 1 for values in role_sets)
        constant_roles += (len({a["background"] for a in all_analyses}) == 1
                           and all(len(values) == 1 for values in role_sets))
        candidates = [
            order for order in permutations(range(4))
            if all(np.array_equal(_composite(analysis, order), pair["output"])
                   for analysis, pair in zip(all_analyses[:train_count], puzzle["train"]))
        ]
        _require(len(candidates) == 1, "self-test training order ambiguity")
        order = candidates[0]
        order_set.add(order)
        train_signatures = {
            int(x) for analysis in all_analyses[:train_count] for x in analysis["signatures"].flat
        }
        test_signatures = {
            int(x) for analysis in all_analyses[train_count:] for x in analysis["signatures"].flat
        }
        unseen_test_signatures += bool(test_signatures - train_signatures)
        for index, (analysis, pair) in enumerate(zip(all_analyses, all_pairs)):
            shape = analysis["shape"]
            shapes.add(shape)
            backgrounds.add(analysis["background"])
            signature_values = tuple(int(x) for x in analysis["signatures"].flat)
            signatures.update(signature_values)
            layouts.add((shape, signature_values))
            blanks += any(color is None for color in analysis["colors"])
            anchors += shape == (4, 4)
            wide += shape[1] >= 2 * shape[0]
            tall += shape[0] >= 2 * shape[1]
            large += max(shape) >= 10
            small += max(shape) <= 4
            overlaps = sum(x.bit_count() >= 2 for x in signature_values)
            triple = any(x.bit_count() >= 3 for x in signature_values)
            if index < train_count:
                train_triples += triple
                densities = [np.count_nonzero(panel != analysis["background"]) / panel.size
                             for panel in analysis["panels"]]
                train_near_empty += any(0 < density <= 0.10 for density in densities)
                train_near_filled += any(density >= 0.85 for density in densities)
            else:
                test_triples += triple
                test_sparse_overlap += overlaps / len(signature_values) < 0.15
                test_blank_front += analysis["colors"][order[0]] is None
            for q in order[1:]:
                color = analysis["colors"][q]
                if color is None:
                    continue
                occupied = np.count_nonzero(analysis["panels"][q] != analysis["background"])
                if occupied:
                    visible = np.count_nonzero(pair["output"] == color) / occupied
                    visibility_bins.add(0 if visible < 0.30 else 1 if visible < 0.70 else 2)
    _require(len(shapes) >= 25 and len(layouts) >= 200, "shape or layout diversity")
    _require(len(backgrounds) >= 8 and signatures == set(range(16)), "color or overlap diversity")
    _require(changing_shapes >= 25 and changing_palettes >= 20, "within-puzzle variation")
    _require(blanks >= 5 and anchors >= 5 and wide >= 5 and tall >= 5
             and large >= 10 and small >= 5, "mask or size extremes")
    _require(changed_roles >= 15 and changed_backgrounds >= 8 and constant_roles >= 5
             and combined_changes >= 20, "palette roles or combined variation")
    _require(len(order_set) >= 8, "depth order diversity")
    _require(train_triples >= 20 and test_triples >= 10
             and test_sparse_overlap >= 10 and unseen_test_signatures >= 3,
             "overlap distribution diversity")
    _require(train_near_empty >= 20 and train_near_filled >= 10
             and test_blank_front >= 2 and visibility_bins == {0, 1, 2},
             "mask densities, absent layers, or exposure diversity")
    _require(_puzzles_equal(generate_puzzle(314159, 6, 2), generate_puzzle(314159, 6, 2)),
             "explicit determinism")
    _require(_puzzles_equal(generate_puzzle(271828), generate_puzzle(271828)),
             "automatic determinism")
    _require(not _puzzles_equal(puzzles[0], puzzles[1]), "cross-seed difference")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(5000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(6000 + index, train, test)
        _require((train is None or len(puzzle["train"]) == train) and
                 (test is None or len(puzzle["test"]) == test), "partial counts")
    for index, request in enumerate(({"num_train": 1}, {"num_train": 11},
                                     {"num_test": 0}, {"num_test": 5},
                                     {"num_train": 2, "num_test": 2},
                                     {"num_train": 2, "num_test": 3},
                                     {"num_train": 3, "num_test": 4},
                                     {"num_train": 2.5, "num_test": 1})):
        try:
            generate_puzzle(7000 + index, **request)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {request}")
