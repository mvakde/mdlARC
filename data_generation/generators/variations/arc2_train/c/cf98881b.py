import numpy as np


_SIGNATURE_SEQUENCE = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
_ALL_SIGNATURES = frozenset(_SIGNATURE_SEQUENCE)

def _ensure(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.ascontiguousarray(grid, dtype=np.int8)
    return (
        int(normalized.shape[0]),
        int(normalized.shape[1]),
        normalized.tobytes(),
    )


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_EXAMPLES = (
    (
        (
            "00042000920000",
            "44042909020000",
            "40442099021101",
            "04442090021111",
        ),
        ("0004", "4494", "4944", "1444"),
    ),
    (
        (
            "44442909020001",
            "44002990021000",
            "40442000920101",
            "00002009021010",
        ),
        ("4444", "4400", "4144", "1090"),
    ),
    (
        (
            "44402990920101",
            "04042009020100",
            "04042009921001",
            "40442999020001",
        ),
        ("4449", "0494", "1494", "4944"),
    ),
    (
        (
            "40402000020001",
            "44442000921100",
            "04442099021101",
            "04402009020101",
        ),
        ("4041", "4444", "1444", "0441"),
    ),
    (
        (
            "04042990020000",
            "04002009920100",
            "40002000021110",
            "44442909021101",
        ),
        ("9404", "0499", "4110", "4444"),
    ),
    (
        (
            "00402909021100",
            "44042999021110",
            "00002099921101",
            "04402909921000",
        ),
        ("9140", "4494", "1999", "9449"),
    ),
)

_SOURCE_INPUT_KEYS = frozenset(
    _grid_key(_rows_to_grid(input_rows))
    for input_rows, _ in _SOURCE_EXAMPLES
)

_SOURCE_PAIR_KEYS = frozenset(
    (
        _grid_key(_rows_to_grid(input_rows)),
        _grid_key(_rows_to_grid(output_rows)),
    )
    for input_rows, output_rows in _SOURCE_EXAMPLES
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _validate_count_argument(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_count_argument(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_count_argument(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % number_of_train_choices
        )
    elif train_count is None:
        number_of_train_choices = 10 - test_count
        train_count = test_count + 1 + seed % number_of_train_choices
    else:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("the requested pair counts are impossible")

    return int(train_count), int(test_count)



def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {
        "roles": tuple(int(x) for x in rng.choice(10, size=5, replace=False)),
        "vary_palette": bool(rng.random() < 0.38),
        "vary_size": bool(rng.random() < 0.58),
        "size": _sample_size(rng),
    }


def _sample_size(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < 0.23:
        return (4, 4)
    return (int(rng.integers(4, 13)), int(rng.integers(3, 10)))


def _sample_mask(
    rng: np.random.Generator, height: int, width: int
) -> np.ndarray:
    area = height * width
    band = int(rng.integers(0, 3))
    ranges = ((0.08, 0.27), (0.27, 0.57), (0.57, 0.84))
    low, high = ranges[band]
    count = max(1, min(area - 1, int(round(rng.uniform(low, high) * area))))
    style = int(rng.integers(0, 6))
    score = rng.uniform(-0.5, 0.5, size=(height, width))

    if style in (1, 5):  # bars
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                row = int(rng.integers(height))
                a = int(rng.integers(width))
                b = int(rng.integers(a + 1, width + 1))
                score[row, a:b] += 2.0
            else:
                col = int(rng.integers(width))
                a = int(rng.integers(height))
                b = int(rng.integers(a + 1, height + 1))
                score[a:b, col] += 2.0
    if style in (2, 5):  # rectangles, sometimes overlapping
        for _ in range(int(rng.integers(1, 4))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            score[r0:r1, c0:c1] += 2.0
    if style == 3:  # a soft connected patch with irregular edges
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        rows, cols = np.indices((height, width))
        score += 2.5 - 0.7 * (abs(rows - row) + abs(cols - col))
    if style == 4:  # holes cut from a filled sheet
        score += 1.5
        for _ in range(int(rng.integers(1, 4))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            score[r0:r1, c0:c1] -= 3.0
    # Style zero is a random mosaic. All styles use the same independent
    # occupancy range, so shape and density are not locked together.
    order = np.argsort(score.reshape(-1), kind="stable")
    flat = np.zeros(area, dtype=bool)
    flat[order[-count:]] = True
    return flat.reshape(height, width)


def _apply_interaction(
    rng: np.random.Generator, masks: np.ndarray, mode: str
) -> None:
    height, width = masks.shape[1:]
    area = height * width
    if mode == "hidden_middle":
        masks[1] &= masks[0]
        if not masks[1].any():
            row, col = divmod(int(rng.choice(np.flatnonzero(masks[0]))), width)
            masks[1, row, col] = True
    elif mode == "hidden_right":
        union = masks[0] | masks[1]
        masks[2] &= union
        if not masks[2].any():
            row, col = divmod(int(rng.choice(np.flatnonzero(union))), width)
            masks[2, row, col] = True
    elif mode == "broad_right":
        cells = np.flatnonzero(masks[2])
        if len(cells) < max(3, area // 5):
            extra = rng.choice(area, size=max(3, area // 5), replace=False)
            masks[2].flat[extra] = True
            cells = np.flatnonzero(masks[2])
        chosen = rng.choice(cells, size=min(len(cells), max(3, area // 5)), replace=False)
        masks[0].flat[chosen] = False
        masks[1].flat[chosen] = False
    elif mode == "exclusive":
        for row, col in np.argwhere(np.sum(masks, axis=0) > 1):
            owner = int(rng.choice(np.flatnonzero(masks[:, row, col])))
            masks[:, row, col] = False
            masks[owner, row, col] = True
    elif mode == "overlap":
        selected = rng.choice(area, size=max(2, area // 4), replace=False)
        masks.reshape(3, -1)[:, selected] = True


def _repair_layers(rng: np.random.Generator, masks: np.ndarray, mode: str) -> None:
    height, width = masks.shape[1:]
    if mode == "hidden_middle":
        masks[1] &= masks[0]
    elif mode == "hidden_right":
        masks[2] &= masks[0] | masks[1]
    for layer in range(3):
        if not masks[layer].any():
            if layer == 1 and mode == "hidden_middle":
                candidates = np.flatnonzero(masks[0])
            elif layer == 2 and mode == "hidden_right":
                candidates = np.flatnonzero(masks[0] | masks[1])
            else:
                candidates = np.arange(height * width)
            masks[layer].flat[int(rng.choice(candidates))] = True
        if masks[layer].all():
            index = int(rng.integers(height * width))
            masks[layer].flat[index] = False
            if layer == 0 and mode == "hidden_middle":
                masks[1].flat[index] = False
            if layer in (0, 1) and mode == "hidden_right":
                if not (masks[0].flat[index] or masks[1].flat[index]):
                    masks[2].flat[index] = False


def _apply_witnesses(
    rng: np.random.Generator, masks: np.ndarray, signatures: tuple
) -> np.ndarray:
    area = masks.shape[1] * masks.shape[2]
    selected = rng.choice(area, size=len(signatures), replace=False)
    signatures = [signatures[int(i)] for i in rng.permutation(len(signatures))]
    for index, signature in zip(selected, signatures):
        masks.reshape(3, -1)[:, int(index)] = signature
    return np.asarray(selected, dtype=np.int64)


def _row_policy(
    rng: np.random.Generator, masks: np.ndarray, policy: str,
    protected: np.ndarray
) -> None:
    height, width = masks.shape[1:]
    locked = set(int(x) for x in protected)
    if policy == "empty":
        possible = [
            row for row in range(height)
            if not any(
                row * width + col in locked and masks[0, row, col]
                for col in range(width)
            )
        ]
        if possible:
            row = int(rng.choice(possible))
            masks[0, row] = False
    elif policy == "active":
        for row in range(height):
            if not masks[0, row].any():
                possible = [col for col in range(width) if row * width + col not in locked]
                if possible:
                    masks[0, row, int(rng.choice(possible))] = True


def _construct_grids(
    masks: np.ndarray, roles: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    background, divider, *payloads = roles
    height, width = masks.shape[1:]
    panels = []
    for layer in range(3):
        panel = np.full((height, width), background, dtype=np.int8)
        panel[masks[layer]] = payloads[layer]
        panels.append(panel)
    separator = np.full((height, 1), divider, dtype=np.int8)
    input_grid = np.concatenate(
        (panels[0], separator, panels[1], separator, panels[2]), axis=1
    )
    output = np.full((height, width), background, dtype=np.int8)
    for layer in (2, 1, 0):
        output[masks[layer]] = payloads[layer]
    return input_grid, output


def _inspect_pair(pair: dict) -> tuple[tuple, np.ndarray]:
    source, target = pair["input"], pair["output"]
    height, width = target.shape
    _ensure(source.shape == (height, 3 * width + 2), "incorrect panel dimensions")
    _ensure(3 <= width <= 9 and 4 <= height <= 12, "unsupported panel size")
    divider_a = source[:, width]
    divider_b = source[:, 2 * width + 1]
    _ensure(np.all(divider_a == divider_a[0]) and np.array_equal(divider_a, divider_b),
            "dividers differ")
    divider = int(divider_a[0])
    panels = np.stack((source[:, :width], source[:, width + 1:2 * width + 1],
                       source[:, 2 * width + 2:]), axis=0)
    color_sets = [set(int(c) for c in np.unique(panel)) for panel in panels]
    _ensure(all(len(colors) == 2 for colors in color_sets), "panel needs background and payload")
    shared = color_sets[0] & color_sets[1] & color_sets[2]
    _ensure(len(shared) == 1, "background is not identifiable")
    background = next(iter(shared))
    payloads = tuple(next(iter(colors - {background})) for colors in color_sets)
    _ensure(len({background, divider, *payloads}) == 5, "color roles collide")
    masks = panels != background
    expected = np.full((height, width), background, dtype=np.int8)
    for layer in (2, 1, 0):
        expected[masks[layer]] = payloads[layer]
    _ensure(np.array_equal(target, expected), "incorrect priority composite")
    return (height, width, background, divider, *payloads), masks


def _signature_set(masks: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in bits) for bits in masks.reshape(3, -1).T}


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
            "puzzle schema")
    _ensure(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
            "pair lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen = set()
    training_signatures = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"},
                    "pair schema")
            for grid in pair.values():
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2,
                        "grid type or rank")
                _ensure(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _ensure(all(1 <= dim <= 30 for dim in grid.shape), "grid bounds")
                _ensure(0 <= int(grid.min()) <= int(grid.max()) <= 9, "ARC colors")
            key = _grid_key(pair["input"])
            _ensure(key not in seen and key not in _SOURCE_INPUT_KEYS,
                    "duplicate or source input")
            seen.add(key)
            _ensure((key, _grid_key(pair["output"])) not in _SOURCE_PAIR_KEYS,
                    "source pair")
            _, masks = _inspect_pair(pair)
            if section == "train":
                training_signatures.update(_signature_set(masks))
    _ensure(training_signatures == _ALL_SIGNATURES,
            "training lacks a visible priority or background witness")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    parameters = sample_puzzle_parameters(rng)
    total = train_count + test_count
    evidence_pairs = rng.choice(train_count, size=2, replace=False)
    signatures = (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)),
        ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)),
    )
    modes = ("free", "hidden_middle", "hidden_right", "broad_right",
             "exclusive", "overlap")
    puzzle = {"train": [], "test": []}
    seen = set()
    pair_seeds = rng.integers(0, 2**32, size=(total, 4), dtype=np.uint32)
    for index in range(total):
        pair_rng = np.random.default_rng(np.random.SeedSequence(
            [int(v) for v in pair_seeds[index]] + [index]))
        section = "train" if index < train_count else "test"
        witness = None
        for slot, evidence_index in enumerate(evidence_pairs):
            if index == int(evidence_index):
                witness = signatures[slot]
        for _ in range(30):
            height, width = (_sample_size(pair_rng) if parameters["vary_size"]
                             else parameters["size"])
            roles = (tuple(int(x) for x in pair_rng.choice(10, size=5, replace=False))
                     if parameters["vary_palette"] else parameters["roles"])
            masks = np.stack([_sample_mask(pair_rng, height, width) for _ in range(3)])
            mode = "free" if witness is not None else str(pair_rng.choice(modes))
            _apply_interaction(pair_rng, masks, mode)
            protected = (np.empty(0, dtype=np.int64) if witness is None
                         else _apply_witnesses(pair_rng, masks, witness))
            policy = str(pair_rng.choice(("empty", "active", "natural"),
                                         p=(0.28, 0.32, 0.40)))
            _row_policy(pair_rng, masks, policy, protected)
            _repair_layers(pair_rng, masks, mode)
            source, target = _construct_grids(masks, roles)
            key = _grid_key(source)
            if key not in seen and key not in _SOURCE_INPUT_KEYS:
                break
        else:
            raise RuntimeError("could not construct a unique pair")
        seen.add(key)
        puzzle[section].append({"input": source, "output": target})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((section, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                  for p in puzzle[section])) for section in ("train", "test"))


def _has_isolated_cell(mask: np.ndarray) -> bool:
    adjacent = np.zeros_like(mask)
    adjacent[1:] |= mask[:-1]
    adjacent[:-1] |= mask[1:]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    return bool(np.any(mask & ~adjacent))


def self_test() -> None:
    for seed, counts in ((100, (None, None)), (101, (7, 3)),
                         (102, (2, 1)), (103, (10, 1)), (104, (10, 4)),
                         (105, (None, 1)), (106, (None, 4)),
                         (107, (2, None)), (108, (10, None))):
        first = generate_puzzle(seed, *counts)
        second = generate_puzzle(seed, *counts)
        _ensure(_puzzle_key(first) == _puzzle_key(second), "nondeterministic puzzle")
        validate_generated_puzzle(first)
        if counts[0] is not None:
            _ensure(len(first["train"]) == counts[0], "train count ignored")
        if counts[1] is not None:
            _ensure(len(first["test"]) == counts[1], "test count ignored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(200, *counts)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    empty_train = empty_test = active_test = broad_right = overlap = 0
    hidden_middle = hidden_right = exclusive = bars = blocks = holes = isolated = 0
    edge_background = pairwise = combined = 0
    sparse_train = sparse_test = dense_train = dense_test = 0
    varying_shape = varying_palette = fixed_both = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_shapes, local_palettes = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                roles, masks = _inspect_pair(pair)
                height, width = roles[:2]
                shape = height, width
                shapes.add(shape)
                local_shapes.add(shape)
                palettes.add(roles[2:])
                local_palettes.add(roles[2:])
                layouts.add((shape, np.packbits(masks.reshape(-1)).tobytes()))
                top, middle, right = masks
                edge_background += bool(np.any(~top[:, 0]))
                hidden_middle += bool(not np.any(middle & ~top))
                hidden_right += bool(not np.any(right & ~top & ~middle))
                exclusive += bool(np.sum(np.sum(masks, axis=0) == 1) >= height * width // 2)
                pairwise += bool(np.any(top & middle) or np.any(top & right)
                                 or np.any(middle & right))
                bars += any(np.any(np.sum(mask[:, j:j+4], axis=1) == 4)
                            for mask in masks for j in range(max(0, width-3))) or any(
                                np.any(np.sum(mask[i:i+4, :], axis=0) == 4)
                                for mask in masks for i in range(max(0, height-3)))
                blocks += any(np.any(mask[:-1, :-1] & mask[1:, :-1]
                                     & mask[:-1, 1:] & mask[1:, 1:]) for mask in masks)
                holes += any(np.any((~mask[1:-1, 1:-1]) & mask[:-2, 1:-1]
                                    & mask[2:, 1:-1] & mask[1:-1, :-2]
                                    & mask[1:-1, 2:]) for mask in masks)
                isolated += any(_has_isolated_cell(mask) for mask in masks)
                occupied = [int(mask.sum()) / (height * width) for mask in masks]
                combined += bool(np.any(~top.any(axis=1))
                                 and any(x >= .60 for x in occupied[1:]))
                if section == "train":
                    empty_train += bool(np.any(~top.any(axis=1)))
                    sparse_train += any(x <= .28 for x in occupied[1:])
                    dense_train += any(x >= .60 for x in occupied[1:])
                else:
                    empty_test += bool(np.any(~top.any(axis=1)))
                    active_test += bool(np.all(top.any(axis=1)))
                    sparse_test += any(x <= .28 for x in occupied[1:])
                    dense_test += any(x >= .60 for x in occupied[1:])
                broad_right += int(np.sum(right & ~top & ~middle) >= max(3, height * width // 6))
                overlap += int(np.sum(top & middle & right) >= max(2, height * width // 6))
        varying_shape += len(local_shapes) > 1
        varying_palette += len(local_palettes) > 1
        fixed_both += len(local_shapes) == 1 and len(local_palettes) == 1
    _ensure(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _ensure(len(shapes) >= 18 and (4, 4) in shapes, "size diversity")
    _ensure(any(h >= 10 and w <= 4 for h, w in shapes), "no tall thin panel")
    _ensure(len(layouts) >= 230 and len(palettes) >= 30, "layout or palette diversity")
    _ensure(varying_shape >= 10 and varying_palette >= 6 and fixed_both >= 5,
            "pair-local or fixed-convention diversity")
    _ensure(min(empty_train, empty_test, active_test, hidden_middle,
                hidden_right, broad_right, overlap, exclusive, bars, blocks,
                holes, isolated, edge_background, pairwise, combined, sparse_train,
                sparse_test, dense_train, dense_test) > 0,
            "reviewed regimes lack coverage")
