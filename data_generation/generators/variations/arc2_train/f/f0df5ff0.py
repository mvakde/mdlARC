import numpy as np


_SOURCE_PAIRS = (
    (
        "006200060000004/000002000260400/630104000006000/"
        "004060010000300/603000000322004/420202000060060/"
        "000026060040000/060000400046000/000606003340660/"
        "460313004002600/003204060043600/040000020004000/"
        "000100030300220/600000200010043/000003400200000",
        "006200060000004/001112000260400/631114111006000/"
        "004160111000300/603000111322004/420202000060060/"
        "000026060040000/060000400046000/000616003340660/"
        "460313004002600/003214060043600/041110020004000/"
        "001110030311220/601110200111043/000003400211000",
    ),
    (
        "300000092302330/222300357000270/032200070500050/"
        "000020000900292/800300128200000/300320007020300/"
        "003000300560200/010236002302060/028030006070030/"
        "023080030100600/703002000000670/002052000700000/"
        "090200000200300/002023300010062/029005230000200",
        "300000092302330/222300357000270/032200070500050/"
        "000021110900292/800301128200000/300321117020300/"
        "113000300560200/111236002302060/128030006170030/"
        "023080031110600/703002001110670/002052000700000/"
        "090200000211300/002023300111062/029005230111200",
    ),
    (
        "390000000839308/000404003027702/033090900002000/"
        "000090403033010/010088030293000/090800003007003/"
        "007224709000008/040070008033270/033020102330004/"
        "000308000703010/080030990073900/443030780407309/"
        "701330701700409/300078800809000/007009800483000",
        "390000000839308/000404003027702/033090900002111/"
        "111090403033111/111088030293111/191800003007003/"
        "007224709000008/040071118033270/033021112330114/"
        "000308110703111/080030990073911/443130781407309/"
        "711330711700409/311178811809000/007009800483000",
    ),
    (
        "000700607000003/204037007070008/000780627010272/"
        "010020026000078/600601000200864/004660048008087/"
        "876000077448007/300100300007080/008686761660407/"
        "087078070800804/440003002003848/008010837678087/"
        "000080060300300/060000631030013/460000800022006",
        "000700607000003/204037007171008/111780627111272/"
        "111021126111078/611611100200864/004661148008087/"
        "876110077448007/301110311107080/008686761660407/"
        "087078071800804/440113002003848/008111837678087/"
        "000181061300311/060000631130113/460000811122116",
    ),
)


def _compact_grid(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    def checked(
        name: str,
        value: int | None,
        lower: int,
        upper: int,
    ) -> int | None:
        if value is None:
            return None

        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{name} must be an integer or None")

        value = int(value)
        if not lower <= value <= upper:
            raise ValueError(
                f"{name} must be between {lower} and {upper}"
            )
        return value

    num_train = checked("num_train", num_train, 2, 10)
    num_test = checked("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    return {
        "active": colors[0],
        "background": colors[1],
        "vary_background": bool(rng.random() < 0.68),
        "novel_passive": colors[2] if rng.random() < 0.38 else None,
        "allow_overlap": bool(rng.random() < 0.56),
        "allow_boundary": bool(rng.random() < 0.72),
    }


def _stamp_mask(shape: tuple[int, int], seeds: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for row, col in seeds:
        mask[max(0, row - 1):min(height, row + 2),
             max(0, col - 1):min(width, col + 2)] = True
    return mask


def _apply_stamp(grid: np.ndarray, background: int, active: int) -> np.ndarray:
    result = grid.copy()
    seeds = [tuple(map(int, p)) for p in np.argwhere(grid == active)]
    result[_stamp_mask(grid.shape, seeds) & (grid == background)] = active
    return result


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    bands = ((7, 11), (12, 20), (21, 30))
    sizes = []
    for _ in range(2):
        lo, hi = bands[int(rng.choice(3, p=[0.24, 0.45, 0.31]))]
        sizes.append(int(rng.integers(lo, hi + 1)))
    return sizes[0], sizes[1]


def _position_pool(shape: tuple[int, int], kind: str) -> list[tuple[int, int]]:
    height, width = shape
    if kind == "corner":
        return [(0, 0), (0, width - 1), (height - 1, 0),
                (height - 1, width - 1)]
    if kind == "edge":
        return [(r, c) for r in range(height) for c in range(width)
                if (r in (0, height - 1) or c in (0, width - 1))
                and (r not in (0, height - 1) or c not in (0, width - 1))]
    if kind == "inset":
        return [(r, c) for r in range(1, height - 1)
                for c in range(1, width - 1)
                if r in (1, height - 2) or c in (1, width - 2)]
    return [(r, c) for r in range(2, height - 2)
            for c in range(2, width - 2)]


def _seed_positions(rng: np.random.Generator, shape: tuple[int, int],
                    count: int, kind: str, overlap: bool,
                    exemplar: bool, permit_boundary: bool) -> list[tuple[int, int]]:
    pool = _position_pool(shape, kind)
    first = pool[int(rng.integers(len(pool)))]
    seeds = [first]
    if overlap and count >= 2:
        choices = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                   if max(abs(r - first[0]), abs(c - first[1])) == 2
                   and (permit_boundary or
                        (0 < r < shape[0] - 1 and 0 < c < shape[1] - 1))]
        seeds.append(choices[int(rng.integers(len(choices)))])
    spacing = 3 if exemplar or not overlap else 2
    candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                  if permit_boundary or (0 < r < shape[0] - 1
                                         and 0 < c < shape[1] - 1)]
    for index in rng.permutation(len(candidates)):
        if len(seeds) >= count:
            break
        position = candidates[int(index)]
        if all(max(abs(position[0] - row), abs(position[1] - col))
               >= spacing for row, col in seeds):
            seeds.append(position)
    return seeds


def _choose_background(rng: np.random.Generator, params: dict,
                       split: str, index: int) -> int:
    if not params["vary_background"]:
        return params["background"]
    choices = [c for c in range(10) if c != params["active"]
               and c != params["novel_passive"]]
    if split == "train" and index == 0:
        return params["background"]
    if split == "train" and index == 1:
        choices = [c for c in choices if c != params["background"]]
    return choices[int(rng.integers(len(choices)))]


def _generate_pair(rng: np.random.Generator, regime: str, params: dict,
                   index: int) -> tuple[np.ndarray, np.ndarray]:
    shape = _sample_shape(rng)
    height, width = shape
    exemplar = regime == "source_train" and index == 0
    force_overlap = (regime == "source_train" and index == 1
                     and params["allow_overlap"])
    overlap = force_overlap or (params["allow_overlap"] and not exemplar
                                and rng.random() < 0.38)
    force_boundary = (regime == "source_train" and index == 1
                      and params["allow_boundary"])
    boundary_allowed = params["allow_boundary"] and not exemplar
    if force_boundary:
        kind = "corner" if rng.random() < 0.43 else "edge"
    elif boundary_allowed and rng.random() < (0.48 if regime == "target_test" else 0.26):
        kind = "corner" if rng.random() < 0.25 else "edge"
    else:
        kind = "inset" if not exemplar and rng.random() < 0.37 else "interior"
    maximum = min(11, max(2, height * width // 17))
    if exemplar:
        desired = int(rng.integers(1, min(4, maximum) + 1))
    else:
        desired = int(rng.integers(1, maximum + 1))
        if regime == "target_test" and rng.random() < 0.48:
            desired = min(maximum, desired + int(rng.integers(1, 3)))
    if force_overlap:
        desired = max(2, desired)
    seeds = _seed_positions(rng, shape, desired, kind, overlap, exemplar,
                            params["allow_boundary"] and not exemplar)
    stamp = _stamp_mask(shape, seeds)
    active = params["active"]
    background = _choose_background(rng, params,
                                    "train" if regime == "source_train" else "test",
                                    index)
    grid = np.full(shape, background, dtype=np.int8)
    for row, col in seeds:
        grid[row, col] = active
    protected: set[tuple[int, int]] = set()
    adjacent = [(r, c) for r in range(height) for c in range(width)
                if max(abs(r - seeds[0][0]), abs(c - seeds[0][1])) == 1
                and grid[r, c] == background]
    protected.add(adjacent[int(rng.integers(len(adjacent)))])
    if exemplar:
        orthogonal = [(r, c) for r, c in adjacent
                      if r == seeds[0][0] or c == seeds[0][1]]
        diagonal = [(r, c) for r, c in adjacent
                    if r != seeds[0][0] and c != seeds[0][1]]
        protected.add(orthogonal[int(rng.integers(len(orthogonal)))])
        protected.add(diagonal[int(rng.integers(len(diagonal)))])
    outside = [(r, c) for r in range(height) for c in range(width)
               if not stamp[r, c]]
    if outside:
        protected.add(outside[int(rng.integers(len(outside)))])
    reserved = params["novel_passive"]
    color_pool = [c for c in range(10) if c not in (background, active)
                  and (regime != "source_train" or c != reserved)]
    count = int(rng.integers(1, len(color_pool) + 1))
    selected = [int(c) for c in rng.choice(color_pool, size=count, replace=False)]
    if regime == "target_test" and reserved is not None and index == 0:
        if reserved not in selected:
            selected[-1] = reserved
    prefilled: set[tuple[int, int]] = set()
    if exemplar:
        blocker_choices = [p for p in adjacent if p not in protected]
        blocker = blocker_choices[int(rng.integers(len(blocker_choices)))]
        grid[blocker] = selected[0]
        prefilled.add(blocker)
    free = [(r, c) for r in range(height) for c in range(width)
            if grid[r, c] == background and (r, c) not in protected]
    order = rng.permutation(len(free))
    free = [free[int(i)] for i in order]
    remaining_colors = selected[1:] if exemplar else selected
    for color in remaining_colors:
        row, col = free.pop()
        grid[row, col] = color
        prefilled.add((row, col))
    # Short connected passive runs or patches coexist with isolated dots.
    for _ in range(int(rng.integers(0, 4))):
        if not free:
            break
        center = free[int(rng.integers(len(free)))]
        direction = int(rng.integers(0, 3))
        patch = ([(center[0], center[1] + d) for d in range(3)]
                 if direction == 0 else
                 [(center[0] + d, center[1]) for d in range(3)]
                 if direction == 1 else
                 [(center[0] + dr, center[1] + dc)
                  for dr in range(2) for dc in range(2)])
        available = [p for p in patch if p in free]
        if len(available) >= 2:
            color = selected[int(rng.integers(len(selected)))]
            for row, col in available:
                grid[row, col] = color
                prefilled.add((row, col))
                free.remove((row, col))
    bands = ((0.04, 0.17), (0.17, 0.34), (0.34, 0.49))
    probs = [0.19, 0.43, 0.38] if regime == "source_train" else [0.12, 0.35, 0.53]
    low, high = bands[int(rng.choice(3, p=probs))]
    density = float(rng.uniform(low, high))
    area = height * width
    target = max(len(prefilled), int(round(density * area)))
    target = min(target, (area - len(seeds) - 2) // 2)
    local_weights = np.ones(shape, dtype=float)
    for row, col in seeds:
        weight = float(rng.choice([0.12, 0.4, 1.0, 2.8, 7.0]))
        local_weights[max(0, row - 1):min(height, row + 2),
                      max(0, col - 1):min(width, col + 2)] = weight
    if free and target > len(prefilled):
        scores = [float(-np.log(max(rng.random(), 1e-12)) /
                        local_weights[row, col]) for row, col in free]
        chosen = np.argsort(scores)[:target - len(prefilled)]
        for i in chosen:
            row, col = free[int(i)]
            grid[row, col] = selected[int(rng.integers(len(selected)))]
    return grid, _apply_stamp(grid, background, active)


def _infer_active(train: list[dict]) -> int:
    colors = set()
    for pair in train:
        changed = pair["input"] != pair["output"]
        assert np.any(changed)
        after = np.unique(pair["output"][changed])
        assert len(after) == 1
        colors.add(int(after[0]))
    assert len(colors) == 1
    return colors.pop()


def _modal_background(grid: np.ndarray, active: int) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert background != active
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    return background


def _independent_prediction(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    active = _infer_active(train)
    background = _modal_background(test_input, active)
    output = test_input.copy()
    height, width = output.shape
    for row in range(height):
        for col in range(width):
            if test_input[row, col] != background:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < height and 0 <= cc < width and test_input[rr, cc] == active:
                        output[row, col] = active
    return output


def _pair_features(pair: dict, active: int) -> dict:
    grid = pair["input"]
    background = _modal_background(grid, active)
    seeds = [tuple(map(int, p)) for p in np.argwhere(grid == active)]
    mask = _stamp_mask(grid.shape, seeds)
    passive = (grid != background) & (grid != active)
    changed = grid != pair["output"]
    overlap = any(max(abs(r - s), abs(c - t)) == 2
                  for i, (r, c) in enumerate(seeds)
                  for s, t in seeds[i + 1:])
    boundary = any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                   for r, c in seeds)
    orth = diag = False
    for r, c in np.argwhere(changed):
        for sr, sc in seeds:
            dr, dc = abs(int(r) - sr), abs(int(c) - sc)
            if dr + dc == 1:
                orth = True
            if dr == dc == 1:
                diag = True
    return {"background": background, "seeds": seeds,
            "overlap": overlap, "boundary": boundary,
            "blocker": bool(np.any(mask & passive)),
            "outside": bool(np.any(~mask & (grid == background))),
            "orth": orth, "diag": diag,
            "passive": {int(x) for x in np.unique(grid[passive])}}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count, regime in (("train", train_count, "source_train"),
                                 ("test", test_count, "target_test")):
        for index in range(count):
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                grid, output = _generate_pair(pair_rng, regime, params, index)
                key = _grid_key(grid)
                if key not in used:
                    used.add(key)
                    puzzle[split].append({"input": grid, "output": output})
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            ikey, pkey = _grid_key(pair["input"]), _pair_key(pair)
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert (_compact_grid(pair["input"]),
                    _compact_grid(pair["output"])) not in _SOURCE_PAIRS
    active = _infer_active(puzzle["train"])
    train_features = []
    test_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid, output = pair["input"], pair["output"]
            background = _modal_background(grid, active)
            assert np.any(grid == active)
            changed = grid != output
            assert np.any(changed)
            assert np.all(grid[changed] == background)
            assert np.all(output[changed] == active)
            assert np.array_equal(output, _apply_stamp(grid, background, active))
            assert np.any((grid == background) &
                          ~_stamp_mask(grid.shape, [tuple(map(int, p)) for p in np.argwhere(grid == active)]))
            feature = _pair_features(pair, active)
            for i, (r, c) in enumerate(feature["seeds"]):
                for s, t in feature["seeds"][i + 1:]:
                    assert max(abs(r - s), abs(c - t)) >= 2
            (train_features if split == "train" else test_features).append(feature)
    assert any(f["blocker"] for f in train_features)
    assert any(f["orth"] and f["diag"] for f in train_features)
    assert all(f["outside"] for f in train_features + test_features)
    assert not any(f["overlap"] for f in test_features) or any(f["overlap"] for f in train_features)
    assert not any(f["boundary"] for f in test_features) or any(f["boundary"] for f in train_features)
    for pair in puzzle["test"]:
        assert np.array_equal(_independent_prediction(puzzle["train"], pair["input"]),
                              pair["output"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(31415, 7, 3), generate_puzzle(31415, 7, 3))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    shapes, roles, seed_counts, palettes, layouts = set(), set(), set(), set(), set()
    densities, kinds = {"train": [], "test": []}, set()
    positions = {"train": set(), "test": set()}
    backgrounds = novelty = fewer = more = equal = False
    fixed_background = palette_variation = separated = both_overlap = False
    clear_stamp = heavy_stamp = multicolor_stamp = passive_run = False
    small = large = elongated = combined = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        active = _infer_active(puzzle["train"])
        roles.add(active)
        features = {s: [_pair_features(p, active) for p in puzzle[s]]
                    for s in ("train", "test")}
        train_passive = set().union(*(f["passive"] for f in features["train"]))
        test_passive = set().union(*(f["passive"] for f in features["test"]))
        novelty |= bool(test_passive - train_passive)
        backgrounds |= len({f["background"] for f in features["train"]}) > 1
        fixed_background |= len({f["background"] for f in features["train"]}) == 1
        palette_variation |= len({tuple(sorted(f["passive"]))
                                  for f in features["train"]}) > 1
        train_counts = [len(f["seeds"]) for f in features["train"]]
        test_counts = [len(f["seeds"]) for f in features["test"]]
        fewer |= min(test_counts) < max(train_counts)
        more |= max(test_counts) > max(train_counts)
        equal |= bool(set(test_counts) & set(train_counts))
        separated |= not any(f["overlap"] for fs in features.values() for f in fs)
        both_overlap |= (any(f["overlap"] for f in features["train"])
                         and any(f["overlap"] for f in features["test"]))
        for split in ("train", "test"):
            for pair, f in zip(puzzle[split], features[split]):
                grid = pair["input"]
                height, width = grid.shape
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple(sorted(f["seeds"]))))
                seed_counts.add(len(f["seeds"]))
                palettes.add(len(f["passive"]))
                density = np.mean((grid != f["background"]) & (grid != active))
                densities[split].append(density)
                kinds.add((f["overlap"], f["boundary"]))
                small |= min(height, width) <= 11
                large |= max(height, width) >= 25
                elongated |= max(height, width) / min(height, width) >= 2
                combined |= density >= 0.35 and f["boundary"] and len(f["passive"]) >= 7
                passive_run |= any(grid[row, col] == grid[row, col + 1]
                                   and grid[row, col] not in (f["background"], active)
                                   for row in range(height) for col in range(width - 1))
                for row, col in f["seeds"]:
                    if row in (0, height - 1) and col in (0, width - 1):
                        positions[split].add("corner")
                    elif row in (0, height - 1) or col in (0, width - 1):
                        positions[split].add("edge")
                    elif row in (1, height - 2) or col in (1, width - 2):
                        positions[split].add("inset")
                    else:
                        positions[split].add("interior")
                    patch = grid[max(0, row - 1):min(height, row + 2),
                                 max(0, col - 1):min(width, col + 2)]
                    blockers = (patch != f["background"]) & (patch != active)
                    blocked_fraction = np.mean(blockers)
                    clear_stamp |= blocked_fraction <= 0.125
                    heavy_stamp |= blocked_fraction >= 0.5
                    multicolor_stamp |= len(np.unique(patch[blockers])) >= 2
                assert np.array_equal(_independent_prediction(puzzle["train"], grid), pair["output"])
    assert len(shapes) >= 80 and len(layouts) >= 200 and len(roles) >= 8
    assert min(seed_counts) <= 2 and max(seed_counts) >= 8
    assert min(palettes) == 1 and max(palettes) >= 7
    for split in ("train", "test"):
        assert min(densities[split]) < 0.15 and max(densities[split]) > 0.42
        assert positions[split] == {"corner", "edge", "inset", "interior"}
    assert (False, False) in kinds and (True, True) in kinds
    assert backgrounds and fixed_background and palette_variation and novelty
    assert fewer and more and equal and separated and both_overlap
    assert clear_stamp and heavy_stamp and multicolor_stamp and passive_run
    assert small and large and elongated and combined
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(20000 + 100 * train_count + test_count,
                            train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for i, (train_count, test_count) in enumerate(((None, 1), (None, 4),
                                                   (2, None), (10, None))):
        p = generate_puzzle(30000 + i, train_count, test_count)
        assert train_count is None or len(p["train"]) == train_count
        assert test_count is None or len(p["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (False, 1)):
        try:
            generate_puzzle(40000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
