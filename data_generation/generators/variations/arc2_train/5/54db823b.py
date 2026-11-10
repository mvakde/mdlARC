import numpy as np


_SOURCE_INPUT_STRINGS = {
    "".join((
        "000000000000333",
        "033333000000393",
        "033933033330393",
        "039333039930339",
        "000000033330933",
        "000000033930000",
        "000000000000000",
        "393330000000330",
        "333930000000390",
        "393330033330330",
        "333930039390390",
        "000000033330930",
        "000000039930000",
        "000000033390000",
        "000000000000000",
    )),
    "".join((
        "033300000000000",
        "093900333330000",
        "039300333930000",
        "033900333330000",
        "033300333330000",
        "000000333330000",
        "003930000000000",
        "009330000000000",
        "000000003333333",
        "033333003333333",
        "033333003939933",
        "039333003333333",
        "033333003393333",
        "033393003333339",
        "000000000000000",
    )),
    "".join((
        "000000000000000",
        "339333300033939",
        "333339300033333",
        "393933300039339",
        "333393300033333",
        "339333300000000",
        "000000000000000",
        "000000333333300",
        "033330333339300",
        "039390393333300",
        "033390333933300",
        "093930333333300",
        "033330000000000",
        "000000000000000",
        "000000000000000",
    )),
    "".join((
        "033933000000000",
        "033333039330000",
        "093393033390000",
        "000000093930000",
        "000000033330000",
        "339000000000000",
        "933003930003393",
        "393003390009333",
        "333000000003393",
        "933000000003933",
        "339033393003393",
        "000093933003339",
        "000033333009333",
        "000000000000000",
        "000000000000000",
    )),
    "".join((
        "000000000333330",
        "003333930393330",
        "003933330333330",
        "003333390333330",
        "000000000333330",
        "333330000339330",
        "393930000000000",
        "333330000000000",
        "393930339930000",
        "339330333330000",
        "333330393930333",
        "000000333330339",
        "339300000000333",
        "393900000000399",
        "933300000000333",
    )),
}


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validate_seed(seed: int) -> int:
    valid_type = isinstance(seed, (int, np.integer))
    is_boolean = isinstance(seed, (bool, np.bool_))
    if not valid_type or is_boolean or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _validate_requested_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    valid_type = isinstance(value, (int, np.integer))
    is_boolean = isinstance(value, (bool, np.bool_))
    if not valid_type or is_boolean:
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_requested_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_requested_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This modular schedule guarantees broad count diversity for
        # consecutive seeds while remaining deterministic.
        test_count = 1 + seed % 4
        lowest_train = max(2, test_count + 1)
        number_of_choices = 11 - lowest_train
        train_count = (
            lowest_train
            + (seed // 4) % number_of_choices
        )
        return train_count, test_count

    if train_count is None:
        lowest_train = max(2, int(test_count) + 1)
        if lowest_train > 10:
            raise ValueError("no legal num_train exists for num_test")
        train_count = int(rng.integers(lowest_train, 11))
        return train_count, int(test_count)

    largest_test = min(4, train_count - 1)
    if largest_test < 1:
        raise ValueError("no legal num_test exists for num_train")
    test_count = int(rng.integers(1, largest_test + 1))
    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {"vary_palette": bool(rng.integers(0, 4) == 0),
            "palette": tuple(int(x) for x in rng.choice(10, 3, replace=False))}


def _palette(rng: np.random.Generator, parameters: dict) -> tuple[int, int, int]:
    if parameters["vary_palette"]:
        return tuple(int(x) for x in rng.choice(10, 3, replace=False))
    return parameters["palette"]


def _dimensions(rng: np.random.Generator, object_count: int,
                large_required: bool) -> tuple[int, int]:
    mode = int(rng.integers(0, 5))
    if mode == 0 and not large_required and object_count <= 5:
        return 15, 15
    if mode <= 1 and not large_required:
        side = int(rng.integers(16, 24))
        return side, side + int(rng.integers(-2, 3))
    if mode == 2 and not large_required:
        return int(rng.integers(16, 24)), int(rng.integers(20, 29))
    if mode == 3 and not large_required:
        return int(rng.integers(20, 29)), int(rng.integers(16, 24))
    return int(rng.integers(23, 31)), int(rng.integers(23, 31))


def _shape_options(max_side: int, min_area: int, max_area: int,
                   exact_area: int | None = None) -> list[tuple[int, int]]:
    return [(h, w) for h in range(2, max_side + 1)
            for w in range(2, max_side + 1)
            if min_area <= h * w <= max_area
            and (exact_area is None or h * w == exact_area)]


def _choose_shape(rng: np.random.Generator, max_side: int,
                  min_area: int, max_area: int,
                  exact_area: int | None = None) -> tuple[int, int]:
    options = _shape_options(max_side, min_area, max_area, exact_area)
    if not options:
        raise ValueError("no rectangle shape fits the requested area")
    # Independently favor thin, square, wide, and tall silhouettes.
    style = int(rng.integers(0, 5))
    if style == 0:
        preferred = [s for s in options if min(s) == 2]
    elif style == 1:
        preferred = [s for s in options if abs(s[0] - s[1]) <= 1]
    elif style == 2:
        preferred = [s for s in options if s[0] * 2 <= s[1]]
    elif style == 3:
        preferred = [s for s in options if s[1] * 2 <= s[0]]
    else:
        preferred = options
    pool = preferred or options
    return pool[int(rng.integers(len(pool)))]


def _make_specs(rng: np.random.Generator, object_count: int,
                height: int, width: int, target_count: int,
                special: str) -> list[tuple[int, int, int]]:
    compact = min(height, width) <= 18
    max_side = 6 if compact else 9
    area_cap = 30 if compact else 63
    if special == "source_like":
        target_shape = (6, 5)
    elif special == "training_anchor":
        target_shape = _choose_shape(rng, max_side, 19, min(36, area_cap))
    else:
        low = max(3 * target_count, 6)
        upper = min(area_cap, 36 if compact else 49)
        target_shape = _choose_shape(rng, max_side, low, upper)
    target_area = target_shape[0] * target_shape[1]
    specs = [(target_shape[0], target_shape[1], target_count)]

    if special == "training_anchor":
        # This one demonstration directly refutes area and density shortcuts.
        small_count = target_count + 1
        small_area_max = min(target_area - 1, 24)
        small = _choose_shape(rng, max_side, 3 * small_count,
                              small_area_max)
        specs.append((small[0], small[1], small_count))
        density_count = target_count + 1
        density_min_area = max(3 * density_count,
                               target_area * density_count // target_count + 1)
        density = _choose_shape(rng, max_side, density_min_area, area_cap)
        specs.append((density[0], density[1], density_count))
    else:
        # Independent optional count traps: a larger minimum, an equal-area
        # comparison, and a lower-density nonminimum object.
        traps = list(rng.permutation(3))
        for trap in traps:
            if len(specs) >= object_count:
                break
            count = target_count + int(rng.integers(1, 4))
            if trap == 0 and target_area > 3 * count + 1 and rng.random() < .60:
                shape = _choose_shape(rng, max_side, 3 * count,
                                      target_area - 1)
            elif trap == 1 and target_area >= 3 * count and rng.random() < .55:
                shape = _choose_shape(rng, max_side, target_area,
                                      target_area, target_area)
            elif trap == 2 and rng.random() < .75:
                lower = max(3 * count,
                            target_area * count // target_count + 1)
                options = _shape_options(max_side, lower, area_cap)
                if not options:
                    continue
                shape = options[int(rng.integers(len(options)))]
            else:
                continue
            specs.append((shape[0], shape[1], count))

    while len(specs) < object_count:
        if len(specs) > 2 and rng.random() < .38:
            count = specs[int(rng.integers(1, len(specs)))][2]
        else:
            count = target_count + int(rng.integers(1, 5))
        maximum_count = min(7, area_cap // 3)
        count = min(count, maximum_count)
        shape = _choose_shape(rng, max_side, 3 * count, area_cap)
        specs.append((shape[0], shape[1], count))
    return specs


def _can_place(box: tuple[int, int, int, int],
               existing: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    # Half-open rectangles with one visible background cell of separation.
    return all(r1 < other_r0 or other_r1 < r0
               or c1 < other_c0 or other_c1 < c0
               for other_r0, other_c0, other_r1, other_c1 in existing)


def _position_ok(box: tuple[int, int, int, int],
                 shape: tuple[int, int], location: str) -> bool:
    r0, c0, r1, c1 = box
    height, width = shape
    cr, cc = r0 + r1 - 1, c0 + c1 - 1
    if location == "upper_right":
        return cr < height and cc >= width
    if location == "lower_left":
        return cr >= height and cc < width
    if location == "interior":
        return r0 > 0 and c0 > 0 and r1 < height and c1 < width
    return True


def _place_specs(rng: np.random.Generator, height: int, width: int,
                 specs: list[tuple[int, int, int]], location: str
                 ) -> list[tuple[int, int, int, int]] | None:
    order = [0] + sorted(range(1, len(specs)),
                         key=lambda i: specs[i][0] * specs[i][1], reverse=True)
    for _ in range(16):
        boxes: list[tuple[int, int, int, int]] = []
        result = [None] * len(specs)
        for index in order:
            h, w, _ = specs[index]
            candidates = []
            for r0 in range(height - h + 1):
                for c0 in range(width - w + 1):
                    box = (r0, c0, r0 + h, c0 + w)
                    if index == 0 and not _position_ok(box, (height, width), location):
                        continue
                    if _can_place(box, boxes):
                        candidates.append(box)
            if not candidates:
                break
            if boxes and rng.random() < .55:
                # Select among nearby fits, creating irregular close clusters.
                def gap(box):
                    return min(
                        max(0, b[0]-box[2], box[0]-b[2])
                        + max(0, b[1]-box[3], box[1]-b[3])
                        for b in boxes
                    )
                candidates.sort(key=gap)
                candidates = candidates[:max(4, len(candidates)//3)]
            box = candidates[int(rng.integers(len(candidates)))]
            boxes.append(box)
            result[index] = box
        else:
            return result
    return None


def _generate_pair(rng: np.random.Generator, regime: str,
                   parameters: dict, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
    if regime == "source_train" and pair_index == 0:
        object_count, target_count = 6, 2
        special, location = "training_anchor", "upper_right"
    elif regime == "source_train" and pair_index == 1:
        object_count, target_count = int(rng.integers(3, 6)), 1
        special, location = "ordinary", "lower_left"
    else:
        object_count = int(rng.integers(3, 8))
        target_count = int(rng.integers(1, 4))
        source_like = regime == "target_test" and rng.random() < .24
        if source_like:
            object_count = 6
        special = "source_like" if source_like else "ordinary"
        location = "upper_right" if source_like else (
            "interior" if rng.random() < .30 else "any")
    for attempt in range(45):
        large_required = special in ("training_anchor", "source_like")
        height, width = _dimensions(rng, object_count, large_required)
        if min(height, width) <= 18 and object_count > 5:
            if attempt < 12:
                continue
            height = max(height, 21)
            width = max(width, 21)
        try:
            specs = _make_specs(rng, object_count, height, width,
                                target_count, special)
        except ValueError:
            continue
        boxes = _place_specs(rng, height, width, specs, location)
        if boxes is None:
            continue
        background, carrier, accent = _palette(rng, parameters)
        grid = np.full((height, width), background, dtype=np.int8)
        for box, (_, _, count) in zip(boxes, specs):
            r0, c0, r1, c1 = box
            grid[r0:r1, c0:c1] = carrier
            positions = rng.choice((r1-r0)*(c1-c0), count, replace=False)
            offsets = np.unravel_index(positions, (r1-r0, c1-c0))
            grid[r0 + offsets[0], c0 + offsets[1]] = accent
        # The ambient background must be readily identifiable in each input.
        if np.count_nonzero(grid == background) <= grid.size // 2:
            continue
        output = grid.copy()
        r0, c0, r1, c1 = boxes[0]
        output[r0:r1, c0:c1] = background
        return grid, output
    raise ValueError("could not construct a separated rectangle scene")


def _grid_signature(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def _input_string(grid: np.ndarray) -> str:
    return "".join(str(int(value)) for value in grid.ravel())


def _foreground_components(grid: np.ndarray, background: int
                           ) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    foreground = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for start_r in range(height):
        for start_c in range(width):
            if not foreground[start_r, start_c] or seen[start_r, start_c]:
                continue
            seen[start_r, start_c] = True
            stack = [(start_r, start_c)]
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if (0 <= nr < height and 0 <= nc < width
                            and foreground[nr,nc] and not seen[nr,nc]):
                        seen[nr,nc] = True
                        stack.append((nr,nc))
            components.append(component)
    return components


def _inspect_input(grid: np.ndarray, background: int) -> dict:
    colors = {int(x) for x in np.unique(grid)}
    _check(len(colors) == 3 and background in colors, "three visible roles required")
    components = _foreground_components(grid, background)
    _check(3 <= len(components) <= 7, "three to seven rectangles required")
    foreground = list(colors - {background})
    objects = []
    for cells in components:
        rows, columns = zip(*cells)
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(columns), max(columns)+1
        area = (r1-r0)*(c1-c0)
        _check(len(cells) == area, "every component must fill its rectangle")
        block = grid[r0:r1,c0:c1]
        counts = {color:int(np.count_nonzero(block == color)) for color in foreground}
        objects.append({"bbox":(r0,c0,r1,c1), "area":area, "counts":counts})
    sparse = [color for color in foreground if all(
        0 < obj["counts"][color] <= obj["area"]//3 for obj in objects)]
    _check(len(sparse) == 1, "accent role must be uniquely sparse")
    accent = sparse[0]
    carrier = next(color for color in foreground if color != accent)
    counts = [obj["counts"][accent] for obj in objects]
    _check(counts.count(min(counts)) == 1, "minimum accent count must be unique")
    for obj, count in zip(objects, counts):
        obj["accent_count"] = count
    return {"roles":(background, carrier, accent), "objects":objects,
            "target_index":counts.index(min(counts)),
            "target":objects[counts.index(min(counts))]}


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _check(input_grid.shape == output_grid.shape, "grid dimensions changed")
    changed = input_grid != output_grid
    _check(np.any(changed), "pair must change")
    changed_values = np.unique(output_grid[changed])
    _check(len(changed_values) == 1, "deleted cells must become background")
    background = int(changed_values[0])
    analysis = _inspect_input(input_grid, background)
    expected = input_grid.copy()
    r0,c0,r1,c1 = analysis["target"]["bbox"]
    expected[r0:r1,c0:c1] = background
    _check(np.array_equal(expected, output_grid),
           "output must erase only the unique minimum-count rectangle")
    return analysis


def _predict_from_demonstrations(training: list[dict], test_input: np.ndarray
                                 ) -> np.ndarray:
    # Infer from only the exposed examples and the test input.
    demonstrated = []
    for pair in training:
        changed = pair["input"] != pair["output"]
        background = int(np.unique(pair["output"][changed])[0])
        demonstrated.append(_inspect_input(pair["input"], background)["roles"])
    colors = {int(x) for x in np.unique(test_input)}
    possible = []
    for background in colors:
        try:
            analysis = _inspect_input(test_input, background)
        except AssertionError:
            continue
        roles = analysis["roles"]
        if len(set(demonstrated)) == 1 and roles != demonstrated[0]:
            continue
        result = test_input.copy()
        r0,c0,r1,c1 = analysis["target"]["bbox"]
        result[r0:r1,c0:c1] = background
        possible.append(result)
    _check(len(possible) == 1, "test input does not determine one output")
    return possible[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train":[], "test":[]}
    seen = set()
    for section, count in (("train",train_count),("test",test_count)):
        regime = "source_train" if section == "train" else "target_test"
        for pair_index in range(count):
            for _ in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                try:
                    grid, output = _generate_pair(
                        pair_rng, regime, parameters, pair_index)
                except ValueError:
                    continue
                signature = _grid_signature(grid)
                if signature in seen or (grid.shape == (15,15)
                        and _input_string(grid) in _SOURCE_INPUT_STRINGS):
                    continue
                if (parameters["vary_palette"] and section == "train"
                        and pair_index == 1
                        and _analyze_pair(grid, output)["roles"]
                        == _analyze_pair(puzzle["train"][0]["input"],
                                         puzzle["train"][0]["output"])["roles"]):
                    continue
                if section == "test":
                    try:
                        predicted = _predict_from_demonstrations(
                            puzzle["train"], grid.copy())
                    except AssertionError:
                        continue
                    if not np.array_equal(predicted, output):
                        continue
                seen.add(signature)
                puzzle[section].append({"input":grid,"output":output})
                break
            else:
                raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train","test"},
           "puzzle schema")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train,list) and isinstance(test,list), "pair lists required")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4
           and len(train) > len(test), "illegal pair counts")
    seen = set()
    analyses = {"train":[],"test":[]}
    for section in ("train","test"):
        for pair in puzzle[section]:
            _check(isinstance(pair,dict) and set(pair) == {"input","output"},
                   "pair schema")
            for grid in pair.values():
                _check(isinstance(grid,np.ndarray) and grid.ndim == 2
                       and np.issubdtype(grid.dtype,np.integer), "integer grid required")
                _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                       "grid size out of bounds")
                _check(0 <= int(grid.min()) and int(grid.max()) <= 9,
                       "illegal color")
            grid = pair["input"]
            signature = _grid_signature(grid)
            _check(signature not in seen, "duplicate input")
            seen.add(signature)
            _check(grid.shape != (15,15) or _input_string(grid)
                   not in _SOURCE_INPUT_STRINGS, "source pair copied")
            analysis = _analyze_pair(grid,pair["output"])
            _check(np.count_nonzero(grid == analysis["roles"][0]) > grid.size//2,
                   "background insufficiently visible")
            analyses[section].append(analysis)
    train_roles = [a["roles"] for a in analyses["train"]]
    test_roles = [a["roles"] for a in analyses["test"]]
    if len(set(train_roles)) == 1:
        _check(all(role == train_roles[0] for role in test_roles),
               "new test-only palette convention")
    _check(len(analyses["train"][0]["objects"]) == 6,
           "training must demonstrate six objects")
    _check(_is_upper_right(analyses["train"][0]["target"]["bbox"],
                           train[0]["input"].shape),
           "training must demonstrate upper-right deletion")
    counts = {a["target"]["accent_count"] for a in analyses["train"]}
    _check(1 in counts and 2 in counts, "training must vary target counts")
    has_smaller = any(any(obj["area"] < a["target"]["area"]
                          for j,obj in enumerate(a["objects"])
                          if j != a["target_index"])
                      for a in analyses["train"])
    has_lower_density = any(any(
        obj["accent_count"] * a["target"]["area"]
        < a["target"]["accent_count"] * obj["area"]
        for j,obj in enumerate(a["objects"]) if j != a["target_index"])
        for a in analyses["train"])
    _check(has_smaller and has_lower_density,
           "training must refute area and density shortcuts")
    for pair in test:
        predicted = _predict_from_demonstrations(train, pair["input"].copy())
        _check(np.array_equal(predicted,pair["output"]),
               "test output not recoverable from demonstrations")


def _is_upper_right(bbox: tuple[int,int,int,int], shape: tuple[int,int]) -> bool:
    r0,c0,r1,c1 = bbox
    return r0+r1-1 < shape[0] and c0+c1-1 >= shape[1]


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((section,tuple((_grid_signature(pair["input"]),
                                  _grid_signature(pair["output"]))
                                 for pair in puzzle[section]))
                 for section in ("train","test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({len(p["train"]) for p in puzzles}) >= 4,
           "train counts lack diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3,
           "test counts lack diversity")
    _check(len({_puzzle_signature(p) for p in puzzles}) >= 40,
           "puzzles repeat across seeds")
    for seed, nt, ns in ((100,2,1),(101,10,1),(102,10,4),
                         (103,None,1),(104,None,4),(105,2,None),
                         (106,10,None),(107,7,3)):
        puzzle = generate_puzzle(seed,nt,ns)
        _check(nt is None or len(puzzle["train"]) == nt, "train count ignored")
        _check(ns is None or len(puzzle["test"]) == ns, "test count ignored")
    for nt,ns in ((1,None),(11,None),(None,0),(None,5),
                  (2,2),(3,4),(4,4),(2,4)):
        try:
            generate_puzzle(900,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _check(_puzzle_signature(generate_puzzle(777))
           == _puzzle_signature(generate_puzzle(777)),
           "automatic generation not deterministic")
    _check(_puzzle_signature(generate_puzzle(778,8,3))
           == _puzzle_signature(generate_puzzle(778,8,3)),
           "explicit generation not deterministic")
    shapes, palettes, object_counts, target_counts = set(),set(),set(),set()
    target_sizes, object_sizes, locations = set(),set(),set()
    count_traps = {section:{"equal_area":0,"low_density":0,
                            "smaller_area":0,"ties":0}
                   for section in ("train","test")}
    train_six = train_ur = test_non_ur = compact = two_thick = 0
    source_like_tests = wide_objects = tall_objects = square_objects = 0
    varying_counts = varying_sizes = varying_palettes = varying_objects = 0
    for puzzle in puzzles:
        puzzle_counts, puzzle_sizes, puzzle_palettes, puzzle_objects = (
            set(),set(),set(),set())
        for section in ("train","test"):
            for pair in puzzle[section]:
                a = _analyze_pair(pair["input"], pair["output"])
                objects, target = a["objects"], a["target"]
                shapes.add(pair["input"].shape)
                palettes.add(a["roles"])
                object_counts.add(len(objects))
                target_counts.add(target["accent_count"])
                r0,c0,r1,c1 = target["bbox"]
                target_sizes.add((r1-r0,c1-c0))
                puzzle_counts.add(target["accent_count"])
                puzzle_sizes.add((r1-r0,c1-c0))
                puzzle_palettes.add(a["roles"])
                puzzle_objects.add(len(objects))
                locations.add((r0+r1 < pair["input"].shape[0],
                               c0+c1 < pair["input"].shape[1]))
                compact += pair["input"].shape == (15,15)
                for obj in objects:
                    b = obj["bbox"]
                    height, width = b[2]-b[0], b[3]-b[1]
                    object_sizes.add((height,width))
                    two_thick += min(height,width) == 2
                    wide_objects += width >= 2*height
                    tall_objects += height >= 2*width
                    square_objects += height == width
                other = [obj for i,obj in enumerate(objects)
                         if i != a["target_index"]]
                traps = count_traps[section]
                traps["equal_area"] += any(
                    obj["area"] == target["area"]
                    and obj["accent_count"] > target["accent_count"]
                    for obj in other)
                traps["low_density"] += any(
                    obj["accent_count"]*target["area"]
                    < target["accent_count"]*obj["area"] for obj in other)
                traps["smaller_area"] += any(
                    obj["area"] < target["area"] for obj in other)
                traps["ties"] += (
                    len({obj["accent_count"] for obj in other}) < len(other))
                if section == "train":
                    train_six += len(objects) == 6
                    train_ur += _is_upper_right(target["bbox"],pair["input"].shape)
                else:
                    test_non_ur += not _is_upper_right(target["bbox"],pair["input"].shape)
                    source_like_tests += (
                        len(objects) == 6 and (r1-r0,c1-c0) == (6,5)
                        and target["accent_count"] == 2
                        and _is_upper_right(target["bbox"],pair["input"].shape))
        varying_counts += len(puzzle_counts) > 1
        varying_sizes += len(puzzle_sizes) > 1
        varying_palettes += len(puzzle_palettes) > 1
        varying_objects += len(puzzle_objects) > 1
    _check(len(shapes) >= 35 and len(palettes) >= 24,
           "size or palette diversity inadequate")
    _check({3,4,5,6}.issubset(object_counts) and len(target_sizes) >= 12,
           "object or target size diversity inadequate")
    _check({1,2,3}.issubset(target_counts), "target counts inadequate")
    _check(compact >= 5 and two_thick >= 20,
           "compact or two-thick examples missing")
    _check(train_six >= 42 and train_ur >= 42 and test_non_ur >= 10,
           "test-only signature persists")
    _check(source_like_tests >= 3 and len(object_sizes) >= 20,
           "source-like test or rectangle vocabulary missing")
    _check(min(wide_objects,tall_objects,square_objects) >= 20,
           "rectangle aspect ratios inadequate")
    _check(min(varying_counts,varying_sizes,varying_objects) >= 30
           and varying_palettes >= 3,
           "within-puzzle variation inadequate")
    for section in ("train","test"):
        _check(min(count_traps[section].values()) >= 10,
               f"{section} lacks counting counterexamples")
