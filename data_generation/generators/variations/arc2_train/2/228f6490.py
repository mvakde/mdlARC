"""Generate exact key-to-cavity ARC puzzles."""

import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SOURCE_PAIRS = {
    ("5555500000/5000509999/5550509999/5555500000/0000000606/3330006600/0035555550/0005000050/6605000050/6605555550",
     "5555500000/5333500000/5553500000/5555500000/0000000606/0000006600/0005555550/0005999950/6605999950/6605555550"),
    ("7000000077/0555550000/0500550660/0500550000/0555550000/0555550070/0000000000/0000755555/0880055005/0880055555",
     "7000000077/0555550000/0588550000/0588550000/0555550000/0555550070/0000000000/0000755555/0000055665/0000055555"),
    ("2200555555/2220500055/0000555005/0440555555/0040040000/5555500440/5555500000/5005500004/5000508880/5555500088",
     "0000555555/0000588855/0000555885/0440555555/0040040000/5555500440/5555500000/5225500004/5222500000/5555500000"),
    ("0055555002/2050005000/0055055444/0055555000/0055555002/7770020200/0700000000/0005555555/0205000555/2005555555",
     "0055555002/2057775000/0055755000/0055555000/0055555002/0000020200/0000000000/0005555555/0205444555/2005555555"),
}


def _compact(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _normalize(cells):
    lo_r = min(r for r, _ in cells)
    lo_c = min(c for _, c in cells)
    return tuple(sorted((r - lo_r, c - lo_c) for r, c in cells))


def _components(cells):
    remaining = set(cells)
    answer = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        answer.append(part)
    return answer


def _cells(grid, color):
    return {tuple(map(int, p)) for p in np.argwhere(grid == color)}


def _shape(rng, compact=False, area=None):
    """Sample a connected silhouette, then choose a common donor/cavity orientation."""
    if compact:
        area = int(rng.integers(2, 4)) if area is None else area
        mode = int(rng.integers(0, 3))
    else:
        area = int(rng.integers(2, 10)) if area is None else area
        mode = int(rng.integers(0, 5))
    if mode == 0:
        shape = {(0, c) for c in range(area)}
    elif mode == 1 and area >= 4:
        divisors = [(h, area // h) for h in range(2, 5) if area % h == 0 and area // h <= 5]
        if divisors:
            h, w = divisors[int(rng.integers(len(divisors)))]
            shape = {(r, c) for r in range(h) for c in range(w)}
        else:
            mode = 2
    if mode == 3 and area >= 4:
        splits = [(a, area - a) for a in range(2, 5) if 2 <= area - a <= 4]
        if splits:
            upper, lower = splits[int(rng.integers(len(splits)))]
            offsets = [c for c in range(upper) if c + lower <= 5]
            offset = offsets[int(rng.integers(len(offsets)))]
            shape = {(0, c) for c in range(upper)} | {
                (1, offset + c) for c in range(lower)}
        else:
            mode = 2
    if mode == 4 and area >= 4:
        rectangles = [(h, w) for h in range(2, 5) for w in range(2, 6)
                      if area < h * w <= area + 2]
        if rectangles:
            h, w = rectangles[int(rng.integers(len(rectangles)))]
            shape = {(r, c) for r in range(h) for c in range(w)}
            while len(shape) > area:
                removable = [p for p in sorted(shape)
                             if p[0] in (0, h - 1) or p[1] in (0, w - 1)
                             if len(_components(shape - {p})) == 1]
                shape.remove(removable[int(rng.integers(len(removable)))])
        else:
            mode = 2
    if mode == 2 or (mode == 1 and area < 4) or (mode in (3, 4) and area < 4):
        shape = {(0, 0)}
        max_side = 3 if compact else 5
        for _ in range(area - 1):
            frontier = set()
            for r, c in shape:
                for dr, dc in _DIRS:
                    p = (r + dr, c + dc)
                    if p not in shape:
                        trial = shape | {p}
                        nr = _normalize(trial)
                        if max(r for r, _ in nr) < max_side and max(c for _, c in nr) < max_side:
                            frontier.add(p)
            shape.add(sorted(frontier)[int(rng.integers(len(frontier)))])
    if int(rng.integers(2)):
        shape = {(c, r) for r, c in shape}
    if int(rng.integers(2)):
        shape = {(r, -c) for r, c in shape}
    if int(rng.integers(2)):
        shape = {(-r, c) for r, c in shape}
    return _normalize(shape)


def _choose_shapes(rng, count, compact, same_area):
    for _ in range(100):
        first = _shape(rng, compact, int(rng.integers(3, 8)) if same_area else None)
        shapes = [first]
        for _index in range(1, count):
            area = len(first) if same_area else None
            for _retry in range(30):
                candidate = _shape(rng, compact, area)
                if candidate not in shapes:
                    shapes.append(candidate)
                    break
            else:
                break
        if len(shapes) == count:
            return shapes
    raise ValueError("Could not sample distinct silhouettes")


def _free(mask, cells):
    height, width = mask.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                y, x = r + dr, c + dc
                if 0 <= y < height and 0 <= x < width and mask[y, x]:
                    return False
    return True


def _place_shape(mask, shape, rng):
    height, width = mask.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    if sh > height or sw > width:
        return None
    positions = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
    for index in rng.permutation(len(positions)):
        br, bc = positions[int(index)]
        cells = {(br + r, bc + c) for r, c in shape}
        if _free(mask, cells):
            for r, c in cells:
                mask[r, c] = True
            return cells
    return None


def _mold(mask, shape, walls, rng):
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    top, bottom, left, right = walls
    mh, mw = top + sh + bottom, left + sw + right
    rect = tuple((r, c) for r in range(mh) for c in range(mw))
    placed = _place_shape(mask, rect, rng)
    if placed is None:
        return None
    br = min(r for r, _ in placed)
    bc = min(c for _, c in placed)
    cavity = {(br + top + r, bc + left + c) for r, c in shape}
    return placed, cavity


def _pair(rng, regime, index, previous_roles, familiar_shapes=()):
    forced_clutter = regime == "source_train" and index == 0
    same_area = (regime == "target_test" and index == 0) or (
        regime == "source_train" and index == 1)
    for attempt in range(120):
        if forced_clutter or same_area:
            count = 2
            height = int(rng.integers(16, 23))
            width = int(rng.integers(16, 23))
        else:
            count = int(rng.choice((1, 2, 2, 2, 3)))
            compact = count == 2 and int(rng.integers(5)) == 0
            if compact:
                height = width = 10
            elif count == 1:
                height, width = (int(rng.integers(10, 19)), int(rng.integers(10, 20)))
            elif count == 2:
                height, width = (int(rng.integers(14, 24)), int(rng.integers(13, 24)))
            else:
                height, width = (int(rng.integers(19, 27)), int(rng.integers(19, 27)))
        compact = height == width == 10
        shapes = None
        if regime == "target_test" and familiar_shapes:
            reuse = int(rng.integers(4)) == 0 if index == 0 else int(rng.integers(2)) == 0
            if reuse:
                if same_area:
                    eligible = [s for s in familiar_shapes
                                if sum(len(other) == len(s) for other in familiar_shapes) >= count]
                    if eligible:
                        area = len(eligible[int(rng.integers(len(eligible)))])
                        eligible = [s for s in familiar_shapes if len(s) == area]
                else:
                    eligible = list(familiar_shapes)
                if eligible and len(eligible) >= count:
                    indices = rng.choice(len(eligible), size=count, replace=False)
                    shapes = [eligible[int(j)] for j in indices]
        if shapes is None:
            try:
                shapes = _choose_shapes(rng, count, compact, same_area)
            except ValueError:
                continue
        colors = [int(v) for v in rng.permutation(10)]
        background = colors[0]
        distractor = colors[1]
        donor_colors = colors[2:2 + count]
        structure_colors = colors[2 + count:3 + count]
        if count > 1 and int(rng.integers(3)) == 0:
            structure_colors = colors[2 + count:2 + 2 * count]
        else:
            structure_colors = structure_colors * count
        roles = (background, distractor, tuple(donor_colors), tuple(structure_colors))
        if index == 1 and regime == "source_train" and roles == previous_roles:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        occupied = np.zeros((height, width), dtype=bool)
        cavities = []
        okay = True
        for shape, structure in zip(shapes, structure_colors):
            if compact:
                walls = (1, 1, 1, 1)
            else:
                walls = tuple(int(rng.choice((1, 1, 1, 2, 2, 3))) for _ in range(4))
            result = _mold(occupied, shape, walls, rng)
            if result is None:
                okay = False
                break
            rectangle, cavity = result
            for r, c in rectangle:
                grid[r, c] = structure
            for r, c in cavity:
                grid[r, c] = background
            cavities.append(cavity)
        if not okay:
            continue
        keys = []
        for shape, color in zip(shapes, donor_colors):
            key = _place_shape(occupied, shape, rng)
            if key is None:
                okay = False
                break
            for r, c in key:
                grid[r, c] = color
            keys.append(key)
        if not okay:
            continue
        distractor_shapes = []
        if forced_clutter:
            distractor_shapes.append(shapes[int(rng.integers(count))])
        extra = int(rng.integers(1, 4)) if forced_clutter else int(rng.integers(2, 6))
        for _ in range(extra):
            if compact or int(rng.integers(4)) == 0:
                distractor_shapes.append(((0, 0),))
            else:
                distractor_shapes.append(_shape(rng, False, int(rng.integers(2, 6))))
        if not forced_clutter and len(distractor_shapes) < 2:
            distractor_shapes.append(((0, 0),))
        distractors = []
        for shape in distractor_shapes:
            fragment = _place_shape(occupied, shape, rng)
            if fragment is None:
                okay = False
                break
            distractors.append(fragment)
            for r, c in fragment:
                grid[r, c] = distractor
        if not okay:
            continue
        output = grid.copy()
        for color, key, cavity in zip(donor_colors, keys, cavities):
            for r, c in key:
                output[r, c] = background
            for r, c in cavity:
                output[r, c] = color
        try:
            inspection = _inspect_input(grid)
        except AssertionError:
            continue
        if not np.array_equal(output, inspection["predicted"]):
            continue
        if (_compact(grid), _compact(output)) in _SOURCE_PAIRS:
            continue
        return {"input": grid, "output": output}, roles
    raise ValueError("Could not place an unambiguous pair")


def _inspect_input(grid):
    """Recover the complete move using only visible input cells."""
    height, width = grid.shape
    candidates = []
    for background in (int(v) for v in np.unique(grid)):
        cavities = []
        structure_colors = []
        for region in _components(_cells(grid, background)):
            if any(r in (0, height - 1) or c in (0, width - 1) for r, c in region):
                continue
            boundary = set()
            for r, c in region:
                for dr, dc in _DIRS:
                    p = (r + dr, c + dc)
                    if p not in region:
                        if not (0 <= p[0] < height and 0 <= p[1] < width):
                            boundary = set()
                            break
                        boundary.add(p)
            colors = {int(grid[p]) for p in boundary}
            if len(colors) != 1 or background in colors:
                continue
            structure = next(iter(colors))
            components = _components(_cells(grid, structure))
            matches = [component for component in components if boundary <= component]
            if len(matches) != 1:
                continue
            whole = matches[0] | region
            r0, r1 = min(r for r, _ in whole), max(r for r, _ in whole)
            c0, c1 = min(c for _, c in whole), max(c for _, c in whole)
            if len(whole) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if not all(r0 < r < r1 and c0 < c < c1 for r, c in region):
                continue
            cavities.append(region)
            structure_colors.append(structure)
        if cavities:
            candidates.append((background, cavities, structure_colors))
    assert len(candidates) == 1
    background, cavities, structures = candidates[0]
    assert 1 <= len(cavities) <= 3
    shapes = [_normalize(cavity) for cavity in cavities]
    assert len(set(shapes)) == len(shapes)
    excluded = {background, *structures}
    donor_by_shape = {}
    distractors = []
    for color in (int(v) for v in np.unique(grid)):
        if color in excluded:
            continue
        components = _components(_cells(grid, color))
        if len(components) == 1 and _normalize(components[0]) in shapes:
            shape = _normalize(components[0])
            assert shape not in donor_by_shape
            donor_by_shape[shape] = (color, components[0])
        else:
            distractors.append((color, components))
    assert len(donor_by_shape) == len(cavities)
    assert len(distractors) == 1
    distractor, fragments = distractors[0]
    assert len(fragments) >= 2
    predicted = grid.copy()
    for cavity, shape in zip(cavities, shapes):
        donor, key = donor_by_shape[shape]
        for r, c in key:
            predicted[r, c] = background
        for r, c in cavity:
            predicted[r, c] = donor
    assert not np.array_equal(grid, predicted)
    return {"predicted": predicted, "background": background,
            "structures": tuple(structures), "shapes": tuple(shapes),
            "donors": tuple(v[0] for v in donor_by_shape.values()),
            "distractor": distractor, "fragments": fragments,
            "cavities": cavities}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    inspections = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = pair["input"], pair["output"]
            for grid in (before, after):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert before.shape == after.shape
            inp, out = _signature(before), _signature(after)
            assert inp not in seen_inputs and (inp, out) not in seen_pairs
            seen_inputs.add(inp)
            seen_pairs.add((inp, out))
            assert (_compact(before), _compact(after)) not in _SOURCE_PAIRS
            inspection = _inspect_input(before)
            assert np.array_equal(after, inspection["predicted"])
            inspections[split].append(inspection)
    assert len(inspections["train"][0]["cavities"]) == 2
    assert any(any(len(fragment) > 1 for fragment in item["fragments"])
               for item in inspections["train"])
    assert any(_normalize(fragment) in item["shapes"]
               for item in inspections["train"] for fragment in item["fragments"])
    first, second = inspections["train"][:2]
    assert (first["background"], first["distractor"], first["donors"], first["structures"]) != (
        second["background"], second["distractor"], second["donors"], second["structures"])
    second_shapes = second["shapes"]
    assert len(second_shapes) == 2
    assert len(second_shapes[0]) == len(second_shapes[1]) and second_shapes[0] != second_shapes[1]
    assert len(inspections["test"][0]["cavities"]) == 2
    test_shapes = inspections["test"][0]["shapes"]
    assert len(test_shapes[0]) == len(test_shapes[1]) and test_shapes[0] != test_shapes[1]


def _count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} outside legal range")
    return value


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        legal = [(tr, te) for tr in range(2, 11) for te in range(1, 5) if tr > te]
        num_train, num_test = legal[int(rng.integers(len(legal)))]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    # The puzzle-level convention is exact translation matching. Literal role
    # colors and geometry are observable pair-level properties, as in the source.
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_roles = None
    familiar_shapes = set()
    for split, count, regime in (("train", num_train, "source_train"),
                                 ("test", num_test, "target_test")):
        for index in range(count):
            for _ in range(20):
                child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair, roles = _pair(child_rng, regime, index, previous_roles,
                                    tuple(sorted(familiar_shapes)))
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise ValueError("Could not generate unique input")
            seen.add(signature)
            puzzle[split].append(pair)
            if split == "train":
                familiar_shapes.update(_inspect_input(pair["input"])["shapes"])
            if split == "train" and index == 0:
                previous_roles = roles
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    train_counts, test_counts, sizes, pair_counts = set(), set(), set(), set()
    shapes, shape_areas, palettes, layout_signatures = set(), set(), set(), set()
    compact_two, clustered, near_match, differing_walls = 0, 0, 0, 0
    familiar_tests, novel_tests = 0, 0
    all_signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _puzzle_signature(puzzle)
        assert signature not in all_signatures
        all_signatures.add(signature)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        learned_shapes = {shape for pair in puzzle["train"]
                          for shape in _inspect_input(pair["input"])["shapes"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _inspect_input(grid)
                assert np.array_equal(info["predicted"], pair["output"])
                if split == "test":
                    familiar_tests += all(shape in learned_shapes for shape in info["shapes"])
                    novel_tests += any(shape not in learned_shapes for shape in info["shapes"])
                sizes.add(grid.shape)
                pair_counts.add(len(info["cavities"]))
                palettes.add((info["background"], info["distractor"], info["donors"], info["structures"]))
                shapes.update(info["shapes"])
                shape_areas.update(len(s) for s in info["shapes"])
                layout_signatures.add((grid.shape, tuple(sorted(info["shapes"])),
                                       tuple(sorted((len(f) for f in info["fragments"])))))
                compact_two += grid.shape == (10, 10) and len(info["cavities"]) == 2
                clustered += any(len(f) > 1 for f in info["fragments"])
                near_match += any(_normalize(f) in info["shapes"] for f in info["fragments"])
                differing_walls += len(set(info["structures"])) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 20 and len(pair_counts) == 3
    assert len(shapes) >= 30 and len(shape_areas) >= 6
    assert len(palettes) >= 30 and len(layout_signatures) >= 80
    assert compact_two > 0 and clustered >= 42 and near_match >= 42 and differing_walls > 0
    assert familiar_tests >= 5 and novel_tests >= 20
    assert _puzzle_signature(generate_puzzle(12345, 6, 3)) == _puzzle_signature(generate_puzzle(12345, 6, 3))
    assert _puzzle_signature(generate_puzzle(17)) != _puzzle_signature(generate_puzzle(18))
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(500 + tr + te, tr, te)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (tr, te)
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(600 + (tr or 0) + (te or 0), tr, te)
        if tr is not None:
            assert len(puzzle["train"]) == tr
        if te is not None:
            assert len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(700, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid count was accepted")
