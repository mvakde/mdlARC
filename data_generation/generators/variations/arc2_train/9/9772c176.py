"""Complete independently cropped, axis-aligned filled diamonds."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "1680254898998472181958acc288bc3e45a6c80434042e2c443c50fa55ef120c",
    "64b1d4c7f71bf3239c8300217974fe0ff2e7394b965a7dfc4fb311ee5da18279",
    "0eb7f15198d37f8ffd935f434967a9267bf9332d967e4fbfd7546ef811cac63a",
}


def _seed_key(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside its legal range")
    return value


def _resolve_counts(seed_key, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed_key % 4
        train = test + 1 + (seed_key // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed_key % (10 - test)
    elif test is None:
        test = 1 + seed_key % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _extents(ulo, uhi, vlo, vhi):
    return ((ulo + vlo + 1) // 2, (uhi + vhi) // 2,
            (ulo - vhi + 1) // 2, (uhi - vlo) // 2)


def _template(rng, diameter):
    """Return a crop whose four diagonal support lines remain observable."""
    parity = int(rng.integers(0, 2))
    bounds = (0, diameter, parity, parity + diameter)
    fr0, fr1, fc0, fc1 = _extents(*bounds)
    full = tuple((r, c) for r in range(fr0, fr1 + 1)
                 for c in range(fc0, fc1 + 1)
                 if 0 <= r + c <= diameter
                 and parity <= r - c <= parity + diameter)
    limit = min(7, max(2, diameter // 3))
    for _ in range(90):
        trims = [int(rng.integers(1, limit + 1)) for _ in range(4)]
        if rng.random() < 0.22:
            trims[int(rng.integers(0, 4))] = 0
        top, bottom, left, right = trims
        r0, r1 = fr0 + top, fr1 - bottom
        c0, c1 = fc0 + left, fc1 - right
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        visible = tuple((r, c) for r, c in full if r0 <= r <= r1 and c0 <= c <= c1)
        if len(visible) < 10:
            continue
        us = [r + c for r, c in visible]
        vs = [r - c for r, c in visible]
        if (min(us), max(us), min(vs), max(vs)) != bounds:
            continue
        if min(*(us.count(x) for x in bounds[:2]),
               *(vs.count(x) for x in bounds[2:])) < 2:
            continue
        if (min(r for r, _ in visible), max(r for r, _ in visible),
            min(c for _, c in visible), max(c for _, c in visible)) != (r0, r1, c0, c1):
            continue
        flat = [sum(r == r0 for r, _ in visible), sum(r == r1 for r, _ in visible),
                sum(c == c0 for _, c in visible), sum(c == c1 for _, c in visible)]
        if any(n < 3 for n, trim in zip(flat, trims) if trim > 0):
            continue
        return {"diameter": diameter, "full": full, "visible": visible,
                "full_bbox": (fr0, fr1, fc0, fc1), "crop_bbox": (r0, r1, c0, c1)}
    return None


def _placement_options(template, height, width, mode):
    fr0, fr1, fc0, fc1 = template["full_bbox"]
    cr0, cr1, cc0, cc1 = template["crop_bbox"]
    options = []
    for dr in range(-cr0, height - cr1):
        for dc in range(-cc0, width - cc1):
            ext = (fr0 + dr, fr1 + dr, fc0 + dc, fc1 + dc)
            crossing = ext[0] < 0 or ext[1] >= height or ext[2] < 0 or ext[3] >= width
            touching = ext[0] == 0 or ext[1] == height - 1 or ext[2] == 0 or ext[3] == width - 1
            if ((mode == "cross" and crossing)
                or (mode == "touch" and touching and not crossing)
                or (mode == "inside" and not crossing and not touching)
                or mode == "free"):
                options.append((dr, dc))
    return options


def _instantiate(template, shift, height, width):
    dr, dc = shift
    visible = {(r + dr, c + dc) for r, c in template["visible"]}
    full = {(r + dr, c + dc) for r, c in template["full"]
            if 0 <= r + dr < height and 0 <= c + dc < width}
    return visible, full


def _diameters(rng, count):
    if count == 1:
        return [int(rng.integers(7, 24))]
    if count == 3:
        return [int(rng.integers(7, 12)) for _ in range(3)]
    relation = int(rng.choice(4, p=[0.19, 0.38, 0.28, 0.15]))
    if relation == 0:
        first = int(rng.integers(7, 23))
        second = first
    elif relation == 1:
        first = int(rng.integers(7, 22))
        second = min(23, first + int(rng.integers(1, 3)))
    elif relation == 2:
        first = int(rng.integers(7, 14))
        second = int(rng.integers(first + 5, 24))
    else:
        first, second = int(rng.integers(7, 24)), int(rng.integers(7, 24))
    return [first, second] if rng.random() < 0.5 else [second, first]


def _dimensions(rng, regime, count, diameters):
    low = 16 if regime == "source_train" else 14
    if count == 3:
        low = 25
    elif max(diameters) >= 18:
        low = max(low, 22)
    height = int(rng.integers(low, 31))
    width = height if rng.random() < (0.28 if regime == "source_train" else 0.43) else int(rng.integers(low, 31))
    return height, width


def _object_count(rng):
    return int(rng.choice([1, 2, 3], p=[0.09, 0.79, 0.12]))


def _modes(rng, regime, count, force_clip):
    weights = [0.54, 0.25, 0.21] if regime == "source_train" else [0.68, 0.20, 0.12]
    modes = [str(rng.choice(["cross", "touch", "inside"], p=weights)) for _ in range(count)]
    if force_clip:
        modes[int(rng.integers(0, count))] = "cross"
    return modes


def _place_all(rng, templates, modes, height, width):
    options = [_placement_options(t, height, width, mode)
               for t, mode in zip(templates, modes)]
    if any(not x for x in options):
        return None

    def search(index, used, blocked):
        if index == len(templates):
            return used
        choices = options[index]
        order = rng.permutation(len(choices))
        limit = 34 if index == 0 else 130
        for k in order[:limit]:
            visible, full = _instantiate(templates[index], choices[int(k)], height, width)
            if len(full - visible) < 2 or full & blocked:
                continue
            expanded = {(r + dr, c + dc) for r, c in full
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            result = search(index + 1, used + [(visible, full)], blocked | expanded)
            if result is not None:
                return result
        return None

    return search(0, [], set())


def _generate_pair(rng, regime, colors, force_clip=False, force_count=None):
    count = force_count if force_count is not None else _object_count(rng)
    for _ in range(230):
        diameters = _diameters(rng, count)
        height, width = _dimensions(rng, regime, count, diameters)
        templates = [_template(rng, d) for d in diameters]
        if any(t is None for t in templates):
            continue
        modes = _modes(rng, regime, count, force_clip)
        objects = _place_all(rng, templates, modes, height, width)
        if objects is None:
            continue
        background, source, completion = colors
        inp = np.full((height, width), background, dtype=np.int8)
        for visible, _ in objects:
            for r, c in visible:
                inp[r, c] = source
        if np.count_nonzero(inp == source) * 2 >= height * width:
            continue
        out = inp.copy()
        for _, full in objects:
            for r, c in full:
                if inp[r, c] == background:
                    out[r, c] = completion
        if not np.array_equal(inp, out):
            return {"input": inp, "output": out}
    raise ValueError("could not construct separated, recoverable diamonds")


def _input_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_key = _seed_key(seed)
    rng = np.random.default_rng(seed_key)
    train_count, test_count = _resolve_counts(seed_key, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, destination in (("source_train", train_count, puzzle["train"]),
                                       ("target_test", test_count, puzzle["test"])):
        for index in range(count):
            child_rng = np.random.default_rng(int(rng.integers(0, 1 << 63, dtype=np.uint64)))
            for _ in range(12):
                pair = _generate_pair(child_rng, regime, colors,
                                      force_clip=regime == "source_train" and index == 0,
                                      force_count=2 if regime == "source_train" and index == 0 else None)
                key = _input_key(pair["input"])
                if key not in seen:
                    destination.append(pair)
                    seen.add(key)
                    break
            else:
                raise ValueError("could not create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    remaining = {tuple(x) for x in np.argwhere(mask)}
    components = []
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    changed = inp != out
    assert np.any(changed)
    bg_values = np.unique(inp[changed])
    completion_values = np.unique(out[changed])
    assert len(bg_values) == len(completion_values) == 1
    background, completion = int(bg_values[0]), int(completion_values[0])
    assert background != completion
    input_colors = {int(x) for x in np.unique(inp)}
    source_colors = input_colors - {background}
    assert len(input_colors) == 2 and len(source_colors) == 1
    source = source_colors.pop()
    assert completion not in input_colors
    assert np.count_nonzero(inp == background) > np.count_nonzero(inp == source)
    assert np.all(out[inp == source] == source)
    assert set(int(x) for x in np.unique(out)) == {background, source, completion}
    components = _components(inp == source)
    assert 1 <= len(components) <= 3
    height, width = inp.shape
    info, full_sets = [], []
    for component in components:
        rows, cols = [r for r, _ in component], [c for _, c in component]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        us = [r + c for r, c in component]
        vs = [r - c for r, c in component]
        bounds = min(us), max(us), min(vs), max(vs)
        ulo, uhi, vlo, vhi = bounds
        diameter = uhi - ulo
        assert diameter == vhi - vlo and 7 <= diameter <= 23
        assert min(us.count(ulo), us.count(uhi), vs.count(vlo), vs.count(vhi)) >= 2
        expected_visible = {(r, c) for r in range(r0, r1 + 1)
                            for c in range(c0, c1 + 1)
                            if ulo <= r + c <= uhi and vlo <= r - c <= vhi}
        assert expected_visible == component
        ext = _extents(*bounds)
        assert ext[0] <= r0 and ext[1] >= r1 and ext[2] <= c0 and ext[3] >= c1
        assert any(a != b for a, b in zip(ext, (r0, r1, c0, c1)))
        flat = [sum(r == r0 for r, _ in component), sum(r == r1 for r, _ in component),
                sum(c == c0 for _, c in component), sum(c == c1 for _, c in component)]
        for n, trimmed in zip(flat, (ext[0] < r0, ext[1] > r1,
                                    ext[2] < c0, ext[3] > c1)):
            if trimmed:
                assert n >= 3
        full = {(r, c) for r in range(height) for c in range(width)
                if ulo <= r + c <= uhi and vlo <= r - c <= vhi}
        assert len(full - component) >= 2
        crossing = (ext[0] < 0, ext[1] >= height, ext[2] < 0, ext[3] >= width)
        touching = (ext[0] == 0, ext[1] == height - 1,
                    ext[2] == 0, ext[3] == width - 1)
        info.append({"diameter": diameter, "crop_bbox": (r0, r1, c0, c1),
                     "full_bbox": ext, "crossing": crossing, "touching": touching,
                     "full": full})
        full_sets.append(full)
    for i, first in enumerate(full_sets):
        expanded = {(r + dr, c + dc) for r, c in first
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        for second in full_sets[i + 1:]:
            assert not expanded.intersection(second)
    predicted = inp.copy()
    for full in full_sets:
        for r, c in full:
            if predicted[r, c] == background:
                predicted[r, c] = completion
    assert np.array_equal(predicted, out)
    return {"colors": (background, source, completion), "diamonds": info}


def _solve_from_training(train, inp):
    first = train[0]
    changed = first["input"] != first["output"]
    background = int(np.unique(first["input"][changed])[0])
    completion = int(np.unique(first["output"][changed])[0])
    source = next(int(x) for x in np.unique(first["input"]) if int(x) != background)
    assert set(int(x) for x in np.unique(inp)) == {background, source}
    predicted = inp.copy()
    for component in _components(inp == source):
        ulo = min(r + c for r, c in component)
        uhi = max(r + c for r, c in component)
        vlo = min(r - c for r, c in component)
        vhi = max(r - c for r, c in component)
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                if (ulo <= r + c <= uhi and vlo <= r - c <= vhi
                    and predicted[r, c] == background):
                    predicted[r, c] = completion
    return predicted


def _pair_digest(inp, out):
    data = bytes(inp.shape) + inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()
    return hashlib.sha256(data).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, colors = set(), set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_key = _input_key(inp)
            pair_key = (input_key, _input_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _pair_digest(inp, out) not in _SOURCE_DIGESTS
            analysis = _analyze_pair(inp, out)
            analyses[split].append(analysis)
            colors.add(analysis["colors"])
    assert len(colors) == 1
    assert any(any(d["crossing"] for d in a["diamonds"]) for a in analyses["train"])
    assert len(analyses["train"][0]["diamonds"]) == 2
    for pair in puzzle["test"]:
        assert np.array_equal(_solve_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    diameters, counts, modes, trims = set(), set(), set(), set()
    within_shape, within_scale = 0, 0
    within_mode, within_crop = 0, 0
    review = set()
    side_relations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_shapes, local_diameters, local_modes, local_crops = set(), set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_pair(inp, out)
                palettes.add(info["colors"])
                shapes.add(inp.shape)
                local_shapes.add(inp.shape)
                counts.add(len(info["diamonds"]))
                review.add((split, "square" if inp.shape[0] == inp.shape[1] else "nonsquare"))
                if inp.shape in ((22, 21), (26, 29), (20, 20)):
                    review.add(("source", "dimensions_reused"))
                normalized = (inp == info["colors"][1]).astype(np.int8)
                layouts.add((inp.shape, normalized.tobytes()))
                assert _pair_digest(inp, out) not in _SOURCE_DIGESTS
                if split == "test":
                    assert np.array_equal(_solve_from_training(puzzle["train"], inp), out)
                for d in info["diamonds"]:
                    diameters.add(d["diameter"])
                    local_diameters.add(d["diameter"])
                    mode = "cross" if any(d["crossing"]) else ("touch" if any(d["touching"]) else "inside")
                    modes.add(mode)
                    local_modes.add(mode)
                    review.add((split, mode))
                    crop, full = d["crop_bbox"], d["full_bbox"]
                    these_trims = (crop[0] - full[0], full[1] - crop[1],
                                   crop[2] - full[2], full[3] - crop[3])
                    trims.update(these_trims)
                    local_crops.add(these_trims)
                    for side, extent, edge in (("top", full[0], 0),
                                               ("bottom", full[1], inp.shape[0] - 1),
                                               ("left", full[2], 0),
                                               ("right", full[3], inp.shape[1] - 1)):
                        relation = ("on" if extent == edge else "outside" if
                                    (extent < edge if side in ("top", "left") else extent > edge)
                                    else "inside")
                        side_relations.add((side, relation))
                if len(info["diamonds"]) == 2:
                    first, second = info["diamonds"]
                    gap = abs(first["diameter"] - second["diameter"])
                    review.add((split, "equal" if gap == 0 else "close" if gap <= 2
                                else "wide" if gap >= 5 else "medium"))
                    review.add((split, "same_parity" if first["diameter"] % 2 ==
                                second["diameter"] % 2 else "opposite_parity"))
                    clipped = [d for d in (first, second) if any(d["crossing"])]
                    if len(clipped) == 1:
                        other = second if clipped[0] is first else first
                        review.add((split, "smaller_clipped" if clipped[0]["diameter"] <
                                    other["diameter"] else "equal_clipped" if
                                    clipped[0]["diameter"] == other["diameter"] else
                                    "larger_clipped"))
                if (len(info["diamonds"]) >= 2 and len({d["diameter"] for d in
                    info["diamonds"]}) > 1 and any(any(d["crossing"]) for d in
                    info["diamonds"]) and any(0 in (
                    d["crop_bbox"][0] - d["full_bbox"][0],
                    d["full_bbox"][1] - d["crop_bbox"][1],
                    d["crop_bbox"][2] - d["full_bbox"][2],
                    d["full_bbox"][3] - d["crop_bbox"][3]) for d in info["diamonds"])):
                    review.add((split, "combined"))
        within_shape += len(local_shapes) > 1
        within_scale += len(local_diameters) > 1
        within_mode += len(local_modes) > 1
        within_crop += len(local_crops) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 20 and len(shapes) >= 35 and len(layouts) >= 120
    assert min(diameters) <= 8 and max(diameters) >= 20
    assert counts == {1, 2, 3} and modes == {"cross", "touch", "inside"}
    assert 0 in trims and max(trims) >= 5
    assert within_shape >= 25 and within_scale >= 35
    assert within_mode >= 30 and within_crop >= 35
    assert len(side_relations) == 12
    for split in ("train", "test"):
        for feature in ("square", "nonsquare", "cross", "touch", "inside",
                        "equal", "close", "wide", "same_parity", "opposite_parity",
                        "smaller_clipped", "larger_clipped", "combined"):
            assert (split, feature) in review
    assert ("source", "dimensions_reused") in review
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1101, None, 1), (1102, None, 4),
                              (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(generate_puzzle(20260827, 6, 2), generate_puzzle(20260827, 6, 2))
    assert not _puzzles_equal(generate_puzzle(3001, 5, 2), generate_puzzle(3002, 5, 2))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(71, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
