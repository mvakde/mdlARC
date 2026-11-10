import itertools

import numpy as np


_SOURCE_PAIR_KEYS = frozenset(
    {
        (
            (
                "0088350053900934004",
                "0880355053990930044",
                "8880305503990034000",
                "8800300003000034440",
            ),
            ("9089", "9949", "9980", "4440"),
        ),
        (
            (
                "0088355003099934040",
                "8888305053090934040",
                "8808350553000930404",
                "0880300053900930000",
            ),
            ("4999", "4949", "8459", "9889"),
        ),
        (
            (
                "8800355503909934404",
                "8808355553090030044",
                "8000305053900934004",
                "8088350503000030040",
            ),
            ("9499", "8944", "9509", "8048"),
        ),
        (
            (
                "0800300003000934040",
                "0800355003099034004",
                "8880350053999034040",
                "0000355553009030000",
            ),
            ("4849", "4994", "9995", "5595"),
        ),
        (
            (
                "8880355503999034444",
                "8080355553990930000",
                "0008355003000030440",
                "0800305553900034444",
            ),
            ("9994", "9989", "5448", "9444"),
        ),
        (
            (
                "0880355553990934004",
                "8080305053000934044",
                "8808300003990930404",
                "8808355003990030000",
            ),
            ("9989", "4549", "9909", "9908"),
        ),
        (
            (
                "8880350553900030000",
                "8088355553009934404",
                "8008350553000930404",
                "0088300553999030440",
            ),
            ("9885", "4499", "8459", "9998"),
        ),
    }
)


def _is_plain_int(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_plain_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_plain_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        choices = list(range(max(2, num_test + 1), 11))
        num_train = choices[(seed // 4) % len(choices)]
    elif num_train is None:
        choices = list(range(max(2, num_test + 1), 11))
        num_train = choices[seed % len(choices)]
    elif num_test is None:
        choices = list(range(1, min(4, num_train - 1) + 1))
        num_test = choices[seed % len(choices)]

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    layers = int(rng.choice([2, 3, 4, 5, 6], p=[0.10, 0.13, 0.48, 0.14, 0.15]))
    colors = tuple(int(x) for x in rng.permutation(10)[:layers + 2])
    priority = (2, 3, 0, 1) if layers == 4 and rng.random() < 0.22 else tuple(int(x) for x in rng.permutation(layers))
    return {
        "layers": layers,
        "priority": priority,
        "shift_mode": str(rng.choice(["ordinary", "increase", "decrease"], p=[0.62, 0.25, 0.13])) if layers >= 3 else "ordinary",
        "orientation_mode": str(rng.choice(["horizontal", "vertical", "mixed"], p=[0.45, 0.28, 0.27])),
        "palette_mode": str(rng.choice(["stable", "variable"], p=[0.3, 0.7])),
        "colors": colors,
    }


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_source_pair(pair: dict) -> bool:
    return (_rows_key(pair["input"]), _rows_key(pair["output"])) in _SOURCE_PAIR_KEYS


def _choose_shape(rng, layers, orientation, teaching):
    packed_max = (30 - layers + 1) // layers
    assert packed_max >= 4
    if teaching:
        # A compact demonstration still needs room for the rank chain.
        for _ in range(20):
            along = int(rng.integers(2, packed_max + 1))
            across = int(rng.choice([2, 3, 4, 5, 6, 8, 10, 14, 20, 30]))
            if along * across >= layers + 3:
                break
    else:
        along = int(rng.integers(2, packed_max + 1))
        across = int(rng.choice([2, 3, 4, 5, 6, 8, 10, 14, 20, 30]))
    return (across, along) if orientation == "horizontal" else (along, across)


def _single_mask(rng, height, width, kind, density):
    mask = np.zeros((height, width), dtype=bool)
    if kind == "scatter":
        return rng.random((height, width)) < density
    if kind == "blob":
        y0, y1 = sorted(int(x) for x in rng.integers(0, height + 1, size=2))
        x0, x1 = sorted(int(x) for x in rng.integers(0, width + 1, size=2))
        y1 = max(y0 + 1, y1)
        x1 = max(x0 + 1, x1)
        mask[y0:y1, x0:x1] = True
        return mask
    if kind == "bar":
        if rng.random() < 0.5:
            y = int(rng.integers(height))
            y2 = min(height, y + int(rng.integers(1, min(4, height - y) + 1)))
            mask[y:y2, :] = True
        else:
            x = int(rng.integers(width))
            x2 = min(width, x + int(rng.integers(1, min(4, width - x) + 1)))
            mask[:, x:x2] = True
        return mask
    if kind == "hollow":
        y0 = int(rng.integers(max(1, height - 2)))
        x0 = int(rng.integers(max(1, width - 2)))
        y1 = int(rng.integers(min(height, y0 + 3), height + 1)) if height - y0 >= 3 else height
        x1 = int(rng.integers(min(width, x0 + 3), width + 1)) if width - x0 >= 3 else width
        mask[y0:y1, x0:x1] = True
        if y1 - y0 >= 3 and x1 - x0 >= 3:
            mask[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = False
        return mask
    if kind == "clusters":
        for _ in range(int(rng.integers(2, 5))):
            y, x = int(rng.integers(height)), int(rng.integers(width))
            mask[y:min(height, y + int(rng.integers(1, 4))), x:min(width, x + int(rng.integers(1, 4)))] = True
        return mask
    raise AssertionError(kind)


def _make_masks(rng, layers, height, width, priority, teaching, split, shift_mode):
    weights = [0.27, 0.57, 0.16] if split == "source_train" else [0.16, 0.45, 0.39]
    regime = str(rng.choice(["sparse", "mixed", "dense"], p=weights))
    kinds = ["scatter", "blob", "bar", "hollow", "clusters"]
    masks = np.zeros((layers, height, width), dtype=bool)
    for layer in range(layers):
        density = float(rng.uniform(*{"sparse": (0.10, 0.38), "mixed": (0.25, 0.75), "dense": (0.58, 0.92)}[regime]))
        kind = str(rng.choice(kinds, p=[0.36, 0.19, 0.15, 0.13, 0.17]))
        masks[layer] = _single_mask(rng, height, width, kind, density)

    relation = str(rng.choice(["none", "identical", "containment", "disjoint", "covered", "blank", "solid"],
                              p=[0.28, 0.12, 0.16, 0.13, 0.12, 0.10, 0.09]))
    a, b = (int(x) for x in rng.choice(layers, 2, replace=False))
    if relation == "identical":
        masks[b] = masks[a]
    elif relation == "containment":
        masks[b] = masks[a] & (rng.random((height, width)) < rng.uniform(0.2, 0.8))
    elif relation == "disjoint":
        masks[b] &= ~masks[a]
    elif relation == "covered":
        upper, lower = sorted((a, b), key=lambda x: priority.index(x))
        if not np.any(masks[upper]):
            masks[upper, int(rng.integers(height)), int(rng.integers(width))] = True
        masks[lower] = masks[upper] & (rng.random((height, width)) < 0.38)
        if not np.any(masks[lower]):
            y, x = rng.choice(np.argwhere(masks[upper]))
            masks[lower, y, x] = True
    elif relation == "blank":
        masks[a] = False
    elif relation == "solid":
        masks[a] = True

    # Vary uncovered area independently of geometry and overlap regime.
    blank_regime = str(rng.choice(["none", "few", "many"], p=[0.23, 0.43, 0.34]))
    if blank_regime == "none":
        uncovered = ~np.any(masks, axis=0)
        masks[int(rng.integers(layers)), uncovered] = True
    elif blank_regime == "many":
        clear = rng.random((height, width)) < rng.uniform(0.18, 0.55)
        masks[:, clear] = False
    else:
        clear = rng.random((height, width)) < rng.uniform(0.01, 0.1)
        masks[:, clear] = False

    if teaching:
        coordinates = rng.choice(height * width, size=layers - 1, replace=False)
        for rank, flat in enumerate(coordinates):
            y, x = divmod(int(flat), width)
            masks[:, y, x] = False
            masks[priority[rank], y, x] = True
            masks[priority[rank + 1], y, x] = True
    if shift_mode == "increase" and split == "source_train" or shift_mode == "decrease" and split == "target_test":
        for y, x in np.argwhere(masks.sum(axis=0) > 2):
            active = np.flatnonzero(masks[:, y, x])
            keep = rng.choice(active, size=2, replace=False)
            masks[:, y, x] = False
            masks[keep, y, x] = True
    if shift_mode == "increase" and split == "target_test":
        y, x = int(rng.integers(height)), int(rng.integers(width))
        masks[:, y, x] = False
        masks[list(priority[:3]), y, x] = True
    return masks


def _compose(masks, payloads, background, priority):
    output = np.full(masks.shape[1:], background, dtype=np.int8)
    unresolved = np.ones(masks.shape[1:], dtype=bool)
    for layer in priority:
        selected = unresolved & masks[layer]
        if payloads[layer] is not None:
            output[selected] = payloads[layer]
        unresolved &= ~masks[layer]
    return output


def _generate_pair(rng, parameters, orientation, teaching, split):
    layers = parameters["layers"]
    priority = parameters["priority"]
    height, width = _choose_shape(rng, layers, orientation, teaching)
    masks = _make_masks(rng, layers, height, width, priority, teaching, split, parameters["shift_mode"])
    colors = parameters["colors"] if parameters["palette_mode"] == "stable" else tuple(int(x) for x in rng.permutation(10)[:layers + 2])
    background, divider = colors[:2]
    payloads = colors[2:]
    panels = []
    for layer in range(layers):
        panel = np.full((height, width), background, dtype=np.int8)
        panel[masks[layer]] = payloads[layer]
        panels.append(panel)
    separator = np.full((height, 1) if orientation == "horizontal" else (1, width), divider, dtype=np.int8)
    parts = []
    for layer, panel in enumerate(panels):
        parts.append(panel)
        if layer + 1 < layers:
            parts.append(separator)
    input_grid = np.concatenate(parts, axis=1 if orientation == "horizontal" else 0)
    output_grid = _compose(masks, payloads, background, priority)
    return {"input": input_grid, "output": output_grid}


def _parse_input(input_grid, layers):
    candidates = []
    for orientation in ("horizontal", "vertical"):
        if orientation == "horizontal":
            if (input_grid.shape[1] - layers + 1) % layers:
                continue
            width = (input_grid.shape[1] - layers + 1) // layers
            height = input_grid.shape[0]
            if width < 2 or width * layers + layers - 1 > 30:
                continue
            divider_lines = [input_grid[:, (k + 1) * width + k] for k in range(layers - 1)]
            panels = [input_grid[:, k * (width + 1):k * (width + 1) + width] for k in range(layers)]
        else:
            if (input_grid.shape[0] - layers + 1) % layers:
                continue
            height = (input_grid.shape[0] - layers + 1) // layers
            width = input_grid.shape[1]
            if height < 2 or height * layers + layers - 1 > 30:
                continue
            divider_lines = [input_grid[(k + 1) * height + k, :] for k in range(layers - 1)]
            panels = [input_grid[k * (height + 1):k * (height + 1) + height, :] for k in range(layers)]
        if not divider_lines:
            continue
        divider = int(divider_lines[0].flat[0])
        if not all(np.all(line == divider) for line in divider_lines):
            continue
        sets = [set(int(x) for x in np.unique(panel)) for panel in panels]
        backgrounds = [color for color in range(10) if color != divider and sum(color in s for s in sets) >= 2]
        if len(backgrounds) != 1:
            continue
        background = backgrounds[0]
        if any(len(s - {background}) > 1 or divider in s for s in sets):
            continue
        payloads = [next(iter(s - {background}), None) for s in sets]
        active = [p for p in payloads if p is not None]
        if len(active) != len(set(active)):
            continue
        masks = np.stack([np.zeros((height, width), dtype=bool) if p is None else panel == p for panel, p in zip(panels, payloads)])
        candidates.append({"orientation": orientation, "height": height, "width": width, "background": background,
                           "divider": divider, "payloads": tuple(payloads), "masks": masks, "layers": layers})
    return candidates


def _infer_convention(train):
    candidates = []
    for layers in range(2, 7):
        parsed = [_parse_input(pair["input"], layers) for pair in train]
        if not all(len(options) == 1 for options in parsed):
            continue
        analyses = [options[0] for options in parsed]
        for priority in itertools.permutations(range(layers)):
            if all(np.array_equal(pair["output"], _compose(info["masks"], info["payloads"], info["background"], priority))
                   for pair, info in zip(train, analyses)):
                candidates.append((layers, priority, analyses))
    return candidates


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_plain_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = _sample_puzzle_parameters(rng)
    mode = parameters["orientation_mode"]
    if mode == "mixed":
        orientations = ["horizontal", "vertical"] + [str(rng.choice(["horizontal", "vertical"])) for _ in range(num_train + num_test - 2)]
    else:
        orientations = [mode] * (num_train + num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    taught = set()
    for index in range(num_train + num_test):
        split = "train" if index < num_train else "test"
        orientation = orientations[index]
        teaching = split == "train" and orientation not in taught
        if teaching:
            taught.add(orientation)
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        for _ in range(120):
            pair = _generate_pair(pair_rng, parameters, orientation, teaching,
                                  "source_train" if split == "train" else "target_test")
            key = _grid_key(pair["input"])
            if key not in seen and not _is_source_pair(pair) and len(_parse_input(pair["input"], parameters["layers"])) == 1:
                break
        else:
            raise RuntimeError("could not generate a valid unique pair")
        puzzle[split].append(pair)
        seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            ik = _grid_key(pair["input"])
            pk = (ik, _grid_key(pair["output"]))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            assert not _is_source_pair(pair)
    conventions = _infer_convention(puzzle["train"])
    assert len(conventions) == 1, "training must determine panel count and rank order"
    layers, priority, training = conventions[0]
    for orientation in {info["orientation"] for info in training}:
        subset = [pair for pair, info in zip(puzzle["train"], training) if info["orientation"] == orientation]
        consistent = [order for order in itertools.permutations(range(layers))
                      if all(np.array_equal(pair["output"], _compose(info["masks"], info["payloads"], info["background"], order))
                             for pair in subset for info in _parse_input(pair["input"], layers))]
        assert consistent == [priority], "each demonstrated orientation must reveal the rank order"
    for pair, info in zip(puzzle["train"], training):
        assert pair["output"].shape == (info["height"], info["width"])
    for pair in puzzle["test"]:
        parsed = _parse_input(pair["input"], layers)
        assert len(parsed) == 1
        info = parsed[0]
        expected = _compose(info["masks"], info["payloads"], info["background"], priority)
        assert np.array_equal(pair["output"], expected)
    # Both orientations, when present, must be demonstrated by training.
    assert {a["orientation"] for a in training}.issuperset(
        _parse_input(pair["input"], layers)[0]["orientation"] for pair in puzzle["test"])


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(np.array_equal(a[k], b[k]) for a, b in zip(left[s], right[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(321), generate_puzzle(321))
    assert not _puzzles_equal(generate_puzzle(321), generate_puzzle(322))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(805 + (counts[0] or 0) + (counts[1] or 0), *counts)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
        validate_generated_puzzle(puzzle)
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (True, 1)):
        try:
            generate_puzzle(999, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, layers_seen, priorities, sizes, palettes = set(), set(), set(), set(), set(), set()
    orientations, depths, background_counts, visibility_counts = set(), set(), set(), set()
    review = {split: {name: 0 for name in ("pairs", "triple", "fourway", "no_background", "hidden", "blank_panel", "solid_panel", "identical", "containment", "disjoint", "covered", "large", "compact", "hollow", "rectangle", "bar", "disconnected")} for split in ("train", "test")}
    all_keys = set()
    varied_shapes = varied_palettes = varied_depths = mixed_orientation = source_order = 0
    increased_depth = decreased_depth = equal_depth = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        inferred = _infer_convention(puzzle["train"])
        assert len(inferred) == 1
        n, order, _ = inferred[0]
        layers_seen.add(n)
        priorities.add((n, order))
        source_order += n == 4 and order == (2, 3, 0, 1)
        pair_shapes, pair_palettes, pair_depths, pair_orientations = set(), set(), set(), set()
        split_depths = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                key = _grid_key(pair["input"])
                assert key not in all_keys
                all_keys.add(key)
                info = _parse_input(pair["input"], n)[0]
                assert np.array_equal(pair["output"], _compose(info["masks"], info["payloads"], info["background"], order))
                sizes.add((info["height"], info["width"]))
                orientations.add(info["orientation"])
                pair_orientations.add(info["orientation"])
                pair_shapes.add((info["height"], info["width"]))
                palette = (info["background"], info["divider"], info["payloads"])
                pair_palettes.add(palette)
                palettes.add(palette)
                occupancy = info["masks"].sum(axis=0)
                depths.add(int(occupancy.max()))
                pair_depths.add(int(occupancy.max()))
                split_depths[split].append(int(occupancy.max()))
                background_counts.add(int(np.count_nonzero(occupancy == 0)))
                visibility_counts.add(len(np.unique(pair["output"])))
                r = review[split]
                r["pairs"] += 1
                r["triple"] += bool(np.any(occupancy >= 3))
                r["fourway"] += bool(np.any(occupancy >= 4))
                r["no_background"] += not np.any(occupancy == 0)
                r["hidden"] += any(info["payloads"][i] is not None and info["payloads"][i] not in pair["output"] for i in range(n))
                r["blank_panel"] += any(not np.any(m) for m in info["masks"])
                r["solid_panel"] += any(np.all(m) for m in info["masks"])
                r["large"] += max(info["height"], info["width"]) >= 14
                r["compact"] += info["height"] <= 3 and info["width"] <= 3
                for m in info["masks"]:
                    if not np.any(m):
                        continue
                    r["bar"] += bool(np.any(m.all(axis=0)) or np.any(m.all(axis=1)))
                    ys, xs = np.where(m)
                    box = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    r["rectangle"] += box.shape[0] > 1 and box.shape[1] > 1 and bool(np.all(box))
                    if m.shape[0] >= 3 and m.shape[1] >= 3:
                        r["hollow"] += any(np.all(z[0]) and np.all(z[-1]) and np.all(z[:, 0]) and np.all(z[:, -1]) and not z[1, 1]
                                           for y in range(m.shape[0] - 2) for x in range(m.shape[1] - 2) for z in (m[y:y + 3, x:x + 3],))
                    occupied = set(map(tuple, np.argwhere(m)))
                    components = 0
                    while occupied:
                        components += 1
                        frontier = [occupied.pop()]
                        while frontier:
                            y, x = frontier.pop()
                            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                                if neighbor in occupied:
                                    occupied.remove(neighbor)
                                    frontier.append(neighbor)
                    r["disconnected"] += components > 1
                relations = set()
                masks = info["masks"]
                for i, j in itertools.combinations(range(n), 2):
                    a, b = masks[i], masks[j]
                    if np.any(a) and np.any(b):
                        if np.array_equal(a, b):
                            relations.add("identical")
                        elif np.all(~a | b) or np.all(~b | a):
                            relations.add("containment")
                        if not np.any(a & b):
                            relations.add("disjoint")
                        upper, lower = sorted((i, j), key=lambda layer: order.index(layer))
                        if np.all(~masks[lower] | masks[upper]):
                            relations.add("covered")
                for relation in relations:
                    r[relation] += 1
        varied_shapes += len(pair_shapes) > 1
        varied_palettes += len(pair_palettes) > 1
        varied_depths += len(pair_depths) > 1
        mixed_orientation += len(pair_orientations) > 1
        increased_depth += max(split_depths["test"]) > max(split_depths["train"])
        decreased_depth += max(split_depths["test"]) < max(split_depths["train"])
        equal_depth += max(split_depths["test"]) == max(split_depths["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert layers_seen == {2, 3, 4, 5, 6} and len(priorities) >= 15 and source_order >= 2
    assert len(sizes) >= 25 and varied_shapes >= 30
    assert len(palettes) >= 60 and varied_palettes >= 15
    assert varied_depths >= 25 and mixed_orientation >= 5
    assert increased_depth >= 4 and decreased_depth >= 4 and equal_depth >= 4
    assert orientations == {"horizontal", "vertical"}
    assert 1 in depths and max(depths) >= 4
    assert 0 in background_counts and max(background_counts) >= 4
    assert min(visibility_counts) < max(visibility_counts)
    for split in ("train", "test"):
        r = review[split]
        assert r["triple"] >= 30 and r["fourway"] >= 10
        assert r["no_background"] >= 15 and r["hidden"] >= 20
        assert r["blank_panel"] >= 8 and r["solid_panel"] >= 3
        assert r["identical"] >= 5 and r["containment"] >= 20
        assert r["disjoint"] >= 10 and r["covered"] >= 10
        assert r["large"] >= 15 and r["compact"] >= 4
        assert r["hollow"] >= 8 and r["rectangle"] >= 10
        assert r["bar"] >= 20 and r["disconnected"] >= 30
