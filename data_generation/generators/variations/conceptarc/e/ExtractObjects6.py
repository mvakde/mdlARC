import numpy as np


_SOURCE_INPUTS = (
    ("000600045333355", "006060045355355", "060006045355355",
     "600000645355355", "060006045355355", "006060045355355",
     "000600045333355"),
    ("77777777", "77727777", "77727777", "77727777", "72222277",
     "77777777", "77777777", "44444444", "88838888", "88333888",
     "88838888", "88888888", "88888888"),
    ("111111", "111111", "118811", "118811", "111111", "111111",
     "111111", "111111", "111111", "444444", "775577", "775577",
     "777777"),
    ("07704333313333", "07704333111333", "07704331111133",
     "07704311111113"),
    ("999222299999", "999222299999", "999999999999", "444444444444",
     "777777777777", "777777777777", "777777777777", "777777777777",
     "777722227777", "777722227777", "777777777777", "777777777777"),
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _join_crops(crops, orientation):
    if orientation == "vertical":
        result = np.zeros((max(c.shape[0] for c in crops),
                           sum(c.shape[1] for c in crops)), dtype=np.int64)
        column = 0
        for crop in crops:
            row = (result.shape[0] - crop.shape[0]) // 2
            result[row:row + crop.shape[0],
                   column:column + crop.shape[1]] = crop
            column += crop.shape[1]
    else:
        result = np.zeros((sum(c.shape[0] for c in crops),
                           max(c.shape[1] for c in crops)), dtype=np.int64)
        row = 0
        for crop in crops:
            column = (result.shape[1] - crop.shape[1]) // 2
            result[row:row + crop.shape[0],
                   column:column + crop.shape[1]] = crop
            row += crop.shape[0]
    return result


_KINDS = (
    "frame", "solid", "ell", "tee", "plus", "line", "diagonal",
    "stair", "triangle", "outline_triangle", "diamond",
    "filled_diamond", "strokes", "scatter",
)


def _raw_mask(rng, height, width, kind):
    mask = np.zeros((height, width), dtype=bool)
    if height == 1 or width == 1:
        mask[:] = True
        if max(height, width) >= 4 and rng.random() < 0.3:
            if height == 1:
                mask[0, 1:-1] = rng.random(width - 2) < 0.6
            else:
                mask[1:-1, 0] = rng.random(height - 2) < 0.6
        return mask
    if kind == "frame":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == "solid":
        mask[:] = True
    elif kind == "ell":
        mask[:, 0] = True
        mask[-1, :] = True
        if width >= 4 and rng.random() < 0.5:
            mask[0, :int(rng.integers(2, width))] = True
    elif kind == "tee":
        mask[0, :] = True
        mask[:, int(rng.integers(1, width - 1))] = True
    elif kind == "plus":
        mask[int(rng.integers(1, height - 1)), :] = True
        mask[:, int(rng.integers(1, width - 1))] = True
    elif kind == "line":
        mask[int(rng.integers(height)), :] = True
        mask[:, int(rng.integers(width))] = True
    elif kind in ("diagonal", "stair"):
        for row in range(height):
            col = int(round(row * (width - 1) / (height - 1)))
            mask[row, col] = True
            if kind == "stair":
                mask[row, :col + 1] = rng.random() < 0.35
                mask[row, col] = True
    elif kind in ("triangle", "outline_triangle"):
        for row in range(height):
            radius = int(round((width - 1) * row / (2 * (height - 1))))
            middle = (width - 1) / 2
            left = max(0, int(np.floor(middle - radius)))
            right = min(width - 1, int(np.ceil(middle + radius)))
            if kind == "triangle" or row == height - 1:
                mask[row, left:right + 1] = True
            else:
                mask[row, left] = mask[row, right] = True
    elif kind in ("diamond", "filled_diamond"):
        for row in range(height):
            distance = abs((2 * row - height + 1) / (height - 1))
            radius = int(round((1 - distance) * (width - 1) / 2))
            middle = (width - 1) / 2
            left = max(0, int(np.floor(middle - radius)))
            right = min(width - 1, int(np.ceil(middle + radius)))
            if kind == "filled_diamond":
                mask[row, left:right + 1] = True
            else:
                mask[row, left] = mask[row, right] = True
        mask[height // 2, 0] = mask[height // 2, -1] = True
    elif kind == "strokes":
        for _ in range(int(rng.integers(2, 6))):
            r0, r1 = rng.integers(0, height, size=2)
            c0, c1 = rng.integers(0, width, size=2)
            for t in np.linspace(0, 1, max(abs(r1 - r0), abs(c1 - c0)) + 1):
                mask[int(round(r0 + t * (r1 - r0))),
                     int(round(c0 + t * (c1 - c0)))] = True
    elif kind == "scatter":
        for _ in range(int(rng.integers(2, max(3, height * width // 3)))):
            mask[int(rng.integers(height)), int(rng.integers(width))] = True
    else:
        raise ValueError("unknown motif")
    # Disconnected marks are allowed, but the requested dimensions are the bbox.
    for axis, extent in ((0, height), (1, width)):
        for edge in (0, extent - 1):
            if not np.any(np.take(mask, edge, axis=axis)):
                other = int(rng.integers(width if axis == 0 else height))
                if axis == 0:
                    mask[edge, other] = True
                else:
                    mask[other, edge] = True
    return mask


def _make_mask(rng, height, width, kind):
    rotation = int(rng.integers(4))
    if rotation % 2:
        raw = _raw_mask(rng, width, height, kind)
    else:
        raw = _raw_mask(rng, height, width, kind)
    mask = np.rot90(raw, rotation)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _sample_colors(rng, divider, normalization=False):
    nondivider = [c for c in range(1, 10) if c != divider]
    backgrounds = [
        0 if rng.random() < 0.34 else int(rng.choice(nondivider))
        for _ in range(2)
    ]
    if rng.random() < 0.22:
        backgrounds[1] = backgrounds[0]
    if normalization:
        colored = int(rng.integers(2))
        backgrounds[colored] = int(rng.choice(nondivider))
        backgrounds[1 - colored] = 0
    shared = rng.random() < 0.30
    if shared:
        choices = [c for c in nondivider if c not in backgrounds]
        color = int(rng.choice(choices))
        foregrounds = [color, color]
    else:
        foregrounds = [
            int(rng.choice([c for c in nondivider if c != backgrounds[i]]))
            for i in range(2)
        ]
        if (not normalization and backgrounds[0] != backgrounds[1]
                and backgrounds[1] != 0 and rng.random() < 0.25):
            foregrounds[0] = backgrounds[1]
    return backgrounds, foregrounds


def _sample_extent(rng):
    if rng.random() < 0.12:
        return 1
    if rng.random() < 0.10:
        return int(rng.integers(9, 13))
    return int(rng.integers(2, 9))


def _sample_margins(rng):
    return [int(rng.choice((0, 0, 0, 1, 1, 2, 2, 3, 4)))
            for _ in range(2)]


def _generate_pair(rng, divider, orientation, shifted, center_demo,
                   normalization, regime):
    seamless = not center_demo and not normalization and rng.random() < 0.13
    perpendicular = [_sample_extent(rng), _sample_extent(rng)]
    if seamless:
        side = int(rng.integers(2, 9))
        perpendicular = [side, side]
    elif center_demo:
        larger = int(rng.integers(5, 11))
        difference = int(rng.choice((2, 4)))
        perpendicular = [larger, larger - difference]
        if rng.random() < 0.5:
            perpendicular.reverse()
    else:
        second_choices = [n for n in range(1, 13)
                          if n % 2 == perpendicular[0] % 2]
        perpendicular[1] = int(rng.choice(second_choices))
    along = ([int(rng.integers(2, 7)) for _ in range(2)]
             if seamless else [_sample_extent(rng), _sample_extent(rng)])
    if normalization:
        along[int(rng.integers(2))] = int(rng.integers(4, 10))
    edge_panel = None
    if (not shifted and not normalization and not seamless
            and max(perpendicular) >= 5 and rng.random() < 0.10):
        edge_panel = int(np.argmax(perpendicular))
        along[edge_panel] = max(along[edge_panel], 5)
    extra = int(rng.integers(0, 7 if not shifted else 11))
    common = max(perpendicular) + (0 if edge_panel is not None else extra)
    if (common - perpendicular[0]) % 2:
        common += 1
    if shifted:
        common = max(common, max(perpendicular) + 3)
        limits = [common - n for n in perpendicular]
        offsets = [int(rng.integers(n + 1)) for n in limits]
        if 2 * offsets[0] + perpendicular[0] == 2 * offsets[1] + perpendicular[1]:
            offsets[1] = (offsets[1] + 1) % (limits[1] + 1)
    else:
        offsets = [(common - n) // 2 for n in perpendicular]
    margins = [_sample_margins(rng) for _ in range(2)]
    if edge_panel is not None:
        margins[edge_panel] = [0, 0]
    if seamless:
        margins = [[max(2, value) for value in pair] for pair in margins]
    while sum(along) + sum(sum(m) for m in margins) + 1 > 30:
        i = int(np.argmax([margins[0][0], margins[0][1],
                           margins[1][0], margins[1][1]]))
        margins[i // 2][i % 2] -= 1
    backgrounds, foregrounds = _sample_colors(rng, divider, normalization)
    if seamless:
        choices = [c for c in range(1, 10)
                   if c != divider and c not in backgrounds]
        foregrounds = [int(rng.choice(choices))] * 2
    panels, crops = [], []
    normalized_panel = None
    if normalization:
        normalized_panel = 0 if backgrounds[0] != 0 else 1
    for i in range(2):
        if orientation == "vertical":
            height, width = perpendicular[i], along[i]
        else:
            height, width = along[i], perpendicular[i]
        if i == normalized_panel:
            height = max(height, 4)
            width = max(width, 4)
            # Preserve the perpendicular extent and common panel size.
            if orientation == "vertical":
                height = perpendicular[i]
                if height < 4:
                    raise ValueError("normalization motif too short")
            else:
                width = perpendicular[i]
                if width < 4:
                    raise ValueError("normalization motif too narrow")
            kind = "frame"
        else:
            kind = ("diamond" if i == edge_panel else
                    "solid" if seamless else str(rng.choice(_KINDS)))
        mask = _make_mask(rng, height, width, kind)
        before, after = margins[i]
        if orientation == "vertical":
            panel = np.full((common, before + width + after),
                            backgrounds[i], dtype=np.int64)
            row, col = offsets[i], before
        else:
            panel = np.full((before + height + after, common),
                            backgrounds[i], dtype=np.int64)
            row, col = before, offsets[i]
        rr, cc = np.where(mask)
        panel[row + rr, col + cc] = foregrounds[i]
        panels.append(panel)
        crops.append(np.where(mask, foregrounds[i], 0).astype(np.int64))
    if orientation == "vertical":
        separator = np.full((common, 1), divider, dtype=np.int64)
        input_grid = np.concatenate((panels[0], separator, panels[1]), axis=1)
    else:
        separator = np.full((1, common), divider, dtype=np.int64)
        input_grid = np.concatenate((panels[0], separator, panels[1]), axis=0)
    return input_grid, _join_crops(crops, orientation)


def _panel_colors(panel, divider):
    colors, counts = np.unique(panel, return_counts=True)
    _check(len(colors) == 2 and divider not in colors,
           "each panel needs two local colors distinct from its divider")
    if counts[0] == counts[1]:
        corners = (panel[0, 0], panel[0, -1], panel[-1, 0], panel[-1, -1])
        corner_counts = [sum(value == color for value in corners)
                         for color in colors]
        _check(corner_counts[0] != corner_counts[1],
               "local background is ambiguous")
        background = int(colors[int(np.argmax(corner_counts))])
    else:
        background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[0] if colors[1] == background else colors[1])
    _check(foreground != 0, "foreground must be nonzero")
    return background, foreground


def _inspect_input(input_grid):
    height, width = input_grid.shape
    candidates = []
    for row in range(1, height - 1):
        color = int(input_grid[row, 0])
        if (color != 0 and np.all(input_grid[row] == color)
                and np.count_nonzero(input_grid == color) == width):
            candidates.append(("horizontal", row, color))
    for col in range(1, width - 1):
        color = int(input_grid[0, col])
        if (color != 0 and np.all(input_grid[:, col] == color)
                and np.count_nonzero(input_grid == color) == height):
            candidates.append(("vertical", col, color))
    _check(len(candidates) == 1, "divider is not unique")
    orientation, position, divider = candidates[0]
    if orientation == "horizontal":
        panels = (input_grid[:position], input_grid[position + 1:])
    else:
        panels = (input_grid[:, :position], input_grid[:, position + 1:])
    info, crops = [], []
    for panel in panels:
        _check(panel.size > 0, "empty panel")
        background, foreground = _panel_colors(panel, divider)
        rows, cols = np.where(panel == foreground)
        box = (int(rows.min()), int(rows.max()),
               int(cols.min()), int(cols.max()))
        r0, r1, c0, c1 = box
        crop = np.where(panel[r0:r1 + 1, c0:c1 + 1] == foreground,
                        foreground, 0).astype(np.int64)
        info.append({"background": background, "foreground": foreground,
                     "box": box, "panel_shape": panel.shape, "crop": crop})
        crops.append(crop)
    axis = 0 if orientation == "vertical" else 1
    _check(abs(crops[0].shape[axis] - crops[1].shape[axis]) % 2 == 0,
           "centering must be integral")
    return {"orientation": orientation, "divider": divider,
            "panels": info, "expected": _join_crops(crops, orientation)}


def _centers_align(info):
    indices = (0, 1) if info["orientation"] == "vertical" else (2, 3)
    a, b = (panel["box"] for panel in info["panels"])
    return sum(a[i] for i in indices) == sum(b[i] for i in indices)


def _perpendicular_difference(info):
    axis = 0 if info["orientation"] == "vertical" else 1
    a, b = info["panels"]
    return abs(a["crop"].shape[axis] - b["crop"].shape[axis])


def _margins(info):
    result = []
    for panel in info["panels"]:
        r0, r1, c0, c1 = panel["box"]
        height, width = panel["panel_shape"]
        result.append((r0, height - r1 - 1, c0, width - c1 - 1))
    return result


def _normalization_visible(info):
    return any(p["background"] != 0 and np.any(p["crop"] == 0)
               for p in info["panels"])


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(type(puzzle) is dict and set(puzzle) == {"train", "test"},
           "puzzle must have train and test")
    _check(type(puzzle["train"]) is list and type(puzzle["test"]) is list,
           "groups must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
           "invalid pair counts")
    inspected = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for group in ("train", "test"):
        for pair in puzzle[group]:
            _check(type(pair) is dict and set(pair) == {"input", "output"},
                   "invalid pair")
            for name in ("input", "output"):
                grid = pair[name]
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a two-dimensional array")
                _check(all(1 <= n <= 30 for n in grid.shape),
                       "grid dimension out of range")
                _check(np.issubdtype(grid.dtype, np.integer),
                       "grid must have integer dtype")
                _check(np.all((grid >= 0) & (grid <= 9)),
                       "invalid ARC color")
            inp, out = pair["input"], pair["output"]
            key = (inp.shape, inp.astype(np.int64).tobytes())
            pair_key = (key, out.shape, out.astype(np.int64).tobytes())
            _check(key not in seen_inputs and pair_key not in seen_pairs,
                   "duplicate generated pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            rows = tuple("".join(str(int(c)) for c in row) for row in inp)
            _check(rows not in _SOURCE_INPUTS, "source input copied")
            info = _inspect_input(inp)
            _check(np.array_equal(info["expected"], out),
                   "crop, normalize, center, join rule mismatch")
            _check(not np.array_equal(inp, out), "unchanged output")
            inspected[group].append(info)
    _check(len({i["divider"] for values in inspected.values()
                for i in values}) == 1, "divider varies within puzzle")
    train, test = inspected["train"], inspected["test"]
    _check(any(_normalization_visible(i) for i in train),
           "training must reveal colored-background normalization")
    _check(any(p["background"] == 0 for i in train for p in i["panels"]),
           "training must show zero background")
    _check(any(_perpendicular_difference(i) > 0 for i in train),
           "training must demonstrate centering")
    _check(any(_centers_align(i) for i in train)
           and any(not _centers_align(i) for i in train),
           "training must include aligned and shifted crops")
    _check({i["orientation"] for i in test}
           <= {i["orientation"] for i in train},
           "test orientation was not demonstrated")


def _validate_count(value, minimum, maximum, name):
    if value is None:
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(name)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _validate_count(num_train, 2, 10, "num_train")
    _validate_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_train = int(rng.choice((2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10)))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train is None:
        choices = list(range(max(2, num_test + 1), 11))
        num_train = int(rng.choice(choices))
    divider = int(rng.integers(1, 10))
    orientations = ["horizontal", "vertical"]
    if rng.random() < 0.22:
        only = str(rng.choice(orientations))
        train_orientations = [only] * num_train
    else:
        train_orientations = [*orientations]
        train_orientations += [str(rng.choice(orientations))
                              for _ in range(num_train - 2)]
        rng.shuffle(train_orientations)
    available = tuple(orientation for orientation in orientations
                      if orientation in train_orientations)
    test_orientations = [str(rng.choice(available)) for _ in range(num_test)]
    if len(available) == 2 and num_test >= 2 and rng.random() < 0.6:
        test_orientations[:2] = orientations
        rng.shuffle(test_orientations)
    aligned_slot = int(rng.integers(num_train))
    shifted_slot = int(rng.integers(num_train - 1))
    if shifted_slot >= aligned_slot:
        shifted_slot += 1
    normalization_slot = int(rng.integers(num_train))
    center_slot = int(rng.integers(num_train))
    puzzle = {"train": [], "test": []}
    seen = set()
    for group, count, planned in (
        ("train", num_train, train_orientations),
        ("test", num_test, test_orientations),
    ):
        for index in range(count):
            for _ in range(180):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if group == "train":
                    shifted = (index == shifted_slot
                               or (index != aligned_slot and pair_rng.random() < 0.48))
                    center_demo = index == center_slot
                    normalization = index == normalization_slot
                else:
                    shifted = pair_rng.random() < 0.58
                    center_demo = pair_rng.random() < 0.45
                    normalization = pair_rng.random() < 0.30
                try:
                    inp, out = _generate_pair(
                        pair_rng, divider, planned[index], shifted,
                        center_demo, normalization, group)
                    if not all(1 <= n <= 30 for n in (*inp.shape, *out.shape)):
                        continue
                    info = _inspect_input(inp)
                    if (not np.array_equal(info["expected"], out)
                            or (center_demo and _perpendicular_difference(info) == 0)
                            or (normalization and not _normalization_visible(info))
                            or (_centers_align(info) == shifted)):
                        continue
                    key = (inp.shape, inp.tobytes())
                    if key in seen:
                        continue
                except (AssertionError, ValueError):
                    continue
                seen.add(key)
                puzzle[group].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not construct a unique clear pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[g]) == len(b[g]) and all(
        np.array_equal(x[n], y[n])
        for x, y in zip(a[g], b[g]) for n in ("input", "output"))
        for g in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, dividers = set(), set(), set()
    layouts, motifs, fingerprints = set(), set(), set()
    features = {name: 0 for name in (
        "edge", "divider_contact", "thin", "large", "shared_foreground",
        "shared_background", "cross_color", "aligned_test", "shifted_test",
        "equal_perpendicular", "wide_margin", "normalization", "seamless",
    )}
    one_orientation = mixed_orientation = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_same_puzzle(puzzle, generate_puzzle(seed)),
               "automatic generation is not deterministic")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprint = tuple((p["input"].shape, p["input"].tobytes(),
                             p["output"].shape, p["output"].tobytes())
                            for g in ("train", "test") for p in puzzle[g])
        fingerprints.add(fingerprint)
        orientations = set()
        for group in ("train", "test"):
            for pair in puzzle[group]:
                info = _inspect_input(pair["input"])
                _check(np.array_equal(info["expected"], pair["output"]),
                       "independent reconstruction mismatch")
                orientations.add(info["orientation"])
                dividers.add(info["divider"])
                layouts.add((info["orientation"], pair["input"].shape,
                             pair["output"].shape,
                             tuple(p["box"] for p in info["panels"])))
                panel_margins = _margins(info)
                if info["orientation"] == "vertical":
                    divider_contact = (panel_margins[0][3] == 0
                                       or panel_margins[1][2] == 0)
                else:
                    divider_contact = (panel_margins[0][1] == 0
                                       or panel_margins[1][0] == 0)
                features["divider_contact"] += int(divider_contact)
                for panel, margins in zip(info["panels"], panel_margins):
                    crop = panel["crop"]
                    motifs.add((crop.shape, (crop != 0).tobytes()))
                    features["edge"] += int(0 in margins)
                    features["thin"] += int(min(crop.shape) == 1)
                    features["large"] += int(max(crop.shape) >= 7)
                    features["wide_margin"] += int(max(margins) >= 3)
                a, b = info["panels"]
                features["shared_foreground"] += int(
                    a["foreground"] == b["foreground"])
                features["shared_background"] += int(
                    a["background"] == b["background"])
                features["cross_color"] += int(
                    a["foreground"] == b["background"]
                    or b["foreground"] == a["background"])
                features["equal_perpendicular"] += int(
                    _perpendicular_difference(info) == 0)
                features["normalization"] += int(
                    _normalization_visible(info))
                features["seamless"] += int(
                    a["foreground"] == b["foreground"]
                    and np.all(pair["output"] != 0))
                if group == "test":
                    features["aligned_test"] += int(_centers_align(info))
                    features["shifted_test"] += int(not _centers_align(info))
        one_orientation += len(orientations) == 1
        mixed_orientation += len(orientations) == 2
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "pair count diversity collapsed")
    _check(len(fingerprints) == 42 and len(dividers) >= 5,
           "seed or palette diversity collapsed")
    _check(len(layouts) >= 120 and len(motifs) >= 80,
           "structural diversity collapsed")
    _check(one_orientation >= 2 and mixed_orientation >= 2,
           "orientation regimes collapsed")
    _check(all(value >= 2 for value in features.values()),
           "one review property is missing")
    for n_train, n_test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(100 + n_train * 10 + n_test, n_train, n_test)
        _check((len(puzzle["train"]), len(puzzle["test"]))
               == (n_train, n_test), "explicit counts ignored")
        validate_generated_puzzle(puzzle)
        _check(_same_puzzle(puzzle, generate_puzzle(
            100 + n_train * 10 + n_test, n_train, n_test)),
            "explicit generation is not deterministic")
    for n_train, n_test in ((None, 1), (None, 4),
                            (2, None), (10, None)):
        puzzle = generate_puzzle(200, n_train, n_test)
        validate_generated_puzzle(puzzle)
        _check(n_train is None or len(puzzle["train"]) == n_train,
               "partial train count ignored")
        _check(n_test is None or len(puzzle["test"]) == n_test,
               "partial test count ignored")
    for n_train, n_test in ((1, None), (11, None), (None, 0),
                            (None, 5), (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(300, n_train, n_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
