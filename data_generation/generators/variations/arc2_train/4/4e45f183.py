import numpy as np


_PANEL_ORIGINS = (1, 7, 13)
_LATTICE_LINES = (0, 6, 12, 18)

# Rejecting any grid whose output exactly matches a source output is sufficient
# to prevent exact duplication of a source input/output pair.
_SOURCE_OUTPUT_KEYS = (
    (
        "0000000000000000000"
        "0233330332330333320"
        "0323330332330333230"
        "0333330333330333330"
        "0333330333330333330"
        "0333330333330333330"
        "0000000000000000000"
        "0333330233320333330"
        "0333330323230333330"
        "0223330333330333220"
        "0333330323230333330"
        "0333330233320333330"
        "0000000000000000000"
        "0333330333330333330"
        "0333330333330333330"
        "0333330333330333330"
        "0323330332330333230"
        "0233330332330333320"
        "0000000000000000000"
    ),
    (
        "0000000000000000000"
        "0881110881880111880"
        "0811110811180111180"
        "0111110111110111110"
        "0111110111110111110"
        "0111110111110111110"
        "0000000000000000000"
        "0881110881880111880"
        "0811110811180111180"
        "0111110111110111110"
        "0811110811180111180"
        "0881110881880111880"
        "0000000000000000000"
        "0111110111110111110"
        "0111110111110111110"
        "0111110111110111110"
        "0811110811180111180"
        "0881110881880111880"
        "0000000000000000000"
    ),
    (
        "0000000000000000000"
        "0668880666660888660"
        "0668880888880888660"
        "0888880888880888880"
        "0888880888880888880"
        "0888880888880888880"
        "0000000000000000000"
        "0688880668660888860"
        "0688880668660888860"
        "0688880888880888860"
        "0688880668660888860"
        "0688880668660888860"
        "0000000000000000000"
        "0888880888880888880"
        "0888880888880888880"
        "0888880888880888880"
        "0668880888880888660"
        "0668880666660888660"
        "0000000000000000000"
    ),
    (
        "0000000000000000000"
        "0114440411140444110"
        "0144440441440444410"
        "0444440444440444440"
        "0444440444440444440"
        "0444440444440444440"
        "0000000000000000000"
        "0444440114110444440"
        "0144440144410444410"
        "0114440444440444110"
        "0144440144410444410"
        "0444440114110444440"
        "0000000000000000000"
        "0444440444440444440"
        "0444440444440444440"
        "0444440444440444440"
        "0144440441440444410"
        "0114440411140444110"
        "0000000000000000000"
    ),
)


def _validate_count_argument(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _validate_count_argument(num_train, "num_train", 2, 10)
    num_test = _validate_count_argument(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _axis_class(value, size):
    fraction = value / (size - 1)
    if fraction <= 0.36:
        return 0
    if 0.43 <= fraction <= 0.57:
        return 1
    if fraction >= 0.64:
        return 2
    return None


def _classify_mask(mask):
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        return None
    row_class = _axis_class(float(coordinates[:, 0].mean()), mask.shape[0])
    col_class = _axis_class(float(coordinates[:, 1].mean()), mask.shape[1])
    if row_class is None or col_class is None:
        return None
    return 3 * row_class + col_class


def _component_count(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _is_solid_rectangle(mask):
    points = np.argwhere(mask)
    if len(points) < 4:
        return False
    low = points.min(axis=0)
    high = points.max(axis=0)
    return bool(np.all(mask[low[0]:high[0] + 1, low[1]:high[1] + 1]))


def _is_thin_line(mask):
    points = np.argwhere(mask)
    if len(points) < 2:
        return False
    rows, cols = points[:, 0], points[:, 1]
    if len(set(rows)) == 1:
        return len(set(cols)) == len(cols) and max(cols) - min(cols) + 1 == len(cols)
    if len(set(cols)) == 1:
        return len(set(rows)) == len(rows) and max(rows) - min(rows) + 1 == len(rows)
    if len(set(rows)) == len(points) and len(set(cols)) == len(points):
        ordered = points[np.argsort(rows)]
        return bool(np.all(np.diff(ordered[:, 0]) == 1) and
                    np.all(np.abs(np.diff(ordered[:, 1])) == 1) and
                    len(set(np.diff(ordered[:, 1]))) == 1)
    return False


def _trim(mask):
    points = np.argwhere(mask)
    low, high = points.min(axis=0), points.max(axis=0)
    return mask[low[0]:high[0] + 1, low[1]:high[1] + 1]


def _base_shape(rng, family, height, width):
    bh = int(rng.integers(2, min(height, 5) + 1))
    bw = int(rng.integers(2, min(width, 5) + 1))
    mask = np.zeros((bh, bw), dtype=bool)
    if family == "stroke":
        direction = int(rng.integers(0, 4))
        if direction == 0:
            mask[:int(rng.integers(1, min(3, bh) + 1)), :] = True
        elif direction == 1:
            mask[:, :int(rng.integers(1, min(3, bw) + 1))] = True
        else:
            length = min(bh, bw)
            for i in range(length):
                mask[i, i if direction == 2 else length - 1 - i] = True
    elif family == "block":
        mask[:, :] = True
    elif family == "L":
        thickness = int(rng.integers(1, min(3, bh, bw) + 1))
        mask[:thickness, :] = True
        mask[:, :thickness] = True
        if rng.integers(2):
            mask = mask[::-1, :]
        if rng.integers(2):
            mask = mask[:, ::-1]
    elif family == "T":
        mask[0, :] = True
        center = (bw - 1) // 2
        mask[:, center:center + int(rng.integers(1, min(3, bw - center + 1)))] = True
        mask = np.rot90(mask, int(rng.integers(4)))
    elif family == "hollow":
        bh = max(3, bh)
        bw = max(3, bw)
        mask = np.zeros((bh, bw), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif family == "irregular":
        row, col = int(rng.integers(bh)), int(rng.integers(bw))
        mask[row, col] = True
        target = int(rng.integers(4, min(13, bh * bw + 1)))
        for _ in range(target - 1):
            boundary = set()
            for r, c in np.argwhere(mask):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < bh and 0 <= nc < bw and not mask[nr, nc]:
                        boundary.add((nr, nc))
            if not boundary:
                break
            choices = sorted(boundary)
            mask[choices[int(rng.integers(len(choices)))]] = True
    elif family == "cluster":
        count = int(rng.integers(2, min(6, bh * bw) + 1))
        cells = rng.choice(bh * bw, size=count, replace=False)
        mask.flat[cells] = True
        if rng.integers(2) and bh >= 3 and bw >= 3:
            row, col = int(rng.integers(bh - 1)), int(rng.integers(bw - 1))
            mask[row:row + 2, col:col + 2] = True
    else:
        raise ValueError("unknown motif family")
    return _trim(mask)


def _center_special(rng, height, width, kind):
    mask = np.zeros((height, width), dtype=bool)
    if kind == "corners":
        if min(height, width) < 5:
            kind = "hollow"
        else:
            primitive = ((0, 0),) if rng.integers(5) == 0 else (
                ((0, 0), (1, 1)) if rng.integers(3) == 0 else
                ((0, 0), (0, 1), (1, 0)) if rng.integers(2) == 0 else
                ((0, 0), (0, 1), (1, 0), (1, 1))
            )
            ro = int(rng.integers(0, max(1, height // 2 - 1)))
            co = int(rng.integers(0, max(1, width // 2 - 1)))
            for r, c in primitive:
                r, c = r + ro, c + co
                for rr in (r, height - 1 - r):
                    for cc in (c, width - 1 - c):
                        mask[rr, cc] = True
            if _classify_mask(mask) == 4:
                return mask
            mask[:] = False
            kind = "hollow"
    if kind == "small":
        rh = 1 if height % 2 else 2
        rw = 1 if width % 2 else 2
    else:
        rh = min(height, 5 if height % 2 else 4)
        rw = min(width, 5 if width % 2 else 4)
        if kind == "solid" and rh >= 4:
            rh -= 2
        if kind == "solid" and rw >= 4:
            rw -= 2
    r0, c0 = (height - rh) // 2, (width - rw) // 2
    mask[r0:r0 + rh, c0:c0 + rw] = True
    if kind == "hollow" and rh >= 3 and rw >= 3:
        mask[r0 + 1:r0 + rh - 1, c0 + 1:c0 + rw - 1] = False
    assert _classify_mask(mask) == 4
    return mask


def _make_mask(rng, sector, height, width, family):
    if sector == 4 and family in ("corners", "small", "solid", "hollow_center"):
        kind = "hollow" if family == "hollow_center" else family
        return _center_special(rng, height, width, kind)
    if family == "hollow_center":
        family = "hollow"
    for _ in range(90):
        base = _base_shape(rng, family, height, width)
        bh, bw = base.shape
        placements = []
        for r0 in range(height - bh + 1):
            for c0 in range(width - bw + 1):
                mask = np.zeros((height, width), dtype=bool)
                mask[r0:r0 + bh, c0:c0 + bw] = base
                if int(mask.sum()) < height * width and _classify_mask(mask) == sector:
                    placements.append(mask)
        if placements:
            return placements[int(rng.integers(len(placements)))]
    return _minimal_mask(sector, height, width)


def _minimal_mask(sector, height, width):
    # A central mark or boundary mark has a valid location at every size.
    mask = np.zeros((height, width), dtype=bool)
    row_class, col_class = divmod(sector, 3)
    row = (0, (height - 1) // 2, height - 1)[row_class]
    col = (0, (width - 1) // 2, width - 1)[col_class]
    mask[row, col] = True
    if row_class == 1 and height % 2 == 0:
        mask[row + 1, col] = True
    if col_class == 1 and width % 2 == 0:
        mask[row, col + 1] = True
        if row_class == 1 and height % 2 == 0:
            mask[row + 1, col + 1] = True
    assert _classify_mask(mask) == sector
    return mask


_FAMILIES = ("stroke", "block", "L", "T", "irregular", "cluster", "hollow")
_CENTER_FAMILIES = ("corners", "small", "solid", "hollow_center",
                    "stroke", "block", "L", "T", "irregular", "cluster")


def _make_masks(rng, height, width, regime, index, simple):
    for _ in range(20):
        coordinated = bool(rng.integers(4) < (3 if simple else 1))
        common_family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
        masks = []
        for sector in range(9):
            if sector == 4:
                if (regime == "source_train" and index == 0) or (
                    regime == "target_test" and height == width == 5
                ):
                    family = "corners"
                else:
                    family = _CENTER_FAMILIES[int(rng.integers(len(_CENTER_FAMILIES)))]
            elif regime == "source_train" and index == 0 and sector == 0:
                family = "stroke"
            elif regime == "source_train" and index == 0 and sector == 8:
                family = "block"
            elif coordinated and rng.integers(4):
                family = common_family
            else:
                family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
            masks.append(_make_mask(rng, sector, height, width, family))
        if sum(int(mask.sum()) for mask in masks) < 0.44 * 9 * height * width:
            return masks
    masks = [_minimal_mask(sector, height, width) for sector in range(9)]
    if (regime == "source_train" and index == 0) or (
        regime == "target_test" and height == width == 5
    ):
        masks[4][:] = False
        for row in (0, height - 1):
            for col in (0, width - 1):
                masks[4][row, col] = True
    return masks


def _sample_permutation(rng, regime, index):
    mode = "full" if regime == "source_train" and index == 0 else (
        "full", "swap", "cycle", "partial", "identity"
    )[int(rng.choice(5, p=(0.35, 0.23, 0.18, 0.20, 0.04)))]
    permutation = list(range(9))
    if mode == "identity":
        return tuple(permutation)
    if mode == "full":
        for _ in range(20):
            permutation = list(map(int, rng.permutation(9)))
            if sum(x != i for i, x in enumerate(permutation)) >= 3:
                return tuple(permutation)
    elif mode == "swap":
        a, b = map(int, rng.choice(9, size=2, replace=False))
        permutation[a], permutation[b] = permutation[b], permutation[a]
    else:
        size = 3 if mode == "cycle" else int(rng.integers(4, 8))
        positions = list(map(int, rng.choice(9, size=size, replace=False)))
        rotated = positions[1:] + positions[:1]
        for a, b in zip(positions, rotated):
            permutation[a] = b
    return tuple(permutation)


def _sample_dimensions(rng, regime, index, simple):
    if regime == "source_train" and index == 0:
        return 5, 5
    if simple:
        if regime == "source_train" and index == 1:
            return ((5, 6), (6, 5))[int(rng.integers(2))]
        return ((5, 5), (5, 6), (6, 5))[int(rng.integers(3))]
    if regime == "source_train" and index == 1:
        choices = ((4, 4), (5, 6), (6, 5), (5, 7), (7, 5),
                   (6, 6), (4, 7), (7, 4), (8, 5), (5, 8))
        return choices[int(rng.integers(len(choices)))]
    if regime == "target_test":
        height = int(rng.choice((5, 6, 7, 8)))
        width = int(rng.choice((5, 6, 7, 8)))
    else:
        height = int(rng.choice((4, 5, 5, 5, 6, 6, 7, 8)))
        width = int(rng.choice((4, 5, 5, 5, 6, 6, 7, 8)))
    return height, width


def _sample_colors(rng, regime, index, history, simple):
    if regime == "source_train" and index == 0:
        roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
        return roles[0], roles[1], (roles[2],)
    if simple:
        first = history[0]
        palette = (first[0], first[1], first[2][0])
        if regime == "source_train" and index == 1:
            return palette[1], palette[2], (palette[0],)
        roles = tuple(map(int, rng.permutation(palette)))
        return roles[0], roles[1], (roles[2],)
    if regime == "source_train" and index == 1:
        if rng.integers(2):
            first = history[0]
            separator, fill = first[1], first[2][0]
            primary = first[0]
        else:
            for _ in range(10):
                separator, fill, primary = map(int, rng.choice(10, size=3, replace=False))
                if (separator, fill) != history[0][:2]:
                    break
            if (separator, fill) == history[0][:2]:
                separator, fill, primary = history[0][1], history[0][2][0], history[0][0]
    else:
        if rng.integers(4) == 0:
            separator, fill = history[int(rng.integers(len(history)))][:2]
        else:
            separator, fill = map(int, rng.choice(10, size=2, replace=False))
        if regime == "target_test":
            mode = int(rng.integers(3))
            if mode == 0:
                pool = {c for _, _, inks in history for c in inks}
            elif mode == 1:
                pool = {f for _, f, _ in history}
            else:
                used = {c for s, f, inks in history for c in (s, f, *inks)}
                pool = set(range(10)) - used
            pool -= {separator, fill}
            primary = int(rng.choice(sorted(pool))) if pool else int(
                rng.choice([c for c in range(10) if c not in (separator, fill)]))
        else:
            primary = int(rng.choice([c for c in range(10) if c not in (separator, fill)]))
    choices = [c for c in range(10) if c not in (separator, fill, primary)]
    if regime == "source_train" and index == 1:
        extra_count = int(rng.integers(1, 4))
    else:
        extra_count = 0 if rng.integers(5) < 2 else int(rng.integers(1, 5))
    extras = tuple(map(int, rng.choice(choices, size=extra_count, replace=False)))
    return int(separator), int(fill), (int(primary),) + extras


def _render_panels(panels, separator, fill):
    height, width = panels[0].shape
    grid = np.full((3 * (height + 1) + 1, 3 * (width + 1) + 1),
                   separator, dtype=np.int8)
    for position, panel in enumerate(panels):
        row, col = divmod(position, 3)
        r0, c0 = 1 + row * (height + 1), 1 + col * (width + 1)
        grid[r0:r0 + height, c0:c0 + width] = panel
    return grid


def _generate_pair(rng, regime, index, history, simple):
    height, width = _sample_dimensions(rng, regime, index, simple)
    masks = _make_masks(rng, height, width, regime, index, simple)
    separator, fill, inks = _sample_colors(rng, regime, index, history, simple)
    sector_inks = list(map(int, rng.choice(inks, size=9, replace=True)))
    if len(inks) > 1:
        order = rng.permutation(9)
        for k, color in enumerate(inks):
            sector_inks[int(order[k])] = color
    panels = []
    for mask, color in zip(masks, sector_inks):
        panel = np.full((height, width), fill, dtype=np.int8)
        panel[mask] = color
        panels.append(panel)
    permutation = _sample_permutation(rng, regime, index)
    input_grid = _render_panels([panels[label] for label in permutation], separator, fill)
    output_grid = _render_panels(panels, separator, fill)
    return input_grid, output_grid, (separator, fill, inks)


def _array_key(array):
    return array.shape, array.astype(np.int8, copy=False).tobytes()


def _digit_key(grid):
    return "".join(str(int(value)) for value in grid.flat)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    simple = bool(rng.integers(5) == 0)
    puzzle = {"train": [], "test": []}
    history = []
    seen = set()
    for regime, count, section in (("source_train", num_train, "train"),
                                   ("target_test", num_test, "test")):
        for index in range(count):
            for _ in range(30):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                input_grid, output_grid, roles = _generate_pair(child, regime, index, history, simple)
                key = _array_key(input_grid)
                if key not in seen and _digit_key(output_grid) not in _SOURCE_OUTPUT_KEYS:
                    seen.add(key)
                    puzzle[section].append({"input": input_grid, "output": output_grid})
                    history.append(roles)
                    break
            else:
                raise RuntimeError("unable to create a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    rows, cols = input_grid.shape
    assert 16 <= rows <= 28 and 16 <= cols <= 28
    assert (rows - 1) % 3 == 0 and (cols - 1) % 3 == 0
    height, width = (rows - 1) // 3 - 1, (cols - 1) // 3 - 1
    assert 4 <= height <= 8 and 4 <= width <= 8
    lattice = np.zeros(input_grid.shape, dtype=bool)
    lattice[::height + 1, :] = True
    lattice[:, ::width + 1] = True
    separator = int(input_grid[0, 0])
    assert np.all(input_grid[lattice] == separator)
    assert np.all(output_grid[lattice] == separator)
    assert np.all(input_grid[~lattice] != separator)
    assert np.all(output_grid[~lattice] != separator)
    values, counts = np.unique(input_grid[~lattice], return_counts=True)
    assert len(values) >= 2
    fill = int(values[np.argmax(counts)])
    assert int(max(counts)) > int(sum(counts) - max(counts))
    assert fill != separator
    assert np.array_equal(np.bincount(input_grid.flat, minlength=10),
                          np.bincount(output_grid.flat, minlength=10))
    panels = [None] * 9
    labels = []
    motif_colors = set()
    motif_counts = []
    for position in range(9):
        row, col = divmod(position, 3)
        r0, c0 = 1 + row * (height + 1), 1 + col * (width + 1)
        panel = input_grid[r0:r0 + height, c0:c0 + width]
        mask = panel != fill
        assert 1 <= int(mask.sum()) < height * width
        colors = set(map(int, np.unique(panel[mask])))
        assert len(colors) == 1
        motif_colors.update(colors)
        label = _classify_mask(mask)
        assert label is not None and panels[label] is None
        panels[label] = panel.copy()
        labels.append(label)
        motif_counts.append(int(mask.sum()))
    assert sorted(labels) == list(range(9))
    expected = _render_panels(panels, separator, fill)
    assert np.array_equal(output_grid, expected)
    return {"shape": (height, width), "roles": (separator, fill),
            "motif_colors": frozenset(motif_colors), "labels": tuple(labels),
            "masks": tuple(panel != fill for panel in panels),
            "motif_counts": tuple(motif_counts)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _digit_key(pair["output"]) not in _SOURCE_OUTPUT_KEYS
            info = _inspect_pair(pair["input"], pair["output"])
            infos[section].append(info)
    train = infos["train"]
    assert train[0]["shape"] == (5, 5)
    assert len({info["shape"] for info in train}) >= 2
    assert len({info["roles"] for info in train}) >= 2
    assert any(info["labels"] != tuple(range(9)) for info in train)
    assert any(_component_count(info["masks"][4]) >= 4 and
               not info["masks"][4][2, 2] for info in train)
    if any(len(info["motif_colors"]) >= 2 for info in infos["test"]):
        assert any(len(info["motif_colors"]) >= 2 for info in train)
    for info in infos["test"]:
        assert min(info["shape"]) >= 5
        assert max(info["shape"]) >= 6 or _component_count(info["masks"][4]) >= 2


def _puzzles_equal(first, second):
    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for a, b in zip(first[section], second[section]):
            if not all(np.array_equal(a[key], b[key]) for key in ("input", "output")):
                return False
    return True


def self_test() -> None:
    for counts in ((2, 1), (10, 1), (10, 4), (7, 3),
                   (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(20000 + 17 * (counts[0] or 0) + (counts[1] or 0),
                                 num_train=counts[0], num_test=counts[1])
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                {"num_train": 2, "num_test": 3},
                {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(77, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    assert _puzzles_equal(generate_puzzle(918273, 6, 2),
                          generate_puzzle(918273, 6, 2))
    assert not _puzzles_equal(generate_puzzle(1001, 5, 2),
                              generate_puzzle(1002, 5, 2))
    train_counts, test_counts, shapes, aspect_ratios = set(), set(), set(), set()
    layouts, centers, motif_counts, colors, palettes = set(), set(), set(), set(), set()
    family_features = set()
    fixed_points, moved_counts, multi_ink, rectangular = set(), set(), 0, 0
    ink_over_half, center_no_middle = 0, 0
    shared_palette_puzzles = varied_palette_puzzles = 0
    center_counts = {8: 0, 12: 0, 16: 0}
    test_center_counts = {8: 0, 12: 0, 16: 0}
    connected_center = disconnected_center = 0
    reused_test_ink = former_fill_test_ink = new_test_ink = 0
    combined_variations = 0
    features_by_split = {"train": set(), "test": set()}
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_palettes = {
            tuple(map(int, np.unique(pair["input"])))
            for section in ("train", "test") for pair in puzzle[section]
        }
        shared_palette_puzzles += len(puzzle_palettes) == 1
        varied_palette_puzzles += len(puzzle_palettes) > 1
        train_colors = {
            int(color) for pair in puzzle["train"]
            for color in np.unique(pair["input"])
        }
        train_infos = [
            _inspect_pair(pair["input"], pair["output"])
            for pair in puzzle["train"]
        ]
        train_fills = {info["roles"][1] for info in train_infos}
        train_inks = set().union(*(info["motif_colors"] for info in train_infos))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _inspect_pair(pair["input"], pair["output"])
                total_pairs += 1
                shapes.add(info["shape"])
                aspect_ratios.add(info["shape"][0] / info["shape"][1])
                layouts.add(info["labels"])
                centers.add((info["shape"], info["masks"][4].tobytes()))
                motif_counts.update(info["motif_counts"])
                colors.update(map(int, np.unique(pair["input"])))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                fixed_points.add(sum(label == i for i, label in enumerate(info["labels"])))
                moved_counts.add(sum(label != i for i, label in enumerate(info["labels"])))
                multi_ink += len(info["motif_colors"]) >= 2
                rectangular += info["shape"][0] != info["shape"][1]
                center = info["masks"][4]
                center_size = int(center.sum())
                if center_size in center_counts:
                    center_counts[center_size] += 1
                    if section == "test":
                        test_center_counts[center_size] += 1
                connected_center += _component_count(center) == 1
                disconnected_center += _component_count(center) >= 2
                fixed = sum(label == i for i, label in enumerate(info["labels"]))
                combined_variations += (
                    info["shape"][0] != info["shape"][1]
                    and len(info["motif_colors"]) >= 2
                    and _component_count(center) >= 2
                    and 0 < fixed < 9
                )
                if section == "test":
                    reused_test_ink += bool(info["motif_colors"] & train_inks)
                    former_fill_test_ink += bool(info["motif_colors"] & train_fills)
                    new_test_ink += bool(info["motif_colors"] - train_colors)
                for mask in info["masks"]:
                    features = (_is_thin_line(mask), _is_solid_rectangle(mask),
                                _component_count(mask) >= 2)
                    family_features.add(features)
                    features_by_split[section].add(features)
                    ink_over_half += int(mask.sum()) > mask.size / 2
                center_no_middle += not info["masks"][4][info["shape"][0] // 2,
                                                         info["shape"][1] // 2]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 15 and len(aspect_ratios) >= 10
    assert len(layouts) >= 80 and len(centers) >= 25
    assert len(motif_counts) >= 14 and len(colors) == 10
    assert len(palettes) >= 40 and len(family_features) >= 4
    assert 0 in fixed_points and any(n >= 2 for n in fixed_points)
    assert 2 in moved_counts and max(moved_counts) >= 8
    assert multi_ink >= 42 and rectangular >= 42
    assert ink_over_half > 0 and center_no_middle >= 42
    assert shared_palette_puzzles >= 2 and varied_palette_puzzles >= 20
    assert all(center_counts[size] >= 10 for size in (8, 12, 16))
    assert all(test_center_counts[size] >= 2 for size in (8, 12, 16))
    assert connected_center >= 20 and disconnected_center >= 20
    assert reused_test_ink >= 20 and former_fill_test_ink >= 10 and new_test_ink >= 2
    assert combined_variations >= 5
    assert len(features_by_split["train"]) >= 4
    assert len(features_by_split["test"]) >= 4
    assert total_pairs >= 100
