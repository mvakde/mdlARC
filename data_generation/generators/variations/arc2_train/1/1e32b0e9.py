"""Registered-stencil completion puzzles with divider-colored additions."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = frozenset({
    "1db1991c8afb6cbfbf2233ec3bc253b30e518e6fdf4dda39ca7ed9b04f7eb16d",
    "54448f3ccfb12da6f444badd60560ed6d888c091b05797ca27094b4e562b85bb",
    "d7b0102160fd17d22fbacb4798f01490e556b3a1f7caca24816e030b82e19f32",
    "a71ceeecff06322539585aaa7051766eff9374301ae2cae18536be608fe6b98e",
})
_CATEGORIES = ("filled", "plus", "diamond", "frame", "bent", "growth", "hole", "pieces")
_TRAIN_WEIGHTS = (15, 9, 9, 14, 15, 17, 9, 12)
_TEST_WEIGHTS = (7, 5, 5, 17, 18, 20, 13, 15)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        if num_train <= 1:
            raise ValueError("no legal test count")
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _sample_dimension(rng, count):
    maximum = (31 - count) // count
    mode = int(rng.integers(3))
    if mode == 0:
        upper = min(maximum, 5)
        return int(rng.integers(3, upper + 1))
    if mode == 1:
        return int(rng.integers(3, maximum + 1))
    lower = max(3, maximum - 2)
    return int(rng.integers(lower, maximum + 1))


def _sample_geometry(rng):
    mode = rng.random()
    if mode < 0.10:
        return (2, 2, int(rng.integers(3, 5)), int(rng.integers(3, 5)))
    if mode < 0.17:
        nrows = int(rng.integers(4, 6))
        ncols = int(rng.integers(4, 6))
        return (nrows, ncols, (31 - nrows) // nrows, (31 - ncols) // ncols)
    nrows = int(rng.integers(2, 6))
    ncols = int(rng.integers(2, 6))
    return (nrows, ncols, _sample_dimension(rng, nrows), _sample_dimension(rng, ncols))


def _extent(rng, limit, low=2):
    maximum = min(limit, 8)
    if int(rng.integers(4)) == 0:
        maximum = limit
    return int(rng.integers(low, maximum + 1))


def _local_motif(rng, height, width, category):
    if category == "filled":
        sh, sw = _extent(rng, height), _extent(rng, width)
        local = np.ones((sh, sw), dtype=bool)
    elif category == "plus":
        local = np.zeros((3, 3), dtype=bool)
        local[1, :] = True
        local[:, 1] = True
    elif category == "diamond":
        radius = 2 if min(height, width) >= 5 and rng.random() < 0.35 else 1
        side = 2 * radius + 1
        local = np.zeros((side, side), dtype=bool)
        for row in range(side):
            for col in range(side):
                local[row, col] = abs(row - radius) + abs(col - radius) == radius
    elif category in ("frame", "hole"):
        sh, sw = _extent(rng, height, 3), _extent(rng, width, 3)
        local = np.zeros((sh, sw), dtype=bool)
        local[0, :] = local[-1, :] = True
        local[:, 0] = local[:, -1] = True
        if category == "hole" and (sh - 2) * (sw - 2) >= 2:
            interior = [(r, c) for r in range(1, sh - 1) for c in range(1, sw - 1)]
            count = int(rng.integers(1, len(interior)))
            for idx in rng.choice(len(interior), count, replace=False):
                local[interior[int(idx)]] = True
    elif category == "bent":
        sh, sw = _extent(rng, height, 3), _extent(rng, width, 3)
        local = np.zeros((sh, sw), dtype=bool)
        bend_row = int(rng.integers(1, sh - 1))
        bend_col = int(rng.integers(1, sw - 1))
        local[bend_row, :bend_col + 1] = True
        local[bend_row:, bend_col] = True
        if rng.random() < 0.4:
            local[-1, bend_col:] = True
        if rng.random() < 0.5:
            local = np.flip(local, axis=0)
        if rng.random() < 0.5:
            local = np.flip(local, axis=1)
    elif category == "growth":
        sh, sw = _extent(rng, height, 3), _extent(rng, width, 3)
        local = np.zeros((sh, sw), dtype=bool)
        start = (int(rng.integers(sh)), int(rng.integers(sw)))
        local[start] = True
        target = int(rng.integers(4, max(5, int(0.8 * sh * sw) + 1)))
        for _ in range(target - 1):
            frontier = set()
            for row, col in np.argwhere(local):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(row + dr), int(col + dc)
                    if 0 <= nr < sh and 0 <= nc < sw and not local[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            points = sorted(frontier)
            local[points[int(rng.integers(len(points)))]] = True
    else:  # separated pieces with a full empty row or column between them
        horizontal = bool(rng.integers(2))
        sh = _extent(rng, height, 3) if horizontal else _extent(rng, height)
        sw = _extent(rng, width) if horizontal else _extent(rng, width, 3)
        local = np.zeros((sh, sw), dtype=bool)
        if horizontal:
            for row in (0, sh - 1):
                start = int(rng.integers(max(1, sw - 1)))
                local[row, start:start + 2] = True
        else:
            for col in (0, sw - 1):
                start = int(rng.integers(max(1, sh - 1)))
                local[start:start + 2, col] = True
    return local


def _sample_stencil(rng, height, width, regime):
    weights = _TRAIN_WEIGHTS if regime == "source_train" else _TEST_WEIGHTS
    category = _CATEGORIES[int(rng.choice(len(_CATEGORIES), p=np.array(weights) / sum(weights)))]
    local = _local_motif(rng, height, width, category)
    if int(local.sum()) < 4:
        local = _local_motif(rng, height, width, "plus")
    sh, sw = local.shape
    placement = int(rng.integers(6))
    if placement == 0 and local[0, 0]:
        top, left = 0, 0
    elif placement == 1:
        top, left = (height - sh) // 2, (width - sw) // 2
    else:
        top = int(rng.integers(height - sh + 1))
        left = int(rng.integers(width - sw + 1))
    stencil = np.zeros((height, width), dtype=bool)
    stencil[top:top + sh, left:left + sw] = local
    return stencil


def _partial_fragment(rng, stencil, mode):
    coordinates = np.argwhere(stencil)
    size = len(coordinates)
    fragment = stencil.copy()
    if mode == "single":
        fragment[:] = False
        point = coordinates[int(rng.integers(size))]
        fragment[tuple(point)] = True
    elif mode == "near":
        removed = rng.choice(size, size=int(rng.integers(1, min(4, size))), replace=False)
        for idx in removed:
            fragment[tuple(coordinates[int(idx)])] = False
    elif mode == "row":
        axis = int(rng.integers(2))
        positions = np.unique(coordinates[:, axis])
        location = int(positions[int(rng.integers(len(positions)))])
        if axis == 0:
            fragment[location, :] = False
        else:
            fragment[:, location] = False
    elif mode == "patch":
        point = coordinates[int(rng.integers(size))]
        radius = int(rng.integers(1, 3))
        r, c = map(int, point)
        fragment[max(0, r - radius):r + radius + 1,
                 max(0, c - radius):c + radius + 1] = False
    else:
        count = int(rng.integers(1, size))
        fragment[:] = False
        selected = rng.choice(size, size=count, replace=False)
        fragment[coordinates[selected, 0], coordinates[selected, 1]] = True
    count = int(fragment.sum())
    if count == 0 or count == size:
        fragment = stencil.copy()
        fragment[tuple(coordinates[int(rng.integers(size))])] = False
    return fragment


def _make_fragments(rng, stencil, count):
    order = [int(v) for v in rng.permutation(count)]
    complete_count = int(rng.integers(1, min(3, count - 1) + 1))
    remaining = count - complete_count
    if remaining == 1 or rng.random() < 0.35:
        empty_count = 0
    else:
        empty_count = int(rng.integers(1, remaining))
    fragments = [None] * count
    for index in order[:complete_count]:
        fragments[index] = stencil.copy()
    for index in order[complete_count:complete_count + empty_count]:
        fragments[index] = np.zeros_like(stencil)
    modes = ("single", "near", "row", "patch", "scattered")
    for index in order[complete_count + empty_count:]:
        mode = modes[int(rng.integers(len(modes)))]
        fragments[index] = _partial_fragment(rng, stencil, mode)
    return fragments


def _draw_pair(rng, geometry, colors, regime):
    nrows, ncols, ch, cw = geometry
    background, divider, accent = colors
    stencil = _sample_stencil(rng, ch, cw, regime)
    fragments = _make_fragments(rng, stencil, nrows * ncols)
    height, width = nrows * (ch + 1) - 1, ncols * (cw + 1) - 1
    source = np.full((height, width), background, dtype=np.int8)
    for row in range(ch, height, ch + 1):
        source[row, :] = divider
    for col in range(cw, width, cw + 1):
        source[:, col] = divider
    target = source.copy()
    for index, fragment in enumerate(fragments):
        top, left = divmod(index, ncols)
        top *= ch + 1
        left *= cw + 1
        source[top:top + ch, left:left + cw][fragment] = accent
        block = target[top:top + ch, left:left + cw]
        block[stencil] = divider
        block[fragment] = accent
    return {"input": source, "output": target}


def _pair_digest(pair):
    source = pair["input"].astype(np.uint8, copy=False)
    target = pair["output"].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(source.shape) + source.tobytes() + target.tobytes()).hexdigest()


def _candidate_from_input(grid, divider):
    height, width = grid.shape
    horizontal = [r for r in range(1, height - 1) if np.all(grid[r, :] == divider)]
    vertical = [c for c in range(1, width - 1) if np.all(grid[:, c] == divider)]
    if not horizontal or not vertical:
        return None
    ch, cw = horizontal[0], vertical[0]
    if ch < 3 or cw < 3 or horizontal != list(range(ch, height, ch + 1)) or vertical != list(range(cw, width, cw + 1)):
        return None
    nrows, ncols = len(horizontal) + 1, len(vertical) + 1
    if not (2 <= nrows <= 5 and 2 <= ncols <= 5):
        return None
    if height != nrows * (ch + 1) - 1 or width != ncols * (cw + 1) - 1:
        return None
    if len(np.unique(grid)) != 3:
        return None
    inside = np.ones(grid.shape, dtype=bool)
    inside[horizontal, :] = False
    inside[:, vertical] = False
    colors, counts = np.unique(grid[inside], return_counts=True)
    if len(colors) != 2 or int(colors[0]) == divider or int(colors[1]) == divider or counts[0] == counts[1]:
        return None
    background = int(colors[int(np.argmax(counts))])
    accent = int(colors[int(np.argmin(counts))])
    fragments = []
    for row in range(nrows):
        for col in range(ncols):
            block = grid[row * (ch + 1):row * (ch + 1) + ch,
                         col * (cw + 1):col * (cw + 1) + cw]
            if not np.all((block == background) | (block == accent)):
                return None
            fragments.append(block == accent)
    stencil = np.logical_or.reduce(fragments)
    size = int(stencil.sum())
    fragment_sizes = tuple(int(fragment.sum()) for fragment in fragments)
    if size < 4 or not any(x == size for x in fragment_sizes) or not any(0 < x < size for x in fragment_sizes):
        return None
    target = grid.copy()
    for index, fragment in enumerate(fragments):
        row, col = divmod(index, ncols)
        block = target[row * (ch + 1):row * (ch + 1) + ch,
                       col * (cw + 1):col * (cw + 1) + cw]
        block[stencil & ~fragment] = divider
    if np.array_equal(target, grid):
        return None
    return target, {
        "geometry": (nrows, ncols, ch, cw), "background": background,
        "divider": divider, "accent": accent, "stencil": stencil,
        "fragment_sizes": fragment_sizes, "fragments": tuple(fragments),
    }


def _reconstruct(grid):
    choices = []
    for color in np.unique(grid):
        result = _candidate_from_input(grid, int(color))
        if result is not None:
            choices.append(result)
    assert len(choices) == 1, "lattice or color roles are ambiguous"
    return choices[0]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The source convention is fixed: same registered stencil, divider-colored fill.
    # Every varying color and geometry is visible in its own input.
    puzzle = {"train": [], "test": []}
    used_geometry = set()
    used_inputs = set()
    for index in range(num_train + num_test):
        regime = "source_train" if index < num_train else "target_test"
        for attempt in range(64):
            geometry = _sample_geometry(rng)
            if geometry in used_geometry:
                continue
            colors = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
            pair = _draw_pair(rng, geometry, colors, regime)
            if _pair_digest(pair) in _SOURCE_PAIR_DIGESTS:
                continue
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature in used_inputs:
                continue
            try:
                predicted, info = _reconstruct(pair["input"])
            except AssertionError:
                continue
            if not np.array_equal(predicted, pair["output"]):
                continue
            used_geometry.add(info["geometry"])
            used_inputs.add(signature)
            puzzle["train" if index < num_train else "test"].append(pair)
            break
        else:
            raise RuntimeError(f"could not construct pair {index} for seed {seed}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
        signatures = set()
        pair_signatures = set()
        geometries = set()
        train_has_completion = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                source, target = pair["input"], pair["output"]
                for grid in (source, target):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert int(grid.min()) >= 0 and int(grid.max()) <= 9
                assert source.shape == target.shape
                signature = (source.shape, source.astype(np.uint8).tobytes())
                pair_signature = (signature, target.astype(np.uint8).tobytes())
                assert signature not in signatures and pair_signature not in pair_signatures
                signatures.add(signature)
                pair_signatures.add(pair_signature)
                assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
                predicted, info = _reconstruct(source)
                assert np.array_equal(predicted, target)
                assert set(np.unique(target)) <= set(np.unique(source))
                geometries.add(info["geometry"])
                if split == "train":
                    train_has_completion |= any(0 < n < int(info["stencil"].sum()) for n in info["fragment_sizes"])
        assert train_has_completion
        assert len(geometries) >= 2
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("invalid generated puzzle") from error


def _signature(puzzle):
    return tuple((split, pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
                 for split in ("train", "test") for pair in puzzle[split])


def _components(mask, selected):
    groups = []
    visited = set()
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if bool(mask[row, col]) != selected or (row, col) in visited:
                continue
            stack = [(row, col)]
            group = set()
            visited.add((row, col))
            while stack:
                r, c = stack.pop()
                group.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                            and (nr, nc) not in visited and bool(mask[nr, nc]) == selected):
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            groups.append(group)
    return groups


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(123)) == _signature(generate_puzzle(123))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (7, 3)):
        pair = generate_puzzle(1000 + ntrain * 10 + ntest, ntrain, ntest)
        assert (len(pair["train"]), len(pair["test"])) == (ntrain, ntest)
    for ntrain, ntest in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (ntrain or 0) * 10 + (ntest or 0), ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    geometries, backgrounds, stencils, normalized, palettes = set(), set(), set(), set(), set()
    row_counts, col_counts, heights, widths = set(), set(), set(), set()
    counts = {key: 0 for key in (
        "black", "corner", "black_corner", "multiple", "one_complete", "no_blank",
        "many_blanks", "singleton", "near", "row_gap", "patch_gap", "scattered_gap",
        "tiny", "large", "padded", "filled", "asymmetric", "separated", "hole",
        "bent_path", "irregular_region",
        "test_hollow_square", "train_filled", "mixed_backgrounds", "mixed_geometries",
        "mixed_palettes", "size_overlap", "size_increase", "combined",
    )}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = {split: [_reconstruct(pair["input"])[1] for pair in puzzle[split]]
                 for split in ("train", "test")}
        all_infos = infos["train"] + infos["test"]
        counts["mixed_backgrounds"] += len({v["background"] for v in all_infos}) > 1
        counts["mixed_geometries"] += len({v["geometry"] for v in all_infos}) > 1
        counts["mixed_palettes"] += len({(v["background"], v["divider"], v["accent"]) for v in all_infos}) > 1
        max_train = max(int(t["stencil"].sum()) for t in infos["train"])
        counts["size_overlap"] += any(int(v["stencil"].sum()) <= max_train for v in infos["test"])
        counts["size_increase"] += any(int(v["stencil"].sum()) > max_train for v in infos["test"])
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                geometry = info["geometry"]
                geometries.add(geometry)
                row_counts.add(geometry[0])
                col_counts.add(geometry[1])
                heights.add(geometry[2])
                widths.add(geometry[3])
                backgrounds.add(info["background"])
                palettes.add((info["background"], info["divider"], info["accent"]))
                stencil = info["stencil"]
                stencils.add((stencil.shape, stencil.tobytes()))
                points = np.argwhere(stencil)
                top, left = points.min(axis=0)
                bottom, right = points.max(axis=0)
                crop = stencil[top:bottom + 1, left:right + 1]
                normalized.add((crop.shape, crop.tobytes()))
                groups = _components(crop, True)
                holes = any(all(0 < r < crop.shape[0] - 1 and 0 < c < crop.shape[1] - 1
                                for r, c in group) for group in _components(crop, False))
                cells = set(map(tuple, np.argwhere(crop)))
                degrees = [sum((r + dr, c + dc) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                           for r, c in cells]
                counts["black"] += info["background"] == 0
                counts["corner"] += bool(stencil[0, 0])
                counts["black_corner"] += info["background"] == 0 and bool(stencil[0, 0])
                counts["padded"] += top > 0 and left > 0 and bottom < stencil.shape[0] - 1 and right < stencil.shape[1] - 1
                counts["filled"] += bool(np.all(crop))
                counts["asymmetric"] += not (np.array_equal(crop, crop[::-1, :]) or np.array_equal(crop, crop[:, ::-1]))
                counts["separated"] += len(groups) > 1
                counts["hole"] += holes
                counts["bent_path"] += len(groups) == 1 and crop.shape[0] >= 2 and crop.shape[1] >= 2 and degrees.count(1) == 2 and max(degrees) <= 2
                counts["irregular_region"] += len(groups) == 1 and crop.shape[0] >= 3 and crop.shape[1] >= 3 and len(cells) >= 8 and not np.all(crop) and not holes and max(degrees) == 4
                counts["test_hollow_square"] += split == "test" and crop.shape == (3, 3) and int(crop.sum()) == 8 and not crop[1, 1]
                counts["train_filled"] += split == "train" and bool(np.all(crop))
                sizes = info["fragment_sizes"]
                size = int(stencil.sum())
                counts["multiple"] += sizes.count(size) > 1
                counts["one_complete"] += sizes.count(size) == 1
                counts["no_blank"] += 0 not in sizes
                counts["many_blanks"] += sizes.count(0) >= max(2, len(sizes) // 3)
                counts["singleton"] += 1 in sizes
                counts["near"] += any(size - 2 <= n < size for n in sizes)
                counts["tiny"] += pair["input"].shape[0] <= 9 and pair["input"].shape[1] <= 9
                counts["large"] += pair["input"].shape[0] >= 25 and pair["input"].shape[1] >= 25
                counts["combined"] += sizes.count(size) > 1 and 0 not in sizes
                for fragment in info["fragments"]:
                    n = int(fragment.sum())
                    if not 0 < n < size:
                        continue
                    missing = stencil & ~fragment
                    counts["row_gap"] += any(np.any(stencil[r]) and np.all(missing[r, stencil[r]])
                                             for r in range(stencil.shape[0]))
                    missing_points = np.argwhere(missing)
                    span = missing_points.max(axis=0) - missing_points.min(axis=0) + 1
                    counts["patch_gap"] += len(missing_points) >= 2 and np.all(span <= 3)
                    counts["scattered_gap"] += len(missing_points) >= 3 and np.any(span > 3)
    assert counts["mixed_backgrounds"] == counts["mixed_geometries"] == counts["mixed_palettes"] == 42
    assert len(geometries) >= 100 and len(backgrounds) == 10 and len(palettes) >= 100
    assert len(stencils) >= 150 and len(normalized) >= 100
    assert row_counts == col_counts == {2, 3, 4, 5}
    assert min(heights) == min(widths) == 3 and max(heights) >= 10 and max(widths) >= 10
    assert all(counts[key] >= 10 for key in (
        "black", "corner", "multiple", "one_complete", "no_blank", "many_blanks",
        "singleton", "near", "tiny", "large", "padded", "filled", "asymmetric",
        "separated", "hole", "bent_path", "irregular_region",
        "train_filled", "combined", "size_overlap"))
    assert counts["black_corner"] >= 1 and counts["test_hollow_square"] >= 1
    assert counts["size_increase"] >= 1
    assert all(counts[key] >= 5 for key in ("row_gap", "patch_gap", "scattered_gap"))
