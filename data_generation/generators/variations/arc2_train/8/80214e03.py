import hashlib

import numpy as np


_TRAIN_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
_TEST_SHAPES = _TRAIN_SHAPES + ((3, 4), (4, 2), (4, 3), (4, 4))
_SOURCE_INPUTS = {
    ((20, 14), "4e4ebacf5cb9ddda7b447125461a66206c8694edbdc4a267c344226f1735a4c6"),
    ((22, 11), "bf25f809e2e810774bfff59dd9f1069aa141beca177cd802164625647d99f538"),
    ((18, 19), "b9003d4ad5e3c80551465fa78bf5121dab77cfb00c545eabef964f3ee9df89ea"),
    ((22, 19), "074a1dfb5a2f6397cc9d30e604b362549d92e8ae93859e9738ebe6ab0cd7fe3c"),
}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        low = max(2, test + 1)
        train = low + (int(seed) // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _fusion(grid):
    return (bool(np.any(grid[:, :-1] == grid[:, 1:])),
            bool(np.any(grid[:-1, :] == grid[1:, :])))


def _boundary_evidence(grid):
    return bool(np.all(np.any(grid[:-1, :] != grid[1:, :], axis=1))
                and np.all(np.any(grid[:, :-1] != grid[:, 1:], axis=0)))


def _construct_latent(shape, labels, mode, rng):
    h, w = shape
    k = len(labels)
    base = np.fromfunction(lambda y, x: (y + x) % 2, (h, w), dtype=int).astype(int)
    protected = set()
    if mode == 1:  # horizontal fusion only
        if k == 2:
            row = np.array([0, 0] + [1 if x % 2 == 0 else 0 for x in range(2, w)])
            base = (row[None, :] + np.arange(h)[:, None]) % 2
        else:
            base[0, :2] = 2
            protected.update(((0, 0), (0, 1)))
    elif mode == 2:  # vertical fusion only
        if k == 2:
            column = np.array([0, 0] + [1 if y % 2 == 0 else 0 for y in range(2, h)])
            base = (column[:, None] + np.arange(w)[None, :]) % 2
        else:
            base[:2, 0] = 2
            protected.update(((0, 0), (1, 0)))
    elif mode == 3:
        base[0, 0] = base[0, 1] = base[1, 0] = 0
        base[1, 1] = 1
        protected.update(((0, 0), (0, 1), (1, 0), (1, 1)))
    first_extra = 3 if mode in (1, 2) and k >= 3 else 2
    if mode == 3:
        first_extra = 2
    if k > first_extra:
        candidates = [(y, x) for y in range(h) for x in range(w)
                      if (y, x) not in protected]
        rng.shuffle(candidates)
        keep = set()
        for value in range(first_extra):
            possible = [cell for cell in candidates if base[cell] == value]
            if possible:
                keep.add(possible[0])
        candidates = [cell for cell in candidates if cell not in keep]
        for value, cell in zip(range(first_extra, k), candidates):
            base[cell] = value
    grid = np.asarray(labels, dtype=np.int8)[base]
    if bool(rng.integers(0, 2)):
        grid = grid[::-1, :]
    if bool(rng.integers(0, 2)):
        grid = grid[:, ::-1]
    return grid.copy()


def _latent(rng, shape, background, regime, demonstrator=False, allowed_modes=None, forced_mode=None, excluded_k=None):
    n = shape[0] * shape[1]
    upper = min(n, 6 if regime == "source_train" else 9)
    if forced_mode == 3:
        upper = min(upper, n - 2)
    if demonstrator:
        k = upper
    else:
        choices = [value for value in range(2, upper + 1) if value != excluded_k]
        k = int(rng.choice(choices))
    available = np.array([c for c in range(10) if c != background], dtype=np.int8)
    rng.shuffle(available)
    if background != 0 and regime == "source_train" and demonstrator:
        labels = np.concatenate((np.array([0], dtype=np.int8), available[available != 0][:k-1]))
    else:
        labels = available[:k]
    rng.shuffle(labels)
    if demonstrator:
        mode = 0
    elif forced_mode is not None:
        mode = forced_mode
    else:
        modes = [0]
        if 3 <= k <= n - 1:
            modes.append(1)
        if 3 <= k <= n - 1:
            modes.append(2)
        if k <= n - 2:
            modes.append(3)
        if allowed_modes is not None:
            modes = [value for value in modes if value in allowed_modes]
        mode = int(rng.choice(modes))
    for _ in range(100):
        flat = np.concatenate((np.arange(k), rng.integers(0, k, size=n-k)))
        rng.shuffle(flat)
        candidate = labels[flat.reshape(shape)]
        if _boundary_evidence(candidate) and _fusion(candidate) == (bool(mode & 1), bool(mode & 2)):
            if not demonstrator or not np.array_equal(candidate, candidate[:, ::-1]):
                return candidate.astype(np.int8, copy=True)
    candidate = _construct_latent(shape, labels, mode, rng)
    assert _boundary_evidence(candidate)
    assert _fusion(candidate) == (bool(mode & 1), bool(mode & 2))
    assert len(np.unique(candidate)) == k
    return candidate


def _bands(rng, count):
    weights = np.array([0.18, 0.23, 0.23, 0.20, 0.10, 0.06])
    if bool(rng.random() < 0.22):
        choices = [v for v in range(3, 9) if count * v <= 28]
        value = int(rng.choice(choices))
        return np.full(count, value, dtype=int)
    for _ in range(50):
        bands = rng.choice(np.arange(3, 9), size=count, p=weights)
        if int(np.sum(bands)) <= 28:
            return bands.astype(int)
    return np.full(count, 3, dtype=int)


def _padding(rng, patch_h, patch_w):
    def axis(total):
        values = [(a, b) for a in range(1, 9) for b in range(1, 9)
                  if total + a + b <= 30]
        return values[int(rng.integers(0, len(values)))]
    top, bottom = axis(patch_h)
    left, right = axis(patch_w)
    return top, bottom, left, right


def _damage_tile(rng, tile, protected, background, severity):
    h, w = tile.shape
    area = h * w
    if severity == 0:
        fraction = float(rng.uniform(0.0, 0.07))
    elif severity == 1:
        fraction = float(rng.uniform(0.08, 0.17))
    else:
        fraction = float(rng.uniform(0.19, 0.30))
    target = min(int(np.rint(area * fraction)), int(np.floor(0.30 * area)))
    available = {(y, x) for y in range(h) for x in range(w)
                 if (y, x) not in protected}
    target = min(target, len(available))
    chosen = set()
    style = int(rng.integers(0, 3))
    if target >= 2 and style in (1, 2):
        paths = []
        for y in range(h):
            for x in range(w):
                if style == 2 and y not in (0, h-1) and x not in (0, w-1):
                    continue
                for dy, dx in ((0, 1), (1, 0)):
                    length = min(target, int(rng.integers(2, min(6, target) + 1)))
                    path = [(y + t*dy, x + t*dx) for t in range(length)]
                    if all(cell in available for cell in path):
                        paths.append(path)
        if paths:
            chosen.update(paths[int(rng.integers(0, len(paths)))])
    remainder = list(available - chosen)
    rng.shuffle(remainder)
    chosen.update(remainder[:target-len(chosen)])
    for y, x in chosen:
        tile[y, x] = background


def _expand(rng, latent, background, row_sizes, col_sizes, severity):
    row_bounds = np.r_[0, np.cumsum(row_sizes)].astype(int)
    col_bounds = np.r_[0, np.cumsum(col_sizes)].astype(int)
    patch_h, patch_w = int(row_bounds[-1]), int(col_bounds[-1])
    patch = np.empty((patch_h, patch_w), dtype=np.int8)
    for i in range(latent.shape[0]):
        for j in range(latent.shape[1]):
            patch[row_bounds[i]:row_bounds[i+1], col_bounds[j]:col_bounds[j+1]] = latent[i, j]
    protected = set()
    # Preserve the bounding box and a foreground witness on every physical line.
    for y in range(patch_h):
        protected.add((y, int(rng.integers(0, patch_w))))
    for x in range(patch_w):
        protected.add((int(rng.integers(0, patch_h)), x))
    for i in range(latent.shape[0]):
        for j in range(latent.shape[1]):
            protected.add((int(row_bounds[i]), int(col_bounds[j])))
    for i in range(1, latent.shape[0]):
        js = np.flatnonzero(latent[i-1] != latent[i])
        j = int(rng.choice(js))
        x = int(rng.integers(col_bounds[j], col_bounds[j+1]))
        protected.update(((int(row_bounds[i]-1), x), (int(row_bounds[i]), x)))
    for j in range(1, latent.shape[1]):
        rows = np.flatnonzero(latent[:, j-1] != latent[:, j])
        i = int(rng.choice(rows))
        y = int(rng.integers(row_bounds[i], row_bounds[i+1]))
        protected.update(((y, int(col_bounds[j]-1)), (y, int(col_bounds[j]))))
    for i in range(latent.shape[0]):
        for j in range(latent.shape[1]):
            y0, y1 = int(row_bounds[i]), int(row_bounds[i+1])
            x0, x1 = int(col_bounds[j]), int(col_bounds[j+1])
            local = {(y-y0, x-x0) for y, x in protected if y0 <= y < y1 and x0 <= x < x1}
            local_severity = severity if rng.random() < 0.75 else int(rng.integers(0, 3))
            _damage_tile(rng, patch[y0:y1, x0:x1], local, background, local_severity)
    top, bottom, left, right = _padding(rng, patch_h, patch_w)
    canvas = np.full((patch_h+top+bottom, patch_w+left+right), background, dtype=np.int8)
    canvas[top:top+patch_h, left:left+patch_w] = patch
    return canvas


def _recover(input_grid):
    background = int(input_grid[0, 0])
    points = np.argwhere(input_grid != background)
    assert points.size
    y0, x0 = np.min(points, axis=0).astype(int)
    y1, x1 = np.max(points, axis=0).astype(int)
    assert 1 <= y0 <= y1 < input_grid.shape[0] - 1
    assert 1 <= x0 <= x1 < input_grid.shape[1] - 1
    patch = input_grid[y0:y1+1, x0:x1+1]
    assert np.all(np.any(patch != background, axis=1))
    assert np.all(np.any(patch != background, axis=0))
    row_change = (patch[:-1] != background) & (patch[1:] != background) & (patch[:-1] != patch[1:])
    col_change = (patch[:, :-1] != background) & (patch[:, 1:] != background) & (patch[:, :-1] != patch[:, 1:])
    rows = np.r_[0, np.flatnonzero(np.any(row_change, axis=1)) + 1, patch.shape[0]]
    cols = np.r_[0, np.flatnonzero(np.any(col_change, axis=0)) + 1, patch.shape[1]]
    assert 2 <= len(rows)-1 <= 4 and 2 <= len(cols)-1 <= 4
    latent = np.empty((len(rows)-1, len(cols)-1), dtype=np.int8)
    holes = []
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
            tile = patch[rows[i]:rows[i+1], cols[j]:cols[j+1]]
            labels = np.unique(tile[tile != background])
            assert labels.size == 1
            latent[i, j] = labels[0]
            holes.append((int(np.count_nonzero(tile == background)), int(tile.size)))
    assert _boundary_evidence(latent)
    return {
        "background": background, "latent": latent, "row_sizes": tuple(np.diff(rows).tolist()),
        "col_sizes": tuple(np.diff(cols).tolist()), "holes": holes,
        "padding": (int(y0), int(input_grid.shape[0]-y1-1), int(x0), int(input_grid.shape[1]-x1-1)),
        "patch": patch,
    }


def _variants(grid):
    return [grid, grid[:, ::-1], grid[::-1, :], grid[::-1, ::-1],
            grid.T, grid.T[:, ::-1], grid.T[::-1, :], grid.T[::-1, ::-1]]


def _predict_from_training(train, test_input):
    options = set(range(8))
    for pair in train:
        latent = _recover(pair["input"])["latent"]
        options = {i for i in options if np.array_equal(_variants(latent)[i], pair["output"])}
    assert len(options) == 1
    return _variants(_recover(test_input)["latent"])[options.pop()].copy()


def _grid_key(grid):
    return (tuple(grid.shape), grid.astype(np.int8).tobytes())


def _source_copy(grid):
    key = (tuple(grid.shape), hashlib.sha256(grid.astype(np.int8).tobytes()).hexdigest())
    return key in _SOURCE_INPUTS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # The only convention shared by pairs is the visible local-background,
    # compact-then-reflect rule. Background and label assignments are local.
    demo_index = int(rng.integers(0, train_count))
    demo_shape = _TRAIN_SHAPES[int(rng.integers(0, 3))]
    source_like_progression = bool(rng.random() < 0.24)
    train_shapes = []
    for index in range(train_count):
        if index == demo_index:
            train_shapes.append(demo_shape)
        else:
            options = ([shape for shape in _TRAIN_SHAPES if shape != demo_shape]
                       if index == (demo_index + 1) % train_count else _TRAIN_SHAPES)
            train_shapes.append(options[int(rng.integers(0, len(options)))])
    # The two training backgrounds make the local role of 0 explicit.
    other_backgrounds = [int(x) for x in rng.permutation(np.arange(1, 10))]
    background_train = [0] * train_count
    background_train[demo_index] = other_backgrounds[0]
    zero_bg_index = (demo_index + 1) % train_count
    for index in range(train_count):
        if index not in (demo_index, zero_bg_index):
            background_train[index] = 0 if rng.random() < 0.55 else other_backgrounds[(index+1) % 9]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(25):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                if split == "train":
                    shape = train_shapes[index]
                    background = background_train[index]
                    demo = index == demo_index
                    severity = int(pair_rng.integers(0, 3))
                else:
                    if source_like_progression and index == 0:
                        shape = ((3, 4), (4, 2), (4, 3), (4, 4))[int(pair_rng.integers(0, 4))]
                    else:
                        shape = _TEST_SHAPES[int(pair_rng.choice(len(_TEST_SHAPES), p=[.04,.05,.05,.11,.18,.18,.24,.15]))]
                    background = 0 if pair_rng.random() < 0.55 else int(pair_rng.integers(1, 10))
                    demo = False
                    severity = int(pair_rng.choice(3, p=[.18,.36,.46]))
                latent = _latent(
                    pair_rng, shape, background,
                    "source_train" if split == "train" else "target_test", demo,
                    allowed_modes=(0, 1) if source_like_progression and split == "train" else None,
                    forced_mode=3 if source_like_progression and split == "test" and index == 0 else None,
                    excluded_k=demo_shape[0] * demo_shape[1]
                    if split == "train" and index == (demo_index + 1) % train_count else None,
                )
                row_sizes = _bands(pair_rng, shape[0])
                col_sizes = _bands(pair_rng, shape[1])
                input_grid = _expand(pair_rng, latent, background, row_sizes, col_sizes, severity)
                if _grid_key(input_grid) not in seen and not _source_copy(input_grid):
                    seen.add(_grid_key(input_grid))
                    puzzle[split].append({"input": input_grid, "output": latent[:, ::-1].copy()})
                    break
            else:
                raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_info = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9
            input_grid, output_grid = pair["input"], pair["output"]
            assert not np.array_equal(input_grid, output_grid)
            key = _grid_key(input_grid)
            pair_key = (key, _grid_key(output_grid))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _source_copy(input_grid)
            info = _recover(input_grid)
            latent = info["latent"]
            assert output_grid.shape == latent.shape
            assert np.array_equal(output_grid, latent[:, ::-1])
            assert info["background"] not in np.unique(latent)
            assert all(3 <= n <= 8 for n in info["row_sizes"] + info["col_sizes"])
            assert all(loss <= int(np.floor(0.30*area)) for loss, area in info["holes"])
            assert set(np.unique(input_grid)) == set(np.unique(latent)) | {info["background"]}
            if split == "train":
                assert tuple(latent.shape) in _TRAIN_SHAPES
                assert len(np.unique(latent)) <= 6
                train_info.append(info)
            else:
                assert tuple(latent.shape) in _TEST_SHAPES
    assert any(info["background"] == 0 for info in train_info)
    assert any(info["background"] != 0 and 0 in info["latent"] for info in train_info)
    assert len({tuple(info["latent"].shape) for info in train_info}) >= 2
    assert any(len(np.unique(info["latent"])) == info["latent"].size for info in train_info)
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes = {s: set() for s in ("train", "test")}
    fusion_modes = set()
    backgrounds = set()
    palette_sizes = set()
    canvas_shapes = set()
    band_sizes = set()
    padding = set()
    loss_fractions = set()
    within_background = 0
    within_shapes = 0
    within_palette = 0
    within_fusion = 0
    even = odd = zero_foreground = connected_holes = 0
    uniform_rows = uniform_cols = wide_padding = long_bands = 0
    clean_tiles = heavy_tiles = connected_tiles = edge_tiles = 0
    symmetric = separated_repeats = same_size_tests = new_size_tests = 0
    source_like = combined = 0
    keys = set()
    total_pairs = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        backgrounds_here = set()
        shapes_here = set()
        palettes_here = set()
        fusions_here = set()
        train_infos = [_recover(pair["input"]) for pair in puzzle["train"]]
        test_infos = [_recover(pair["input"]) for pair in puzzle["test"]]
        source_like += int(
            all(not _fusion(info["latent"])[1] for info in train_infos)
            and any(_fusion(info["latent"]) == (True, True) for info in test_infos)
        )
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                inp, out = pair["input"], pair["output"]
                info = _recover(inp)
                latent = info["latent"]
                assert np.array_equal(out, latent[:, ::-1])
                assert not _source_copy(inp)
                shapes[split].add(tuple(latent.shape))
                shapes_here.add(tuple(latent.shape))
                mode = _fusion(latent)
                fusion_modes.add(mode)
                fusions_here.add(mode)
                backgrounds.add(info["background"])
                backgrounds_here.add(info["background"])
                colors = len(np.unique(latent))
                palette_sizes.add(colors)
                palettes_here.add(colors)
                canvas_shapes.add(tuple(inp.shape))
                band_sizes.update(info["row_sizes"] + info["col_sizes"])
                padding.update(info["padding"])
                loss_fractions.update(round(loss/area, 2) for loss, area in info["holes"])
                uniform_rows += int(len(set(info["row_sizes"])) == 1)
                uniform_cols += int(len(set(info["col_sizes"])) == 1)
                wide_padding += int(max(info["padding"]) >= 6)
                long_bands += int(max(info["row_sizes"] + info["col_sizes"]) >= 7)
                symmetric += int(np.array_equal(latent, latent[:, ::-1]))
                separated_repeats += int(any(
                    np.count_nonzero(latent == c) > 1
                    and not np.any((latent == c)[:, :-1] & (latent == c)[:, 1:])
                    and not np.any((latent == c)[:-1] & (latent == c)[1:])
                    for c in np.unique(latent)
                ))
                if split == "test":
                    same_size_tests += int(tuple(latent.shape) in _TRAIN_SHAPES)
                    new_size_tests += int(tuple(latent.shape) not in _TRAIN_SHAPES)
                zero_foreground += int(0 in latent)
                even += int(inp.shape[0] % 2 == 0)
                odd += int(inp.shape[0] % 2 == 1)
                combined += int(info["background"] != 0 and 0 in latent and mode[1]
                                and inp.shape[0] % 2 == 0)
                holes = info["patch"] == info["background"]
                connected_holes += int(np.any(holes[:-1] & holes[1:]) or np.any(holes[:, :-1] & holes[:, 1:]))
                row_bounds = np.r_[0, np.cumsum(info["row_sizes"])]
                col_bounds = np.r_[0, np.cumsum(info["col_sizes"])]
                for i in range(latent.shape[0]):
                    for j in range(latent.shape[1]):
                        tile = holes[row_bounds[i]:row_bounds[i+1], col_bounds[j]:col_bounds[j+1]]
                        lost = int(np.count_nonzero(tile))
                        clean_tiles += int(lost == 0)
                        heavy_tiles += int(lost / tile.size > .18)
                        connected_tiles += int(np.any(tile[:-1] & tile[1:]) or np.any(tile[:, :-1] & tile[:, 1:]))
                        edge_tiles += int(lost > 0 and (np.any(tile[0]) or np.any(tile[-1])
                                                       or np.any(tile[:, 0]) or np.any(tile[:, -1])))
                key = _grid_key(inp)
                assert key not in keys
                keys.add(key)
        within_background += int(len(backgrounds_here) > 1)
        within_shapes += int(len(shapes_here) > 1)
        within_palette += int(len(palettes_here) > 1)
        within_fusion += int(len(fusions_here) > 1)
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes["train"]) == 4 and len(shapes["test"]) >= 7
    assert len(fusion_modes) == 4
    assert len(backgrounds) >= 7 and len(palette_sizes) >= 6
    assert len(canvas_shapes) >= 40 and band_sizes == set(range(3, 9))
    assert max(padding) >= 7 and len(loss_fractions) >= 20
    assert within_background == 42 and within_shapes == 42
    assert within_palette == 42 and within_fusion >= 30
    assert even >= 20 and odd >= 20 and zero_foreground >= 42
    assert connected_holes >= 50 and len(keys) == total_pairs
    assert uniform_rows >= 10 and uniform_cols >= 10
    assert wide_padding >= 30 and long_bands >= 20
    assert clean_tiles >= 30 and heavy_tiles >= 50
    assert connected_tiles >= 50 and edge_tiles >= 50
    assert symmetric >= 2 and separated_repeats >= 20
    assert same_size_tests >= 3 and new_size_tests >= 20
    assert source_like >= 3 and combined >= 2
    a = generate_puzzle(314159, 6, 2)
    assert _same_puzzle(a, generate_puzzle(314159, 6, 2))
    assert not _same_puzzle(a, generate_puzzle(314160, 6, 2))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000+train+test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000+(train or 0)+(test or 0), train, test)
        assert (train is None or len(p["train"]) == train)
        assert (test is None or len(p["test"]) == test)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")
