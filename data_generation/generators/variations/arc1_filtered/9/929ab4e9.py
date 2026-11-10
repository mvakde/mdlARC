"""Generate square D4-symmetry restoration puzzles."""

import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = {
    "15f245785ea1d76bdfa07e80ca6464fb91d3d0fe56f4c2cf0d44893345934b97",
    "a89fcf80779d14f208764d85d0bd6786c127155fb08a31c73da712f727e5ca75",
    "197e6fec478975814534f7e4236854cf36726c564e6f261278ebb0dc2abed051",
    "824b77c1d6442d5ef600f8b016e07ad38262b1f9030e8768bce792f339f3c17c",
    "8e050c95bcd34aef35579cf605f881f8988bb79b7071d5b861e09a84e0b940cb",
}


def _orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({
        (r, c), (c, n - 1 - r), (n - 1 - r, n - 1 - c),
        (n - 1 - c, r), (r, n - 1 - c), (n - 1 - r, c),
        (c, r), (n - 1 - c, n - 1 - r),
    }))


def _transforms(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r, c), (c, n - 1 - r), (n - 1 - r, n - 1 - c),
        (n - 1 - c, r), (r, n - 1 - c), (n - 1 - r, c),
        (c, r), (n - 1 - c, n - 1 - r),
    )


def _orbits(n: int):
    seen = set()
    for r in range(n):
        for c in range(n):
            if (r, c) not in seen:
                orbit = _orbit(n, r, c)
                seen.update(orbit)
                yield orbit


def _make_image(rng: np.random.Generator, n: int,
                palette: tuple[int, ...]) -> np.ndarray:
    """Color canonical orbit cells with several independent motif processes."""
    m = (n + 1) // 2
    k = len(palette)
    u, v = np.indices((m, m))
    style = int(rng.integers(0, 6))
    if style == 0:
        values = rng.integers(0, k, size=(m, m))
    elif style == 1:
        period = int(rng.integers(2, min(7, m + 1)))
        tile = rng.integers(0, k, size=(period, period))
        values = tile[u % period, v % period].copy()
    elif style == 2:
        block = int(rng.integers(2, min(6, m + 1)))
        coarse = rng.integers(0, k, size=((m + block - 1) // block,) * 2)
        values = coarse[u // block, v // block].copy()
    elif style == 3:
        a = int(rng.integers(1, 5))
        b = int(rng.integers(1, 5))
        values = ((u // a) + (v // b) + (u * v) % 3) % k
    elif style == 4:
        width = int(rng.integers(1, 4))
        values = ((u // width) + ((v - u) // (width + 1))) % k
    else:
        values = np.full((m, m), int(rng.integers(0, k)))
        density = float(rng.uniform(0.08, 0.38))
        marks = rng.random((m, m)) < density
        values[marks] = rng.integers(0, k, size=int(np.count_nonzero(marks)))

    for _ in range(int(rng.integers(0, 1 + max(2, m // 2)))):
        r0 = int(rng.integers(0, m))
        c0 = int(rng.integers(0, m))
        h = int(rng.integers(1, min(5, m - r0) + 1))
        w = int(rng.integers(1, min(5, m - c0) + 1))
        values[r0:r0 + h, c0:c0 + w] = int(rng.integers(0, k))

    image = np.empty((n, n), dtype=np.int8)
    for orbit in _orbits(n):
        r, c = orbit[0]
        a = min(r, n - 1 - r)
        b = min(c, n - 1 - c)
        color = palette[int(values[min(a, b), max(a, b)])]
        for rr, cc in orbit:
            image[rr, cc] = color
    return image


def _patch_shape(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    patch = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        shape = int(rng.integers(0, 4))
        if shape == 1:
            dh = int(rng.integers(1, max(2, h // 2 + 1)))
            dw = int(rng.integers(1, max(2, w // 2 + 1)))
            rs = slice(0, dh) if rng.random() < 0.5 else slice(h - dh, h)
            cs = slice(0, dw) if rng.random() < 0.5 else slice(w - dw, w)
            patch[rs, cs] = False
        elif shape == 2:
            if rng.random() < 0.5:
                patch[0, 1:-1] = False
            else:
                patch[1:-1, -1] = False
        elif shape == 3 and h >= 4 and w >= 4:
            patch[h // 2, :max(1, w // 3)] = False
    return patch


def _patch_origin(rng: np.random.Generator, n: int, h: int, w: int,
                  quadrant_only: bool, target: bool) -> tuple[int, int]:
    if quadrant_only:
        side = n // 2
        return (int(rng.integers(0, side - h + 1)),
                int(rng.integers(0, side - w + 1)))
    weights = (0.25, 0.17, 0.20, 0.18, 0.20) if target else (
        0.36, 0.20, 0.15, 0.16, 0.13)
    kind = int(rng.choice(5, p=weights))
    r0 = int(rng.integers(0, n - h + 1))
    c0 = int(rng.integers(0, n - w + 1))
    if kind == 1:
        edge = int(rng.integers(0, 4))
        if edge == 0:
            r0 = 0
        elif edge == 1:
            r0 = n - h
        elif edge == 2:
            c0 = 0
        else:
            c0 = n - w
    elif kind == 2:
        if rng.random() < 0.5:
            r0 = int(np.clip((n - h) // 2 + rng.integers(-1, 2), 0, n - h))
        else:
            c0 = int(np.clip((n - w) // 2 + rng.integers(-1, 2), 0, n - w))
    elif kind == 3:
        if rng.random() < 0.5:
            c0 = int(np.clip(r0 + rng.integers(-w + 1, h), 0, n - w))
        else:
            c0 = int(np.clip(n - r0 - h + rng.integers(-w + 1, h),
                             0, n - w))
    elif kind == 4:
        r0 = int(np.clip((n - h) // 2 + rng.integers(-n // 5, n // 5 + 1),
                         0, n - h))
        c0 = int(np.clip((n - w) // 2 + rng.integers(-n // 5, n // 5 + 1),
                         0, n - w))
    return r0, c0


def _global_copy_exists(mask: np.ndarray) -> bool:
    n = mask.shape[0]
    damaged = list(zip(*np.where(mask)))
    return any(all(not mask[_transforms(n, int(r), int(c))[t]]
                   for r, c in damaged) for t in range(1, 8))


def _axis_accessible(mask: np.ndarray) -> bool:
    n = mask.shape[0]
    return all(not (mask[r, n - 1 - c] and mask[n - 1 - r, c])
               for r, c in zip(*np.where(mask)))


def _add_hard_anchor(rng: np.random.Generator, mask: np.ndarray,
                     excluded: tuple[tuple[int, int], ...] | None = None
                     ) -> tuple[tuple[int, int], ...]:
    n = mask.shape[0]
    candidates = [orbit for orbit in _orbits(n)
                  if len(orbit) == 8 and orbit != excluded]
    orbit = candidates[int(rng.integers(0, len(candidates)))]
    count = int(rng.integers(4, 8))
    selected = None
    for _ in range(80):
        indexes = rng.choice(8, size=count, replace=False)
        trial = np.zeros_like(mask)
        for i in indexes:
            trial[orbit[int(i)]] = True
        if not _global_copy_exists(trial) and not _axis_accessible(trial):
            selected = set(int(i) for i in indexes)
            break
    if selected is None:
        selected = set(range(8)) - {int(rng.integers(0, 8))}
    keeper = next(i for i in range(8) if i not in selected)
    for i in selected:
        mask[orbit[i]] = True
        if rng.random() < 0.55:
            r, c = orbit[i]
            r0 = max(0, min(n - 2, r + int(rng.integers(-1, 1))))
            c0 = max(0, min(n - 2, c + int(rng.integers(-1, 1))))
            mask[r0:r0 + 2, c0:c0 + 2] = True
    mask[orbit[keeper]] = False
    return orbit


def _make_mask(rng: np.random.Generator, n: int,
               mode: str, target: bool) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    quadrant_only = mode == "axis"
    patch_count = int(rng.integers(1, 7))
    fraction = float(rng.uniform(0.015, 0.30))
    if rng.random() < 0.22:
        fraction = float(rng.uniform(0.012, 0.055))
    desired = max(3, int(round(fraction * n * n)))
    area = max(2.0, desired / patch_count)
    limit = n // 2 if quadrant_only else min(n - 1, 15)
    for _ in range(patch_count):
        ratio = float(np.exp(rng.uniform(-1.0, 1.0)))
        h = int(np.clip(round(np.sqrt(area * ratio)), 1, limit))
        w = int(np.clip(round(np.sqrt(area / ratio)), 1, limit))
        r0, c0 = _patch_origin(rng, n, h, w, quadrant_only, target)
        patch = _patch_shape(rng, h, w)
        mask[r0:r0 + h, c0:c0 + w] |= patch
    if mode == "hard":
        first_orbit = _add_hard_anchor(rng, mask)
        if rng.random() < 0.55:
            _add_hard_anchor(rng, mask, first_orbit)
    for orbit in _orbits(n):
        if all(mask[cell] for cell in orbit):
            mask[orbit[int(rng.integers(0, len(orbit)))]] = False
    if not np.any(mask):
        mask[0, 0] = True
    return mask


def _restore(input_grid: np.ndarray, erasure: int) -> np.ndarray:
    n = input_grid.shape[0]
    restored = input_grid.copy()
    for orbit in _orbits(n):
        evidence = {int(input_grid[cell]) for cell in orbit
                    if input_grid[cell] != erasure}
        if not evidence or len(evidence) != 1:
            raise AssertionError("Missing or conflicting symmetry evidence")
        value = evidence.pop()
        for cell in orbit:
            if input_grid[cell] == erasure:
                restored[cell] = value
    return restored


def _infer_erasure(train: list[dict]) -> int:
    colors = set()
    for pair in train:
        changed = pair["input"] != pair["output"]
        colors.update(int(x) for x in np.unique(pair["input"][changed]))
    assert len(colors) == 1
    return colors.pop()


def _pair(rng: np.random.Generator, erasure: int,
          available: tuple[int, ...], mode: str, target: bool) -> dict:
    for _ in range(12):
        n = int(rng.integers(12, 31))
        if len(available) >= 6 and rng.random() < 0.72:
            k = int(rng.integers(6, len(available) + 1))
            min_colors = 5
        else:
            k = int(rng.integers(2, min(5, len(available)) + 1))
            min_colors = 2
        palette = tuple(int(x) for x in rng.choice(available, size=k,
                                                  replace=False))
        output_grid = _make_image(rng, n, palette)
        if len(np.unique(output_grid)) >= min_colors:
            break
    else:
        raise RuntimeError("Could not construct a multicolor image")
    mask = _make_mask(rng, n, mode, target)
    input_grid = output_grid.copy()
    input_grid[mask] = erasure
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None and (type(num_train) is not int or
                                  not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 through 10")
    if num_test is not None and (type(num_test) is not int or
                                 not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    erasure = int(rng.integers(0, 10))
    all_colors = [c for c in range(10) if c != erasure]
    pool_size = (int(rng.integers(7, 10)) if rng.random() < 0.75
                 else int(rng.integers(5, 7)))
    pool = tuple(int(c) for c in rng.choice(all_colors, size=pool_size,
                                            replace=False))
    modes = ["axis", "hard"] + [
        ("axis", "general", "hard")[int(rng.choice(3, p=(0.22, 0.42, 0.36)))]
        for _ in range(num_train - 2)
    ]
    rng.shuffle(modes)

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for mode in modes:
        for _ in range(12):
            pair = _pair(rng, erasure, pool, mode, False)
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in used_inputs:
                used_inputs.add(signature)
                puzzle["train"].append(pair)
                break
        else:
            raise RuntimeError("Could not generate a distinct training pair")

    seen_colors = tuple(sorted({int(x) for pair in puzzle["train"]
                                for x in np.unique(pair["output"])}))
    for _ in range(num_test):
        mode = ("axis", "general", "hard")[int(
            rng.choice(3, p=(0.16, 0.36, 0.48)))]
        for _ in range(12):
            pair = _pair(rng, erasure, seen_colors, mode, True)
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in used_inputs:
                used_inputs.add(signature)
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("Could not generate a distinct test pair")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs = set()
    used_pairs = set()
    train_colors = set()
    erasure = None
    axis_train = False
    hard_train = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and inp.shape[0] == inp.shape[1]
            n = inp.shape[0]
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.tobytes())
            assert input_key not in used_inputs and pair_key not in used_pairs
            used_inputs.add(input_key)
            used_pairs.add(pair_key)
            if n == 24:
                digest = hashlib.sha256(np.asarray(inp, dtype=np.uint8).tobytes()).hexdigest()
                assert digest not in _SOURCE_INPUT_HASHES

            changed = inp != out
            assert np.any(changed)
            erased = np.unique(inp[changed])
            assert len(erased) == 1
            color = int(erased[0])
            assert not np.any(out == color)
            assert np.array_equal(changed, inp == color)
            if erasure is None:
                erasure = color
            else:
                assert erasure == color
            assert len(np.unique(out)) >= 2
            for orbit in _orbits(n):
                values = {int(out[cell]) for cell in orbit}
                assert len(values) == 1
                assert any(not changed[cell] for cell in orbit)
            assert np.array_equal(_restore(inp, color), out)
            if split == "train":
                train_colors.update(int(x) for x in np.unique(out))
                axis_train |= _axis_accessible(changed)
                hard_train |= not _global_copy_exists(changed)
            else:
                assert set(int(x) for x in np.unique(out)) <= train_colors

    assert axis_train and hard_train
    assert _infer_erasure(puzzle["train"]) == erasure
    for pair in puzzle["test"]:
        assert np.array_equal(_restore(pair["input"], erasure), pair["output"])


def _fingerprint(puzzle: dict) -> tuple:
    return tuple(tuple((pair["input"].shape, pair["input"].tobytes(),
                        pair["output"].tobytes()) for pair in puzzle[split])
                 for split in ("train", "test"))


def _component_groups(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    unseen = set(zip(*np.where(mask)))
    groups = []
    while unseen:
        group = {unseen.pop()}
        stack = list(group)
        while stack:
            r, c = stack.pop()
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in unseen:
                    unseen.remove(cell)
                    group.add(cell)
                    stack.append(cell)
        groups.append(group)
    return groups


def self_test() -> None:
    train_counts, test_counts, sizes, erasures = set(), set(), set(), set()
    palette_sizes, mask_sizes, components = set(), [], set()
    image_keys, mask_keys, puzzle_keys = set(), set(), set()
    mixed_sizes = mixed_palettes = mixed_areas = mixed_components = 0
    test_smaller = test_larger = test_comparable = 0
    border = axis = diagonal = one_visible = 0
    test_global = test_non_global = 0
    train_small = train_broad = few_colors = many_colors = 0
    unequal_test = notched = axis_cross = diagonal_cross = 0
    border_hard = axis_notched = odd_one_visible = size24_hard = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        puzzle_keys.add(_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        erasures.add(_infer_erasure(puzzle["train"]))
        pairs = puzzle["train"] + puzzle["test"]
        mixed_sizes += len({p["input"].shape for p in pairs}) > 1
        mixed_palettes += len({tuple(np.unique(p["output"])) for p in pairs}) > 1
        mixed_areas += len({int(np.count_nonzero(p["input"] != p["output"]))
                            for p in pairs}) > 1
        mixed_components += len({len(_component_groups(p["input"] != p["output"]))
                                 for p in pairs}) > 1
        train_areas = [int(np.count_nonzero(p["input"] != p["output"]))
                       for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                mask = inp != out
                n = inp.shape[0]
                sizes.add(n)
                palette_sizes.add(len(np.unique(out)))
                area = int(np.count_nonzero(mask))
                mask_sizes.append(area)
                groups = _component_groups(mask)
                components.add(len(groups))
                image_keys.add((out.shape, out.tobytes()))
                mask_keys.add((mask.shape, mask.tobytes()))
                touches_border = bool(np.any(mask[0]) or np.any(mask[-1]) or
                                      np.any(mask[:, 0]) or np.any(mask[:, -1]))
                touches_axis = bool(np.any(mask[n // 2]) or
                                    np.any(mask[:, n // 2]))
                touches_diagonal = bool(np.any(np.diag(mask)) or
                                        np.any(np.diag(np.fliplr(mask))))
                has_one_visible = any(
                    sum(not mask[cell] for cell in orbit) == 1
                    and any(mask[cell] for cell in orbit)
                    for orbit in _orbits(n))
                has_notch = any(
                    len(group) >= 4 and
                    (max(r for r, _ in group) - min(r for r, _ in group) + 1) *
                    (max(c for _, c in group) - min(c for _, c in group) + 1)
                    > len(group) for group in groups)
                crosses_axis = any(
                    min(r for r, _ in group) < n / 2 <=
                    max(r for r, _ in group) or
                    min(c for _, c in group) < n / 2 <=
                    max(c for _, c in group) for group in groups)
                crosses_diagonal = any(
                    min(r - c for r, c in group) < 0 <
                    max(r - c for r, c in group) or
                    min(r + c - n + 1 for r, c in group) < 0 <
                    max(r + c - n + 1 for r, c in group)
                    for group in groups)
                border += touches_border
                axis += touches_axis
                diagonal += touches_diagonal
                one_visible += has_one_visible
                notched += has_notch
                axis_cross += crosses_axis
                diagonal_cross += crosses_diagonal
                border_hard += touches_border and not _global_copy_exists(mask)
                axis_notched += crosses_axis and has_notch
                odd_one_visible += n % 2 == 1 and has_one_visible
                size24_hard += n == 24 and not _global_copy_exists(mask)
                few_colors += len(np.unique(out)) <= 3
                many_colors += len(np.unique(out)) >= 7
                if split == "train":
                    train_small += area <= 12
                    train_broad += area >= 96
                if split == "test":
                    unequal_test += (len(groups) == 2 and
                                     len(groups[0]) != len(groups[1]))
                    test_global += _global_copy_exists(mask)
                    test_non_global += not _global_copy_exists(mask)
                    test_smaller += area < min(train_areas)
                    test_larger += area > max(train_areas)
                    test_comparable += min(train_areas) <= area <= max(train_areas)
                    assert np.array_equal(
                        _restore(inp, _infer_erasure(puzzle["train"])), out)

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 12 and 24 in sizes and any(n % 2 for n in sizes)
    assert len(erasures) >= 6 and len(palette_sizes) >= 5
    assert min(mask_sizes) <= 12 and max(mask_sizes) >= 75
    assert len(components) >= 4 and max(components) >= 4
    assert mixed_sizes >= 25 and mixed_palettes >= 25
    assert mixed_areas >= 25 and mixed_components >= 25
    assert min(test_smaller, test_larger, test_comparable) >= 3
    assert min(border, axis, diagonal, one_visible) >= 10
    assert min(train_small, train_broad, few_colors, many_colors) >= 5
    assert min(unequal_test, notched, axis_cross, diagonal_cross) >= 3
    assert min(border_hard, axis_notched, odd_one_visible, size24_hard) >= 2
    assert test_global >= 3 and test_non_global >= 3
    assert len(image_keys) >= 100 and len(mask_keys) >= 100
    assert len(puzzle_keys) == 42

    cases = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
             (2, None), (10, None)]
    for i, (requested_train, requested_test) in enumerate(cases):
        puzzle = generate_puzzle(1000 + i, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(
            generate_puzzle(1000 + i, requested_train, requested_test))
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (True, 1), (2, False)):
        try:
            generate_puzzle(17, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts must raise ValueError")
