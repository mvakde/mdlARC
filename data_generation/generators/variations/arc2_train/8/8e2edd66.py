"""Self-referential two-color ARC block-substitution puzzles."""

from itertools import combinations, islice

import numpy as np


_SIZES = ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (5, 5))
_OTHER_SIZES = tuple(size for size in _SIZES if size != (3, 3))
_SOURCE_INPUTS = frozenset(((0, 8, 332), (0, 9, 348), (0, 7, 322), (0, 1, 172)))
_KINDS = ("random", "connected", "scattered", "row", "column", "diagonal", "symmetric", "center_corner")


def _code(mask):
    return tuple(int(v) for v in mask.flat)


def _orbit(mask):
    variants = []
    for turns in range(4):
        turned = np.rot90(mask, turns)
        variants.extend((_code(turned), _code(np.fliplr(turned))))
    return (mask.shape, min(variants))


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    sizes = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        count = 1
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    count += 1
        sizes.append(count)
    return sizes


def _matches_kind(mask, kind):
    h, w = mask.shape
    if kind == "connected":
        return len(_components(mask)) == 1
    if kind == "scattered":
        return len(_components(mask)) >= 2
    if kind == "row":
        return bool(np.any(np.all(mask, axis=1)))
    if kind == "column":
        return bool(np.any(np.all(mask, axis=0)))
    if kind == "diagonal":
        return bool(all(mask[i, i] for i in range(min(h, w))) or
                    all(mask[i, w - 1 - i] for i in range(min(h, w))))
    if kind == "symmetric":
        return (np.array_equal(mask, np.fliplr(mask)) or
                np.array_equal(mask, np.flipud(mask)) or
                (h == w and np.array_equal(mask, np.rot90(mask))))
    if kind == "center_corner":
        centers = [(r, c) for r in range(h) for c in range(w)
                   if abs(r - (h - 1) / 2) <= .5 and abs(c - (w - 1) / 2) <= .5]
        corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
        edges = [(r, c) for r in range(h) for c in range(w)
                 if (r in (0, h - 1) or c in (0, w - 1)) and (r, c) not in corners]
        return (any(mask[p] for p in centers) and any(mask[p] for p in corners)
                and (not edges or any(mask[p] for p in edges)))
    return True


def _random_mask(rng, size, count, kind):
    h, w = size
    n = h * w
    mask = np.zeros(size, dtype=bool)
    if kind == "connected":
        first = int(rng.integers(n))
        cells = {divmod(first, w)}
        while len(cells) < count:
            boundary = set()
            for r, c in cells:
                boundary.update((rr, cc) for rr, cc in
                                ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells)
            choices = sorted(boundary)
            cells.add(choices[int(rng.integers(len(choices)))])
        for cell in cells:
            mask[cell] = True
        return mask
    if kind in ("row", "column", "diagonal"):
        if kind == "row":
            mask[int(rng.integers(h)), :] = True
        elif kind == "column":
            mask[:, int(rng.integers(w))] = True
        else:
            for i in range(min(h, w)):
                mask[i, i if rng.integers(2) else w - 1 - i] = True
        if int(mask.sum()) <= count:
            available = np.flatnonzero(~mask)
            extra = rng.choice(available, size=count - int(mask.sum()), replace=False)
            mask.flat[extra] = True
            return mask
    if kind == "center_corner":
        centers = [(r, c) for r in range(h) for c in range(w)
                   if abs(r - (h - 1) / 2) <= .5 and abs(c - (w - 1) / 2) <= .5]
        corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
        edges = [(r, c) for r in range(h) for c in range(w)
                 if (r in (0, h - 1) or c in (0, w - 1)) and (r, c) not in corners]
        mask[centers[int(rng.integers(len(centers)))]] = True
        mask[corners[int(rng.integers(4))]] = True
        if edges and count >= 3:
            mask[edges[int(rng.integers(len(edges)))]] = True
        remaining = count - int(mask.sum())
        if remaining >= 0:
            available = np.flatnonzero(~mask)
            mask.flat[rng.choice(available, size=remaining, replace=False)] = True
            return mask
    # A requested full line or center/edge/corner core may exceed the
    # selected density. Discard that partial core before the random fallback.
    mask[:] = False
    mask.flat[rng.choice(n, size=count, replace=False)] = True
    return mask


def _mask_candidates(rng, size, count, kind):
    if size == (3, 3):
        candidates = []
        for code in range(1, 511):
            if code.bit_count() == count:
                mask = np.array([(code >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
                if _matches_kind(mask, kind):
                    candidates.append(mask)
        return candidates
    candidates = []
    for _ in range(36):
        mask = _random_mask(rng, size, count, kind)
        if _matches_kind(mask, kind):
            candidates.append(mask)
    return candidates


def _source_copy(mask, background, active):
    if mask.shape != (3, 3):
        return False
    code = sum(int(v) << i for i, v in enumerate(mask.flat))
    return (background, active, code) in _SOURCE_INPUTS


def _choose_mask(rng, size, count, kind, background, active, used, used_orbits, prefer_new):
    candidates = _mask_candidates(rng, size, count, kind)
    candidates = [m for m in candidates if (size, _code(m)) not in used
                  and not _source_copy(m, background, active)]
    if not candidates:
        candidates = _mask_candidates(rng, size, count, "random")
        candidates = [m for m in candidates if (size, _code(m)) not in used
                      and not _source_copy(m, background, active)]
    if not candidates:
        # The random pool can miss an unused small motif. The first 32
        # combinations suffice because a puzzle contains at most 14 motifs.
        for chosen in islice(combinations(range(size[0] * size[1]), count), 32):
            mask = np.zeros(size, dtype=bool)
            mask.flat[list(chosen)] = True
            if (size, _code(mask)) not in used and not _source_copy(mask, background, active):
                candidates = [mask]
                break
        if not candidates:
            raise AssertionError("no unused motif for a legal size and density")
    if prefer_new:
        fresh = [m for m in candidates if _orbit(m) not in used_orbits]
        if fresh:
            candidates = fresh
    mask = candidates[int(rng.integers(len(candidates)))]
    used.add((size, _code(mask)))
    used_orbits.add(_orbit(mask))
    return mask


def _make_pair(mask, background, active):
    source = np.where(mask, background, active).astype(np.int8)
    expanded = np.kron(mask.astype(np.int8), mask.astype(np.int8)).astype(bool)
    target = np.where(expanded, active, background).astype(np.int8)
    return {"input": source, "output": target}


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(f"{name} is outside its legal range")
    if num_train is None and num_test is None:
        num_train = int(rng.choice((2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10)))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _density(rng, size):
    n = size[0] * size[1]
    if n == 4:
        return 2
    low = max(2, int(np.ceil(.22 * n)))
    high = min(n - 2, int(np.floor(.78 * n)))
    return int(rng.integers(low, high + 1))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    background = int(rng.integers(10))
    palette = [int(v) for v in rng.permutation([c for c in range(10) if c != background])]
    train_palette = palette[:min(4, num_train)]
    train_colors = train_palette[:2] + [int(rng.choice(train_palette)) for _ in range(num_train - 2)]
    rng.shuffle(train_colors)
    new_palette = [c for c in palette if c not in train_palette]
    test_colors = [int(rng.choice(train_palette if rng.integers(2) else new_palette))
                   for _ in range(num_test)]

    mixed_size = bool(rng.random() < .55)
    train_sizes = [(3, 3)] * num_train
    if mixed_size:
        train_sizes[1] = _OTHER_SIZES[int(rng.integers(len(_OTHER_SIZES)))]
        for i in range(2, num_train):
            if rng.random() < .65:
                train_sizes[i] = _OTHER_SIZES[int(rng.integers(len(_OTHER_SIZES)))]
    rng.shuffle(train_sizes)
    test_sizes = []
    for _ in range(num_test):
        if not mixed_size or rng.random() < .35:
            test_sizes.append((3, 3))
        elif rng.random() < .55:
            test_sizes.append(train_sizes[int(rng.integers(num_train))])
        else:
            unseen = [size for size in _OTHER_SIZES if size not in train_sizes]
            choices = unseen if unseen else _OTHER_SIZES
            test_sizes.append(choices[int(rng.integers(len(choices)))])
    # The tiny 2x2 vocabulary is useful once or twice, not as a repeated template.
    tiny_count = 0
    for sizes in (train_sizes, test_sizes):
        for i, size in enumerate(sizes):
            if size == (2, 2):
                tiny_count += 1
                if tiny_count > 2:
                    sizes[i] = (3, 3)
    if mixed_size and all(size == (3, 3) for size in train_sizes):
        train_sizes[0] = (4, 4)

    test_counts = [int(rng.choice((2, 3, 3, 4, 4, 5, 5, 6))) if size == (3, 3)
                   else _density(rng, size) for size in test_sizes]
    train_counts = [int(rng.choice((2, 3, 3, 4, 4, 5, 5, 6))) if size == (3, 3)
                    else _density(rng, size) for size in train_sizes]
    extremes = [k for size, k in zip(test_sizes, test_counts) if size == (3, 3) and k in (2, 6)]
    three_indices = [i for i, size in enumerate(train_sizes) if size == (3, 3)]
    for k, i in zip(dict.fromkeys(extremes), three_indices):
        train_counts[i] = k
    if not mixed_size and num_train >= 2 and not extremes:
        train_counts[0], train_counts[1] = 3, 5

    kinds = list(_KINDS)
    rng.shuffle(kinds)
    used, used_orbits = set(), set()
    train, test = [], []
    for split, sizes, counts, colors in ((train, train_sizes, train_counts, train_colors),
                                          (test, test_sizes, test_counts, test_colors)):
        for i, (size, count, color) in enumerate(zip(sizes, counts, colors)):
            kind = kinds[(i + (0 if split is train else num_train)) % len(kinds)]
            mask = _choose_mask(rng, size, count, kind, background, color,
                                used, used_orbits, split is test and rng.random() < .85)
            split.append(_make_pair(mask, background, color))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_background(train):
    common = set(int(v) for v in np.unique(train[0]["input"]))
    for pair in train[1:]:
        common.intersection_update(int(v) for v in np.unique(pair["input"]))
    assert len(common) == 1, "training must reveal a unique common background"
    return common.pop()


def _independent_prediction(train, source):
    background = _infer_background(train)
    colors = set(int(v) for v in np.unique(source))
    assert background in colors and len(colors) == 2
    active = (colors - {background}).pop()
    h, w = source.shape
    result = np.full((h * h, w * w), background, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if int(source[r, c]) == background:
                for y in range(h):
                    for x in range(w):
                        if int(source[y, x]) == background:
                            result[r * h + y, c * w + x] = active
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    background = _infer_background(puzzle["train"])
    inputs, pairs, motifs = set(), set(), set()
    train_colors = set()
    train_sizes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            h, w = source.shape
            assert (h, w) in _SIZES and target.shape == (h * h, w * w)
            colors = set(int(v) for v in np.unique(source))
            assert len(colors) == 2 and background in colors
            active = (colors - {background}).pop()
            mask = source == background
            count = int(mask.sum())
            assert 0 < count < h * w
            if (h, w) == (2, 2):
                assert count == 2
            elif (h, w) == (3, 3):
                assert 2 <= count <= 6
            else:
                assert max(2, int(np.ceil(.22 * h * w))) <= count <= min(h * w - 2, int(np.floor(.78 * h * w)))
            assert set(int(v) for v in np.unique(target)) == {background, active}
            assert int(np.count_nonzero(target == active)) == count * count
            # Inspect each block directly; this does not call the construction helper.
            for r in range(h):
                for c in range(w):
                    block = target[r * h:(r + 1) * h, c * w:(c + 1) * w]
                    for y in range(h):
                        for x in range(w):
                            expected = (active if source[r, c] == background and
                                        source[y, x] == background else background)
                            assert int(block[y, x]) == expected
            source_signature = (source.shape, source.tobytes())
            pair_signature = (source_signature, target.shape, target.tobytes())
            motif_signature = ((h, w), _code(mask))
            assert source_signature not in inputs and pair_signature not in pairs
            assert motif_signature not in motifs
            assert not _source_copy(mask, background, active)
            inputs.add(source_signature)
            pairs.add(pair_signature)
            motifs.add(motif_signature)
            if split == "train":
                train_colors.add(active)
                train_sizes.add((h, w))
    assert len(train_colors) >= 2
    assert (3, 3) in train_sizes
    if any(pair["input"].shape != (3, 3) for pair in puzzle["test"]):
        assert len(train_sizes) >= 2
    for pair in puzzle["test"]:
        assert np.array_equal(_independent_prediction(puzzle["train"], pair["input"]), pair["output"])


def _signature(puzzle):
    return tuple((split, tuple((p["input"].shape, p["input"].tobytes(),
                                p["output"].shape, p["output"].tobytes())
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    probe_rng = np.random.default_rng(1729)
    for size in _SIZES:
        cells = size[0] * size[1]
        low = 2 if size == (2, 2) else max(2, int(np.ceil(.22 * cells)))
        high = 2 if size == (2, 2) else min(cells - 2, int(np.floor(.78 * cells)))
        for count in range(low, high + 1):
            for kind in _KINDS:
                for _ in range(3):
                    mask = _random_mask(probe_rng, size, count, kind)
                    assert mask.shape == size and int(mask.sum()) == count
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert sum(len(p["train"]) <= 4 for p in automatic) >= 10
    assert len({_signature(p) for p in automatic}) == 42
    backgrounds = set()
    sizes = set()
    test_densities = set()
    symmetric_tests = asymmetric_tests = reused_colors = new_colors = 0
    motif_orbits = set()
    novel_test_orbits = 0
    motif_kinds = {kind: 0 for kind in _KINDS[1:]}
    combined_regimes = set()
    mixed_puzzles = varying_density_puzzles = active_zero_tests = 0
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        background = _infer_background(puzzle["train"])
        backgrounds.add(background)
        train_sizes = {p["input"].shape for p in puzzle["train"]}
        mixed_puzzles += len(train_sizes) > 1
        observed_densities = {(p["input"].shape, int(np.count_nonzero(p["input"] == background)))
                              for split in ("train", "test") for p in puzzle[split]}
        varying_density_puzzles += len(observed_densities) > 1
        train_active = {next(iter(set(int(v) for v in np.unique(p["input"])) - {background}))
                        for p in puzzle["train"]}
        train_orbits = {_orbit(p["input"] == background) for p in puzzle["train"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                sizes.add(source.shape)
                motif_orbits.add(_orbit(source == background))
                assert np.array_equal(_independent_prediction(puzzle["train"], source), target)
                for kind in motif_kinds:
                    motif_kinds[kind] += _matches_kind(source == background, kind)
                if split == "test":
                    if _orbit(source == background) not in train_orbits:
                        novel_test_orbits += 1
                    if source.shape == (3, 3):
                        test_densities.add(int(np.count_nonzero(source == background)))
                    if _matches_kind(source == background, "symmetric"):
                        symmetric_tests += 1
                    else:
                        asymmetric_tests += 1
                    active = (set(int(v) for v in np.unique(source)) - {background}).pop()
                    active_zero_tests += active == 0
                    combined_regimes.add((source.shape not in train_sizes,
                                          active in train_active,
                                          _matches_kind(source == background, "symmetric")))
                    if active in train_active:
                        reused_colors += 1
                    else:
                        new_colors += 1
    assert len(backgrounds) >= 7 and any(v != 0 for v in backgrounds)
    assert len(sizes) >= 6 and (3, 3) in sizes and (5, 5) in sizes
    assert {2, 3, 4, 5, 6}.issubset(test_densities)
    assert symmetric_tests >= 4 and asymmetric_tests >= 10
    assert reused_colors >= 8 and new_colors >= 8
    assert len(motif_orbits) >= 35 and novel_test_orbits >= 25
    assert mixed_puzzles >= 10 and varying_density_puzzles >= 35
    assert active_zero_tests >= 1 and len(combined_regimes) == 8
    assert all(count >= 20 for count in motif_kinds.values())
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(3000 + ntrain * 10 + ntest, ntrain, ntest)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
        assert _signature(puzzle) == _signature(generate_puzzle(3000 + ntrain * 10 + ntest, ntrain, ntest))
    for ntrain, ntest in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(4000 + (ntrain or 0) * 10 + (ntest or 0), ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, ntrain, ntest)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
