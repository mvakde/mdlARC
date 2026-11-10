import hashlib

import numpy as np


# Each prototype is an occupied-cell stencil. Dihedral duplicates are removed
# below, so sampling is over geometric families rather than orientation names.
_PROTOTYPES = (
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),  # long-stem T, source train 4
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 3), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 3), (1, 2), (2, 0), (2, 3), (3, 1)),
    ((0, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1)),
    ((0, 1), (1, 0), (1, 2), (2, 0), (2, 2), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3)),
)

_SOURCE_PAIR_HASHES = frozenset((
    "0aeed04d5c96285fd8cf1cccb589d552fd395d08a53bd7c20240f8f58a2080f1",
    "44a534db424b36b07dc88ad48e5f23f604ab4935ef6f8f69b8a0eb88ccef2258",
    "92218daf56ba9cc105405f99a909f6bb676bb4b29352471a8e4dc5ae603600b8",
    "0f628061d0628e972c326e2e9cbb4dd2fdc743f639f3c3810d9644000f7cfddb",
    "2d00486891a2d81756fdf1b606a4fe22283a891e638dc5ca534c5c04eea44f68",
))


def _normalize(points):
    points = [(int(r), int(c)) for r, c in points]
    if not points:
        raise ValueError("empty stencil")
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _rotation_variants(shape):
    return tuple(sorted({
        _normalize([(r, c) for r, c in shape]),
        _normalize([(c, -r) for r, c in shape]),
        _normalize([(-r, -c) for r, c in shape]),
        _normalize([(-c, r) for r, c in shape]),
    }))


def _dihedral_variants(shape):
    rotations = set(_rotation_variants(shape))
    reflected = [(r, -c) for r, c in shape]
    return tuple(sorted(rotations | set(_rotation_variants(reflected))))


def _family(shape):
    return min(_dihedral_variants(shape))


_FAMILIES = tuple(sorted({_family(shape) for shape in _PROTOTYPES}))
_CHIRAL = tuple(shape for shape in _FAMILIES
                if len(_dihedral_variants(shape)) > len(_rotation_variants(shape)))


def _find_matches(grid, color, stencil):
    """Find all complete foreground matches, regardless of gaps or neighbors."""
    height, width = grid.shape
    found = set()
    for variant in _dihedral_variants(stencil):
        bh = max(r for r, _ in variant) + 1
        bw = max(c for _, c in variant) + 1
        nr, nc = height - bh + 1, width - bw + 1
        if nr < 1 or nc < 1:
            continue
        valid = np.ones((nr, nc), dtype=bool)
        for dr, dc in variant:
            valid &= grid[dr:dr + nr, dc:dc + nc] == color
        for ar, ac in np.argwhere(valid):
            found.add(frozenset((int(ar) + dr, int(ac) + dc)
                                for dr, dc in variant))
    return found


def _solve_grid(grid, exemplar, candidate):
    stencil = _normalize(np.argwhere(grid == exemplar))
    result = grid.copy()
    for cells in _find_matches(grid, candidate, stencil):
        for r, c in cells:
            result[r, c] = exemplar
    return result


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _sample_roles(rng):
    # The original 8 -> 3 mapping and palette recur, but neither is locked.
    if rng.random() < 0.25:
        exemplar, candidate = 8, 3
    else:
        exemplar = int(rng.integers(10))
        candidate = int(_choose(rng, [c for c in range(10) if c != exemplar]))
    available = [c for c in range(10) if c not in (exemplar, candidate)]
    if rng.random() < 0.60 and exemplar == 8 and candidate == 3:
        neutrals = (0, 2, 4, 7)
    else:
        count = int(rng.integers(4, 7))
        neutrals = tuple(int(x) for x in rng.choice(available, count, replace=False))
    return exemplar, candidate, neutrals


def _pair_hash(inp, out):
    a = np.asarray(inp, dtype=np.uint8)
    b = np.asarray(out, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _oriented_copies(rng, shape, count, require_mirror, allow_mirror):
    rotations = _rotation_variants(shape)
    mirrors = tuple(x for x in _dihedral_variants(shape) if x not in rotations)
    if require_mirror and not mirrors:
        return None
    orientations = []
    mirror_count = int(rng.integers(1, count + 1)) if require_mirror else 0
    for index in range(count):
        if index < mirror_count:
            pool = mirrors
        elif allow_mirror and mirrors and rng.random() < 0.28:
            pool = mirrors
        else:
            pool = rotations
        if orientations and rng.random() < 0.30 and orientations[0] in pool:
            orientations.append(orientations[0])
        else:
            orientations.append(_choose(rng, pool))
    rng.shuffle(orientations)
    return orientations


def _place_copies(rng, shape, count, height, width, exemplar_cells,
                  require_mirror, allow_mirror):
    oriented = _oriented_copies(rng, shape, count, require_mirror, allow_mirror)
    if oriented is None:
        return None
    placements = []
    occupied = set(exemplar_cells)
    for variant in oriented:
        bh = max(r for r, _ in variant) + 1
        bw = max(c for _, c in variant) + 1
        candidates = [(r, c) for r in range(height - bh + 1)
                      for c in range(width - bw + 1)]
        rng.shuffle(candidates)
        chosen = None
        for ar, ac in candidates:
            cells = frozenset((ar + dr, ac + dc) for dr, dc in variant)
            if cells.isdisjoint(occupied):
                chosen = cells
                break
        if chosen is None:
            return None
        placements.append(chosen)
        occupied.update(chosen)
    return set(placements) if len(set(placements)) == count else None


def _try_candidate(grid, r, c, candidate, stencil, desired):
    old = int(grid[r, c])
    grid[r, c] = candidate
    if _find_matches(grid, candidate, stencil) == desired:
        return True
    grid[r, c] = old
    return False


def _add_far_fragment(rng, grid, candidate, exemplar, stencil, desired, union):
    height, width = grid.shape
    variants = list(_dihedral_variants(stencil))
    rng.shuffle(variants)
    for variant in variants:
        bh = max(r for r, _ in variant) + 1
        bw = max(c for _, c in variant) + 1
        anchors = [(r, c) for r in range(height - bh + 1)
                   for c in range(width - bw + 1)]
        rng.shuffle(anchors)
        for ar, ac in anchors[:100]:
            cells = [(ar + dr, ac + dc) for dr, dc in variant]
            if any(grid[r, c] == exemplar for r, c in cells):
                continue
            if any(max(abs(r - mr), abs(c - mc)) <= 2
                   for r, c in cells for mr, mc in union):
                continue
            rng.shuffle(cells)
            added = []
            for r, c in cells[:-1]:
                if grid[r, c] == candidate:
                    continue
                old = int(grid[r, c])
                if _try_candidate(grid, r, c, candidate, stencil, desired):
                    added.append((r, c, old))
                else:
                    break
            else:
                if len(added) >= max(2, len(stencil) - 2):
                    return True
            for r, c, old in added:
                grid[r, c] = old
    return False


def _generate_pair(rng, regime, roles, shape, copy_count, require_mirror,
                   allow_mirror):
    exemplar, candidate, neutrals = roles
    for attempt in range(90):
        height = int(rng.integers(9, 31))
        width = int(rng.integers(9, 31))
        if max(height, width) / min(height, width) > 2.5 and rng.random() < 0.6:
            continue
        stencil = _choose(rng, _dihedral_variants(shape))
        bh = max(r for r, _ in stencil) + 1
        bw = max(c for _, c in stencil) + 1
        if height < bh or width < bw:
            continue
        weights = rng.uniform(0.72, 1.28, len(neutrals))
        weights /= weights.sum()
        grid = rng.choice(np.asarray(neutrals, dtype=np.int8),
                          size=(height, width), p=weights).astype(np.int8)
        ar = int(rng.integers(height - bh + 1))
        ac = int(rng.integers(width - bw + 1))
        exemplar_cells = frozenset((ar + dr, ac + dc) for dr, dc in stencil)
        for r, c in exemplar_cells:
            grid[r, c] = exemplar
        desired = _place_copies(rng, stencil, copy_count, height, width,
                                exemplar_cells, require_mirror, allow_mirror)
        if desired is None:
            continue
        union = set().union(*desired)
        for r, c in union:
            grid[r, c] = candidate
        if _find_matches(grid, candidate, stencil) != desired:
            continue

        dense = rng.random() < (0.78 if regime == "target_test" else 0.62)
        density = rng.uniform(0.155, 0.255) if dense else rng.uniform(0.075, 0.145)
        target = max(len(union) + 4, int(round(density * height * width)))
        if dense:
            _add_far_fragment(rng, grid, candidate, exemplar, stencil,
                              desired, union)
            if rng.random() < 0.55:
                _add_far_fragment(rng, grid, candidate, exemplar, stencil,
                                  desired, union)

        positions = [(r, c) for r in range(height) for c in range(width)
                     if grid[r, c] != exemplar and grid[r, c] != candidate]
        rng.shuffle(positions)
        for r, c in positions:
            if int(np.count_nonzero(grid == candidate)) >= target:
                break
            _try_candidate(grid, r, c, candidate, stencil, desired)
        if np.count_nonzero(grid == candidate) < target:
            continue
        if any(np.count_nonzero(grid == color) <= len(stencil) + 2
               for color in neutrals):
            continue
        if _find_matches(grid, candidate, stencil) != desired:
            continue
        output = _solve_grid(grid, exemplar, candidate)
        return grid, output
    raise RuntimeError("could not construct an unambiguous pair")


def _coerce_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _coerce_count("num_train", num_train, 2, 10)
    test = _coerce_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        low = max(2, test + 1)
        train = low + (seed // 5) % (11 - low)
    elif test is None:
        test = 1 + (seed // 7) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _mapping_from_train(train):
    mapping = None
    for pair in train:
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape
        changed = inp != out
        assert changed.any()
        transitions = set(zip(inp[changed].tolist(), out[changed].tolist()))
        assert len(transitions) == 1
        one = next(iter(transitions))
        assert one[0] != one[1]
        if mapping is None:
            mapping = one
        else:
            assert mapping == one
    return mapping


def _pair_features(pair, candidate, exemplar):
    inp, out = pair["input"], pair["output"]
    stencil = _normalize(np.argwhere(inp == exemplar))
    assert 4 <= len(stencil) <= 7
    assert max(r for r, _ in stencil) < 4
    assert max(c for _, c in stencil) < 4
    matches = _find_matches(inp, candidate, stencil)
    assert 1 <= len(matches) <= 5
    assert all(a.isdisjoint(b) for a in matches for b in matches if a is not b)
    assert np.array_equal(out, _solve_grid(inp, exemplar, candidate))
    union = set().union(*matches)
    assert np.count_nonzero(inp == candidate) - len(union) >= 4
    colors = set(np.unique(inp).tolist())
    assert 5 <= len(colors) <= 8
    assert candidate in colors and exemplar in colors
    assert sum(np.count_nonzero(inp == c) > len(stencil) + 2
               for c in colors - {candidate, exemplar}) == len(colors) - 2
    assert _pair_hash(inp, out) not in _SOURCE_PAIR_HASHES
    rotations = set(_rotation_variants(stencil))
    mirrored = sum(_normalize(cells) not in rotations for cells in matches)
    return {"family": _family(stencil), "count": len(matches),
            "mirrored": mirrored, "chiral": len(rotations) < len(_dihedral_variants(stencil))}


def _has_far_partial(inp, candidate, stencil, matches):
    far = np.ones(inp.shape, dtype=bool)
    for cells in matches:
        for r, c in cells:
            far[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = False
    for variant in _dihedral_variants(stencil):
        nr = inp.shape[0] - max(r for r, _ in variant)
        nc = inp.shape[1] - max(c for _, c in variant)
        candidate_count = np.zeros((nr, nc), dtype=np.int8)
        wholly_far = np.ones((nr, nc), dtype=bool)
        for dr, dc in variant:
            candidate_count += inp[dr:dr + nr, dc:dc + nc] == candidate
            wholly_far &= far[dr:dr + nr, dc:dc + nc]
        if np.any((candidate_count == len(stencil) - 1) & wholly_far):
            return True
    return False


def _is_connected(stencil):
    occupied = set(stencil)
    reached = {next(iter(occupied))}
    frontier = list(reached)
    while frontier:
        r, c = frontier.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in occupied and neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return len(reached) == len(occupied)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair["input"])
            both = (key, _grid_key(pair["output"]))
            assert key not in inputs and both not in pairs
            inputs.add(key)
            pairs.add(both)
    candidate, exemplar = _mapping_from_train(puzzle["train"])
    demonstrated_colors = set().union(*(set(np.unique(p["input"]).tolist())
                                        for p in puzzle["train"]))
    assert all(set(np.unique(p["input"]).tolist()) <= demonstrated_colors
               for p in puzzle["test"])
    train_info = [_pair_features(p, candidate, exemplar) for p in puzzle["train"]]
    test_info = [_pair_features(p, candidate, exemplar) for p in puzzle["test"]]
    assert len({f["count"] for f in train_info + test_info}) >= 2
    assert len({f["count"] for f in train_info}) >= 2
    assert len({frozenset(np.unique(p["input"]).tolist())
                for p in puzzle["train"] + puzzle["test"]}) >= 2
    train_families = {f["family"] for f in train_info}
    assert all(f["family"] not in train_families for f in test_info)
    # A learned reflection case is a genuine chiral reflected recoloring.
    if any(f["mirrored"] for f in train_info):
        assert any(f["chiral"] and f["mirrored"] for f in train_info)
    for pair in puzzle["test"]:
        inp, out = pair["input"], pair["output"]
        changed = inp != out
        assert set(zip(inp[changed].tolist(), out[changed].tolist())) == {(candidate, exemplar)}


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = _sample_roles(rng)
    teach_reflection = bool(rng.integers(2))
    test_reflection = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    used = set()
    used_train_families = set()
    copy_counts = list(rng.integers(1, 5, size=train_count + test_count))
    if copy_counts[0] == copy_counts[1]:
        copy_counts[1] = 1 + copy_counts[1] % 4
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            global_index = index if section == "train" else train_count + index
            count_copies = copy_counts[global_index]
            must_mirror = ((section == "train" and index == 0 and teach_reflection)
                           or (section == "test" and index == 0 and test_reflection))
            allow_mirror = teach_reflection if section == "train" else test_reflection
            for retry in range(45):
                choices = (_CHIRAL if must_mirror else _FAMILIES)
                if section == "test":
                    choices = tuple(x for x in choices if x not in used_train_families)
                shape = _choose(rng, choices)
                child = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                if global_index == 0:
                    local_neutrals = roles[2]
                elif global_index == 1:
                    local_neutrals = tuple(int(x) for x in
                                           child.choice(roles[2], 3, replace=False))
                else:
                    local_size = int(child.integers(3, len(roles[2]) + 1))
                    local_neutrals = tuple(int(x) for x in
                                           child.choice(roles[2], local_size,
                                                        replace=False))
                try:
                    inp, out = _generate_pair(child, "source_train" if section == "train"
                                              else "target_test",
                                              (roles[0], roles[1], local_neutrals), shape,
                                              count_copies, must_mirror, allow_mirror)
                except RuntimeError:
                    continue
                key = _grid_key(inp)
                if key in used:
                    continue
                used.add(key)
                puzzle[section].append({"input": inp, "output": out})
                if section == "train":
                    used_train_families.add(_family(shape))
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_demonstrations(train, test_input):
    candidate, exemplar = _mapping_from_train(train)
    points = [(int(r), int(c)) for r, c in np.argwhere(test_input == exemplar)]
    orientations = set()
    for reflected in (False, True):
        for turns in range(4):
            transformed = []
            for row, col in points:
                if reflected:
                    col = -col
                for _ in range(turns):
                    row, col = col, -row
                transformed.append((row, col))
            orientations.add(_normalize(transformed))
    prediction = test_input.copy()
    for shape in orientations:
        bh = max(r for r, _ in shape) + 1
        bw = max(c for _, c in shape) + 1
        for ar in range(test_input.shape[0] - bh + 1):
            for ac in range(test_input.shape[1] - bw + 1):
                cells = [(ar + r, ac + c) for r, c in shape]
                if all(test_input[r, c] == candidate for r, c in cells):
                    for r, c in cells:
                        prediction[r, c] = exemplar
    return prediction


def self_test() -> None:
    a = generate_puzzle(123456, 5, 2)
    assert _puzzles_equal(a, generate_puzzle(123456, 5, 2))
    assert not _puzzles_equal(a, generate_puzzle(123457, 5, 2))
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(100 + train * 10 + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
        validate_generated_puzzle(p)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(210 + (train or 0) + (test or 0), train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(33, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    train_counts, test_counts, dimensions, families, mappings = (set() for _ in range(5))
    copy_counts, mirrored_train, mirrored_test = set(), 0, 0
    mirrored_test_counts = set()
    dense, sparse, wide, small, large = 0, 0, 0, 0, 0
    far_partial = far_unmatched = nearby_unmatched = 0
    connected = disconnected = long_t = wide_stencil = 0
    candidate_second_rarest = same_orientation = mixed_orientation = 0
    source_mapping = source_palette = mirror_shift = rotation_only_test = 0
    varying_copies = varying_palettes = 0
    pair_total = 0
    structural = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        candidate, exemplar = _mapping_from_train(p["train"])
        mappings.add((candidate, exemplar))
        source_mapping += (candidate, exemplar) == (3, 8)
        train_mirror_here = test_mirror_here = False
        puzzle_copy_counts = set()
        puzzle_palettes = set()
        for section in ("train", "test"):
            for pair in p[section]:
                inp, out = pair["input"], pair["output"]
                f = _pair_features(pair, candidate, exemplar)
                pair_total += 1
                dimensions.add(inp.shape)
                families.add(f["family"])
                copy_counts.add(f["count"])
                puzzle_copy_counts.add(f["count"])
                palette = frozenset(np.unique(inp).tolist())
                puzzle_palettes.add(palette)
                source_palette += palette == {0, 2, 3, 4, 7, 8}
                if section == "train" and f["mirrored"]:
                    mirrored_train += 1
                    train_mirror_here = True
                if section == "test" and f["mirrored"]:
                    mirrored_test += 1
                    test_mirror_here = True
                if section == "test":
                    mirrored_test_counts.add(f["mirrored"])
                    assert np.array_equal(
                        _predict_from_demonstrations(p["train"], inp), out)
                density = np.count_nonzero(inp == candidate) / inp.size
                dense += density >= 0.15
                sparse += density < 0.15
                wide += max(inp.shape) / min(inp.shape) >= 1.7
                small += inp.size <= 144
                large += inp.size >= 600
                changed = inp != out
                assert changed.any() and np.all(inp[changed] == candidate)
                assert np.all(out[changed] == exemplar)
                stencil = _normalize(np.argwhere(inp == exemplar))
                matches = _find_matches(inp, candidate, stencil)
                union = set().union(*matches)
                unmatched = set(map(tuple, np.argwhere(inp == candidate))) - union
                far_unmatched += any(all(max(abs(r - mr), abs(c - mc)) > 2
                                         for mr, mc in union) for r, c in unmatched)
                nearby_unmatched += any(any(max(abs(r - mr), abs(c - mc)) <= 1
                                            for mr, mc in union) for r, c in unmatched)
                far_partial += _has_far_partial(inp, candidate, stencil, matches)
                connected += _is_connected(stencil)
                disconnected += not _is_connected(stencil)
                long_t += f["family"] == _family(_PROTOTYPES[1])
                wide_stencil += max(r for r, _ in stencil) == 3 and max(
                    c for _, c in stencil) == 3
                color_counts = sorted(np.count_nonzero(inp == color)
                                      for color in palette)
                candidate_second_rarest += (
                    np.count_nonzero(inp == candidate) == color_counts[1])
                orientations = {_normalize(cells) for cells in matches}
                if len(matches) > 1:
                    same_orientation += len(orientations) == 1
                    mixed_orientation += len(orientations) > 1
                structural.add((inp.shape, f["family"], f["count"],
                                tuple(np.argwhere(inp == exemplar).min(axis=0))))
        varying_copies += len(puzzle_copy_counts) >= 2
        varying_palettes += len(puzzle_palettes) >= 2
        mirror_shift += test_mirror_here and not train_mirror_here
        rotation_only_test += not test_mirror_here
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 60 and len(families) >= 12
    assert len(mappings) >= 15 and len(copy_counts) >= 4
    assert mirrored_train and mirrored_test and dense and sparse
    assert wide and small and large and len(structural) >= 100
    assert varying_copies == varying_palettes == 42
    assert dense >= pair_total // 2 and sparse >= pair_total // 5
    assert far_partial >= pair_total // 3
    assert far_unmatched >= pair_total // 2 and nearby_unmatched >= pair_total // 2
    assert connected >= 50 and disconnected >= 30
    assert long_t >= 5 and wide_stencil >= 5
    assert candidate_second_rarest < pair_total * 3 // 4
    assert same_orientation >= 10 and mixed_orientation >= 20
    assert source_mapping and source_palette and mirror_shift and rotation_only_test
    assert {0, 1, 2, 3} <= mirrored_test_counts
