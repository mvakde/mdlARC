"""Compositional corner-key stencil puzzles."""

import hashlib
import numpy as np


_CORNERS = ("tl", "tr", "bl", "br")
_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3), (4, 4))
_SOURCE_DIGESTS = {
    "323b5b3dcebe3efe1f285dbab80aefc17e52e43b9c2a04c3d71af1f42f9f2906",
    "2a703096d6f6c88dbf665d16760bcaa55e9fdcaf6000d102c8838bc42cf1eaf9",
    "a2d4f05175f9715e89fb599a00fe3d1ead10cef41c6fa32e1109105f2b9852df",
    "791239432af0df37ecec626cb5c21714c50d3d5106d431e4ff7185a6062c1bd9",
}


def _geometry(shape, lattice, corner):
    h, w = shape
    rows, cols = lattice
    height = rows * (h + 1) - 1
    width = cols * (w + 1) - 1
    r0 = 0 if corner[0] == "t" else (rows - 2) * (h + 1)
    c0 = 0 if corner[1] == "l" else (cols - 2) * (w + 1)
    kr = r0 + (1 if corner[0] == "t" else 0)
    kc = c0 + (1 if corner[1] == "l" else 0)
    key_rows = slice(kr, kr + 2 * h)
    key_cols = slice(kc, kc + 2 * w)

    structure = np.zeros((height, width), dtype=bool)
    structure[h::h + 1, :] = True
    structure[:, w::w + 1] = True
    structure[r0:r0 + 2 * h + 1, c0:c0 + 2 * w + 1] = False
    outer_r = r0 if corner[0] == "t" else r0 + 2 * h
    outer_c = c0 if corner[1] == "l" else c0 + 2 * w
    structure[outer_r, c0:c0 + 2 * w + 1] = True
    structure[r0:r0 + 2 * h + 1, outer_c] = True

    key_mask = np.zeros_like(structure)
    key_mask[key_rows, key_cols] = True
    positions = []
    ordinary = np.zeros_like(structure)
    excluded_rows = (0, 1) if corner[0] == "t" else (rows - 2, rows - 1)
    excluded_cols = (0, 1) if corner[1] == "l" else (cols - 2, cols - 1)
    for tr in range(rows):
        for tc in range(cols):
            if tr in excluded_rows and tc in excluded_cols:
                continue
            rr = slice(tr * (h + 1), tr * (h + 1) + h)
            cc = slice(tc * (w + 1), tc * (w + 1) + w)
            positions.append((tr, tc, rr, cc))
            ordinary[rr, cc] = True
    return structure, key_mask, ordinary, key_rows, key_cols, positions


def _fully_asymmetric(mask):
    variants = [np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    if mask.shape[0] == mask.shape[1]:
        variants += [np.rot90(mask, k) for k in (1, 3)]
        variants += [np.rot90(np.fliplr(mask), k) for k in (1, 3)]
    return all(not np.array_equal(mask, item) for item in variants)


def _connected(mask):
    points = set(zip(*np.where(mask)))
    if not points:
        return False
    seen = {next(iter(points))}
    pending = list(seen)
    while pending:
        r, c = pending.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in points and q not in seen:
                seen.add(q)
                pending.append(q)
    return seen == points


def _sample_stencil(rng, shape, anchor=False):
    h, w = shape
    area = h * w
    if anchor:
        # A visible orientation witness needs a stencil with distinct rotations.
        for _ in range(500):
            density = int(rng.integers(2, min(6, area)))
            flat = np.zeros(area, dtype=bool)
            flat[rng.choice(area, density, replace=False)] = True
            result = flat.reshape(shape)
            if _fully_asymmetric(result):
                return result
        raise RuntimeError("orientation witness construction failed")

    mode = int(rng.integers(0, 3))
    if mode == 2:
        # Build along a random symmetry orbit. The resulting density is free.
        transform = int(rng.integers(0, 3))
        for _ in range(100):
            raw = rng.random(shape) < float(rng.uniform(0.2, 0.7))
            if transform == 0:
                result = raw | np.flipud(raw)
            elif transform == 1:
                result = raw | np.fliplr(raw)
            else:
                result = raw | np.flipud(np.fliplr(raw))
            if 0 < int(result.sum()) < area:
                return result
        # The fallback is deterministic for this pair's RNG stream.
        result = np.zeros(shape, dtype=bool)
        result[0, 0] = True
        return result
    density = int(rng.integers(1, area))
    flat = np.zeros(area, dtype=bool)
    if mode == 1:
        start = int(rng.integers(area))
        selected = {start}
        while len(selected) < density:
            frontier = set()
            for index in selected:
                r, c = divmod(index, w)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        frontier.add(nr * w + nc)
            selected.add(int(rng.choice(sorted(frontier - selected))))
        flat[list(selected)] = True
    else:
        flat[rng.choice(area, density, replace=False)] = True
    return flat.reshape(shape)


def _sample_layout(rng, shape, anchor=False):
    h, w = shape
    max_rows = min(8, 31 // (h + 1))
    max_cols = min(8, 31 // (w + 1))
    low = 4 if anchor else 3
    return (int(rng.integers(low, max_rows + 1)),
            int(rng.integers(low, max_cols + 1)))


def _make_tiles(rng, stencil, positions, anchor=False):
    area = stencil.size
    active = np.flatnonzero(stencil.ravel())
    inactive = np.flatnonzero(~stencil.ravel())
    count = len(positions)
    if anchor:
        match_count = int(rng.integers(2, min(7, count - 2) + 1))
        near_count = 1
        decoy_count = 1
    else:
        match_count = int(rng.integers(0, min(count - 2, max(9, count // 2)) + 1))
        near_count = int(rng.integers(0, min(4, count - match_count - 1) + 1))
        decoy_count = 0
    order = [positions[int(i)] for i in rng.permutation(count)]
    categories = (["match"] * match_count + ["near"] * near_count
                  + ["decoy"] * decoy_count)
    categories += ["other"] * (count - len(categories))
    rng.shuffle(categories)
    match_number = 0
    exact_bias = float(rng.choice((0.0, 0.2, 0.5, 0.8)))
    tiles = {}
    for (tr, tc, _, _), category in zip(order, categories):
        flat = np.zeros(area, dtype=bool)
        if category == "match":
            flat[active] = True
            if anchor and match_number == 0:
                extra_count = int(rng.integers(1, len(inactive) + 1))
            elif rng.random() < exact_bias:
                extra_count = 0
            elif rng.random() < 0.12:
                extra_count = len(inactive)
            else:
                extra_count = int(rng.integers(1, len(inactive) + 1))
            if extra_count:
                flat[rng.choice(inactive, extra_count, replace=False)] = True
            match_number += 1
        elif category == "decoy":
            flat = np.rot90(stencil).ravel().copy()
        else:
            if category == "near":
                flat[active] = True
                flat[int(rng.choice(active))] = False
            else:
                missing_count = int(rng.integers(1, len(active) + 1))
                flat[active] = True
                flat[rng.choice(active, missing_count, replace=False)] = False
            if rng.random() < 0.18:
                extra_count = len(inactive)
            else:
                extra_count = int(rng.integers(0, len(inactive) + 1))
            if extra_count:
                flat[rng.choice(inactive, extra_count, replace=False)] = True
        tiles[(tr, tc)] = flat.reshape(stencil.shape)
    # Both ordinary roles remain visible even when there is no match.
    if not any(np.any(tile) for tile in tiles.values()):
        first = next(iter(tiles))
        tiles[first].flat[int(inactive[0])] = True
    return tiles


def _palette(rng, profile):
    if profile[0] == "all_shared":
        return profile[1]
    if profile[0] == "base_shared":
        base = profile[1]
        accent = int(rng.choice([c for c in range(10) if c not in base]))
        return (*base, accent)
    return tuple(int(c) for c in rng.choice(10, 4, replace=False))


def _generate_pair(rng, profile, anchor=False):
    if anchor or profile["shape"] == "fixed":
        shape = (3, 3)
    else:
        shape = _SHAPES[int(rng.choice(len(_SHAPES), p=(.17, .18, .12, .33, .20)))]
    lattice = _sample_layout(rng, shape, anchor)
    corner = profile["corner"] if profile["corner"] is not None else _CORNERS[int(rng.integers(4))]
    structural, neutral, foreground, accent = _palette(rng, profile["palette"])
    stencil = _sample_stencil(rng, shape, anchor)
    structure, _, _, key_rows, key_cols, positions = _geometry(shape, lattice, corner)
    tiles = _make_tiles(rng, stencil, positions, anchor)
    input_grid = np.full(structure.shape, neutral, dtype=np.int8)
    input_grid[structure] = structural
    for tr, tc, rr, cc in positions:
        input_grid[rr, cc] = np.where(tiles[(tr, tc)], foreground, neutral)
    key = np.where(stencil, accent, neutral)
    input_grid[key_rows, key_cols] = np.repeat(np.repeat(key, 2, axis=0), 2, axis=1)
    output_grid = input_grid.copy()
    for _, _, rr, cc in positions:
        if np.all(input_grid[rr, cc][stencil] == foreground):
            output_grid[rr, cc][stencil] = accent
    return input_grid, output_grid


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(num_test + 1, 2 + (seed // 4) % 9)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    palette_style = ("all_shared", "base_shared", "local")[int(rng.integers(3))]
    shared = tuple(int(c) for c in rng.choice(10, 4, replace=False))
    palette_profile = (palette_style, shared if palette_style == "all_shared" else shared[:3])
    profile = {
        "palette": palette_profile,
        "shape": "fixed" if rng.random() < .22 else "variable",
        "corner": _CORNERS[int(rng.integers(4))] if rng.random() < .20 else None,
    }
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            # Independent deterministic pair streams; collision retries are bounded.
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                input_grid, output_grid = _generate_pair(
                    pair_rng, profile, anchor=(split == "train" and index == 0))
                signature = (input_grid.shape, input_grid.tobytes())
                if signature not in used_inputs:
                    break
            else:
                raise RuntimeError("could not generate a unique input")
            used_inputs.add(signature)
            puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_digest(input_grid, output_grid):
    return hashlib.sha256(bytes(input_grid.shape) + input_grid.astype(np.uint8).tobytes()
                          + output_grid.astype(np.uint8).tobytes()).hexdigest()


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    assert input_grid.ndim == 2
    colors = set(int(c) for c in np.unique(input_grid))
    assert len(colors) == 4
    candidates = []
    for shape in _SHAPES:
        h, w = shape
        if (input_grid.shape[0] + 1) % (h + 1) or (input_grid.shape[1] + 1) % (w + 1):
            continue
        lattice = ((input_grid.shape[0] + 1) // (h + 1),
                   (input_grid.shape[1] + 1) // (w + 1))
        if min(lattice) < 3:
            continue
        for corner in _CORNERS:
            geometry = _geometry(shape, lattice, corner)
            structure = geometry[0]
            for color in colors:
                if np.array_equal(input_grid == color, structure):
                    candidates.append((shape, lattice, corner, color, geometry))
    assert len(candidates) == 1
    shape, lattice, corner, structural, geometry = candidates[0]
    structure, key_mask, ordinary, key_rows, key_cols, positions = geometry
    assert not np.any(structure & key_mask)
    assert not np.any(structure & ordinary)
    assert not np.any(key_mask & ordinary)
    assert np.all(structure | key_mask | ordinary)
    key = input_grid[key_rows, key_cols]
    small = key[::2, ::2]
    assert np.array_equal(key, np.repeat(np.repeat(small, 2, axis=0), 2, axis=1))
    key_colors = set(int(c) for c in np.unique(key))
    assert len(key_colors) == 2
    accent_candidates = [c for c in key_colors if np.count_nonzero(input_grid == c) == np.count_nonzero(key == c)]
    assert len(accent_candidates) == 1
    accent = accent_candidates[0]
    neutral = next(c for c in key_colors if c != accent)
    foreground_set = colors - {structural, neutral, accent}
    assert len(foreground_set) == 1
    foreground = next(iter(foreground_set))
    assert set(int(c) for c in np.unique(input_grid[ordinary])) == {neutral, foreground}
    stencil = small == accent
    density = int(stencil.sum())
    assert 1 <= density < stencil.size
    expected = input_grid.copy()
    matches = exact = supersets = near = full_tiles = max_extras = orientation_decoys = 0
    for _, _, rr, cc in positions:
        tile = input_grid[rr, cc]
        assert set(int(c) for c in np.unique(tile)) <= {neutral, foreground}
        mask = tile == foreground
        hits = int(np.count_nonzero(mask & stencil))
        if hits == density:
            matches += 1
            extras = int(np.count_nonzero(mask & ~stencil))
            exact += extras == 0
            supersets += extras > 0
            max_extras = max(max_extras, extras)
            full_tiles += bool(np.all(mask))
            expected[rr, cc][stencil] = accent
        else:
            near += hits == density - 1
            if _fully_asymmetric(stencil) and np.array_equal(mask, np.rot90(stencil)):
                orientation_decoys += 1
    assert np.array_equal(output_grid, expected)
    assert _pair_digest(input_grid, output_grid) not in _SOURCE_DIGESTS
    return {"shape": shape, "lattice": lattice, "corner": corner,
            "structural": structural, "neutral": neutral, "foreground": foreground,
            "accent": accent, "stencil": stencil, "density": density,
            "matches": matches, "exact": exact, "supersets": supersets,
            "near": near, "full_tiles": full_tiles, "max_extras": max_extras,
            "connected": _connected(stencil), "asymmetric": _fully_asymmetric(stencil),
            "orientation_decoys": orientation_decoys}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    signatures = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            signature = (pair["input"].shape, pair["input"].astype(np.uint8).tobytes())
            assert signature not in signatures
            signatures.add(signature)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    train = analyses["train"]
    # These training witnesses make wildcard preservation, rejection and
    # the global stencil orientation visible, regardless of test regime.
    assert any(a["supersets"] and a["near"] for a in train)
    assert any(a["asymmetric"] and a["orientation_decoys"] for a in train)


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    analyses = []
    train_counts, test_counts = set(), set()
    all_inputs = set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analyses.append(_analyze_pair(pair["input"], pair["output"]))
                all_inputs.add((pair["input"].shape, pair["input"].tobytes()))
                total += 1
    assert len(all_inputs) == total
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert set(a["shape"] for a in analyses) == set(_SHAPES)
    assert set(a["corner"] for a in analyses) == set(_CORNERS)
    assert any(a["density"] == 1 for a in analyses)
    assert any(a["density"] == a["stencil"].size - 1 for a in analyses)
    assert any(a["matches"] == 0 for a in analyses)
    assert any(a["matches"] >= 7 for a in analyses)
    assert any(a["exact"] >= 2 for a in analyses)
    assert any(a["matches"] and a["exact"] == 0 for a in analyses)
    assert any(a["full_tiles"] for a in analyses)
    assert any(a["max_extras"] >= 4 for a in analyses)
    assert any(a["connected"] for a in analyses)
    assert any(not a["connected"] for a in analyses)
    for seed, ntr, nte in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 7, 3)):
        puzzle = generate_puzzle(seed, ntr, nte)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntr, nte)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _puzzles_equal(generate_puzzle(999, 8, 3), generate_puzzle(999, 8, 3))
    assert not _puzzles_equal(generate_puzzle(999, 8, 3), generate_puzzle(1000, 8, 3))
