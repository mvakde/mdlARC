"""Generate cropped, offset-symmetry mosaic completion puzzles."""

import hashlib

import numpy as np


N = 30
_SOURCE_PAIR_DIGESTS = {
    "fc6476ac25f2c50912dae1dbe370a627953a54a549ec755e1ef1c0521820d694",
    "96671d5a7b3a26af99d119205a6a92812da0c97e40c39630c5036526b78a412e",
    "5d0f4019a324cd09fec767f12c725d958986aaeb8dbb0d200e1334904ab60697",
    "bb9ad1a16078b02154d671895590cb40c4e92d40a95e69ea35fd14fc64d3cf85",
    "c6eaa40e710dfaf8ceb7f29feff2b40d05b5715ba17fd7adcc6c6b596c0b481c",
}


def _orbit(r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = set()
    for x, y in ((r, c), (c, r)):
        for xx in (x, 31 - x):
            for yy in (y, 31 - y):
                if 0 <= xx < N and 0 <= yy < N:
                    cells.add((xx, yy))
    return tuple(sorted(cells))


_ORBITS = tuple(sorted({_orbit(r, c) for r in range(N) for c in range(N)}))


def _zone(a: int, b: int, cut: int) -> int:
    return 0 if b < cut else 1 if a < cut else 2


def _region_palettes(rng: np.random.Generator, total: int) -> list[list[int]]:
    colors = [int(x) for x in rng.choice(np.arange(1, 10), total, replace=False)]
    capacities = [int(x) for x in rng.integers(2, 5, size=3)]
    for _ in range(max(0, total - sum(capacities))):
        available = [z for z in range(3) if capacities[z] < 4]
        capacities[int(rng.choice(available))] += 1
    regions = [[], [], []]
    for color in rng.permutation(colors):
        available = [z for z in range(3) if len(regions[z]) < capacities[z]]
        z = int(rng.choice(available))
        regions[z].append(int(color))
    for z in range(3):
        while len(regions[z]) < capacities[z]:
            choices = [color for color in colors if color not in regions[z]]
            regions[z].append(int(rng.choice(choices)))
        rng.shuffle(regions[z])
    return regions


def _make_mosaic(rng: np.random.Generator, total: int, cut: int) -> np.ndarray:
    palettes = _region_palettes(rng, total)
    wedge = np.zeros((16, 16), dtype=np.int8)
    for z, palette in enumerate(palettes):
        size = len(palette)
        motif = int(rng.integers(0, 6))
        p = int(rng.integers(2, 5))
        q = int(rng.integers(2, 5))
        phase = int(rng.integers(0, size))
        tiles = rng.integers(0, size, size=(9, 9))
        for a in range(16):
            for b in range(a, 16):
                if _zone(a, b, cut) != z:
                    continue
                if motif == 0:  # coherent blocks
                    index = int(tiles[a // p, b // q])
                elif motif == 1:  # diagonal stripes
                    index = (a + b) // p + phase
                elif motif == 2:  # nested square contours
                    index = min(a, b, 15 - a, 15 - b) // p + phase
                elif motif == 3:  # staggered block bands
                    index = a // p + 2 * (b // q) + phase
                elif motif == 4:  # diamond and step contours
                    index = max(a, b) // p + (b - a) // q + phase
                else:  # fine stripes inside wide bands
                    index = b // p + (a + b) // q + phase
                wedge[a, b] = palette[index % size]

        for _ in range(int(rng.integers(0, 4))):
            r = int(rng.integers(0, 15))
            c = int(rng.integers(r, 16))
            h = int(rng.integers(2, 7))
            w = int(rng.integers(2, 7))
            color = int(rng.choice(palette))
            for a in range(r, min(16, r + h)):
                for b in range(max(a, c), min(16, c + w)):
                    if _zone(a, b, cut) == z:
                        wedge[a, b] = color

        # Include each selected regional color without fixing an edge prefix.
        positions = [(a, b) for a in range(16) for b in range(a, 16)
                     if _zone(a, b, cut) == z]
        reserved = set()
        for color in palette:
            witnesses = [(a, b) for a, b in positions if wedge[a, b] == color
                         and (a, b) not in reserved]
            if witnesses:
                reserved.add(witnesses[int(rng.integers(0, len(witnesses)))])
            else:
                available = [cell for cell in positions if cell not in reserved]
                a, b = available[int(rng.integers(0, len(available)))]
                wedge[a, b] = color
                reserved.add((a, b))

    output = np.empty((N, N), dtype=np.int8)
    for r in range(N):
        for c in range(N):
            a, b = sorted((min(r, 31 - r), min(c, 31 - c)))
            output[r, c] = wedge[a, b]
    return output


def _rect(mask: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    mask[max(0, r):min(N, r + h), max(0, c):min(N, c + w)] = True


def _patch(mask: np.ndarray, rng: np.random.Generator, cut: int,
           location: int) -> None:
    # Each patch independently chooses a location and shape, so axes combine.
    if location == 0:  # top edge
        r, c = int(rng.integers(-2, 2)), int(rng.integers(4, 25))
    elif location == 1:  # left edge
        r, c = int(rng.integers(4, 25)), int(rng.integers(-2, 2))
    elif location == 2:  # lower area
        r, c = int(rng.integers(20, 28)), int(rng.integers(5, 23))
    elif location == 3:  # right area
        r, c = int(rng.integers(5, 23)), int(rng.integers(20, 28))
    elif location == 4:  # central axes
        r, c = int(rng.integers(12, 17)), int(rng.integers(12, 17))
    elif location == 5:  # one motif boundary
        r, c = int(rng.integers(cut - 3, cut + 2)), int(rng.integers(4, 24))
    elif location == 6:  # another motif boundary
        r, c = int(rng.integers(4, 24)), int(rng.integers(24 - cut, 29 - cut))
    else:
        r, c = int(rng.integers(3, 26)), int(rng.integers(3, 26))

    shape = int(rng.integers(0, 5))
    if shape == 0:  # broad rectangle
        h, w = int(rng.integers(4, 10)), int(rng.integers(4, 11))
    elif shape == 1:  # horizontal slit
        h, w = int(rng.integers(1, 4)), int(rng.integers(7, 14))
    elif shape == 2:  # vertical slit
        h, w = int(rng.integers(7, 14)), int(rng.integers(1, 4))
    else:
        h, w = int(rng.integers(3, 8)), int(rng.integers(3, 8))
    _rect(mask, r, c, h, w)
    if shape == 3:  # overlapping rectangles make a step
        _rect(mask, r + h // 2, c + w // 2,
              int(rng.integers(2, 7)), int(rng.integers(2, 7)))
    elif shape == 4:  # a joined L shape
        _rect(mask, r + h - 1, c,
              int(rng.integers(3, 8)), int(rng.integers(2, 5)))


def _alternate_count(mask: np.ndarray) -> int:
    count = 0
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        hr, vc = 31 - r, 31 - c
        if (hr < N and vc < N and mask[hr, c] and mask[r, vc]
                and any(not mask[x, y] for x, y in _orbit(r, c))):
            count += 1
    return count


def _make_mask(rng: np.random.Generator, regime: str,
               force_alternate: bool, cut: int) -> np.ndarray:
    mask = np.zeros((N, N), dtype=bool)
    count = int(rng.integers(1, 6 if regime == "source_train" else 7))
    for _ in range(count):
        _patch(mask, rng, cut, int(rng.integers(0, 8)))

    protected = set()
    if force_alternate:
        r = int(rng.integers(6, 12))
        c = int(rng.integers(20, 26))
        for dr in range(2):
            for dc in range(2):
                x, y = r + dr, c + dc
                mask[x, y] = True
                mask[31 - x, y] = True
                mask[x, 31 - y] = True
                protected.add((y, x))  # visible diagonal counterpart
        for x, y in protected:
            mask[x, y] = False

    # The offset crop has small edge orbits, some with only one visible cell.
    for orbit in _ORBITS:
        if all(mask[r, c] for r, c in orbit):
            choices = [cell for cell in orbit if cell not in protected]
            cell = choices[int(rng.integers(0, len(choices)))] if choices else orbit[0]
            mask[cell] = False
    return mask


def _restore(input_grid: np.ndarray) -> np.ndarray:
    result = input_grid.copy()
    for orbit in _ORBITS:
        values = {int(input_grid[r, c]) for r, c in orbit if input_grid[r, c] != 0}
        assert len(values) == 1
        value = values.pop()
        for r, c in orbit:
            if input_grid[r, c] == 0:
                result[r, c] = value
    return result


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        stack = [remaining.pop()]
        cells = set(stack)
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    cells.add(point)
        components.append(cells)
    return components


def _region_profile(output: np.ndarray) -> tuple[int, tuple[int, ...], bool]:
    candidates = []
    for cut in range(6, 10):
        colors = [set(), set(), set()]
        for a in range(16):
            for b in range(a, 16):
                colors[_zone(a, b, cut)].add(int(output[a, b]))
        candidates.append((sum(map(len, colors)), abs(cut - 8),
                           cut, colors))
    _, _, cut, colors = min(candidates, key=lambda x: x[:2])
    reused = any(colors[a] & colors[b]
                 for a in range(3) for b in range(a + 1, 3))
    return cut, tuple(map(len, colors)), reused


def _mask_profile(mask: np.ndarray, cut: int) -> dict:
    parts = _components(mask)
    rectangular = True
    thin = broad = axis_cross = zone_cross = False
    for cells in parts:
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        rectangular &= height * width == len(cells)
        thin |= min(height, width) <= 2
        broad |= min(height, width) >= 4
        axis_cross |= r0 <= 15 < r1 or c0 <= 15 < c1
        zones = set()
        for r, c in cells:
            a, b = sorted((min(r, 31 - r), min(c, 31 - c)))
            zones.add(_zone(a, b, cut))
        zone_cross |= len(zones) > 1
    lower = bool(mask[20:, 10:20].any())
    right = bool(mask[10:20, 20:].any())
    return {
        "components": len(parts),
        "nonrect": not rectangular,
        "thin": thin,
        "broad": broad,
        "top": bool(mask[0].any()),
        "left": bool(mask[:, 0].any()),
        "axis_cross": axis_cross,
        "zone_cross": zone_cross,
        "alternate": _alternate_count(mask) > 0,
        "lower": lower,
        "right": right,
        "lower_right": lower and right,
    }


def _pair_digest(pair: dict) -> str:
    return hashlib.sha256(pair["input"].astype(np.int8).tobytes()
                          + pair["output"].astype(np.int8).tobytes()).hexdigest()


def _generate_pair(rng: np.random.Generator, regime: str,
                   force_alternate: bool, total: int) -> dict:
    for _ in range(100):
        cut = int(rng.integers(6, 10))
        output = _make_mosaic(rng, total, cut)
        mask = _make_mask(rng, regime, force_alternate, cut)
        area = int(np.count_nonzero(mask))
        if not 10 <= area <= 220:
            continue
        if force_alternate and _alternate_count(mask) == 0:
            continue
        input_grid = output.copy()
        input_grid[mask] = 0
        pair = {"input": input_grid, "output": output}
        if _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS:
            return pair
    raise RuntimeError("could not construct a recoverable pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(name)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_test is None:
        maximum = 4 if num_train is None else min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))

    puzzle = {"train": [], "test": []}
    seen = set()
    demonstration = int(rng.integers(0, num_train))
    for regime, group, count in (("source_train", "train", num_train),
                                 ("target_test", "test", num_test)):
        for index in range(count):
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                total = int(pair_rng.integers(4 if group == "train" else 5, 9))
                if group == "train":
                    challenge = index == demonstration or bool(pair_rng.random() < 0.28)
                else:
                    challenge = bool(pair_rng.random() < 0.55)
                pair = _generate_pair(pair_rng, regime, challenge, total)
                signature = pair["input"].tobytes()
                if signature not in seen:
                    seen.add(signature)
                    puzzle[group].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, seen_pairs = set(), set()
    train_alternate = 0
    for group in ("train", "test"):
        for pair in puzzle[group]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape == (N, N)
            assert not np.any(out == 0)
            assert np.array_equal(inp[inp != 0], out[inp != 0])
            mask = inp == 0
            area = int(np.count_nonzero(mask))
            assert 10 <= area <= 220
            assert 4 <= len(np.unique(out)) <= 8
            if group == "test":
                assert len(np.unique(out)) >= 5
            assert np.array_equal(_restore(inp), out)
            for orbit in _ORBITS:
                assert all(out[r, c] == out[orbit[0]] for r, c in orbit)
                assert any(inp[r, c] != 0 for r, c in orbit)
            if group == "train":
                train_alternate += _alternate_count(mask)
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
    assert train_alternate > 0


def _equal_puzzles(a: dict, b: dict) -> bool:
    return all(len(a[group]) == len(b[group])
               and all(np.array_equal(left[name], right[name])
                       for left, right in zip(a[group], b[group])
                       for name in ("input", "output"))
               for group in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    palettes, areas, components, layouts, output_structures = (
        set(), set(), set(), set(), set())
    measured = {key: 0 for key in (
        "regional_three", "all_two", "reuse", "prefix_equal",
        "prefix_different", "nonrect", "thin", "broad", "top", "left",
        "both_edges", "axis_cross", "zone_cross", "alternate_train",
        "alternate_test", "lower_right", "combined")}
    within_palette = within_components = within_area = 0
    shared_components = novel_test_profile = combined_locations = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            first = puzzle
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_palettes, local_components, local_areas = set(), set(), set()
        local_profiles = {"train": [], "test": []}
        for group in ("train", "test"):
            for pair in puzzle[group]:
                inp, out = pair["input"], pair["output"]
                mask = inp == 0
                assert np.array_equal(_restore(inp), out)
                assert np.array_equal(inp[~mask], out[~mask])
                cut, region_counts, reuse = _region_profile(out)
                profile = _mask_profile(mask, cut)
                local_profiles[group].append(profile)
                palette = len(np.unique(out))
                palettes.add(palette)
                area = int(mask.sum())
                areas.add(area)
                components.add(profile["components"])
                layouts.add(mask.tobytes())
                color_map = {}
                output_structures.add(bytes(
                    color_map.setdefault(int(value), len(color_map))
                    for value in out.flat))
                local_palettes.add(palette)
                local_components.add(profile["components"])
                local_areas.add(area)
                measured["regional_three"] += max(region_counts) >= 3
                measured["all_two"] += region_counts == (2, 2, 2)
                measured["reuse"] += reuse
                prefix_equal = bool(out[0, 0] == out[0, 1] == out[0, 2])
                measured["prefix_equal"] += prefix_equal
                measured["prefix_different"] += not prefix_equal
                for name in ("nonrect", "thin", "broad", "top", "left",
                             "axis_cross", "zone_cross", "lower_right"):
                    measured[name] += profile[name]
                measured["both_edges"] += profile["top"] and profile["left"]
                measured["combined"] += all(profile[k] for k in (
                    "nonrect", "thin", "broad", "zone_cross", "alternate"))
                if group == "train":
                    measured["alternate_train"] += profile["alternate"]
                else:
                    assert palette >= 5
                    measured["alternate_test"] += profile["alternate"]
        within_palette += len(local_palettes) >= 2
        within_components += len(local_components) >= 2
        within_area += len(local_areas) >= 2
        train_profiles, test_profiles = local_profiles["train"], local_profiles["test"]
        shared_components += bool(
            {p["components"] for p in train_profiles} &
            {p["components"] for p in test_profiles})
        profile_keys = ("components", "top", "left", "axis_cross",
                        "zone_cross", "alternate", "lower_right", "nonrect")
        train_signatures = {tuple(p[k] for k in profile_keys)
                            for p in train_profiles}
        novel_test_profile += any(tuple(p[k] for k in profile_keys)
                                  not in train_signatures for p in test_profiles)
        combined_locations += (
            any(p["lower"] and not p["right"] for p in train_profiles)
            and any(p["right"] and not p["lower"] for p in train_profiles)
            and any(p["lower_right"] for p in test_profiles))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert palettes == {4, 5, 6, 7, 8}
    assert len(areas) >= 60 and min(areas) <= 20 and max(areas) >= 160
    assert len(components) >= 5
    assert len(layouts) >= 180 and len(output_structures) >= 180
    assert measured["regional_three"] >= 250 and measured["all_two"] >= 3
    assert measured["reuse"] >= 250
    assert measured["prefix_equal"] >= 40 and measured["prefix_different"] >= 40
    assert measured["nonrect"] >= 150 and measured["thin"] >= 100
    assert measured["broad"] >= 150
    assert measured["top"] >= 25 and measured["left"] >= 25
    assert measured["both_edges"] >= 5
    assert measured["axis_cross"] >= 100 and measured["zone_cross"] >= 100
    assert measured["combined"] >= 20
    assert measured["lower_right"] >= 80
    assert measured["alternate_train"] >= 42
    assert measured["alternate_test"] >= 8
    assert within_palette >= 35 and within_components >= 35 and within_area >= 35
    assert shared_components >= 20 and novel_test_profile >= 25
    assert combined_locations >= 5
    assert _equal_puzzles(first, generate_puzzle(0))
    assert not _equal_puzzles(first, generate_puzzle(1))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        p = generate_puzzle(217, *counts)
        validate_generated_puzzle(p)
        assert _equal_puzzles(p, generate_puzzle(217, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(13, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(counts)
