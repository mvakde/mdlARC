import numpy as np


_DIRECTION_ORDER = ("NW", "NE", "SE", "SW")
_VECTORS = {"NW": (-1, -1), "NE": (-1, 1), "SE": (1, 1), "SW": (1, -1)}


def _inside(row, col, height, width):
    return 0 <= row < height and 0 <= col < width


def _diamond_cells(center, radius=2):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius]


def _cap_cells(center, direction, radius=2):
    r, c = center
    vr, vc = _VECTORS[direction]
    return [(r + vr * (k + 1), c + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _background_grid(height, width, background):
    parity = np.indices((height, width)).sum(axis=0) & 1
    return np.asarray(background, dtype=np.int8)[parity]


def _infer_background(grid):
    parity = np.indices(grid.shape).sum(axis=0) & 1
    colors = []
    for p in (0, 1):
        counts = np.bincount(grid[parity == p].astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] > (parity == p).sum() // 2
        colors.append(color)
    assert colors[0] != colors[1]
    return tuple(colors)


def _analyze_input(grid):
    """Recover complete outlines and their caps using only visible pixels."""
    h, w = grid.shape
    background = _infer_background(grid)
    base = _background_grid(h, w, background)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    candidates = []
    # Vectorized outline detection avoids a Python scan over every center.
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        offsets = _diamond_cells((0, 0), radius)
        dr, dc = offsets[0]
        colors = grid[radius + dr:radius + dr + shape[0],
                      radius + dc:radius + dc + shape[1]]
        mask = (colors != background[0]) & (colors != background[1])
        for dr, dc in offsets[1:]:
            mask &= grid[radius + dr:radius + dr + shape[0],
                         radius + dc:radius + dc + shape[1]] == colors
        for rr, cc in np.argwhere(mask):
            center = (int(rr) + radius, int(cc) + radius)
            candidates.append(dict(center=center, radius=radius,
                                   struct=int(colors[rr, cc]), caps={}))
    assert 1 <= len(candidates) <= 4
    covered = set()
    for emitter in candidates:
        cells = set(_diamond_cells(emitter["center"], emitter["radius"]))
        assert not cells & covered
        covered.update(cells)
    for emitter in candidates:
        for direction in _DIRECTION_ORDER:
            visible = [p for p in _cap_cells(emitter["center"], direction, emitter["radius"])
                       if _inside(*p, h, w)]
            if len(visible) < 2:
                continue
            values = [int(grid[p]) for p in visible]
            if (len(set(values)) == 1 and values[0] not in background
                    and values[0] != emitter["struct"]):
                assert not covered.intersection(visible)
                emitter["caps"][direction] = values[0]
                covered.update(visible)
        assert emitter["caps"]
    assert covered == foreground
    candidates.sort(key=lambda e: e["center"])
    return dict(background=background, emitters=candidates)


def _paths(emitter, shape):
    """Full conceptual endpoints, including absent endpoints at clipped caps."""
    h, w = shape
    result = {}
    for direction in emitter["caps"]:
        cap = _cap_cells(emitter["center"], direction, emitter["radius"])
        vr, vc = _VECTORS[direction]
        paths = []
        for r, c in (cap[0], cap[-1]):
            cells = []
            if _inside(r, c, h, w):
                r, c = r + vr, c + vc
                while _inside(r, c, h, w):
                    cells.append((r, c))
                    r, c = r + vr, c + vc
            paths.append(cells)
        result[direction] = paths
    return result


def _derive_output(grid, analysis=None):
    if analysis is None:
        analysis = _analyze_input(grid)
    base = _background_grid(*grid.shape, analysis["background"])
    claimed = {}
    for emitter in analysis["emitters"]:
        for direction, paths in _paths(emitter, grid.shape).items():
            for path in paths:
                for cell in path:
                    assert grid[cell] == base[cell], "ray hits an input object"
                    assert cell not in claimed, "ambiguous ray ownership"
                    claimed[cell] = emitter["caps"][direction]
    assert claimed, "a whole pair must demonstrate a change"
    output = grid.copy()
    for cell, color in claimed.items():
        output[cell] = color
    return output


def _geometric_prediction(grid):
    """Separate, closed-form ray membership check, without endpoint walking.

    In a side's outward coordinates, its two rays have u-v = +/-radius.
    A point past the cap lies on a ray iff min(u,v) >= 2. This computation
    shares input segmentation, but not the renderer's cap/path construction.
    """
    analysis = _analyze_input(grid)
    base = _background_grid(*grid.shape, analysis["background"])
    rows, cols = np.indices(grid.shape)
    output = grid.copy()
    used = np.zeros(grid.shape, dtype=bool)
    for e in analysis["emitters"]:
        for direction, color in e["caps"].items():
            vr, vc = _VECTORS[direction]
            u = (rows - e["center"][0]) * vr
            v = (cols - e["center"][1]) * vc
            mask = (np.abs(u - v) == e["radius"]) & (np.minimum(u, v) >= 2)
            assert not np.any(used & mask)
            assert np.all(grid[mask] == base[mask])
            output[mask] = color
            used |= mask
    return output


def _footprint(emitter, shape):
    h, w = shape
    cells = set(_diamond_cells(emitter["center"], emitter["radius"]))
    if not all(_inside(*p, h, w) for p in cells):
        return None
    for direction in emitter["caps"]:
        cap = [p for p in _cap_cells(emitter["center"], direction, emitter["radius"])
               if _inside(*p, h, w)]
        if len(cap) < 2:
            return None
        cells.update(cap)
    rays = {p for paths in _paths(emitter, shape).values() for path in paths for p in path}
    return cells, rays


def _teaching_evidence(grid, output, analysis):
    """Directions with both endpoints continued and interior lanes left empty."""
    witnesses = set()
    for e in analysis["emitters"]:
        for direction, paths in _paths(e, grid.shape).items():
            if min(map(len, paths)) < 2:
                continue
            vr, vc = _VECTORS[direction]
            middle = _cap_cells(e["center"], direction, e["radius"])[1:-1]
            probes = [(r + vr * k, c + vc * k) for r, c in middle for k in (1, 2)]
            if all(_inside(*p, *grid.shape) and grid[p] == output[p]
                   and int(grid[p]) in analysis["background"] for p in probes):
                witnesses.add((direction, e["radius"]))
    return witnesses


def _render_pair(height, width, background, specs):
    grid = _background_grid(height, width, background)
    occupied, paths = set(), set()
    for e in specs:
        footprint = _footprint(e, grid.shape)
        assert footprint is not None
        cells, rays = footprint
        assert not (cells & (occupied | paths) or rays & (occupied | paths))
        occupied.update(cells)
        paths.update(rays)
        assert e["struct"] not in background
        for p in _diamond_cells(e["center"], e["radius"]):
            grid[p] = e["struct"]
        for direction, color in e["caps"].items():
            assert color not in (*background, e["struct"])
            for p in _cap_cells(e["center"], direction, e["radius"]):
                if _inside(*p, height, width):
                    grid[p] = color
    analysis = _analyze_input(grid)
    assert analysis["emitters"] == sorted(specs, key=lambda e: e["center"])
    return {"input": grid, "output": _derive_output(grid, analysis)}


def _sample_colors(rng, background, masks, params):
    palette = [c for c in range(10) if c not in background]
    rng.shuffle(palette)
    n = len(masks)
    if params["simple"]:
        structs = [params["struct"]] * n
    elif rng.random() < .3:
        structs = [palette[0]] * n
    else:
        bank = palette[:int(rng.integers(2, min(n, 4) + 1))] if n > 1 else palette[:1]
        structs = [bank[i % len(bank)] for i in range(n)]
        rng.shuffle(structs)
    # Choose the extent of cross-role reuse independently of object placement.
    # Selected structural colors will actually occur as another object's cap.
    struct_set = set(structs)
    reuse_count = (int(rng.integers(1, len(struct_set) + 1))
                   if len(struct_set) > 1 and rng.random() < .7 else 0)
    allowed_reuse = set(map(int, rng.choice(sorted(struct_set), reuse_count, replace=False)))
    result = []
    for structural, directions in zip(structs, masks):
        choices = [c for c in palette if c != structural
                   and (c not in struct_set or c in allowed_reuse)]
        count = int(rng.integers(1, min(len(choices), len(directions)) + 1))
        bank = list(map(int, rng.choice(choices, count, replace=False)))
        payloads = bank + [int(rng.choice(bank)) for _ in range(len(directions) - count)]
        rng.shuffle(payloads)
        if params["simple"]:
            payloads = [bank[0]] * len(directions)
        result.append((int(structural), dict(zip(directions, payloads))))
    used_slots = set()
    for color in sorted(allowed_reuse):
        slots = [(i, d) for i, (structural, caps) in enumerate(result)
                 for d in caps if structural != color and (i, d) not in used_slots]
        # Every emitter has a cap; with at least two structural colors there
        # are enough locally valid slots, even for four distinct reused colors.
        if not slots:
            continue
        slot = slots[int(rng.integers(len(slots)))]
        result[slot[0]][1][slot[1]] = color
        used_slots.add(slot)
    return result


def _dimensions(rng, n, radii, teaching):
    low = max((9, 16, 21, 26)[n - 1], 2 * max(radii) + 3)
    if teaching:
        low = max(low, 2 * radii[0] + 7)
    if n == 1 and not teaching and rng.random() < .32:
        if radii[0] == 2 and rng.random() < .3:
            return (9, 9)
        return tuple(int(rng.integers(low, max(low, 14) + 1)) for _ in range(2))
    shapes = [(20, 14), (24, 30), (22, 21), (30, 30)]
    suitable = [s for s in shapes if min(s) >= low]
    if suitable and rng.random() < .18:
        return suitable[int(rng.integers(len(suitable)))]
    return tuple(30 if rng.random() < .12 else int(rng.integers(low, 31)) for _ in range(2))


def _sample_pair(rng, params, n, teaching=None, zero_case=False):
    # Side subsets are chosen before locations, for either split and every count.
    masks = []
    for i in range(n):
        mask = int(rng.integers(1, 16))
        if params["simple"]:
            mask = int(rng.choice([5, 10, 15]))
        masks.append([d for j, d in enumerate(_DIRECTION_ORDER) if mask & (1 << j)])
    radii = [int(rng.integers(2, params["max_radius"] + 1)) for _ in range(n)]
    if teaching:
        masks[0] = list(teaching[0])
        radii[0] = teaching[1]
    zero_direction = _DIRECTION_ORDER[int(rng.integers(4))]
    if zero_case and not teaching:
        # The cap touches two edges: neither endpoint has a next cell.
        masks[0] = list(dict.fromkeys(masks[0] + [zero_direction,
                            _DIRECTION_ORDER[(_DIRECTION_ORDER.index(zero_direction) + 2) % 4]]))
    background = params["background"] if params["fixed_background"] else tuple(
        map(int, rng.choice(10, 2, replace=False)))
    color_specs = _sample_colors(rng, background, masks, params)
    shape = _dimensions(rng, n, radii, teaching)
    clipping = float(rng.choice([0., .2, .55, .85]))
    # Bounded search retains the requested count, masks, and sizes on retries.
    for restart in range(16):
        h, w = shape if restart < 8 else (30, 30)
        specs, occupied, rays = [], set(), set()
        for i in range(n):
            radius = radii[i]
            structural, caps = color_specs[i]
            for attempt in range(180):
                if i == 0 and teaching:
                    center = (int(rng.integers(radius + 3, h - radius - 3)),
                              int(rng.integers(radius + 3, w - radius - 3)))
                elif i == 0 and zero_case:
                    vr, vc = _VECTORS[zero_direction]
                    center = (radius + 1 if vr < 0 else h - radius - 2,
                              radius + 1 if vc < 0 else w - radius - 2)
                else:
                    pos = []
                    for extent in (h, w):
                        if rng.random() < clipping:
                            edge = radius + int(rng.integers(2))
                            pos.append(edge if rng.random() < .5 else extent - 1 - edge)
                        else:
                            pos.append(int(rng.integers(radius, extent - radius)))
                    center = tuple(pos)
                e = dict(center=center, radius=radius, struct=structural, caps=caps)
                if any(sum(abs(a - b) for a, b in zip(center, old["center"]))
                       < radius + old["radius"] + 5 for old in specs):
                    continue
                fp = _footprint(e, (h, w))
                if fp is None:
                    continue
                cells, new_rays = fp
                if cells & (occupied | rays) or new_rays & (occupied | rays):
                    continue
                specs.append(e)
                occupied.update(cells)
                rays.update(new_rays)
                break
            else:
                break
        if len(specs) != n:
            continue
        try:
            pair = _render_pair(h, w, background, specs)
            if teaching:
                evidence = _teaching_evidence(pair["input"], pair["output"], _analyze_input(pair["input"]))
                if not {(d, radii[0]) for d in masks[0]} <= evidence:
                    continue
            if _pair_key(pair) not in _SOURCE_KEYS:
                return pair
        except AssertionError:
            continue
    # A deterministic geometric fallback: one clearly demonstrated emitter.
    # It is used only after the complete bounded free-placement search fails.
    h = w = 30
    radius = radii[0]
    spec = dict(center=(14, 15), radius=radius, struct=color_specs[0][0], caps=color_specs[0][1])
    return _render_pair(h, w, background, [spec])


def _count_value(
    value: int,
    name: str,
    low: int,
    high: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} is outside its legal range"
        )

    return value


def _select_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        num_train = _count_value(
            num_train,
            "num_train",
            2,
            10,
        )

    if num_test is not None:
        num_test = _count_value(
            num_test,
            "num_test",
            1,
            4,
        )

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(num_test + 1, 11)
        )

    elif num_train is None:
        num_train = int(
            rng.integers(
                max(2, num_test + 1),
                11,
            )
        )

    elif num_test is None:
        upper = min(4, num_train - 1)

        if upper < 1:
            raise ValueError(
                "no legal num_test for the requested num_train"
            )

        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _grid_key(grid: np.ndarray) -> tuple:
    # Values are already restricted to 0..9, so normalizing to int8 makes
    # duplicate detection independent of the particular integer dtype.
    normalized = grid.astype(np.int8, copy=False)
    return normalized.shape, normalized.tobytes()




def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


# Exact source grids serve only as regression fixtures and copy exclusions.
_SOURCE_ROWS = (
    (
        '10101010101010/01010101010101/10101010101010/01010101010101/10101010101010/01010101010101/10101010101010/01010108010101/10101030801010/01010303080101/10103010301010/01080303010101/10108030101010/01010801010101/10101010101010/01010101010101/10101010101010/01010101010101/10101010101010/01010101010101',
        '10101010101010/01010101010108/10101010101080/01010101010801/10101010108010/01010101080108/10101010801080/01010108010801/10101030808010/01010303080101/10103010301010/01080303010101/10808030101010/08010801010101/80108010101010/01080101010101/10801010101010/08010101010101/80101010101010/01010101010101',
    ),
    (
        '101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010204010101010101010101/101010102030401010101010101010/010101020303040101010101010101/101010103010301010101010101010/010101040303020101010101010101/101010104030201010101010101010/010101010402010101010101010101/101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010101010101010101010101/101010101010101010101010101010/010101010101010101010101010101',
        '102010201010104010401010101010/010201020101040104010101010101/101020102010401040101010101010/010102010204010401010101010101/101010202030404010101010101010/010101020303040101010101010101/101010103010301010101010101010/010101040303020101010101010101/101010404030202010101010101010/010104010402010201010101010101/101040104010201020101010101010/010401040101020102010101010101/104010401010102010201010101010/040104010101010201020101010101/401040101010101020102010101010/010401010101010102010201010101/104010101010101010201020101010/040101010101010101020102010101/401010101010101010102010201010/010101010101010101010201020101/101010101010101010101020102010/010101010101010101010102010201/101010101010101010101010201020/010101010101010101010101020102',
    ),
    (
        '101010101010101010101/010101010101010101010/101010204010101010101/010102060401010101010/101020606040101010101/010106010601010101010/101090606010101010101/010109060101010101010/101010901010101010101/010101010101010101010/101010101010101010101/010101010101060101010/101010101010306010101/010101010103030601010/101010101030103010101/010101010103030701010/101010101010307010101/010102010101070101010/101020201010101010101/010201020101010101010/106020206010101010101/010602060101010101010',
        '201020101040104010101/020102010401040101010/102010204010401010101/010202060404010101010/101020606040101010106/010106010601010101060/101090606010101010601/010909060101010106010/109010901010101060106/090109010101010601060/901090101010106010601/010901010101060106010/109010101010306060101/090101010103030601010/901010101030103010101/010101010103030701010/101010101010307070101/010102010101070107010/101020201010107010701/010201020101010701070/106020206010101070107/060602060601010107010',
    ),
)
_SOURCE_PAIRS = [dict(zip(("input", "output"),
    [np.array([[int(c) for c in row] for row in text.split("/")], dtype=np.int8)
     for text in rows])) for rows in _SOURCE_ROWS]
_SOURCE_KEYS = {_pair_key(p) for p in _SOURCE_PAIRS}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _select_counts(rng, num_train, num_test)
    background = tuple(map(int, rng.choice(10, 2, replace=False)))
    simple = rng.random() < .16
    params = dict(background=background, simple=simple,
                  fixed_background=simple or rng.random() < .22,
                  struct=int(rng.choice([c for c in range(10) if c not in background])),
                  max_radius=2 if simple else int(rng.choice([2, 3, 4], p=[.2, .35, .45])))
    # Geometry is the puzzle-wide convention; palettes are never secret codes.
    # Two shuffled teaching slots jointly cover every direction at maximum size.
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    first_mask = int(rng.choice([5, 10])) if simple else int(rng.integers(1, 15))
    second_mask = (15 ^ first_mask) | int(rng.integers(0, 16))
    if simple:
        second_mask = 15
    teaching = {slots[0]: ([d for j, d in enumerate(_DIRECTION_ORDER) if first_mask & (1 << j)], params["max_radius"]),
                slots[1]: ([d for j, d in enumerate(_DIRECTION_ORDER) if second_mask & (1 << j)], params["max_radius"])}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            witness = teaching.get(index) if split == "train" else None
            if simple:
                n = 1 if split == "train" else int(pair_rng.integers(1, 4))
            else:
                n = int(pair_rng.integers(1, 5))
            zero_case = not witness and pair_rng.random() < .14
            for _ in range(40):
                pair = _sample_pair(pair_rng, params, n, witness, zero_case)
                key = _grid_key(pair["input"])
                if key not in seen and _pair_key(pair) not in _SOURCE_KEYS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded unique-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, evidence = set(), set(), set()
    test_radii = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _pair_key(pair)
            assert key[0] not in inputs and key not in pairs and key not in _SOURCE_KEYS
            inputs.add(key[0])
            pairs.add(key)
            analysis = _analyze_input(inp)
            assert np.array_equal(out, _derive_output(inp, analysis))
            assert np.array_equal(out, _geometric_prediction(inp))
            base = _background_grid(*inp.shape, analysis["background"])
            foreground = inp != base
            assert np.array_equal(out[foreground], inp[foreground])
            if split == "train":
                evidence.update(_teaching_evidence(inp, out, analysis))
            else:
                test_radii.update(e["radius"] for e in analysis["emitters"])
    assert {d for d, _ in evidence} == set(_DIRECTION_ORDER)
    # The scalable endpoint rule has visible interior-lane counterexamples at
    # least as large as every tested cap, in all four directions.
    for direction in _DIRECTION_ORDER:
        assert max(r for d, r in evidence if d == direction) >= max(test_radii)


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(_pair_key(a) == _pair_key(b) for a, b in zip(first[s], second[s]))
               for s in ("train", "test"))


def _measure_pair(pair):
    """Task features reconstructed from pixels, never generation metadata."""
    grid = pair["input"]
    analysis = _analyze_input(grid)
    emitters = analysis["emitters"]
    lengths, subsets, clipped_caps, clipped_emitters, zero_caps = [], [], 0, 0, 0
    boundary_complete = 0
    for e in emitters:
        clipped_here = 0
        subsets.append(sum(1 << _DIRECTION_ORDER.index(d) for d in e["caps"]))
        for direction, paths in _paths(e, grid.shape).items():
            lengths.extend(map(len, paths))
            zero_caps += int(not any(paths))
            full = _cap_cells(e["center"], direction, e["radius"])
            clipped = any(not _inside(*p, *grid.shape) for p in full)
            clipped_here += int(clipped)
            if not clipped and any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in full):
                boundary_complete += 1
        clipped_caps += clipped_here
        clipped_emitters += int(clipped_here > 0)
    structs = {e["struct"] for e in emitters}
    caps = {c for e in emitters for c in e["caps"].values()}
    overlap = len(structs & caps)
    adjacent_repeats = unequal_opposites = 0
    for e in emitters:
        for j, direction in enumerate(_DIRECTION_ORDER):
            adjacent = _DIRECTION_ORDER[(j + 1) % 4]
            opposite = _DIRECTION_ORDER[(j + 2) % 4]
            if direction in e["caps"]:
                adjacent_repeats += int(adjacent in e["caps"] and
                                        e["caps"][direction] == e["caps"][adjacent])
                unequal_opposites += int(opposite in e["caps"] and
                                         e["caps"][direction] != e["caps"][opposite])
    return dict(count=len(emitters), radii=[e["radius"] for e in emitters],
                subsets=subsets, ray_lengths=lengths, clipped_caps=clipped_caps,
                clipped_emitters=clipped_emitters, zero_caps=zero_caps,
                boundary_complete=boundary_complete,
                background=analysis["background"], structs=sorted(structs),
                reuse=overlap, reuse_category="none" if not overlap else
                    ("full" if overlap == len(structs) else "partial"),
                four_cap_colors=[len(set(e["caps"].values())) for e in emitters if len(e["caps"]) == 4],
                adjacent_repeats=adjacent_repeats, unequal_opposites=unequal_opposites,
                shape=list(grid.shape), changed=int(np.count_nonzero(grid != pair["output"])),
                layout=(grid.shape, tuple((e["center"], e["radius"], mask) for e, mask in zip(emitters, subsets))))


def self_test() -> None:
    # Regression labels are original source evidence, not generated predictions.
    for source in _SOURCE_PAIRS:
        assert np.array_equal(_derive_output(source["input"]), source["output"])
        assert np.array_equal(_geometric_prediction(source["input"]), source["output"])
    fresh_source_sizes = [_render_pair(*shape, (5, 8), [dict(
        center=(8, 7), radius=2, struct=1,
        caps={"NW": 2, "NE": 6, "SE": 9, "SW": 4})])
        for shape in ((20, 14), (24, 30), (22, 21))]
    validate_generated_puzzle(dict(train=fresh_source_sizes[:2], test=fresh_source_sizes[2:]))
    # Explicit edge regression: the NW middle cell must not grow into (0,0).
    edge = _render_pair(13, 15, (1, 0), [dict(center=(3, 3), radius=2, struct=3,
                                            caps={"NW": 4, "SE": 8})])
    assert edge["output"][0, 0] == edge["input"][0, 0] == 1
    assert _measure_pair(edge)["zero_caps"] == 1
    assert np.array_equal(_geometric_prediction(edge["input"]), edge["output"])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                   (None, 4), (2, None), (10, None)), 100):
        p = generate_puzzle(seed, *counts)
        validate_generated_puzzle(p)
        assert all(n is None or len(p[s]) == n for s, n in zip(("train", "test"), counts))
        assert _puzzles_equal(p, generate_puzzle(seed, *counts))
    assert _puzzles_equal(puzzles[0], generate_puzzle(0))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(42, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal count arguments")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    features = {s: [_measure_pair(pair) for p in puzzles for pair in p[s]] for s in ("train", "test")}
    all_features = features["train"] + features["test"]
    for s, rows in features.items():
        assert {f["count"] for f in rows} == {1, 2, 3, 4}
        assert {m for f in rows for m in f["subsets"]} == set(range(1, 16))
        assert {r for f in rows for r in f["radii"]} == {2, 3, 4}
        assert max(max(f["ray_lengths"]) for f in rows) >= 18
        assert any(f["count"] >= 3 and max(f["ray_lengths"]) >= 10 for f in rows)
        assert any(f["count"] > 1 and f["clipped_emitters"] == 0 for f in rows)
        assert any(f["clipped_emitters"] >= 2 for f in rows)
        assert {n for f in rows for n in f["four_cap_colors"]} == {1, 2, 3, 4}
        assert any(f["shape"] == [30, 30] for f in rows)
    assert len({f["layout"] for f in all_features}) >= 200
    assert len({tuple(f["shape"]) for f in all_features}) >= 70
    assert min(min(f["shape"]) for f in all_features) == 9
    assert {f["reuse_category"] for f in all_features} == {"none", "partial", "full"}
    assert max(f["reuse"] for f in all_features) >= 3
    assert sum(f["zero_caps"] for f in all_features) >= 10
    assert sum(f["boundary_complete"] for f in all_features) >= 20
    assert any(len(set(f["radii"])) > 1 for f in all_features)
    assert sum(max(f["shape"]) <= 14 for f in all_features) >= 10
    assert sum(f["adjacent_repeats"] for f in all_features) >= 20
    assert sum(f["unequal_opposites"] for f in all_features) >= 20
    assert any(f["count"] > 1 and len(f["structs"]) == 1 for f in all_features)
    assert any(f["count"] > 1 and len(f["structs"]) > 1 for f in all_features)
    varied_backgrounds = mixed_train_counts = combined = varied_structs = phase_orders = 0
    for p in puzzles:
        rows = [_measure_pair(pair) for s in ("train", "test") for pair in p[s]]
        varied = len({f["background"] for f in rows}) > 1
        varied_backgrounds += varied
        mixed_train_counts += len({_measure_pair(pair)["count"] for pair in p["train"]}) > 1
        varied_structs += len({tuple(_measure_pair(pair)["structs"]) for pair in p["train"]}) > 1
        phase_orders += len({f["background"][0] < f["background"][1] for f in rows}) > 1
        assert len({tuple(f["shape"]) for f in rows}) > 1
        combined += varied and any(f["count"] >= 3 and len(set(f["radii"])) > 1 and
                                   max(f["ray_lengths"]) >= 10 and
                                   any(mask.bit_count() >= 3 for mask in f["subsets"]) for f in rows)
    assert varied_backgrounds >= 20 and mixed_train_counts >= 20 and combined >= 5
    assert varied_structs >= 20 and phase_orders >= 15
    # Deliberate label damage must fail independently of the sampler.
    bad = {s: [{role: g.copy() for role, g in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    changed = np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0]
    cell = tuple(changed)
    bad["test"][0]["output"][cell] = bad["test"][0]["input"][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a missing ray cell")
