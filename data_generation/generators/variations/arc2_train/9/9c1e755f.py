import numpy as np


_SOURCE_PAIR_KEYS = {
    (
        "0000000000"
        "0555555000"
        "0200000000"
        "0100000000"
        "0200000000"
        "0600000000"
        "0100000000"
        "0100000000"
        "0000000000"
        "0000000000",
        "0000000000"
        "0555555000"
        "0222222000"
        "0111111000"
        "0222222000"
        "0666666000"
        "0111111000"
        "0111111000"
        "0000000000"
        "0000000000",
    ),
    (
        "0000000000"
        "5000000000"
        "5000000000"
        "5000000000"
        "5000000000"
        "5000000000"
        "5000000000"
        "5433444000"
        "5737733000"
        "0000000000",
        "0000000000"
        "5433444000"
        "5737733000"
        "5433444000"
        "5737733000"
        "5433444000"
        "5737733000"
        "5433444000"
        "5737733000"
        "0000000000",
    ),
    (
        "0000005555"
        "5000003000"
        "5000003000"
        "5000007000"
        "5000003000"
        "5000007000"
        "5000007000"
        "5636300000"
        "5222200000"
        "0000000000",
        "0000005555"
        "5636303333"
        "5222203333"
        "5636307777"
        "5222203333"
        "5636307777"
        "5222207777"
        "5636300000"
        "5222200000"
        "0000000000",
    ),
    (
        "0000000000"
        "0000000000"
        "0050000000"
        "0050000000"
        "0050000000"
        "0056669990"
        "0000000000"
        "0000000000"
        "0000000000"
        "0000000000",
        "0000000000"
        "0000000000"
        "0056669990"
        "0056669990"
        "0056669990"
        "0056669990"
        "0000000000"
        "0000000000"
        "0000000000"
        "0000000000",
    ),
    (
        "0000000000"
        "5555505000"
        "4000005000"
        "2000005000"
        "2000005000"
        "2000005000"
        "1000005000"
        "0000005868"
        "0000005333"
        "0000000000",
        "0000000000"
        "5555505868"
        "4444405333"
        "2222205868"
        "2222205333"
        "2222205868"
        "1111105333"
        "0000005868"
        "0000005333"
        "0000000000",
    ),
}


def _validate_count_value(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_value("num_train", num_train, 2, 10)
    _validate_count_value("num_test", num_test, 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    else:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("no legal pair-count combination can be selected")

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    return {"background": colors[0], "scaffold": colors[1],
            "payload_colors": tuple(colors[2:])}


def _tile(rng: np.random.Generator, orientation: str,
          period: int, depth: int, palette: tuple[int, ...]) -> np.ndarray:
    shape = (depth, period) if orientation == "H" else (period, depth)
    tile = np.empty(shape, dtype=np.int8)
    # A piecewise random tile gives runs and irregular shapes without a color formula.
    for r in range(shape[0]):
        for c in range(shape[1]):
            if c and rng.random() < .35:
                tile[r, c] = tile[r, c - 1]
            elif r and rng.random() < .35:
                tile[r, c] = tile[r - 1, c]
            else:
                tile[r, c] = int(rng.choice(palette))
    axial = tile.T if orientation == "H" else tile
    if len(np.unique(tile)) < 2 or len({row.tobytes() for row in axial}) != period:
        # Force a visible motif and distinct phase slices. This also guarantees
        # a primitive period for the supported periods of one, two, and three.
        tile.flat[0] = palette[0]
        tile.flat[-1] = palette[-1]
        axial = tile.T if orientation == "H" else tile
        for phase in range(1, period):
            axial[phase, 0] = palette[phase % len(palette)]
            if np.array_equal(axial[phase], axial[0]):
                axial[phase, -1] = palette[(phase + 1) % len(palette)]
    return tile


def _make_object(rng: np.random.Generator, orientation: str,
                 period: int, palette: tuple[int, ...],
                 compact: bool) -> dict:
    low = max(4, period + 2)
    high = 9 if compact else 15
    length = int(rng.integers(low, high + 1))
    depth = int(rng.integers(2, 10 if compact else 11))
    if rng.random() < .2:
        depth = int(rng.integers(max(2, length - 1), min(12, length + 3)))
    side = 1 if rng.random() < .5 else -1
    anchor = 1 if rng.random() < .5 else -1
    # The terminal tile may protrude past its bar as in source train 3.
    overhang = int(rng.integers(0, period)) if period > 1 else 0
    tile_start = -overhang if anchor == 1 else length - period + overhang
    axis_start = min(0, tile_start)
    axis_end = max(length - 1, tile_start + period - 1)
    axis_size = axis_end - axis_start + 1
    tile = _tile(rng, orientation, period, depth, palette)
    return {
        "orientation": orientation, "period": period, "length": length,
        "depth": depth, "side": side, "anchor": anchor,
        "overhang": overhang, "tile": tile,
        "bar_offset": -axis_start, "tile_offset": tile_start - axis_start,
        "bbox_h": depth + 1 if orientation == "H" else axis_size,
        "bbox_w": axis_size if orientation == "H" else depth + 1,
    }


def _pack_objects(rng: np.random.Generator, objects: list[dict],
                  compact: bool) -> tuple[int, int, list[tuple[int, int]]]:
    if len(objects) == 1:
        core_h, core_w = objects[0]["bbox_h"], objects[0]["bbox_w"]
        offsets = [(0, 0)]
    else:
        a, b = objects
        gap = int(rng.integers(1, 4))
        choices = []
        if a["bbox_w"] + gap + b["bbox_w"] <= 30:
            choices.append((max(a["bbox_h"], b["bbox_h"]),
                            a["bbox_w"] + gap + b["bbox_w"],
                            [(0, 0), (0, a["bbox_w"] + gap)], "side"))
        if a["bbox_h"] + gap + b["bbox_h"] <= 30:
            choices.append((a["bbox_h"] + gap + b["bbox_h"],
                            max(a["bbox_w"], b["bbox_w"]),
                            [(0, 0), (a["bbox_h"] + gap, 0)], "stack"))
        if not choices:
            raise ValueError("objects do not fit")
        core_h, core_w, offsets, arrangement = choices[int(rng.integers(len(choices)))]
        if arrangement == "side":
            offsets[0] = (int(rng.integers(0, core_h - a["bbox_h"] + 1)), 0)
            offsets[1] = (int(rng.integers(0, core_h - b["bbox_h"] + 1)), offsets[1][1])
        else:
            offsets[0] = (0, int(rng.integers(0, core_w - a["bbox_w"] + 1)))
            offsets[1] = (offsets[1][0], int(rng.integers(0, core_w - b["bbox_w"] + 1)))
    if core_h > 30 or core_w > 30:
        raise ValueError("object exceeds canvas")
    margin_cap = 2 if compact else 5
    # Select each margin separately so zero-border and off-centre grids occur.
    top = int(rng.integers(0, min(margin_cap, 30 - core_h) + 1))
    bottom = int(rng.integers(0, min(margin_cap, 30 - core_h - top) + 1))
    left = int(rng.integers(0, min(margin_cap, 30 - core_w) + 1))
    right = int(rng.integers(0, min(margin_cap, 30 - core_w - left) + 1))
    return (core_h + top + bottom, core_w + left + right,
            [(r + top, c + left) for r, c in offsets])


def _paint_object(inp: np.ndarray, out: np.ndarray, obj: dict,
                  top: int, left: int, scaffold: int) -> None:
    p, length, depth = obj["period"], obj["length"], obj["depth"]
    bo, to = obj["bar_offset"], obj["tile_offset"]
    tile = obj["tile"]
    if obj["orientation"] == "H":
        bar_r = top if obj["side"] == 1 else top + depth
        band_r = top + 1 if obj["side"] == 1 else top
        inp[bar_r, left + bo:left + bo + length] = scaffold
        out[bar_r, left + bo:left + bo + length] = scaffold
        inp[band_r:band_r + depth, left + to:left + to + p] = tile
        for c in range(obj["bbox_w"]):
            out[band_r:band_r + depth, left + c] = tile[:, (c - to) % p]
    else:
        bar_c = left if obj["side"] == 1 else left + depth
        band_c = left + 1 if obj["side"] == 1 else left
        inp[top + bo:top + bo + length, bar_c] = scaffold
        out[top + bo:top + bo + length, bar_c] = scaffold
        inp[top + to:top + to + p, band_c:band_c + depth] = tile
        for r in range(obj["bbox_h"]):
            out[top + r, band_c:band_c + depth] = tile[(r - to) % p]


def _generate_pair(rng: np.random.Generator, regime: str,
                   params: dict, blueprint: list[tuple[str, int]]) -> dict:
    compact = bool(rng.random() < (.42 if regime == "source_train" else .27))
    pool = (params["payload_colors"][:6] if regime == "source_train"
            else params["payload_colors"])
    objects = []
    shared = bool(len(blueprint) == 2 and rng.random() < .45)
    shared_color = int(rng.choice(pool)) if shared else None
    for orientation, period in blueprint:
        size = int(rng.integers(2, 5))
        choices = [int(x) for x in rng.permutation(pool)]
        if shared_color is not None:
            palette = tuple([shared_color] + [x for x in choices if x != shared_color][:size - 1])
        else:
            palette = tuple(choices[:size])
        objects.append(_make_object(rng, orientation, period, palette, compact))
    height, width, positions = _pack_objects(rng, objects, compact)
    inp = np.full((height, width), params["background"], dtype=np.int8)
    out = inp.copy()
    for obj, (top, left) in zip(objects, positions):
        _paint_object(inp, out, obj, top, left, params["scaffold"])
    return {"input": inp, "output": out}


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            cells, stack = [], [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            components.append(cells)
    return components


def _parse_input(grid: np.ndarray, background: int,
                 scaffold: int) -> tuple[list[dict], np.ndarray]:
    bars = _components(grid == scaffold)
    tiles = _components((grid != background) & (grid != scaffold))
    if not bars or len(bars) != len(tiles) or len(bars) > 2:
        raise ValueError("object count or roles invalid")
    expected = grid.copy()
    objects, used = [], set()
    h, w = grid.shape
    for bar in bars:
        rows = {r for r, _ in bar}
        cols = {c for _, c in bar}
        if len(rows) == 1:
            ori, fixed, axis = "H", next(iter(rows)), sorted(cols)
        elif len(cols) == 1:
            ori, fixed, axis = "V", next(iter(cols)), sorted(rows)
        else:
            raise ValueError("bar not straight")
        if len(axis) < 4 or axis != list(range(axis[0], axis[-1] + 1)):
            raise ValueError("bar not contiguous")
        adjacent = []
        for k, tile_cells in enumerate(tiles):
            tile_set = set(tile_cells)
            if any((r + dr, c + dc) in tile_set for r, c in bar
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                adjacent.append(k)
        if len(adjacent) != 1 or adjacent[0] in used:
            raise ValueError("tile association ambiguous")
        k = adjacent[0]
        used.add(k)
        cells = tiles[k]
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise ValueError("tile not rectangular")
        tile = grid[r0:r1 + 1, c0:c1 + 1]
        if ori == "H":
            if (r1 == fixed - 1) == (r0 == fixed + 1):
                raise ValueError("tile side ambiguous")
            period, depth = c1 - c0 + 1, r1 - r0 + 1
            left = c0 <= axis[0] <= c1 and c1 < axis[-1]
            right = c0 > axis[0] and c0 <= axis[-1] <= c1
            if left == right:
                raise ValueError("tile not terminal")
            extent = range(min(axis[0], c0), max(axis[-1], c1) + 1)
            for c in extent:
                section = expected[r0:r1 + 1, c]
                motif = tile[:, (c - c0) % period]
                if np.any((section != background) & (section != motif)):
                    raise ValueError("overlapping output bands")
                expected[r0:r1 + 1, c] = motif
            side, anchor = (-1 if r1 == fixed - 1 else 1), (1 if left else -1)
        else:
            if (c1 == fixed - 1) == (c0 == fixed + 1):
                raise ValueError("tile side ambiguous")
            period, depth = r1 - r0 + 1, c1 - c0 + 1
            top = r0 <= axis[0] <= r1 and r1 < axis[-1]
            bottom = r0 > axis[0] and r0 <= axis[-1] <= r1
            if top == bottom:
                raise ValueError("tile not terminal")
            extent = range(min(axis[0], r0), max(axis[-1], r1) + 1)
            for r in extent:
                section = expected[r, c0:c1 + 1]
                motif = tile[(r - r0) % period]
                if np.any((section != background) & (section != motif)):
                    raise ValueError("overlapping output bands")
                expected[r, c0:c1 + 1] = motif
            side, anchor = (-1 if c1 == fixed - 1 else 1), (1 if top else -1)
        if period > 3 or period < 1 or depth < 2 or period >= len(axis):
            raise ValueError("unsupported or ambiguous tile")
        axial = tile.T if ori == "H" else tile
        if len(np.unique(tile)) < 2 or len({row.tobytes() for row in axial}) != period:
            raise ValueError("phase not visible")
        objects.append({"orientation": ori, "period": period, "depth": depth,
                        "length": len(axis), "side": side, "anchor": anchor,
                        "overhang": (axis[0] - c0 if ori == "H" and left else
                                     c1 - axis[-1] if ori == "H" else
                                     axis[0] - r0 if top else r1 - axis[-1]),
                        "colors": frozenset(int(x) for x in np.unique(tile)),
                        "tile": tile.copy(), "edge": r0 == 0 or c0 == 0 or
                        r1 == h - 1 or c1 == w - 1 or fixed == 0 or fixed == (h - 1 if ori == "H" else w - 1)})
    if len(used) != len(tiles):
        raise ValueError("unused tile")
    return objects, expected


def _infer_roles_and_parse(puzzle: dict) -> tuple[int, int, list[list[dict]], list[list[dict]]]:
    pairs = puzzle["train"] + puzzle["test"]
    # Background is the modal input color across the puzzle; this is a
    # visible convention and remains stable even when an output band is broad.
    counts = np.zeros(10, dtype=np.int64)
    for pair in pairs:
        counts += np.bincount(pair["input"].ravel().astype(np.int64), minlength=10)
    candidates = []
    for background in range(10):
        if counts[background] != counts.max():
            continue
        for scaffold in range(10):
            if scaffold == background:
                continue
            try:
                parsed = []
                for pair in pairs:
                    objects, expected = _parse_input(pair["input"], background, scaffold)
                    if not np.array_equal(expected, pair["output"]):
                        raise ValueError("wrong output")
                    parsed.append(objects)
            except ValueError:
                continue
            candidates.append((background, scaffold, parsed))
    if len(candidates) != 1:
        raise AssertionError("roles or outputs are not uniquely determined")
    background, scaffold, parsed = candidates[0]
    return background, scaffold, parsed[:len(puzzle["train"])], parsed[len(puzzle["train"]):]


def _digits(grid: np.ndarray) -> str:
    return "".join(str(int(x)) for x in grid.ravel())


def _is_source_pair(pair: dict) -> bool:
    return pair["input"].shape == (10, 10) and pair["output"].shape == (10, 10) and \
        (_digits(pair["input"]), _digits(pair["output"])) in _SOURCE_PAIR_KEYS


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((grid >= 0) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        assert not np.array_equal(pair["input"], pair["output"])
        assert _grid_key(pair["input"]) not in inputs and _pair_key(pair) not in pairs
        assert not _is_source_pair(pair)
        inputs.add(_grid_key(pair["input"]))
        pairs.add(_pair_key(pair))
    background, scaffold, train, test = _infer_roles_and_parse(puzzle)
    assert background != scaffold
    assert all(1 <= len(objects) <= 2 for objects in train + test)
    # Two foundational demonstrations expose extrusion and phase. Other
    # arrangements are free to vary, and novel payload colors need no lookup.
    train_periods = {o["period"] for objects in train for o in objects}
    assert 1 in train_periods and 2 in train_periods
    assert {o["orientation"] for objects in train for o in objects} == {"H", "V"}
    for objects in train + test:
        for o in objects:
            assert 4 <= o["length"] <= 15 and 2 <= o["depth"] <= 12
            assert 1 <= o["period"] <= 3
            assert o["overhang"] >= 0 and o["overhang"] < o["period"]
            assert background not in o["colors"] and scaffold not in o["colors"]


def _blueprints(rng: np.random.Generator, train_count: int,
                test_count: int) -> tuple[list[list[tuple[str, int]]], list[list[tuple[str, int]]]]:
    base_orientation = "H" if rng.random() < .5 else "V"
    other = "V" if base_orientation == "H" else "H"
    fundamentals = [[(base_orientation, 1)],
                    [(other, 2), (base_orientation if rng.random() < .5 else other,
                                  int(rng.integers(1, 4)))]]
    if rng.random() < .5:
        fundamentals.reverse()
    def random_plan(regime: str) -> list[tuple[str, int]]:
        count = 1 if rng.random() < (.5 if regime == "source_train" else .43) else 2
        result = []
        for _ in range(count):
            ori = "H" if rng.random() < .5 else "V"
            period = int(rng.choice((1, 2, 3), p=(.45, .43, .12)))
            result.append((ori, period))
        return result
    training = fundamentals + [random_plan("source_train") for _ in range(train_count - 2)]
    tests = [training[int(rng.integers(len(training)))][:]
             if rng.random() < .38 else random_plan("target_test")
             for _ in range(test_count)]
    return training, tests


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    train_plans, test_plans = _blueprints(rng, n_train, n_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, plans in (("source_train", train_plans), ("target_test", test_plans)):
        for plan in plans:
            for _ in range(150):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    pair = _generate_pair(pair_rng, regime, params, plan)
                    objects, predicted = _parse_input(pair["input"], params["background"], params["scaffold"])
                except ValueError:
                    continue
                if (np.array_equal(predicted, pair["output"]) and
                    _grid_key(pair["input"]) not in seen and not _is_source_pair(pair)):
                    puzzle["train" if regime == "source_train" else "test"].append(pair)
                    seen.add(_grid_key(pair["input"]))
                    break
            else:
                raise ValueError("could not construct valid distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    roles, shapes, layouts, period3, partial, broad, edge, equal_neighbors = set(), set(), set(), 0, 0, 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        bg, sc, train, test = _infer_roles_and_parse(puzzle)
        roles.add((bg, sc))
        for pair, objects in zip(puzzle["train"] + puzzle["test"], train + test):
            shapes.add(pair["input"].shape)
            layouts.add(tuple(sorted((o["orientation"], o["period"], o["length"], o["depth"], o["side"], o["anchor"], o["overhang"]) for o in objects)))
            _, predicted = _parse_input(pair["input"], bg, sc)
            assert np.array_equal(predicted, pair["output"])
            for o in objects:
                period3 += o["period"] == 3
                partial += o["length"] % o["period"] != 0
                broad += o["depth"] >= o["length"]
                edge += o["edge"]
                t = o["tile"]
                equal_neighbors += np.any(t[:, 1:] == t[:, :-1]) or np.any(t[1:] == t[:-1])
    assert len(roles) >= 15 and len(shapes) >= 35 and len(layouts) >= 75
    assert period3 and partial and broad and edge and equal_neighbors
    for nt, ns in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9000 + (nt or 0) * 10 + (ns or 0), nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    assert not _puzzles_equal(generate_puzzle(314159), generate_puzzle(314160))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4)):
        try:
            generate_puzzle(1, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
