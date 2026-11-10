"""One-seed ARC puzzles: complete both diagonals through the seed."""

import numpy as np


_COLORS = tuple((b, s) for b in range(10) for s in range(10) if b != s)
_SOURCE_LAYOUTS = {
    (7, 7, 3, 3), (6, 6, 2, 2), (9, 9, 1, 5),
    (5, 5, 2, 2), (9, 9, 4, 4), (9, 9, 5, 3), (3, 3, 0, 0),
}
_BANDS = ("small", "medium", "large")
_KINDS = ("center", "near", "offset_one", "offset_both", "edge", "corner")


def _center(h: int, w: int, r: int, c: int) -> bool:
    return r in ((h - 1) // 2, h // 2) and c in ((w - 1) // 2, w // 2)


def _kind(h: int, w: int, r: int, c: int) -> str:
    row_edge = r in (0, h - 1)
    col_edge = c in (0, w - 1)
    if row_edge and col_edge:
        return "corner"
    if row_edge or col_edge:
        return "edge"
    if _center(h, w, r, c):
        return "center"
    if min(r, h - 1 - r, c, w - 1 - c) == 1:
        return "near"
    if r not in ((h - 1) // 2, h // 2) and c not in ((w - 1) // 2, w // 2):
        return "offset_both"
    return "offset_one"


def _band(h: int, w: int) -> str:
    longest = max(h, w)
    return "small" if longest <= 5 else "medium" if longest <= 9 else "large"


def _build_candidates() -> dict:
    candidates = {
        (kind, shape, band): []
        for kind in _KINDS for shape in ("square", "rectangle") for band in _BANDS
    }
    for h in range(3, 15):
        for w in range(3, 15):
            if h != w and abs(h - w) < 2:
                continue
            shape = "square" if h == w else "rectangle"
            band = _band(h, w)
            for r in range(h):
                for c in range(w):
                    candidates[_kind(h, w, r, c), shape, band].append((h, w, r, c))
    return candidates


_CANDIDATES = _build_candidates()


def _source_copy(layout: tuple[int, int, int, int], colors: tuple[int, int]) -> bool:
    return colors == (0, 7) and layout in _SOURCE_LAYOUTS


def _choose_layout(
    rng: np.random.Generator,
    kind: str,
    shape: str,
    occupied: set,
    colors: tuple[int, int],
    avoid_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    for band_index in rng.permutation(3):
        available = [
            layout for layout in _CANDIDATES[kind, shape, _BANDS[int(band_index)]]
            if layout not in occupied
            and (avoid_shape is None or layout[:2] != avoid_shape)
            and not _source_copy(layout, colors)
        ]
        if available:
            return available[int(rng.integers(len(available)))]
    raise ValueError("no distinct geometry is available")


def _choose_colors(
    rng: np.random.Generator,
    base: tuple[int, int],
    varied: bool,
    index: int,
) -> tuple[int, int]:
    if not varied or index == 0:
        return base
    if index == 1 and int(rng.integers(3)) == 0:
        return base[1], base[0]
    if index > 1 and int(rng.integers(4)) == 0:
        seed_color = (0, 7)[int(rng.integers(2))]
        backgrounds = [b for b in range(10) if b != seed_color]
        return backgrounds[int(rng.integers(len(backgrounds)))], seed_color
    choices = [colors for colors in _COLORS if index != 1 or colors != base]
    return choices[int(rng.integers(len(choices)))]


def _make_pair(layout: tuple[int, int, int, int], colors: tuple[int, int]) -> dict:
    h, w, r, c = layout
    background, seed_color = colors
    inp = np.full((h, w), background, dtype=np.int8)
    inp[r, c] = seed_color
    rows, cols = np.indices((h, w))
    out = np.where(np.abs(rows - r) == np.abs(cols - c), seed_color, background)
    return {"input": inp, "output": out.astype(np.int8)}


def _check_count(value: int | None, low: int, high: int) -> None:
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError("pair count is outside its legal range")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _check_count(num_train, 2, 10)
    _check_count(num_test, 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    rng = np.random.default_rng(seed)
    if num_test is None:
        maximum = min(4, num_train - 1) if num_train is not None else 4
        num_test = int(rng.integers(1, maximum + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))

    base_colors = _COLORS[int(rng.integers(len(_COLORS)))]
    varied_colors = bool(rng.integers(2))
    rectangles = bool(rng.integers(3) == 0)
    occupied = set()
    train, test = [], []

    # A full X and an asymmetric example establish the geometry before tests.
    first_colors = _choose_colors(rng, base_colors, varied_colors, 0)
    first_sizes = [n for n in (3, 5, 7, 9, 11)
                   if not _source_copy((n, n, n // 2, n // 2), first_colors)]
    n = first_sizes[int(rng.integers(len(first_sizes)))]
    first_layout = (n, n, n // 2, n // 2)
    occupied.add(first_layout)
    train.append(_make_pair(first_layout, first_colors))

    for i in range(1, num_train):
        colors = _choose_colors(rng, base_colors, varied_colors, i)
        kind_options = ("near", "offset_both", "edge", "corner") if i == 1 else _KINDS
        kind = kind_options[int(rng.integers(len(kind_options)))]
        shape = "rectangle" if rectangles and (i == 1 or rng.integers(2)) else "square"
        layout = _choose_layout(rng, kind, shape, occupied, colors,
                                first_layout[:2] if i == 1 else None)
        occupied.add(layout)
        train.append(_make_pair(layout, colors))

    for i in range(num_test):
        colors = _choose_colors(rng, base_colors, varied_colors, num_train + i)
        kind_options = ("near", "offset_both", "edge", "corner") if i == 0 else _KINDS
        kind = kind_options[int(rng.integers(len(kind_options)))]
        shape = "rectangle" if rectangles and (i == 0 or rng.integers(2)) else "square"
        layout = _choose_layout(rng, kind, shape, occupied, colors)
        occupied.add(layout)
        test.append(_make_pair(layout, colors))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    layouts = {"train": [], "test": []}
    palettes = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            h, w = inp.shape
            assert h >= 3 and w >= 3
            input_key = (inp.shape, tuple(map(int, inp.flat)))
            pair_key = (input_key, tuple(map(int, out.flat)))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            values, counts = np.unique(inp, return_counts=True)
            assert len(values) == 2 and sorted(counts.tolist()) == [1, h * w - 1]
            background = int(values[np.argmax(counts)])
            seed_color = int(values[np.argmin(counts)])
            seed_positions = np.argwhere(inp == seed_color)
            assert seed_positions.shape == (1, 2)
            r, c = map(int, seed_positions[0])
            layout = (h, w, r, c)
            assert layout not in seen_layouts
            seen_layouts.add(layout)
            assert not _source_copy(layout, (background, seed_color))

            rows, cols = np.indices(inp.shape)
            expected = np.where(
                np.abs(rows - r) == np.abs(cols - c), seed_color, background
            )
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            assert set(np.unique(out).tolist()) == {background, seed_color}
            layouts[split].append(layout)
            palettes[split].append((background, seed_color))

    first, second = layouts["train"][:2]
    assert first[0] == first[1] and first[0] % 2 == 1
    assert first[0] >= 3 and _center(*first)
    assert not _center(*second) and second[:2] != first[:2]
    assert any(not _center(*layout) for layout in layouts["test"])
    if any(h != w for h, w, _, _ in layouts["test"]):
        assert any(h != w for h, w, _, _ in layouts["train"])
    if len(set(palettes["train"])) == 1:
        assert all(colors == palettes["train"][0] for colors in palettes["test"])


def _same_puzzle(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False
        for pair_a, pair_b in zip(a[split], b[split]):
            for name in ("input", "output"):
                x, y = pair_a[name], pair_b[name]
                if x.dtype != y.dtype or not np.array_equal(x, y):
                    return False
    return True


def self_test() -> None:
    train_counts, test_counts, puzzle_signatures = set(), set(), set()
    palettes, dimensions, layouts = set(), set(), set()
    by_split = {"train": set(), "test": set()}
    kinds = {"train": set(), "test": set()}
    edge_sides = {"train": set(), "test": set()}
    bands = {"train": set(), "test": set()}
    special = {
        "large_corner_train": 0, "large_corner_test": 0,
        "small_interior_train": 0, "small_interior_test": 0,
        "test_above": 0, "test_beside": 0,
        "seed_zero": 0, "seed_seven": 0, "original_palette": 0,
        "rectangle_edge": 0, "rectangle_corner": 0,
        "swapped_palette_puzzles": 0, "variable_rectangles": 0,
    }
    variable_puzzles = stable_puzzles = rectangular_puzzles = 0
    combinations = set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = []
        local_palettes = set()
        local_rectangles = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                values, counts = np.unique(inp, return_counts=True)
                background = int(values[np.argmax(counts)])
                seed_color = int(values[np.argmin(counts)])
                r, c = map(int, np.argwhere(inp == seed_color)[0])
                h, w = inp.shape
                layout = (h, w, r, c)
                kind = _kind(*layout)
                palette = (background, seed_color)
                if kind == "near":
                    assert 0 < r < h - 1 and 0 < c < w - 1
                    assert min(r, h - 1 - r, c, w - 1 - c) == 1
                    assert not _center(*layout)
                if kind == "offset_both":
                    assert min(r, h - 1 - r, c, w - 1 - c) >= 2
                    assert r not in ((h - 1) // 2, h // 2)
                    assert c not in ((w - 1) // 2, w // 2)
                assert all(
                    int(out[y, x]) == (seed_color if abs(y - r) == abs(x - c)
                                       else background)
                    for y in range(h) for x in range(w)
                )
                assert not _source_copy(layout, palette)
                palettes.add(palette)
                local_palettes.add(palette)
                dimensions.add((h, w))
                layouts.add(layout)
                by_split[split].add((h, w))
                kinds[split].add(kind)
                bands[split].add(_band(h, w))
                local_rectangles |= h != w
                combinations.add((split, kind, _band(h, w), h != w))
                if kind == "edge":
                    edge_sides[split].add(
                        "top" if r == 0 else "bottom" if r == h - 1
                        else "left" if c == 0 else "right"
                    )
                special["large_corner_" + split] += (
                    kind == "corner" and _band(h, w) == "large"
                )
                special["small_interior_" + split] += (
                    kind not in ("edge", "corner") and _band(h, w) == "small"
                )
                special["seed_zero"] += seed_color == 0
                special["seed_seven"] += seed_color == 7
                special["original_palette"] += palette == (0, 7)
                special["rectangle_edge"] += h != w and kind == "edge"
                special["rectangle_corner"] += h != w and kind == "corner"
                if split == "test" and kind not in ("center", "corner"):
                    special["test_above"] += r < (h - 1) / 2
                    special["test_beside"] += c != (w - 1) / 2
                signature.append((split, layout, palette))
        variable_puzzles += len(local_palettes) > 1
        stable_puzzles += len(local_palettes) == 1
        rectangular_puzzles += local_rectangles
        special["variable_rectangles"] += (
            len(local_palettes) > 1 and local_rectangles
        )
        special["swapped_palette_puzzles"] += any(
            (seed_color, background) in local_palettes
            for background, seed_color in local_palettes
        )
        puzzle_signatures.add(tuple(signature))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40 and len(layouts) >= 150
    assert len(dimensions) >= 25 and len(palettes) >= 25
    assert stable_puzzles >= 8 and variable_puzzles >= 8
    assert rectangular_puzzles >= 6
    assert set(_KINDS) <= kinds["train"] and set(_KINDS) <= kinds["test"]
    for split in ("train", "test"):
        assert set(_BANDS) <= bands[split]
        assert edge_sides[split] == {"top", "bottom", "left", "right"}
        assert any(h == w and h <= 5 for h, w in by_split[split])
        assert any(h == w and h >= 10 for h, w in by_split[split])
        assert any(h != w for h, w in by_split[split])
        assert any(h % 2 == 0 for h, w in by_split[split])
        assert special["large_corner_" + split] >= 1
        assert special["small_interior_" + split] >= 1
    assert len(combinations) >= 40
    assert all(special[key] >= 1 for key in (
        "test_above", "test_beside", "seed_zero", "seed_seven",
        "original_palette", "rectangle_edge", "rectangle_corner",
        "swapped_palette_puzzles", "variable_rectangles",
    ))

    for i, (nt, nv) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(100 + i, nt, nv)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, nv)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(100 + i, nt, nv))
    for i, (nt, nv) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(200 + i, nt, nv)
        assert nt is None or len(puzzle["train"]) == nt
        assert nv is None or len(puzzle["test"]) == nv
        validate_generated_puzzle(puzzle)
    for nt, nv in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(300, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
