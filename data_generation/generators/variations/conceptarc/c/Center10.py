"""Generate paired L shapes whose missing corners determine one midpoint cell."""

from functools import lru_cache

import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_OPPOSITE = ((0, 3), (3, 0), (1, 2), (2, 1))
_SAME = ((0, 0), (1, 1), (2, 2), (3, 3))
_GAPS = {
    "short_axis": ((0, 2), (2, 0), (0, 4), (4, 0)),
    "short_diag": ((2, 2),),
    "coincident": ((0, 0),),
    "long_axis": ((0, 6), (6, 0), (0, 8), (8, 0)),
    "unequal_diag": ((2, 4), (4, 2), (2, 6), (6, 2), (4, 6), (6, 4)),
    "long_diag": ((4, 4), (6, 6)),
}
_HARD = ("coincident", "long_axis", "unequal_diag", "long_diag")
_ALL = ("short_axis", "short_diag") + _HARD

_SOURCE_ROWS = (
    ("000000", "000000", "550000", "500050", "000550", "000000"),
    ("000550", "000500", "000000", "000000", "000000", "000005", "000055"),
    ("000000", "000550", "000050", "000000", "500000", "550000"),
    ("055000", "050000", "000000", "000500", "005500", "000000"),
    ("000000", "000000", "055000", "050500", "005500", "000000"),
    ("055000", "050000", "000000", "000000", "000000", "000005", "000055", "000000"),
    ("000055", "000005", "000000", "050000", "055000", "000000"),
)
_SOURCE_INPUTS = {
    (len(rows), len(rows[0]), tuple(int(ch) for row in rows for ch in row))
    for rows in _SOURCE_ROWS
}


def _grid_key(grid):
    return grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat)


def _cells(hole, corner):
    top = hole[0] - _CORNERS[corner][0]
    left = hole[1] - _CORNERS[corner][1]
    return frozenset(
        (top + dr, left + dc)
        for i, (dr, dc) in enumerate(_CORNERS)
        if i != corner
    )


def _separate(a, b):
    return not (a & b) and all(
        abs(ar - br) + abs(ac - bc) != 1
        for ar, ac in a for br, bc in b
    )


@lru_cache(maxsize=None)
def _layouts(height, width, delta, corners):
    """Enumerate feasible first holes for a chosen signed gap and canvas."""
    result = []
    dr, dc = delta
    for r in range(height):
        for c in range(width):
            first, second = (r, c), (r + dr, c + dc)
            if not (0 <= second[0] < height and 0 <= second[1] < width):
                continue
            a, b = _cells(first, corners[0]), _cells(second, corners[1])
            cells = a | b
            if not all(0 <= x < height and 0 <= y < width for x, y in cells):
                continue
            if not _separate(a, b):
                continue
            midpoint = (r + dr // 2, c + dc // 2)
            if midpoint in cells:
                continue
            if corners[0] == corners[1]:
                rows = [x for x, _ in cells]
                cols = [y for _, y in cells]
                if (2 * midpoint[0], 2 * midpoint[1]) == (
                    min(rows) + max(rows), min(cols) + max(cols)
                ):
                    continue
            result.append((first, second))
    return tuple(result)


def _make_pair(height, width, first, second, corners, palette):
    bg, obj, marker = palette
    inp = np.full((height, width), bg, dtype=np.int8)
    for r, c in _cells(first, corners[0]) | _cells(second, corners[1]):
        inp[r, c] = obj
    out = inp.copy()
    out[(first[0] + second[0]) // 2, (first[1] + second[1]) // 2] = marker
    return {"input": inp, "output": out}


def _resolve_counts(rng, num_train, num_test):
    for name, value, lo, hi in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not lo <= int(value) <= hi
        ):
            raise ValueError(f"{name} must be an integer from {lo} through {hi}")
    choices = [
        (tr, te) for tr in range(2, 11) for te in range(1, 5)
        if tr > te
        and (num_train is None or tr == int(num_train))
        and (num_test is None or te == int(num_test))
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _choice(rng, options):
    return options[int(rng.integers(len(options)))]


def _pair_palette(rng, index, split, varying, base, marker):
    bg, obj = base
    if not varying:
        return bg, obj, marker
    if split == "train" and index == 0:
        return bg, obj, marker
    if split == "train" and index == 1:
        return obj, bg, marker
    roll = int(rng.integers(5))
    if roll == 0:
        return bg, obj, marker
    if roll == 1:
        return obj, bg, marker
    choices = [x for x in range(10) if x != marker]
    drawn = rng.choice(choices, size=2, replace=False)
    return int(drawn[0]), int(drawn[1]), marker


def _draw_pair(rng, category, kind, palette, used_inputs, used_structure):
    gaps = _GAPS[category]
    corner_options = _OPPOSITE if kind == "opposite" else _SAME
    fallback = None
    for _ in range(90):
        gap = _choice(rng, gaps)
        delta = tuple(
            x * (-1 if x and int(rng.integers(2)) else 1)
            for x in gap
        )
        corners = _choice(rng, corner_options)
        if int(rng.integers(4)) == 0:
            height, width = _choice(rng, ((6, 6), (6, 7), (7, 6), (7, 7)))
        else:
            height = int(rng.integers(6, 13))
            width = int(rng.integers(6, 13))
        layouts = _layouts(height, width, delta, corners)
        if not layouts:
            continue
        first, second = _choice(rng, layouts)
        pair = _make_pair(height, width, first, second, corners, palette)
        key = _grid_key(pair["input"])
        if key in used_inputs or key in _SOURCE_INPUTS:
            continue
        signature = (gap, corners)
        if signature in used_structure:
            fallback = (pair, key, signature)
            continue
        used_inputs.add(key)
        used_structure.add(signature)
        return pair
    if fallback is not None:
        pair, key, signature = fallback
        used_inputs.add(key)
        used_structure.add(signature)
        return pair
    for gap in gaps:
        for sr in (-1, 1):
            for sc in (-1, 1):
                delta = (sr * gap[0], sc * gap[1])
                for corners in corner_options:
                    for height in range(6, 13):
                        for width in range(6, 13):
                            for first, second in _layouts(height, width, delta, corners):
                                pair = _make_pair(
                                    height, width, first, second, corners, palette
                                )
                                key = _grid_key(pair["input"])
                                if key not in used_inputs and key not in _SOURCE_INPUTS:
                                    used_inputs.add(key)
                                    used_structure.add((gap, corners))
                                    return pair
    raise RuntimeError(f"No valid {category} layout found")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    marker, bg, obj = (int(x) for x in rng.choice(10, size=3, replace=False))
    varying_palette = bool(rng.integers(2))
    extended_corners = bool(rng.integers(2))
    regime = _choice(rng, ("shift", "mixed", "coincident", "rich")) if n_train >= 3 else "shift"

    train_categories = ["short_axis", "short_diag"]
    if n_train >= 3:
        third = {
            "shift": _choice(rng, ("short_axis", "short_diag")),
            "mixed": _choice(rng, ("long_axis", "unequal_diag")),
            "coincident": "coincident",
            "rich": _choice(rng, ("long_diag", "unequal_diag")),
        }[regime]
        train_categories.append(third)
    fill = ("short_axis", "short_diag") if regime == "shift" else _ALL
    train_categories.extend(_choice(rng, fill) for _ in range(n_train - len(train_categories)))

    first_test = _choice(rng, _HARD) if regime == "shift" else train_categories[2]
    test_categories = [first_test]
    if n_test >= 2:
        test_categories.append(_choice(rng, ("short_axis", "short_diag")))
    test_categories.extend(_choice(rng, _ALL) for _ in range(n_test - len(test_categories)))

    puzzle = {"train": [], "test": []}
    used_inputs, used_structure = set(), set()
    for split, categories in (("train", train_categories), ("test", test_categories)):
        for index, category in enumerate(categories):
            if category == "coincident" or not extended_corners:
                kind = "opposite"
            elif split == "train" and index == 0:
                kind = "opposite"
            elif split == "train" and index == 1:
                kind = "same"
            else:
                kind = "same" if int(rng.integers(2)) else "opposite"
            palette = _pair_palette(rng, index, split, varying_palette, (bg, obj), marker)
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
            puzzle[split].append(
                _draw_pair(pair_rng, category, kind, palette, used_inputs, used_structure)
            )
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= size <= 30 for size in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2 and 6 in counts
    bg = int(values[np.argmax(counts)])
    obj = int(values[np.argmin(counts)])
    assert int(np.count_nonzero(inp == obj)) == 6

    changed = np.argwhere(inp != out)
    assert changed.shape == (1, 2)
    mr, mc = (int(x) for x in changed[0])
    marker = int(out[mr, mc])
    assert int(inp[mr, mc]) == bg and marker not in (bg, obj)
    assert int(np.count_nonzero(out == marker)) == 1

    remaining = set(zip(*np.where(inp == obj)))
    components = []
    while remaining:
        start = remaining.pop()
        component, frontier = {start}, [start]
        while frontier:
            r, c = frontier.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    assert len(components) == 2
    holes, corners = [], []
    for component in components:
        assert len(component) == 3
        r0, r1 = min(r for r, _ in component), max(r for r, _ in component)
        c0, c1 = min(c for _, c in component), max(c for _, c in component)
        assert (r1 - r0, c1 - c0) == (1, 1)
        missing = {(r, c) for r in (r0, r1) for c in (c0, c1)} - component
        assert len(missing) == 1
        hr, hc = missing.pop()
        holes.append((hr, hc))
        corners.append((hr - r0) * 2 + hc - c0)
    assert corners[0] == corners[1] or sum(corners) == 3
    assert (holes[0][0] + holes[1][0]) % 2 == 0
    assert (holes[0][1] + holes[1][1]) % 2 == 0
    assert (mr, mc) == (
        (holes[0][0] + holes[1][0]) // 2,
        (holes[0][1] + holes[1][1]) // 2,
    )
    dr = abs(holes[0][0] - holes[1][0])
    dc = abs(holes[0][1] - holes[1][1])
    category = next(
        (name for name, gaps in _GAPS.items() if (dr, dc) in gaps), None
    )
    assert category is not None
    if category == "coincident":
        assert corners[0] != corners[1]
    if corners[0] == corners[1]:
        cells = set().union(*components)
        assert (2 * mr, 2 * mc) != (
            min(r for r, _ in cells) + max(r for r, _ in cells),
            min(c for _, c in cells) + max(c for _, c in cells),
        )
    return {
        "palette": (bg, obj, marker),
        "shape": inp.shape,
        "delta": (dr, dc),
        "category": category,
        "same": corners[0] == corners[1],
        "corners": tuple(sorted(corners)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen = set()
    meta = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            item = _analyze_pair(pair)
            key = _grid_key(pair["input"])
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            meta[split].append(item)
    train, test = meta["train"], meta["test"]
    markers = {item["palette"][2] for item in train + test}
    assert len(markers) == 1
    train_palettes = {item["palette"][:2] for item in train}
    if len(train_palettes) == 1:
        assert {item["palette"][:2] for item in test} == train_palettes
    else:
        assert any((b, a) in train_palettes for a, b in train_palettes)
    assert all(item["category"] != "coincident" for item in train[:2])
    assert train[0]["category"] == "short_axis"
    assert train[1]["category"] == "short_diag"
    assert all(not item["same"] for item in test) or any(item["same"] for item in train)
    assert any(not item["same"] for item in train)
    assert test[0]["category"] in _HARD
    if n_test >= 2:
        assert test[1]["category"] in ("short_axis", "short_diag")


def _same_puzzle(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x["input"], y["input"])
            and np.array_equal(x["output"], y["output"])
            for x, y in zip(a[split], b[split])
        ) for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert all(_same_puzzle(p, generate_puzzle(i)) for i, p in enumerate(puzzles))
    assert len({_grid_key(p["train"][0]["input"]) for p in puzzles}) >= 38

    rows = [
        (split, _analyze_pair(pair))
        for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]
    ]
    assert {m["category"] for _, m in rows} == set(_ALL)
    assert {m["category"] for s, m in rows if s == "train"} == set(_ALL)
    assert {m["category"] for s, m in rows if s == "test"} == set(_ALL)
    assert any(m["same"] for s, m in rows if s == "train")
    assert any(m["same"] for s, m in rows if s == "test")
    assert any(m["shape"][0] < m["shape"][1] for _, m in rows)
    assert any(m["shape"][0] > m["shape"][1] for _, m in rows)
    assert any(m["shape"] == (6, 6) for _, m in rows)
    assert len({m["shape"] for _, m in rows}) >= 25
    assert len({m["delta"] for _, m in rows}) >= 13
    assert len({m["palette"] for _, m in rows}) >= 25
    assert any(len({_analyze_pair(x)["palette"] for x in p["train"]}) > 1 for p in puzzles)
    assert any(len({_analyze_pair(x)["palette"] for x in p["train"]}) == 1 for p in puzzles)
    assert any(
        {m["category"] for m in (_analyze_pair(x) for x in p["train"])}
        & {"coincident", "long_axis", "unequal_diag", "long_diag"}
        & {m["category"] for m in (_analyze_pair(x) for x in p["test"])}
        for p in puzzles
    )
    assert any(
        len({(_analyze_pair(x)["delta"], _analyze_pair(x)["corners"])
             for x in p["train"]}) >= 5
        for p in puzzles if len(p["train"]) >= 7
    )
    summaries = [
        ([ _analyze_pair(x) for x in p["train"] ],
         [ _analyze_pair(x) for x in p["test"] ])
        for p in puzzles
    ]
    assert all(
        len({(m["delta"], m["corners"]) for m in train}) >= 3
        for train, _ in summaries if len(train) >= 7
    )
    assert sum(
        all(m["category"] in ("short_axis", "short_diag") for m in train)
        for train, _ in summaries
    ) >= 5
    assert sum(
        any(m["category"] == "coincident" for m in train)
        and any(m["category"] == "coincident" for m in test)
        for train, test in summaries
    ) >= 2
    assert sum(
        bool({m["category"] for m in train}
             & {m["category"] for m in test} & set(_HARD))
        for train, test in summaries
    ) >= 5
    assert any(
        any(m["same"] for m in train)
        and len({m["palette"][:2] for m in train}) > 1
        and any(m["category"] == "coincident" for m in train)
        for train, _ in summaries
    )
    assert {(0, 6), (6, 0), (2, 4), (4, 2)} <= {
        m["delta"] for _, m in rows
    }

    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4), (2, None),
                                  (10, None), (6, 3))):
        puzzle = generate_puzzle(1000 + i, nt, ne)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(1000 + i, nt, ne))
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(3, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {(nt, ne)}")
