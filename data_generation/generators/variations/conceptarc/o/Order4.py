import numpy as np


_SOURCE_INPUTS = {
    ("0000000000000", "0000000000000", "0550220440330",
     "0110220440550", "0110550440330", "0110220550330",
     "0000000000000", "0000000000000"),
    ("000000000000000", "444022206660777", "404022206660777",
     "444022206660777", "444022206660777", "444020206660777",
     "444022206660707", "444022206060777"),
    ("000000000", "602010307", "402010307", "604010307",
     "602040307", "602010407", "602010304", "602010307"),
    ("33333044444055555011111", "33333044444055555021111",
     "32333044444055555021111", "32333044444055555021111",
     "32333044444055555021111", "32333042444055555021111",
     "32333042444055555021111", "32333042444055255021111"),
    ("000000000400000", "000004000400000", "040004000400000",
     "888033307770555", "888033307770555", "888033307770555"),
}


def _counts(rng, num_train, num_test):
    for value, name, low, high in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        options = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return options[int(rng.integers(len(options)))]
    if num_train is None:
        return int(rng.integers(max(2, int(num_test) + 1), 11)), int(num_test)
    return int(num_train), int(rng.integers(1, min(4, int(num_train) - 1) + 1))


def _geometry(rng, n, width, bar_height, above=False):
    top_pad = int(rng.integers(2, min(5, bar_height - 1) + 1)) if above else int(rng.integers(6))
    bottom_pad = int(rng.integers(5))
    minimum_width = n * width + n - 1
    budget = min(30 - minimum_width, 9)
    extras = int(rng.integers(budget + 1))
    amounts = [0] * (n + 1)  # left margin, gaps, right margin
    caps = [4] + [2] * (n - 1) + [4]
    for _ in range(extras):
        available = [i for i, cap in enumerate(caps) if amounts[i] < cap]
        amounts[int(rng.choice(available))] += 1
    left, right = amounts[0], amounts[-1]
    gaps = [1 + v for v in amounts[1:-1]]
    starts = [left]
    for gap in gaps:
        starts.append(starts[-1] + width + gap)
    grid_width = starts[-1] + width + right
    assert grid_width <= 30
    return top_pad, bottom_pad, starts, gaps, grid_width


def _palette(rng, n, previous=None):
    colors = list(range(10))
    if previous is None:
        bg = int(rng.choice(colors))
        ink = int(rng.choice([c for c in colors if c != bg]))
        bars = [int(c) for c in rng.choice(
            [c for c in colors if c not in (bg, ink)], size=n, replace=False
        )]
    else:
        old_bg, old_ink, old_bars = previous
        ink = int(rng.choice(old_bars))
        bg = int(rng.choice([c for c in colors if c not in (old_bg, old_ink, ink)]))
        others = [c for c in colors if c not in (bg, ink, old_ink)]
        bars = [old_ink] + [
            int(c) for c in rng.choice(others, size=n - 1, replace=False)
        ]
        rng.shuffle(bars)
    return bg, ink, bars


def _rank(tops, descending=True, unmarked_first=True):
    return sorted(
        range(len(tops)),
        key=lambda i: (
            0 if unmarked_first else 1, 0
        ) if tops[i] is None else (
            1 if unmarked_first else 0,
            -tops[i] if descending else tops[i],
        ),
    )


def _construct(rng, n, width, bar_height, top_pad, bottom_pad,
               starts, grid_width, palette, marks):
    bg, ink, bars = palette
    height = top_pad + bar_height + bottom_pad
    inp = np.full((height, grid_width), bg, dtype=np.int8)
    for start, color in zip(starts, bars):
        inp[top_pad:top_pad + bar_height, start:start + width] = color
    tops = []
    for i, mark in enumerate(marks):
        if mark is None:
            tops.append(None)
            continue
        kind, top, length, hole = mark
        tops.append(top)
        value = bg if hole else ink
        start = starts[i]
        if kind == "stripe":
            inp[top, start:start + width] = value
        else:
            column = start + int(rng.integers(width))
            stop = top_pad if kind == "above" else top + length
            inp[top:stop, column] = value
    out = np.full_like(inp, bg)
    for start, index in zip(starts, _rank(tops)):
        out[top_pad:top_pad + bar_height, start:start + width] = bars[index]
    return {"input": inp, "output": out}


def _diagnostic_common_bottom(rng):
    n = int(rng.integers(3, 7))
    widths = [w for w in range(2, 6) if n * w + n - 1 <= 30]
    width = int(rng.choice(widths))
    bar_height = int(rng.integers(max(n + 2, 7), 14))
    top_pad, bottom_pad, starts, _, grid_width = _geometry(
        rng, n, width, bar_height
    )
    y0, y1 = top_pad, top_pad + bar_height
    middle = rng.choice(
        np.arange(y0 + 2, y1 - 1), size=n - 2, replace=False
    )
    tops = [y0 + 1, y1 - 1] + [int(v) for v in middle]
    rng.shuffle(tops)
    marks = [("run", top, y1 - top, False) for top in tops]
    pair = _construct(
        rng, n, width, bar_height, top_pad, bottom_pad,
        starts, grid_width, _palette(rng, n), marks
    )
    if np.array_equal(pair["input"], pair["output"]):
        marks[0], marks[1] = marks[1], marks[0]
        pair = _construct(
            rng, n, width, bar_height, top_pad, bottom_pad,
            starts, grid_width, _palette(rng, n), marks
        )
    return pair


def _diagnostic_unmarked(rng, previous_palette, first_count, first_width):
    n = int(rng.choice([v for v in (5, 6) if v != first_count]))
    widths = [
        w for w in range(1, 7)
        if n * w + n - 1 <= 30 and w != first_width
    ]
    width = int(rng.choice(widths))
    bar_height = int(rng.integers(max(n + 4, 9), 15))
    top_pad, bottom_pad, starts, _, grid_width = _geometry(
        rng, n, width, bar_height, above=True
    )
    y0, y1 = top_pad, top_pad + bar_height
    unmarked = int(rng.integers(1, n))
    marked = [i for i in range(n) if i != unmarked]
    rng.shuffle(marked)
    above_i, hole_i, early_i, late_i = marked[:4]
    marks = [None] * n
    above_top = int(rng.integers(y0))
    marks[above_i] = ("above", above_top, y0 - above_top, False)
    early_top = int(rng.integers(y0, y1 - 4))
    late_top = int(rng.integers(early_top + 1, y1 - 2))
    hole_top = int(rng.choice([
        v for v in range(y0, y1) if v not in (early_top, late_top)
    ]))
    marks[hole_i] = ("single", hole_top, 1, True)
    marks[early_i] = ("run", early_top, 2, False)
    marks[late_i] = ("run", late_top, 3, False)
    used = {above_top, hole_top, early_top, late_top}
    for i in marked[4:]:
        top = int(rng.choice([v for v in range(y0, y1) if v not in used]))
        used.add(top)
        kind = str(rng.choice(["single", "stripe", "run"]))
        limit = (
            min(y1 - top, max(1, (bar_height - 1) // 2))
            if width == 1 else min(y1 - top, bar_height - 1)
        )
        length = int(rng.integers(1, limit + 1)) if kind == "run" else 1
        marks[i] = (kind, top, length, bool(rng.integers(2)))
    palette = _palette(rng, n, previous_palette)
    return _construct(
        rng, n, width, bar_height, top_pad, bottom_pad,
        starts, grid_width, palette, marks
    )


def _random_pair(rng, regime):
    n = int(rng.integers(3, 7))
    widths = [w for w in range(1, 7) if n * w + n - 1 <= 30]
    if regime == "target_test":
        weights = np.array([2 if w in (1, 5, 6) else 1 for w in widths], dtype=float)
        weights /= weights.sum()
        width = int(rng.choice(widths, p=weights))
    else:
        width = int(rng.choice(widths))
    bar_height = int(rng.integers(n + 1, 15))
    family = str(rng.choice(["single", "stripe", "run", "above", "mixed"]))
    has_above = family in ("above", "mixed")
    top_pad, bottom_pad, starts, _, grid_width = _geometry(
        rng, n, width, bar_height, above=has_above
    )
    y0, y1 = top_pad, top_pad + bar_height
    unmarked = int(rng.integers(n)) if rng.random() < 0.38 else None
    indices = [i for i in range(n) if i != unmarked]
    rng.shuffle(indices)
    above_count = (
        int(rng.integers(1, min(2, len(indices), y0) + 1))
        if has_above else 0
    )
    above_tops = [
        int(v) for v in rng.choice(np.arange(y0), size=above_count, replace=False)
    ]
    internal_tops = [
        int(v) for v in rng.choice(
            np.arange(y0, y1), size=len(indices) - above_count, replace=False
        )
    ]
    marks = [None] * n
    for i, top in zip(indices[:above_count], above_tops):
        marks[i] = ("above", top, y0 - top, False)
    for i, top in zip(indices[above_count:], internal_tops):
        kind = (
            str(rng.choice(["single", "stripe", "run"]))
            if family in ("above", "mixed") else family
        )
        if kind == "run":
            limit = (
                min(y1 - top, max(1, (bar_height - 1) // 2))
                if width == 1 else min(y1 - top, bar_height - 1)
            )
            length = int(rng.integers(1, limit + 1))
        else:
            length = 1
        marks[i] = (kind, top, length, bool(rng.integers(2)))
    return _construct(
        rng, n, width, bar_height, top_pad, bottom_pad,
        starts, grid_width, _palette(rng, n), marks
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    first = _diagnostic_common_bottom(np.random.default_rng(int(rng.integers(2**63))))
    first_info = _decode_input(first["input"])
    previous_palette = (
        first_info["bg"], first_info["ink"], first_info["colors"]
    )
    second = _diagnostic_unmarked(
        np.random.default_rng(int(rng.integers(2**63))), previous_palette,
        len(first_info["slots"]), first_info["width"],
    )
    train = [first, second]
    for _ in range(train_count - 2):
        train.append(_random_pair(
            np.random.default_rng(int(rng.integers(2**63))), "source_train"
        ))
    rng.shuffle(train)
    test = [
        _random_pair(np.random.default_rng(int(rng.integers(2**63))), "target_test")
        for _ in range(test_count)
    ]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _uniform_groups(columns):
    return 0 if not columns else 1 + sum(
        b != a + 1 for a, b in zip(columns, columns[1:])
    )


def _decode_input(inp):
    candidates = []
    for color in range(10):
        uniform = [x for x in range(inp.shape[1]) if np.all(inp[:, x] == color)]
        if _uniform_groups(uniform) >= 2:
            active = np.flatnonzero(np.any(inp != color, axis=0))
            if len(active) == 0:
                continue
            chunks = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
            widths = {len(chunk) for chunk in chunks}
            gaps = [int(chunks[i + 1][0] - chunks[i][-1] - 1)
                    for i in range(len(chunks) - 1)]
            if (3 <= len(chunks) <= 6 and len(widths) == 1
                    and 1 <= next(iter(widths)) <= 6
                    and all(1 <= gap <= 3 for gap in gaps)):
                candidates.append(color)
    assert len(candidates) == 1
    bg = candidates[0]
    active = np.flatnonzero(np.any(inp != bg, axis=0))
    assert len(active) > 0
    chunks = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    slots = [(int(c[0]), int(c[-1]) + 1) for c in chunks]
    assert 3 <= len(slots) <= 6
    widths = {b - a for a, b in slots}
    assert len(widths) == 1
    width = widths.pop()
    assert 1 <= width <= 6
    gaps = [slots[i + 1][0] - slots[i][1] for i in range(len(slots) - 1)]
    assert all(1 <= gap <= 3 for gap in gaps)
    colors = []
    for a, b in slots:
        counts = np.bincount(inp[:, a:b].ravel(), minlength=10)
        counts[bg] = 0
        assert counts.max() > 0 and np.count_nonzero(counts == counts.max()) == 1
        colors.append(int(np.argmax(counts)))
    assert len(set(colors)) == len(colors)
    base_rows = np.flatnonzero(np.any(np.stack([
        np.any(inp[:, a:b] == color, axis=1)
        for (a, b), color in zip(slots, colors)
    ]), axis=0))
    assert len(base_rows) > 0
    y0, y1 = int(base_rows[0]), int(base_rows[-1]) + 1
    assert 4 <= y1 - y0 <= 14
    marks, tops = [], []
    for (a, b), color in zip(slots, colors):
        footprint = inp[y0:y1, a:b]
        assert np.count_nonzero(footprint == color) > footprint.size / 2
        cells = []
        for y in range(y1):
            for x in range(a, b):
                value = int(inp[y, x])
                if (y < y0 and value != bg) or (y >= y0 and value != color):
                    cells.append((y, x, value))
        assert np.all(inp[y1:, a:b] == bg)
        marks.append(cells)
        tops.append(min((y for y, _, _ in cells), default=None))
    known = [v for v in tops if v is not None]
    assert len(set(known)) == len(known)
    assert tops.count(None) <= 1
    mark_values = {v for mark in marks for _, _, v in mark if v != bg}
    assert len(mark_values) <= 1
    ink = next(iter(mark_values)) if mark_values else None
    assert ink is None or ink not in colors
    for cells in marks:
        if not cells:
            continue
        ys = sorted({y for y, _, _ in cells})
        xs = {x for _, x, _ in cells}
        assert ys == list(range(ys[0], ys[-1] + 1))
        if ys[0] < y0:
            assert ys[-1] == y0 - 1
            assert len(xs) == 1 and len(cells) == len(ys)
            assert all(value != bg for _, _, value in cells)
        elif len(ys) == 1:
            assert len(xs) in (1, width)
            assert len(cells) == len(xs)
        else:
            assert len(xs) == 1 and len(cells) == len(ys)
    return {
        "bg": bg, "ink": ink, "slots": slots, "width": width, "gaps": gaps,
        "colors": colors, "marks": marks, "tops": tops, "y0": y0, "y1": y1,
        "left": slots[0][0], "right": inp.shape[1] - slots[-1][1],
        "bottom": inp.shape[0] - y1,
    }


def _expected(info, shape, descending=True, unmarked_first=True):
    out = np.full(shape, info["bg"], dtype=np.int8)
    for (a, b), index in zip(
        info["slots"], _rank(info["tops"], descending, unmarked_first)
    ):
        out[info["y0"]:info["y1"], a:b] = info["colors"][index]
    return out


def _learn_rule(train):
    consistent = []
    for descending in (False, True):
        for unmarked_first in (False, True):
            if all(np.array_equal(
                pair["output"],
                _expected(_decode_input(pair["input"]), pair["input"].shape,
                          descending, unmarked_first),
            ) for pair in train):
                consistent.append((descending, unmarked_first))
    assert consistent == [(True, True)]
    return consistent[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen = set()
    infos = {"train": [], "test": []}
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            assert tuple("".join(str(int(v)) for v in row) for row in inp) not in _SOURCE_INPUTS
            info = _decode_input(inp)
            assert np.array_equal(out, _expected(info, inp.shape))
            infos[regime].append(info)
    def common_bottom(info):
        lengths = [len({y for y, _, _ in mark}) for mark in info["marks"] if mark]
        return (
            all(info["marks"])
            and len({max(y for y, _, _ in mark) for mark in info["marks"]}) == 1
            and len(set(info["tops"])) == len(info["tops"])
            and all(len({x for _, x, _ in mark}) == 1 for mark in info["marks"])
            and 1 in lengths and max(lengths) >= info["y1"] - info["y0"] - 1
        )

    def unmarked_demonstration(info):
        if info["tops"].count(None) != 1 or info["tops"].index(None) == 0:
            return False
        if not any(top is not None and top < info["y0"] for top in info["tops"]):
            return False
        if not any(any(v == info["bg"] for _, _, v in mark) for mark in info["marks"]):
            return False
        runs = [
            (min(y for y, _, _ in mark), len({y for y, _, _ in mark}))
            for mark in info["marks"]
            if mark and min(y for y, _, _ in mark) >= info["y0"]
            and all(v == info["ink"] for _, _, v in mark)
            and len({y for y, _, _ in mark}) >= 2
        ]
        return any(
            a_top < b_top and a_length < b_length
            for a_top, a_length in runs for b_top, b_length in runs
        )

    assert any(
        common_bottom(a) and unmarked_demonstration(b)
        and a["bg"] != b["bg"]
        and a["ink"] in b["colors"] and b["ink"] in a["colors"]
        for a in infos["train"] for b in infos["train"] if a is not b
    )
    assert len({len(info["slots"]) for info in infos["train"]}) >= 2
    assert len({info["width"] for info in infos["train"]}) >= 2
    learned_rule = _learn_rule(train)
    for pair, info in zip(test, infos["test"]):
        assert np.array_equal(
            pair["output"], _expected(info, pair["input"].shape, *learned_rule)
        )


def _same_puzzle(a, b):
    return all(
        len(a[k]) == len(b[k]) and all(
            np.array_equal(p[x], q[x])
            for p, q in zip(a[k], b[k]) for x in ("input", "output")
        ) for k in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    counts, widths, backgrounds, marker_colors = set(), set(), set(), set()
    shapes, layouts, run_lengths, unmarked_slots = set(), set(), set(), set()
    flags = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                info = _decode_input(pair["input"])
                n = len(info["slots"])
                counts.add(n)
                widths.add(info["width"])
                backgrounds.add(info["bg"])
                if info["ink"] is not None:
                    marker_colors.add(info["ink"])
                shapes.add(pair["input"].shape)
                layouts.add((
                    n, info["width"], info["y0"], info["y1"] - info["y0"],
                    tuple(info["gaps"]), info["left"], info["right"],
                ))
                if info["tops"].count(None):
                    unmarked_slots.add(info["tops"].index(None))
                if info["left"] == info["right"] == 0:
                    flags.add("edge_to_edge")
                if len(set(info["gaps"])) > 1:
                    flags.add("unequal_gaps")
                if info["left"] != info["right"]:
                    flags.add("asymmetric")
                if any(top is not None and top < info["y0"] for top in info["tops"]):
                    flags.add(regime + "_above")
                if _rank(info["tops"]) == list(range(n)):
                    flags.add("already_sorted")
                for mark in info["marks"]:
                    if not mark:
                        continue
                    run_lengths.add(len({y for y, _, _ in mark}))
                    if regime == "test":
                        if len(mark) == info["width"] and len({y for y, _, _ in mark}) == 1:
                            flags.add("test_stripe")
                        if any(v == info["bg"] for _, _, v in mark):
                            flags.add("test_hole")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert counts == {3, 4, 5, 6}
    assert {1, 2, 3, 4, 5}.issubset(widths)
    assert len(backgrounds) >= 8 and len(marker_colors) >= 8
    assert len(shapes) >= 40 and len(layouts) >= 150
    assert max(run_lengths) >= 8 and {1, 2, 3}.issubset(run_lengths)
    assert {0, 1, 2}.issubset(unmarked_slots)
    assert {
        "edge_to_edge", "unequal_gaps", "asymmetric", "train_above",
        "test_above", "test_stripe", "test_hole", "already_sorted",
    }.issubset(flags)
    for a, b in (
        (2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
        (2, None), (10, None), (2, 1), (5, 4),
    ):
        puzzle = generate_puzzle(8101, a, b)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(8101, a, b))
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (2, 3), (2, 4), (4, 4),
    ):
        try:
            generate_puzzle(3, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    assert not _same_puzzle(generate_puzzle(51, 4, 2), generate_puzzle(52, 4, 2))
