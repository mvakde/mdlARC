"""Cyclic hub strips supply a coarse field for exact-span rectangle decoding.

Only Python's standard library and NumPy are used. No evidence is loaded at
runtime, and solve() reads one input only. Running this file runs bounded tests.
"""

import hashlib
import itertools
import json

import numpy as np


AXES = (1, 2, 3, 4, 6, 8)
CANVAS = 26
GAP = 2


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("Expected an integer ARC grid of dimensions 1..30")
    return array.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _runs(row, background):
    """Maximal nonbackground constant-color runs, with exclusive right ends."""
    result = []
    left = 0
    while left < len(row):
        color = int(row[left])
        right = left + 1
        while right < len(row) and int(row[right]) == color:
            right += 1
        if color != background:
            result.append((color, left, right))
        left = right
    return result


def _rectangles(coarse, background):
    rows = [_runs(row, background) for row in coarse]
    result = []
    for top, runs in enumerate(rows):
        previous = set(rows[top - 1]) if top else set()
        for color, left, right in runs:
            if (color, left, right) in previous:
                continue
            bottom = top + 1
            while bottom < len(rows) and (color, left, right) in rows[bottom]:
                bottom += 1
            result.append((color, top, bottom, left, right))
    return result


def _render(coarse, background, merge=True):
    rows, cols = coarse.shape
    if rows not in AXES or cols not in AXES:
        raise ValueError("Coarse axes must divide 24 with pitch greater than two")
    py, px = 24 // rows, 24 // cols
    output = np.full((CANVAS, CANVAS), background, dtype=np.int8)
    if merge:
        rectangles = _rectangles(coarse, background)
    else:
        rectangles = [(color, r, r + 1, left, right)
                      for r, row in enumerate(coarse)
                      for color, left, right in _runs(row, background)]
    for color, top, bottom, left, right in rectangles:
        output[GAP + top * py:bottom * py,
               GAP + left * px:right * px] = color
    return output


def _bridge_render(coarse, background, overlap=False):
    """Author-owned cross-check: paint cell cores and then local bridges.

    overlap=True is an intentionally wrong alternative that connects partial
    overlaps, used to verify that training rejects this tempting shortcut.
    """
    rows, cols = coarse.shape
    py, px = 24 // rows, 24 // cols
    output = np.full((CANVAS, CANVAS), background, dtype=np.int8)
    for r, row in enumerate(coarse):
        for c, value in enumerate(row):
            if value == background:
                continue
            y, x = GAP + r * py, GAP + c * px
            output[y:y + py - GAP, x:x + px - GAP] = value
            if c + 1 < cols and row[c + 1] == value:
                output[y:y + py - GAP, x + px - GAP:x + px] = value
    runs = [_runs(row, background) for row in coarse]
    for r in range(rows - 1):
        for color, left, right in runs[r]:
            for other, lo, hi in runs[r + 1]:
                if color != other:
                    continue
                if (left, right) == (lo, hi) or (overlap and max(left, lo) < min(right, hi)):
                    l, h = max(left, lo), min(right, hi)
                    output[(r + 1) * py:GAP + (r + 1) * py,
                           GAP + l * px:h * px] = color
    return output


def _primitive_period(word):
    """Divisors only: ABA denotes a three-slice word, never a prefix of AB."""
    return next(p for p in range(1, len(word) + 1)
                if len(word) % p == 0
                and np.array_equal(word, word[np.arange(len(word)) % p]))


def _parse_down(grid, background):
    """Parse the canonical, downward-pointing arrangement from input pixels."""
    rows, cols = grid.shape[0] - 4, grid.shape[1] - 2
    if rows not in AXES or cols not in AXES:
        raise ValueError("Invalid coarse field dimensions")
    if not np.all(grid[0] == background):
        raise ValueError("Expected a blank row behind the hubs")
    edges = [int(grid[2, c]) for c in (0, grid.shape[1] - 1)
             if grid[2, c] != background]
    if not edges or len(set(edges)) != 1:
        raise ValueError("An edge-touching hub shoulder must identify the cap")
    hub_color = edges[0]
    face = grid[3, 1:-1]
    lanes = _runs(face, background)
    if not lanes or any(color != hub_color for color, _, _ in lanes):
        raise ValueError("Malformed hub faces")
    expected_cap = np.full((4, cols + 2), background, dtype=np.int8)
    for _, left, right in lanes:
        expected_cap[1, left + 1:right + 1] = hub_color
        expected_cap[2, left:right + 2] = hub_color
        expected_cap[3, left + 1:right + 1] = hub_color
    if any(b[1] - a[2] < 3 for a, b in zip(lanes, lanes[1:])):
        raise ValueError("Hubs need a blank column between their shoulders")
    if not np.array_equal(expected_cap, grid[:4]):
        raise ValueError("Expected separated clipped three-row hubs")
    if not (lanes[0][1] == 0 or lanes[-1][2] == cols):
        raise ValueError("At least one strip must have an edge-aligned shoulder")
    if np.any(grid[4:, (0, grid.shape[1] - 1)] != background):
        raise ValueError("Payload may not enter the side borders")
    visible = grid[4:, 1:-1].copy()
    if np.any(visible == hub_color):
        raise ValueError("Hub color is reserved for the cap")
    claimed = np.zeros(visible.shape, dtype=bool)
    strips = []
    for _, left, right in lanes:
        length = 0
        while length < rows and np.all(visible[length, left:right] != background):
            length += 1
        if not length or np.any(visible[length:, left:right] != background):
            raise ValueError("A visible strip must be one solid rectangular prefix")
        claimed[:length, left:right] = True
        word = visible[:length, left:right].copy()
        strips.append(dict(left=left, right=right, length=length, word=word,
                           period=_primitive_period(word), complete=length == rows))
    if np.any((visible != background) & ~claimed):
        raise ValueError("Unattached material in the coarse field")
    return dict(background=background, hub_color=hub_color, rows=rows, cols=cols,
                visible=visible, strips=strips)


def _decode(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if any(int(grid[r, c]) != background
           for r, c in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("The four corners must identify the background")
    candidates = []
    for upward in (False, True):
        canonical = grid[::-1] if upward else grid
        try:
            info = _parse_down(canonical, background)
        except ValueError:
            continue
        info["upward"] = upward
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("The input must have one unambiguous hub cap")
    return candidates[0]


def _complete(info, prefix_shortcut=False):
    coarse = info["visible"].copy()
    for s in info["strips"]:
        if s["complete"]:
            continue
        word = s["word"]
        if prefix_shortcut:
            # Wrong nondividing-prefix hypothesis, for training sufficiency checks.
            p = next(p for p in range(1, len(word) + 1)
                     if all(np.array_equal(word[i], word[i % p]) for i in range(len(word))))
            word = word[:p]
        for r in range(s["length"], info["rows"]):
            coarse[r, s["left"]:s["right"]] = word[r % len(word)]
    return coarse


def _on_page(coarse, info):
    return coarse[::-1].copy() if info["upward"] else coarse.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Continue hub-relative words, then render horizontal runs in page order."""
    info = _decode(input_grid)
    return _render(_on_page(_complete(info), info), info["background"])


def _palette(rng, minimum):
    colors = [int(c) for c in rng.permutation(10)]
    count = int(rng.integers(minimum, 9))
    return colors[0], colors[1], colors[2:2 + count]


def _unequal_word(rng, width, palette):
    """An interval overlap witness with the cyclic seam P,Q,P | P,Q,P."""
    a, b = [int(x) for x in rng.choice(palette, 2, replace=False)]
    spans = [(l, h) for l in range(width) for h in range(l + 1, width + 1)
             if h - l < width]
    choices = [(p, q) for p in spans for q in spans
               if p != q and max(p[0], q[0]) < min(p[1], q[1])
               and max(p[1] - p[0], q[1] - q[0]) >= 2]
    p, q = choices[int(rng.integers(len(choices)))]
    first, second = np.full(width, b, dtype=np.int8), np.full(width, b, dtype=np.int8)
    first[p[0]:p[1]], second[q[0]:q[1]] = a, a
    return np.array([first, second, first], dtype=np.int8)


def _individual_word(rng, width, palette):
    a, b, c = [int(x) for x in rng.choice(palette, 3, replace=False)]
    length = int(rng.integers(2, width))
    left = int(rng.integers(width - length + 1))
    word = np.array([[b] * width, [c] * width], dtype=np.int8)
    word[:, left:left + length] = a
    return word


def _random_word(rng, width, period, palette):
    word = rng.choice(palette, size=(period, width)).astype(np.int8)
    style = int(rng.integers(5))
    if style == 0:
        word[:] = word[0]  # Long exact stacks, including multicolor slices.
    elif style == 1:
        word[:] = rng.choice(palette, size=(period, 1))
    elif style == 2 and width >= 3 and period >= 3 and len(palette) >= 2:
        first = _unequal_word(rng, width, palette)
        word[:] = first[np.arange(period) % 3]
    elif style == 3 and width >= 3 and len(palette) >= 3:
        first = _individual_word(rng, width, palette)
        word[:] = first[np.arange(period) % 2]
    return word


def _layout(rng, cols, widths):
    if len(widths) == 1:
        return [(0, widths[0])]
    first, second = widths
    start = int(rng.integers(first + 3, cols - second + 1))
    return [(0, first), (start, start + second)]


def _make_pair(rng, rows, kind):
    background, hub, palette = _palette(rng, 3 if kind == "individual" else 2)
    if kind == "seam":
        cols = int(rng.choice((3, 4, 6, 8)))
        widths = [int(rng.integers(3, cols + 1))]
        main = 0
    elif kind == "individual":
        cols = 8
        main_width = int(rng.choice((3, 4)))
        other_width = int(rng.integers(1, 8 - main_width - 3 + 1))
        widths = [main_width, other_width]
        main = 0
        if rng.random() < 0.5:
            widths.reverse()
            main = 1
    else:
        cols = int(rng.choice(AXES))
        if cols >= 6 and rng.random() < 0.55:
            first = int(rng.integers(1, cols - 3))
            second = int(rng.integers(1, cols - first - 3 + 1))
            widths = [first, second]
        else:
            widths = [int(rng.integers(1, cols + 1))]
        main = -1
    layout = _layout(rng, cols, widths)
    visible = np.full((rows, cols), background, dtype=np.int8)
    completed = visible.copy()
    for i, (left, right) in enumerate(layout):
        width = right - left
        if kind == "seam":
            word = _unequal_word(rng, width, palette)
            length = 3
            full = word[np.arange(rows) % 3].copy()
        elif kind == "individual" and i == main:
            word = _individual_word(rng, width, palette)
            length = 2
            full = word[np.arange(rows) % 2].copy()
        elif kind == "individual":
            # A completed, deliberately irregular tail: never regenerate it.
            a, b = [int(x) for x in rng.choice(palette, 2, replace=False)]
            full = np.full((rows, width), a, dtype=np.int8)
            full[-1, int(rng.integers(width))] = b
            length = rows
        else:
            # At least the first test/random lane is unfinished when rows > 1.
            complete = rows == 1 or (i > 0 and rng.random() < 0.4)
            if complete:
                length = rows
                full = _random_word(rng, width, rows, palette)
                if rows >= 3 and rng.random() < 0.5:
                    full[:2] = full[0]
                    full[-1, -1] = next(c for c in palette if c != int(full[0, -1]))
            else:
                period = int(rng.integers(1, min(5, rows - 1) + 1))
                word = _random_word(rng, width, period, palette)
                copies = int(rng.integers(1, (rows - 1) // period + 1))
                length = period * copies
                full = word[np.arange(rows) % period].copy()
        visible[:length, left:right] = full[:length]
        completed[:, left:right] = full
    grid = np.full((rows + 4, cols + 2), background, dtype=np.int8)
    for left, right in layout:
        grid[1, left + 1:right + 1] = hub
        grid[2, left:right + 2] = hub
        grid[3, left + 1:right + 1] = hub
    grid[4:, 1:-1] = visible
    if rng.random() < 0.5:
        grid = grid[:, ::-1].copy()
        completed = completed[:, ::-1].copy()
    if rng.random() < 0.5:
        grid = grid[::-1].copy()
        completed = completed[::-1].copy()
    return {"input": grid, "output": _render(completed, background)}


def _counts(num_train, num_test):
    result = []
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
        result.append(int(value))
    if result[0] <= result[1]:
        raise ValueError("num_train must exceed num_test")
    return result


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    # At most 13 prior inputs exclude at most 13 injective recolorings. Every
    # recoloring preserves geometric roles, witnesses and the intended rule.
    colors = np.unique(pair["input"])
    for values in itertools.permutations(range(10), len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise AssertionError("Impossible: finite recoloring repair exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    anchor_rows = int(rng.choice((6, 8)))
    held_out_rows = 8 if anchor_rows == 6 else 6
    specs = [(anchor_rows, "seam"), (4, "individual")]
    for _ in range(num_train - 2):
        specs.append((int(rng.choice((1, 2, 3, 4, anchor_rows))), "random"))
    puzzle, used = {"train": [], "test": []}, set()
    for split, split_specs in (("train", specs),
                               ("test", [(held_out_rows, "random")] * num_test)):
        for rows, kind in split_specs:
            pair = _unique(_make_pair(rng, rows, kind), used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(input_grid):
    info = _decode(input_grid)
    canonical = _complete(info)
    coarse = _on_page(canonical, info)
    bg = info["background"]
    run_rows = [_runs(row, bg) for row in coarse]
    exact = sum(len(set(a) & set(b)) for a, b in zip(run_rows, run_rows[1:]))
    unequal = sum(ca == cb and (la, ha) != (lb, hb) and max(la, lb) < min(ha, hb)
                  for a, b in zip(run_rows, run_rows[1:])
                  for ca, la, ha in a for cb, lb, hb in b)
    created_exact = created_unequal = 0
    canonical_runs = [_runs(row, bg) for row in canonical]
    for s in info["strips"]:
        for r in range(max(1, s["length"]), info["rows"]):
            previous = [run for run in canonical_runs[r - 1]
                        if s["left"] <= run[1] < run[2] <= s["right"]]
            current = [run for run in canonical_runs[r]
                       if s["left"] <= run[1] < run[2] <= s["right"]]
            created_exact += len(set(previous) & set(current))
            created_unequal += sum(a == b and (l, h) != (lo, hi) and max(l, lo) < min(h, hi)
                                   for a, l, h in previous for b, lo, hi in current)
    return info, coarse, dict(exact=exact, unequal=unequal,
                              created_exact=created_exact, created_unequal=created_unequal)


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("Puzzle needs exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, training, testing = set(), [], []
    evidence = dict(no_continuation=False, no_merge=False, overlap=False,
                    prefix=False, whole_rows=False, completed_tail=False,
                    created_exact=False, created_unequal=False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("Pairs need exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in used, "Duplicate input"
            used.add(_key(inp))
            info, coarse, f = _features(inp)
            bg, hc = info["background"], info["hub_color"]
            assert out.shape == (CANVAS, CANVAS)
            assert np.array_equal(out, _render(coarse, bg)), "Incorrect rectangle output"
            assert np.array_equal(out, _bridge_render(coarse, bg)), "Bridge check disagrees"
            assert not np.any(out == hc), "Hubs are instructions, not rectangle labels"
            assert np.all(out[:2] == bg) and np.all(out[-2:] == bg)
            assert np.all(out[:, :2] == bg) and np.all(out[:, -2:] == bg)
            assert set(np.unique(out)) == set(np.unique(coarse)) | {bg}
            (training if split == "train" else testing).append(info)
            if split == "test":
                assert any(not s["complete"] for s in info["strips"])
                continue
            visible = _on_page(info["visible"], info)
            evidence["no_continuation"] |= not np.array_equal(out, _render(visible, bg))
            evidence["no_merge"] |= not np.array_equal(out, _render(coarse, bg, merge=False))
            evidence["overlap"] |= not np.array_equal(out, _bridge_render(coarse, bg, overlap=True))
            short = _on_page(_complete(info, prefix_shortcut=True), info)
            evidence["prefix"] |= not np.array_equal(out, _render(short, bg))
            runs = [_runs(row, bg) for row in coarse]
            evidence["whole_rows"] |= any(set(a) & set(b) and not np.array_equal(coarse[r], coarse[r + 1])
                                            for r, (a, b) in enumerate(zip(runs, runs[1:])))
            evidence["created_exact"] |= f["created_exact"] > 0
            evidence["created_unequal"] |= f["created_unequal"] > 0
            for s in info["strips"]:
                word = s["word"]
                if (s["complete"] and len(word) >= 3
                        and np.array_equal(word[0], word[1])
                        and not np.array_equal(word[0], word[-1])):
                    normalized = _complete(info)
                    normalized[:, s["left"]:s["right"]] = word[0]
                    evidence["completed_tail"] |= not np.array_equal(
                        out, _render(_on_page(normalized, info), bg))
    assert all(evidence.values()), f"Missing training evidence: {evidence}"
    assert {x["rows"] for x in training}.isdisjoint({x["rows"] for x in testing})


def _normalized(coarse, bg):
    mapping = {bg: 0}
    return coarse.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in coarse.flat)


def self_test():
    """42 streamed seeds; no files, retained puzzles, logs, or execution traces."""
    assert _primitive_period(np.array([[2], [3], [2]])) == 3
    assert _primitive_period(np.array([[2], [3], [2], [3]])) == 2
    shapes, coarse_shapes, layouts, backgrounds, hubs = set(), set(), set(), set(), set()
    widths, periods, counts, directions, depths, rectangle_counts = set(), set(), set(), set(), set(), set()
    signatures, exact, unequal, created_exact, created_unequal = set(), 0, 0, 0, 0
    pairs = clipping = complete = blank_columns = 0
    for seed in range(42):
        # Include the minimum-count curriculum in the same bounded 42-seed run.
        p = generate_puzzle(seed, 2 if seed % 3 == 0 else 4, 1)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                signature.update(bytes(inp.shape))
                signature.update(inp.tobytes())
                info, coarse, f = _features(inp)
                assert np.array_equal(out, solve(inp))
                pairs += 1
                shapes.add(inp.shape)
                coarse_shapes.add(coarse.shape)
                layouts.add(_normalized(coarse, info["background"]))
                backgrounds.add(info["background"])
                hubs.add(info["hub_color"])
                counts.add(len(info["strips"]))
                directions.add(info["upward"])
                depths.update(bottom - top for _, top, bottom, _, _ in _rectangles(coarse, info["background"]))
                rectangle_counts.add(len(_rectangles(coarse, info["background"])))
                exact += f["exact"] > 0
                unequal += f["unequal"] > 0
                created_exact += f["created_exact"] > 0
                created_unequal += f["created_unequal"] > 0
                blank_columns += bool(np.any(np.all(coarse == info["background"], axis=0)))
                for s in info["strips"]:
                    widths.add(s["right"] - s["left"])
                    periods.add(s["period"])
                    complete += s["complete"]
                    clipping += not s["complete"] and info["rows"] % s["period"] != 0
        signatures.add(signature.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 20 and len(coarse_shapes) >= 20
    assert len(layouts) >= 100 and len(backgrounds) == len(hubs) == 10
    assert widths >= {1, 2, 3, 4, 5, 6} and periods >= {1, 2, 3, 4, 5}
    assert counts == {1, 2} and directions == {False, True}
    assert max(depths) >= 6 and len(rectangle_counts) >= 12
    assert min(exact, unequal, created_exact, created_unequal) >= 42
    assert clipping >= 20 and complete >= 42 and blank_columns >= 42
    # All 30 legal count combinations, bounded to one seed each.
    legal_counts = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123456, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
            legal_counts += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    p, q = generate_puzzle(987654, 5, 3), generate_puzzle(987654, 5, 3)
    assert all(_key(a[k]) == _key(b[k]) for split in ("train", "test")
               for a, b in zip(p[split], q[split]) for k in ("input", "output"))
    # All 36 legal field shapes, both vertical directions, both reflections.
    # These direct fixtures also cover one-row/all-completed family members.
    solver_fixtures = 0
    for rows in AXES:
        for cols in AXES:
            period = min(3, max(1, rows - 1))
            word = np.array([[2 + (r + c) % 3 for c in range(cols)]
                             for r in range(period)], dtype=np.int8)
            full = word[np.arange(rows) % period]
            g = np.zeros((rows + 4, cols + 2), dtype=np.int8)
            g[1, 1:-1] = g[3, 1:-1] = 1
            g[2] = 1
            g[4:4 + period, 1:-1] = word
            for upward, mirror in itertools.product((False, True), repeat=2):
                inp = g[:, ::-1] if mirror else g.copy()
                coarse = full[:, ::-1] if mirror else full.copy()
                if upward:
                    inp, coarse = inp[::-1], coarse[::-1]
                assert np.array_equal(solve(inp), _render(coarse, 0))
                solver_fixtures += 1
    # Output mutation and duplicate-input rejection are genuinely exercised.
    damaged = generate_puzzle(31, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Altered output accepted")
    duplicated = generate_puzzle(32, 2, 1)
    duplicated["test"][0] = duplicated["train"][0]
    try:
        validate_generated_puzzle(duplicated)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate input accepted")
    original = generate_puzzle(33, 2, 1)["train"][0]
    repaired = _unique(original, {_key(original["input"])})
    assert _key(original["input"]) != _key(repaired["input"])
    assert np.array_equal(solve(repaired["input"]), repaired["output"])
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), coarse_shapes=len(coarse_shapes),
                color_normalized_layouts=len(layouts), backgrounds=sorted(backgrounds),
                hub_colors=sorted(hubs), strip_counts=sorted(counts), widths=sorted(widths),
                periods=sorted(periods), stack_depths=sorted(depths),
                rectangle_counts=sorted(rectangle_counts), exact_pairs=exact,
                unequal_overlap_pairs=unequal, continuation_created_exact_pairs=created_exact,
                continuation_created_unequal_pairs=created_unequal,
                clipped_strips=clipping, completed_strips=complete,
                blank_column_pairs=blank_columns, legal_count_combinations=legal_counts,
                direct_solver_fixtures=solver_fixtures,
                semantic_verification="author-owned checks only")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
