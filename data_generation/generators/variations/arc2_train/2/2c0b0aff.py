"""Generate ARC scenes that select the rectangle with the most complete pluses."""

import hashlib

import numpy as np


# Exact source pair digests are checked, while source dimensions remain legal.
_SOURCE_DIGESTS = {
    "c35e1b41291176339bb71f52988c390cc20f8b5f842654188ea3eade76d44144",
    "d5de9e32f6b53881d7463a12ab63738c0bedbec940bd8ef4dfe882d0d162c53e",
    "e4f28c1a5f5e65110e9884c1d4c8bc83db95c34cc6e84104864d34706eaaa61c",
    "0b28e926aa1686925ed98431703531aa484c00eb91605b360e720f059a1225d4",
    "9c61ac6e5aa698a9363fd3dd4c6c98842c7b3c872c31d2cc8c19c32674038492",
}


def _count_arg(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    a = _count_arg("num_train", num_train)
    b = _count_arg("num_test", num_test)
    if a is not None and not 2 <= a <= 10:
        raise ValueError("num_train must be in 2..10")
    if b is not None and not 1 <= b <= 4:
        raise ValueError("num_test must be in 1..4")
    if a is None and b is None:
        b = int(rng.integers(1, 5))
        a = int(rng.integers(b + 1, 11))
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _plus_count(tile, accent):
    mask = tile == accent
    if min(mask.shape) < 3:
        return 0
    return int(np.count_nonzero(
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
        & mask[1:-1, :-2] & mask[1:-1, 2:]
    ))


def _plus_cells(y, x):
    return ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))


def _clear(mask, cells):
    h, w = mask.shape
    for y, x in cells:
        if not (0 <= y < h and 0 <= x < w) or mask[y, x]:
            return False
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and (yy, xx) not in cells:
                return False
    return True


def _motif_mask(rng, shape, count):
    h, w = shape
    centers = [(y, x) for y in range(1, h - 1) for x in range(1, w - 1)]
    for _ in range(24):
        mask = np.zeros(shape, dtype=bool)
        placed = 0
        for index in rng.permutation(len(centers)):
            cells = _plus_cells(*centers[int(index)])
            if _clear(mask, cells):
                for y, x in cells:
                    mask[y, x] = True
                placed += 1
                if placed == count:
                    return mask
    return None


def _add_noise(rng, mask, target):
    h, w = mask.shape
    added = 0
    for _ in range(4):
        choices = [(y, x) for y in range(h) for x in range(w)]
        rng.shuffle(choices)
        for y, x in choices:
            if added >= target:
                break
            if target - added >= 2 and rng.random() < 0.43:
                dy, dx = ((0, 1) if rng.random() < 0.5 else (1, 0))
                cells = ((y, x), (y + dy, x + dx))
            else:
                cells = ((y, x),)
            if _clear(mask, cells):
                for yy, xx in cells:
                    mask[yy, xx] = True
                added += len(cells)
        if added >= target:
            break
    return added


def _make_tile(rng, shape, pluses, noise, carrier, accent):
    mask = _motif_mask(rng, shape, pluses) if pluses else np.zeros(shape, bool)
    if mask is None:
        return None
    if _add_noise(rng, mask, noise) < noise:
        return None
    if int(mask.sum()) * 2 >= mask.size:
        return None
    tile = np.full(shape, carrier, dtype=np.int8)
    tile[mask] = accent
    if _plus_count(tile, accent) != pluses:
        return None
    return tile


def _layout(rng, n, witness=False, congruent=False, quad=False):
    if quad:
        groups = [2, 2]
        h = int(rng.integers(8, 12))
        w = int(rng.integers(8, 13))
        heights = [h, h]
        vgap = int(rng.integers(1, 4))
        hgap = int(rng.integers(1, 4))
        canvas_h = int(rng.integers(2 * h + vgap, min(30, 2 * h + vgap + 6) + 1))
        canvas_w = int(rng.integers(2 * w + hgap, min(30, 2 * w + hgap + 6) + 1))
        shapes = [(h, w)] * 4
    elif witness:
        groups = [2, 1] if rng.random() < 0.5 else [1, 2]
        heights = [int(rng.integers(8, 12)) for _ in groups]
        vgap = int(rng.integers(1, 4))
        canvas_h = min(30, sum(heights) + vgap + int(rng.integers(0, 5)))
        canvas_w = int(rng.integers(25, 31))
        shapes = [(int(rng.integers(8, heights[ri] + 1)), int(rng.integers(8, 12)))
                  for ri, group in enumerate(groups) for _ in range(group)]
        hgap = int(rng.integers(1, 4))
        if any(sum(shapes[sum(groups[:ri]) + k][1] for k in range(group))
               + (group - 1) * hgap > canvas_w for ri, group in enumerate(groups)):
            return None
    else:
        rows = int(rng.integers(1, n + 1))
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, n), size=rows - 1, replace=False))
        groups = [b - a for a, b in zip([0] + cuts, cuts + [n])]
        rng.shuffle(groups)
        vgap = int(rng.integers(1, min(3, (30 - 5 * rows) // max(1, rows - 1)) + 1)) if rows > 1 else 0
        min_h = 5 * rows + vgap * (rows - 1)
        canvas_h = int(rng.integers(min_h, min(30, min_h + 11) + 1))
        heights = [5] * rows
        for i in rng.integers(0, rows, size=canvas_h - min_h):
            heights[int(i)] += 1
        max_group = max(groups)
        hgap = int(rng.integers(1, min(3, (30 - 5 * max_group) // max(1, max_group - 1)) + 1)) if max_group > 1 else 0
        min_w = 5 * max_group + hgap * (max_group - 1)
        canvas_w = int(rng.integers(min_w, min(30, min_w + 12) + 1))
        shapes = []
        for ri, group in enumerate(groups):
            row_budget = int(rng.integers(5 * group + hgap * (group - 1), canvas_w + 1))
            widths = [5] * group
            for i in rng.integers(0, group, size=row_budget - 5 * group - hgap * (group - 1)):
                widths[int(i)] += 1
            for width in widths:
                max_h = min(heights[ri], 16)
                max_w = min(width, 17)
                low_h = max(5, max_h - int(rng.integers(2, 6)))
                low_w = max(5, max_w - int(rng.integers(2, 6)))
                shapes.append((int(rng.integers(low_h, max_h + 1)),
                               int(rng.integers(low_w, max_w + 1))))
        if congruent:
            shape = (min(s[0] for s in shapes), min(s[1] for s in shapes))
            shapes = [shape] * n
    if congruent and witness:
        shape = (min(s[0] for s in shapes), min(s[1] for s in shapes))
        shapes = [shape] * n
    positions = []
    top_room = canvas_h - sum(heights) - vgap * (len(groups) - 1)
    y = int(rng.integers(0, top_room + 1))
    index = 0
    for ri, group in enumerate(groups):
        row_shapes = shapes[index:index + group]
        row_w = sum(s[1] for s in row_shapes) + hgap * (group - 1)
        if row_w > canvas_w:
            return None
        x = int(rng.integers(0, canvas_w - row_w + 1))
        for shape in row_shapes:
            yy = y + int(rng.integers(0, heights[ri] - shape[0] + 1))
            positions.append((yy, x))
            x += shape[1] + hgap
        index += group
        y += heights[ri] + vgap
    if not witness and n <= 4 and rng.random() < 0.22:
        scattered = []
        for shape in shapes:
            for _ in range(120):
                yy = int(rng.integers(0, canvas_h - shape[0] + 1))
                xx = int(rng.integers(0, canvas_w - shape[1] + 1))
                if all(yy + shape[0] < py or py + ps[0] < yy or
                       xx + shape[1] < px or px + ps[1] < xx
                       for (py, px), ps in zip(scattered, shapes)):
                    scattered.append((yy, xx))
                    break
            else:
                scattered = []
                break
        if len(scattered) == n:
            positions = scattered
    return (canvas_h, canvas_w), shapes, positions


def _palette(rng, convention, index):
    if convention[0] == "fixed":
        return convention[1]
    accent = convention[1][2]
    if convention[0] == "all_vary":
        return tuple(int(x) for x in rng.choice(10, 3, replace=False))
    choices = [x for x in range(10) if x != accent]
    other = rng.choice(choices, 2, replace=False)
    return int(other[0]), int(other[1]), accent


def _pair(rng, convention, palette, regime, index, first_slot):
    witness = regime == "train" and index == 0
    for _ in range(100):
        if witness:
            n = 3
            congruent = False
            quad = False
        else:
            quad = rng.random() < (.36 if regime == "test" else .12)
            n = 4 if quad else int(rng.choice([2, 3, 4, 5], p=[.16, .33, .34, .17]))
            congruent = quad or rng.random() < (.30 if regime == "test" else .22)
        layout = _layout(rng, n, witness, congruent, quad)
        if layout is None:
            continue
        canvas_shape, shapes, positions = layout
        eligible = [k for k, (h, w) in enumerate(shapes) if h >= 8 and w >= 8]
        challenge = ((quad and rng.random() < (.55 if regime == "test" else .15))
                     or (regime == "test" and not quad and len(eligible) >= 2
                         and rng.random() < .35))
        slots = list(range(n))
        if regime == "train" and index == 1 and first_slot is not None:
            slots = [x for x in slots if x != first_slot]
            if not slots:
                continue
        if challenge:
            slots = [x for x in slots if x in eligible]
            if len(slots) == 0:
                continue
        winner = int(rng.choice(slots))
        counts = [0] * n
        if witness:
            counts[winner] = 3
            distractor = (winner + int(rng.integers(1, n))) % n
            counts[distractor] = 2
            counts[3 - winner - distractor] = int(rng.integers(0, 2))
        elif challenge:
            counts[winner] = 3
            distractor = int(rng.choice([k for k in eligible if k != winner]))
            counts[distractor] = 2
            for k in range(n):
                if k not in (winner, distractor):
                    counts[k] = int(rng.integers(0, 3))
        else:
            cap = min(5, max(1, shapes[winner][0] * shapes[winner][1] // 17))
            counts[winner] = int(rng.integers(1, cap + 1))
            if min(shapes[winner]) >= 7:
                counts[winner] = max(2, counts[winner])
            for k in range(n):
                if k != winner:
                    ceiling = counts[winner]
                    if ceiling >= 3 and rng.random() < .29:
                        ceiling -= 1
                    counts[k] = int(rng.integers(0, ceiling))
        background, carrier, accent = palette
        tiles = []
        for k, shape in enumerate(shapes):
            if (witness or challenge) and k == winner:
                noise = int(rng.integers(0, 2))
            elif (witness or challenge) and k == distractor:
                noise = int(rng.integers(7, 10))
            else:
                noise = int(rng.integers(0, min(9, shape[0] * shape[1] // 7) + 1))
            tile = _make_tile(rng, shape, counts[k], noise, carrier, accent)
            if tile is None:
                break
            tiles.append(tile)
        if len(tiles) != n:
            continue
        if (witness or challenge) and not any(int(np.count_nonzero(t == accent)) >
                               int(np.count_nonzero(tiles[winner] == accent))
                               for k, t in enumerate(tiles) if k != winner):
            continue
        grid = np.full(canvas_shape, background, dtype=np.int8)
        for tile, (y, x) in zip(tiles, positions):
            h, w = tile.shape
            grid[y:y + h, x:x + w] = tile
        pair = {"input": grid, "output": tiles[winner].copy()}
        try:
            record = _check_pair(pair)
        except AssertionError:
            continue
        if regime == "train" and index == 1 and record["winner"] == first_slot:
            continue
        return pair, record["winner"]
    raise RuntimeError("could not construct a valid pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    mode = str(rng.choice(["fixed", "roles_vary", "all_vary"], p=[.40, .45, .15]))
    convention = (mode, colors)
    result = {"train": [], "test": []}
    used = set()
    first_slot = None
    previous_palette = None
    for regime, number in (("train", ntrain), ("test", ntest)):
        for index in range(number):
            for _ in range(20):
                palette = _palette(rng, convention, index)
                if mode != "fixed" and regime == "train" and index == 1 and palette == previous_palette:
                    continue
                pair, slot = _pair(rng, convention, palette, regime, index, first_slot)
                key = _grid_key(pair["input"])
                if key not in used:
                    used.add(key)
                    result[regime].append(pair)
                    if regime == "train" and index == 0:
                        first_slot = slot
                        previous_palette = palette
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(result)
    return result


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_digest(pair):
    digest = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name]
        digest.update(bytes(grid.shape))
        digest.update(np.asarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _rectangles(grid, background):
    mask = grid != background
    seen = np.zeros(grid.shape, bool)
    result = []
    for sy, sx in np.argwhere(mask):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        ys, xs = zip(*cells)
        y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
        assert len(cells) == (y1 - y0) * (x1 - x0)
        result.append((grid[y0:y1, x0:x1], (y0, x0)))
    return result


def _valid_accent_fragments(tile, accent):
    mask = tile == accent
    seen = np.zeros(tile.shape, bool)
    for sy, sx in np.argwhere(mask):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        cells = set()
        while stack:
            y, x = stack.pop()
            cells.add((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if len(cells) in (1, 2):
            continue
        if len(cells) == 5 and any(set(_plus_cells(y, x)) == cells for y, x in cells):
            continue
        return False
    return True


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, b = pair["input"], pair["output"]
    for grid in (a, b):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.all((0 <= grid) & (grid <= 9))
    colors = set(int(x) for x in np.unique(a))
    outcolors = set(int(x) for x in np.unique(b))
    assert len(colors) == 3 and len(outcolors) == 2 and outcolors < colors
    background = next(iter(colors - outcolors))
    rects = _rectangles(a, background)
    assert 2 <= len(rects) <= 5
    assert all(set(int(x) for x in np.unique(t)) == outcolors for t, _ in rects)
    accent_candidates = [c for c in outcolors if all(np.count_nonzero(t == c) * 2 < t.size for t, _ in rects)]
    assert len(accent_candidates) == 1
    accent = accent_candidates[0]
    assert all(_valid_accent_fragments(t, accent) for t, _ in rects)
    pluses = [_plus_count(t, accent) for t, _ in rects]
    assert pluses.count(max(pluses)) == 1 and max(pluses) >= 1
    winner = pluses.index(max(pluses))
    assert np.array_equal(b, rects[winner][0])
    assert a.shape != b.shape
    assert _pair_digest(pair) not in _SOURCE_DIGESTS
    return {"n": len(rects), "winner": winner, "pluses": pluses,
            "rects": rects, "accent": accent, "background": background,
            "colors": (background, next(iter(outcolors - {accent})), accent)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    records = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            record = _check_pair(pair)
            key = _grid_key(pair["input"])
            pkey = (key, _grid_key(pair["output"]))
            assert key not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pkey)
            records.append((regime, pair, record))
    train = [x for x in records if x[0] == "train"]
    assert any(any(int(np.count_nonzero(t == rec["accent"])) >=
                   int(np.count_nonzero(pair["output"] == rec["accent"]))
                   for k, (t, _) in enumerate(rec["rects"]) if k != rec["winner"])
               for _, pair, rec in train)
    assert len({rec["winner"] for _, _, rec in train}) >= 2
    # Color semantics are observable locally: the carrier is the majority color
    # in every rectangle, so recoloring introduces no hidden mapping.


def self_test() -> None:
    records = []
    automatic = []
    signatures = set()
    shapes = set()
    palettes = set()
    train_counts, test_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        automatic.append(puzzle)
        again = generate_puzzle(seed)
        assert all(np.array_equal(a[name], b[name]) for regime in ("train", "test")
                   for a, b in zip(puzzle[regime], again[regime]) for name in ("input", "output"))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                rec = _check_pair(pair)
                records.append((regime, pair, rec))
                shapes.add(pair["output"].shape)
                palettes.add(rec["colors"])
                signatures.add((pair["input"].shape, tuple((t.shape, pos) for t, pos in rec["rects"])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(palettes) >= 30 and len(signatures) >= 150
    assert {rec["n"] for _, _, rec in records} >= {2, 3, 4, 5}
    assert any(len({t.shape for t, _ in rec["rects"]}) == 1
               for regime, pair, rec in records if regime == "test")
    assert any(len({t.shape for t, _ in rec["rects"]}) == 1
               for regime, pair, rec in records if regime == "train")
    observed = {"train": {"counts": set(), "slots": set(), "margins": set(),
                          "congruent": set(), "shape_kind": set(), "area_rank": set(),
                          "pixel_conflict": 0, "pixel_tie": 0, "density_conflict": 0,
                          "zero_loser": 0, "edge": 0, "one_row": 0, "one_column": 0},
                "test": {"counts": set(), "slots": set(), "margins": set(),
                         "congruent": set(), "shape_kind": set(), "area_rank": set(),
                         "pixel_conflict": 0, "pixel_tie": 0, "density_conflict": 0,
                         "zero_loser": 0, "edge": 0, "one_row": 0, "one_column": 0}}
    for regime, pair, rec in records:
        dest = observed[regime]
        tiles = [t for t, _ in rec["rects"]]
        shapes_here = [t.shape for t in tiles]
        positions = [pos for _, pos in rec["rects"]]
        counts = rec["pluses"]
        win = rec["winner"]
        pixels = [int(np.count_nonzero(t == rec["accent"])) for t in tiles]
        densities = [p / t.size for p, t in zip(pixels, tiles)]
        area = tiles[win].size
        dest["counts"].add(rec["n"])
        dest["slots"].add(win)
        dest["margins"].add(max(counts) - max(v for k, v in enumerate(counts) if k != win))
        dest["congruent"].add(len(set(shapes_here)) == 1)
        dest["shape_kind"].add("tall" if shapes_here[win][0] > shapes_here[win][1]
                               else "wide" if shapes_here[win][0] < shapes_here[win][1]
                               else "square")
        dest["area_rank"].add((any(t.size < area for t in tiles),
                               any(t.size > area for t in tiles)))
        dest["pixel_conflict"] += any(v > pixels[win] for k, v in enumerate(pixels) if k != win)
        dest["pixel_tie"] += any(v == pixels[win] for k, v in enumerate(pixels) if k != win)
        dest["density_conflict"] += any(v > densities[win] for k, v in enumerate(densities) if k != win)
        dest["zero_loser"] += 0 in counts
        dest["edge"] += any(y == 0 or x == 0 or y + t.shape[0] == pair["input"].shape[0]
                            or x + t.shape[1] == pair["input"].shape[1]
                            for t, (y, x) in rec["rects"])
        dest["one_row"] += max(y for y, x in positions) < min(y + t.shape[0]
                                                               for t, (y, x) in rec["rects"])
        dest["one_column"] += max(x for y, x in positions) < min(x + t.shape[1]
                                                                  for t, (y, x) in rec["rects"])
    for regime in ("train", "test"):
        dest = observed[regime]
        assert dest["counts"] == {2, 3, 4, 5}
        assert dest["slots"] == {0, 1, 2, 3, 4}
        assert dest["congruent"] == {False, True}
        assert dest["shape_kind"] == {"tall", "wide", "square"}
        assert dest["pixel_conflict"] >= 5 and dest["pixel_tie"] >= 1
        assert dest["density_conflict"] >= 5 and dest["zero_loser"] >= 5
        assert dest["edge"] >= 5 and dest["one_row"] >= 5 and dest["one_column"] >= 5
        assert {1, 2} <= dest["margins"]
        assert {(False, True), (True, False), (True, True)} <= dest["area_rank"]
    assert any(len({_check_pair(pair)["background"] for pair in p["train"]}) > 1
               for p in automatic)
    assert any(len({_check_pair(pair)["accent"] for pair in p["train"]}) > 1
               for p in automatic)
    assert any(any(len({t.shape for t, _ in _check_pair(pair)["rects"]}) == 1
                   for pair in p["train"]) and any(
                       len({t.shape for t, _ in _check_pair(pair)["rects"]}) > 1
                       for pair in p["test"]) for p in automatic)
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(20000 + i, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for i, (a, b) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(30000 + i, a, b)
        if a is not None:
            assert len(p["train"]) == a
        if b is not None:
            assert len(p["test"]) == b
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
                   {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(40000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
