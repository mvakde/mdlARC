"""Register locally exceptional perforated plaques along a unit staircase.

Only standard-library modules and NumPy are used. Generation constructs labels
from its authored bands; solve independently parses the visible input.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("a grid must be a 1..30 integer rectangle with colors 0..9")
    return a


def _runs(flags):
    """Half-open intervals of consecutive true entries."""
    starts, result = None, []
    for i, flag in enumerate(list(flags) + [False]):
        if flag and starts is None:
            starts = i
        elif not flag and starts is not None:
            result.append((starts, i))
            starts = None
    return result


def _connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    todo = [cells.pop()]
    while todo:
        r, c = todo.pop()
        for dr, dc in _DIRS:
            p = (r + dr, c + dc)
            if p in cells:
                cells.remove(p)
                todo.append(p)
    return not cells


def _valid_holes(height, width, holes):
    holes = set(holes)
    if any((r + dr, c + dc) in holes for r, c in holes
           for dr, dc in ((1, 0), (0, 1))):
        return False
    mask = np.ones((height, width), dtype=bool)
    for r, c in holes:
        mask[r, c] = False
    return _connected(mask)


@lru_cache(maxsize=None)
def _hole_options(height, width, count):
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    return tuple(p for p in combinations(cells, count)
                 if _valid_holes(height, width, p))


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the input needs a uniform outer background frame")
    row_runs = _runs(np.any(a != bg, axis=1))
    if len(row_runs) < 2:
        raise ValueError("at least two separated bands are required")
    bands = []
    for top, bottom in row_runs:
        intervals = _runs(np.any(a[top:bottom] != bg, axis=0))
        if len(intervals) < 3:
            raise ValueError("each band needs at least three plaques")
        tiles, dimensions = [], set()
        for left, right in intervals:
            tile = a[top:bottom, left:right]
            height, width = tile.shape
            if height < 3 or width < 3:
                raise ValueError("plaques need an interior")
            fg = int(tile[0, 0])
            if (fg == bg or not np.all(tile[0] == fg) or not np.all(tile[-1] == fg)
                    or not np.all(tile[:, 0] == fg) or not np.all(tile[:, -1] == fg)
                    or not set(map(int, np.unique(tile))) <= {bg, fg}):
                raise ValueError("a plaque needs a complete monochrome border")
            holes = tuple(zip(*np.nonzero(tile == bg)))
            if not _valid_holes(height, width, holes):
                raise ValueError("holes must be isolated and foreground connected")
            dimensions.add(tile.shape)
            tiles.append({"left": left, "right": right, "tile": tile.copy(),
                          "holes": holes, "count": len(holes), "color": fg})
        if len(dimensions) != 1:
            raise ValueError("plaques in a band must have equal dimensions")
        frequencies = Counter(t["count"] for t in tiles)
        if len(frequencies) != 2 or sorted(frequencies.values()) != [1, len(tiles) - 1]:
            raise ValueError("each band must have exactly one minority hole count")
        common = next(k for k, n in frequencies.items() if n == len(tiles) - 1)
        unusual = next(k for k, n in frequencies.items() if n == 1)
        odd = next(i for i, t in enumerate(tiles) if t["count"] == unusual)
        left, right = intervals[0][0], intervals[-1][1]
        bands.append({"top": top, "bottom": bottom, "left": left, "right": right,
                      "strip": a[top:bottom, left:right].copy(), "tiles": tiles,
                      "anchor": tiles[odd]["left"] - left, "odd": odd,
                      "common": common, "exceptional": unusual})
    return bg, bands


def _origins(bands):
    n = len(bands)
    raw = [n - 1 - i - b["anchor"] for i, b in enumerate(bands)]
    minimum = min(raw)
    origins = [x - minimum for x in raw]
    width = max(x + b["strip"].shape[1] for x, b in zip(origins, bands))
    return origins, width


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible geometry and local hole-count frequencies."""
    bg, bands = _parse(input_grid)
    origins, width = _origins(bands)
    top = bands[0]["top"]
    height = bands[-1]["bottom"] - top
    if width > 30:
        raise ValueError("registered arrangement exceeds the ARC width limit")
    out = np.full((height, width), bg, dtype=np.int8)
    for b, x in zip(bands, origins):
        y = b["top"] - top
        h, w = b["strip"].shape
        out[y:y + h, x:x + w] = b["strip"]
    return out


def _integer(name, value, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _key(a):
    return (a.shape, a.astype(np.uint8).tobytes())


def _candidate(rng, band_count, bg, pair_index, is_test, mixed_allowed, safe=False):
    if not is_test and pair_index == 0:
        ab = [1, 2]
    elif not is_test and pair_index == 1:
        ab = list(((1, 3), (2, 3), (0, 2), (0, 3), (1, 2))[int(rng.integers(5))])
    else:
        ab = [int(v) for v in rng.choice(4, size=2, replace=False)]
    rng.shuffle(ab)
    widths = [4, 5]
    rng.shuffle(widths)
    heights = [4, 5]
    rng.shuffle(heights)
    edge_order = [0, -1]
    rng.shuffle(edge_order)
    bands, y = [], 0
    for i in range(band_count):
        force_mixed = mixed_allowed and not is_test and pair_index == 1 and i == 2
        mixed = force_mixed or (mixed_allowed and (is_test or pair_index >= 2)
                                and rng.random() < 0.30)
        n = 4 if mixed else int(rng.choice((3, 4, 5), p=(0.72, 0.24, 0.04)))
        counts = tuple(ab if i == 0 else reversed(ab)) if i < 2 else None
        if safe:
            n = 4 if force_mixed else 3
            mixed = force_mixed
            h, w = ((5, 4), (4, 5))[i] if i < 2 else ((5, 3) if force_mixed else (3, 3))
            if i >= 2:
                counts = (0, 1) if i % 2 == 0 else (1, 0)
        elif not is_test and pair_index == 0:
            n, mixed = 3, False
            h, w = heights[i], widths[i]
        else:
            required = max(counts) if counts is not None else 0
            shapes = [(h, w) for h in range(3, 6) for w in range(3, 7)
                      if n * w + n - 1 <= 22 and _hole_options(h, w, required)]
            if not shapes:
                return None
            h, w = shapes[int(rng.integers(len(shapes)))]
        if counts is None:
            available = [k for k in range(4) if _hole_options(h, w, k)]
            counts = tuple(int(v) for v in rng.choice(available, size=2, replace=False))
        common, exceptional = counts
        if not is_test and pair_index == 0:
            odd = 0 if edge_order[i] == 0 else n - 1
        elif not is_test and pair_index == 1 and i == 0:
            odd = 1 if safe else int(rng.integers(1, n - 1))
        elif safe:
            odd = (0, n - 1)[i] if i < 2 else 1
        else:
            odd = int(rng.integers(n))
        gaps = [1] * (n - 1) if safe else [int(v) for v in rng.integers(1, 3, n - 1)]
        starts = [0]
        for gap in gaps:
            starts.append(starts[-1] + w + gap)
        span = starts[-1] + w
        if span > 28:
            return None
        available_colors = np.array([c for c in range(10) if c != bg])
        colors = [int(c) for c in rng.choice(available_colors, size=2 if mixed else 1, replace=False)]
        color_indices = [j % 2 for j in range(n)] if mixed else [0] * n
        rng.shuffle(color_indices)
        strip = np.full((h, span), bg, dtype=np.int8)
        options = _hole_options(h, w, common)
        ordinary = [j for j in range(n) if j != odd]
        choices = [int(rng.integers(len(options))) for _ in ordinary]
        # A positive-count anchor lane in training has unequal ordinary masks.
        # Other lanes may repeat exact ordinary plaques.
        if not is_test and i < 2 and len(options) > 1:
            choices[:2] = [int(v) for v in rng.choice(len(options), size=2, replace=False)]
        elif len(choices) > 2 and rng.random() < 0.45:
            choices[-1] = choices[0]
        hole_by_index = {j: options[k] for j, k in zip(ordinary, choices)}
        ex_options = _hole_options(h, w, exceptional)
        hole_by_index[odd] = ex_options[int(rng.integers(len(ex_options)))]
        for j, x in enumerate(starts):
            tile = np.full((h, w), colors[color_indices[j]], dtype=np.int8)
            for r, c in hole_by_index[j]:
                tile[r, c] = bg
            strip[:, x:x + w] = tile
        bands.append({"strip": strip, "anchor": starts[odd], "top": y, "bottom": y + h})
        y += h + (1 if safe else int(rng.integers(1, 3)))
    core_height = bands[-1]["bottom"]
    if core_height > 28:
        return None
    # Label construction uses authored bands, without calling solve or its parser.
    raw = [band_count - 1 - i - b["anchor"] for i, b in enumerate(bands)]
    lo = min(raw)
    output_width = max(x + b["strip"].shape[1] for x, b in zip(raw, bands)) - lo
    if output_width > 30:
        return None
    max_span = max(b["strip"].shape[1] for b in bands)
    input_width = int(rng.integers(max_span + 2, min(30, max_span + 8) + 1))
    input_height = int(rng.integers(core_height + 2, min(30, core_height + 6) + 1))
    top_margin = int(rng.integers(1, input_height - core_height))
    inp = np.full((input_height, input_width), bg, dtype=np.int8)
    out = np.full((core_height, output_width), bg, dtype=np.int8)
    for b, x in zip(bands, raw):
        h, w = b["strip"].shape
        left = int(rng.integers(1, input_width - w))
        top = top_margin + b["top"]
        inp[top:top + h, left:left + w] = b["strip"]
        out[b["top"]:b["bottom"], x - lo:x - lo + w] = b["strip"]
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer("seed", seed, 0)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mixed_allowed = bool(rng.integers(2))
    background_order = [int(v) for v in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            is_test = split == "test"
            m = int(rng.integers(4, 6)) if is_test else (2 if i == 0 else 3 if i == 1
                                                       else int(rng.integers(2, 4)))
            bg = background_order[i] if not is_test and i < 2 else int(rng.integers(10))
            for attempt in range(501):
                pair = _candidate(rng, m, bg, i, is_test, mixed_allowed, safe=attempt == 500)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("unable to construct a distinct feasible arrangement")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    bg, bands = _parse(inp)
    origins, width = _origins(bands)
    first_top = bands[0]["top"]
    assert out.shape == (bands[-1]["bottom"] - first_top, width)
    occupied = np.zeros(out.shape, dtype=bool)
    diagonal = []
    for b, x in zip(bands, origins):
        h, w = b["strip"].shape
        y = b["top"] - first_top
        assert np.array_equal(out[y:y + h, x:x + w], b["strip"])
        occupied[y:y + h, x:x + w] = True
        diagonal.append(x + b["anchor"])
        # Every plaque survives unchanged, including its foreground color.
        for t in b["tiles"]:
            xx = x + t["left"] - b["left"]
            assert np.array_equal(out[y:y + h, xx:xx + t["tile"].shape[1]], t["tile"])
        color_counts = Counter(t["color"] for t in b["tiles"])
        assert all(n >= 2 for n in color_counts.values()), "color rarity must not select an anchor"
    assert all(a - b == 1 for a, b in zip(diagonal, diagonal[1:]))
    assert np.all(out[~occupied] == bg)
    assert np.any(out[0] != bg) and np.any(out[-1] != bg)
    assert np.any(out[:, 0] != bg) and np.any(out[:, -1] != bg)
    assert Counter(map(int, inp[inp != bg])) == Counter(map(int, out[out != bg]))
    assert np.array_equal(solve(inp.copy()), out)
    return bg, bands


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            bg, bands = _check_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen
            seen.add(key)
            analyses[split].append((bg, bands))
            signs = {int(np.sign(b["exceptional"] - b["common"])) for b in bands}
            assert signs == {-1, 1}
            roles = {(b["common"], b["exceptional"]) for b in bands}
            assert any((b, a) in roles for a, b in roles)
    train = analyses["train"]
    assert {len(b) for _, b in train} == {2, 3}
    assert all(len(b) in (4, 5) for _, b in analyses["test"])
    assert train[0][0] != train[1][0]
    first = train[0][1]
    assert len({b["strip"].shape[0] for b in first}) == 2
    assert len({b["tiles"][0]["tile"].shape[1] for b in first}) == 2
    assert {b["odd"] == 0 for b in first} == {False, True}
    assert all(b["odd"] in (0, len(b["tiles"]) - 1) for b in first)
    assert any(0 < b["odd"] < len(b["tiles"]) - 1 for b in train[1][1])
    # An exact-pattern singleton strategy cannot identify the anchor here.
    assert any(len(b["tiles"]) == 3 and len({t["holes"] for t in b["tiles"]}) == 3
               for b in first)
    train_mixed = any(len({t["color"] for t in b["tiles"]}) > 1 for _, bs in train for b in bs)
    test_mixed = any(len({t["color"] for t in b["tiles"]}) > 1
                     for _, bs in analyses["test"] for b in bs)
    assert not test_mixed or train_mixed


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _counterfactual(bands, bg, mode):
    """Concrete shortcut outputs, used only to audit training evidence."""
    if mode == "band_edges":
        q = [len(bands) - 1 - i for i in range(len(bands))]
    elif mode == "vertical":
        q = [-b["anchor"] for b in bands]
    elif mode == "centers":
        # Twice the coordinates avoids rounding for unequal plaque widths.
        q = [2 * (len(bands) - 1 - i - b["anchor"])
             - b["tiles"][0]["tile"].shape[1] for i, b in enumerate(bands)]
        return tuple(q)
    else:
        raise ValueError(mode)
    q = [x - min(q) for x in q]
    width = max(x + b["strip"].shape[1] for x, b in zip(q, bands))
    out = np.full((bands[-1]["bottom"] - bands[0]["top"], width), bg, dtype=np.int8)
    for x, b in zip(q, bands):
        y = b["top"] - bands[0]["top"]
        h, w = b["strip"].shape
        out[y:y + h, x:x + w] = b["strip"]
    return out


def self_test():
    backgrounds, shapes, layouts, signatures, count_pairs = set(), set(), set(), set(), set()
    tile_counts, tile_shapes, band_counts, patterns, mixed = set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                bg, bands = _check_pair(pair)
                backgrounds.add(bg)
                shapes.add(pair["input"].shape)
                band_counts.add(len(bands))
                layout = []
                for b in bands:
                    tile_counts.add(len(b["tiles"]))
                    tile_shapes.add(b["tiles"][0]["tile"].shape)
                    count_pairs.add((b["common"], b["exceptional"]))
                    mixed.add(len({t["color"] for t in b["tiles"]}) > 1)
                    patterns.update((t["tile"].shape, t["holes"]) for t in b["tiles"])
                    layout.append((b["top"], b["left"], b["anchor"], b["strip"].shape))
                layouts.add(tuple(layout))
        bg, bands = _parse(p["train"][0]["input"])
        truth = p["train"][0]["output"]
        # Omitting A aligns band edges; omitting B aligns the anomalies vertically.
        assert not np.array_equal(_counterfactual(bands, bg, "band_edges"), truth)
        assert not np.array_equal(_counterfactual(bands, bg, "vertical"), truth)
        origins, _ = _origins(bands)
        centered = _counterfactual(bands, bg, "centers")
        assert len({q - 2 * x for q, x in zip(centered, origins)}) > 1
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
    assert len(signatures) == 42 and len(layouts) >= 180 and len(shapes) >= 40
    assert backgrounds == set(range(10)) and band_counts == {2, 3, 4, 5}
    assert tile_counts == {3, 4, 5} and len(tile_shapes) >= 10
    assert len(patterns) >= 100 and len(count_pairs) == 12 and mixed == {False, True}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Feasible safe construction is exercised rather than left as untested code.
    for m, i, test, allow in ((2, 0, False, False), (3, 1, False, True),
                              (5, 0, True, True)):
        pair = _candidate(np.random.default_rng(123 + m), m, 0, i, test, allow, safe=True)
        assert pair is not None
        _check_pair(pair)
    anchor_pair = generate_puzzle(0)["train"][0]
    inp, expected = anchor_pair["input"], anchor_pair["output"]
    bg, bands = _parse(inp)
    # Canvas translation and independently moving source bands carry no signal.
    padded = np.pad(inp, 1, constant_values=bg)
    assert np.array_equal(solve(padded), expected)
    displaced = inp.copy()
    for i, b in enumerate(bands):
        displaced[b["top"]:b["bottom"]] = bg
        w = b["strip"].shape[1]
        x = 1 if i % 2 else inp.shape[1] - w - 1
        displaced[b["top"]:b["bottom"], x:x + w] = b["strip"]
    assert np.array_equal(solve(displaced), expected)
    permutation = np.random.default_rng(987).permutation(10)
    assert np.array_equal(solve(permutation[inp]), permutation[expected])
    # Solve also supports admissible band/tile counts beyond the sampled 2..5/3..5.
    for m, n in ((7, 3), (2, 7)):
        h, span = 4 * m - 1, 4 * n - 1
        a = np.full((h + 2, span + 2), 9, dtype=np.int8)
        raw = []
        strips = []
        for i in range(m):
            odd = i % n if m == 7 else 3
            strip = np.full((3, span), 9, dtype=np.int8)
            for j in range(n):
                strip[:, 4 * j:4 * j + 3] = 2
                if j != odd:
                    strip[1, 4 * j + 1] = 9
            a[1 + 4 * i:4 + 4 * i, 1:1 + span] = strip
            strips.append(strip)
            raw.append(m - 1 - i - 4 * odd)
        lo = min(raw)
        target = np.full((h, max(raw) - lo + span), 9, dtype=np.int8)
        for i, (strip, q) in enumerate(zip(strips, raw)):
            target[4 * i:4 * i + 3, q - lo:q - lo + span] = strip
        assert np.array_equal(solve(a), target)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "layouts": len(layouts), "tile_shapes": len(tile_shapes),
            "hole_patterns": len(patterns), "count_pairs": len(count_pairs)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
