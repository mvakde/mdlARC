"""Self-similar reservoirs: component counts become stencil-column fill depths.

No parent modules, files, random state, or examples are consulted by solve().
Only NumPy and the Python standard library are used.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIMENSIONS = tuple((h, w) for h in range(2, 6) for w in range(2, 6)
                   if (h, w) != (5, 5))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= x <= 30 for x in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array.astype(np.int8, copy=False)


def _components(mask, diagonals=True):
    """Return cell sets, counting corner contact only when requested."""
    remaining = {tuple(map(int, point)) for point in np.argwhere(mask)}
    groups = []
    directions = DIR8 if diagonals else DIR4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            row, col = stack.pop()
            for dr, dc in directions:
                point = row + dr, col + dc
                if point in remaining:
                    remaining.remove(point)
                    group.add(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _counts(labels, support, diagonals=True):
    return {int(color): len(_components(support & (labels == color), diagonals))
            for color in np.unique(labels[support])}


def _filled(support, depth, geometric=False):
    """Lowest depth available cells PER COLUMN; holes consume no capacity."""
    if geometric:
        return support & (np.arange(support.shape[0])[:, None] >= support.shape[0] - depth)
    result = np.zeros_like(support)
    for col in range(support.shape[1]):
        rows = np.flatnonzero(support[:, col])
        if depth:
            result[rows[-depth:], col] = True
    return result


def _render_output(grid, info, counts=None, *, geometric=False, full=False, uniform=False):
    """Render from a recovered or constructed geometry, also used for ablations."""
    support, labels = info["support"], info["labels"]
    counts = info["counts"] if counts is None else counts
    result = grid.copy()
    py, px = info["pitch"]
    er, ec = info["expansion_origin"]
    for mr, mc in np.argwhere(support):
        color = int(labels[mr, mc])
        depth = min(counts.values()) if uniform else counts[color]
        fill = support if full else _filled(support, depth, geometric)
        rows, cols = np.where(fill)
        result[er + int(mr) * py + rows, ec + int(mc) * px + cols] = color
    return result


def _recover(value):
    """Recover all conventions from the input, checking the two-scale identity."""
    grid = _grid(value)
    frequencies = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(frequencies))
    if frequencies[background] * 2 <= grid.size:
        raise ValueError("the background must occupy more than half of the canvas")
    rest = frequencies.copy()
    rest[background] = -1
    carrier = int(np.argmax(rest))
    seed_mask = (grid != background) & (grid != carrier)
    seed_points = np.argwhere(seed_mask)
    carrier_points = np.argwhere(grid == carrier)
    n = len(seed_points)
    if not 6 <= n <= 20 or len(carrier_points) != n * n:
        raise ValueError("carrier area must be the square of seed support area")
    sr, sc = map(int, seed_points.min(axis=0))
    bottom, right = map(int, seed_points.max(axis=0))
    h, w = bottom - sr + 1, right - sc + 1
    if (h, w) not in DIMENSIONS:
        raise ValueError("unsupported stencil bounding box")
    support = seed_mask[sr:bottom + 1, sc:right + 1].copy()
    labels = grid[sr:bottom + 1, sc:right + 1].copy()
    if not 2 <= len(np.unique(labels[support])) <= 5:
        raise ValueError("the seed must have two through five label colors")
    er, ec = map(int, carrier_points.min(axis=0))
    eb, ee = map(int, carrier_points.max(axis=0))
    dy, dx = eb - er + 1 - h, ee - ec + 1 - w
    if dy % (h - 1) or dx % (w - 1):
        raise ValueError("no integral coarse lattice")
    py, px = dy // (h - 1), dx // (w - 1)
    if not (1 <= py - h <= 4 and 1 <= px - w <= 4):
        raise ValueError("gutters must be one through four cells")
    points = [tuple(map(int, p)) for p in np.argwhere(support)]
    expected = {(er + mr * py + r, ec + mc * px + c)
                for mr, mc in points for r, c in points}
    actual = {tuple(map(int, p)) for p in carrier_points}
    if expected != actual or len(expected) != n * n:
        raise ValueError("micro support and macro support must both equal the seed")
    if right + 1 < ec:
        direction = "left"
    elif ee + 1 < sc:
        direction = "right"
    elif bottom + 1 < er:
        direction = "above"
    elif eb + 1 < sr:
        direction = "below"
    else:
        raise ValueError("seed and expansion require a blank separating row or column")
    return grid, {"background": background, "carrier": carrier,
                  "support": support, "labels": labels,
                  "counts": _counts(labels, support), "pitch": (py, px),
                  "seed_origin": (sr, sc), "expansion_origin": (er, ec),
                  "direction": direction}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count seed islands, then fill their matching stencil reservoirs."""
    grid, info = _recover(input_grid)
    return _render_output(grid, info)


def _witnesses(grid, info):
    output = _render_output(grid, info)
    area = {int(c): int(np.count_nonzero(info["support"] & (info["labels"] == c)))
            for c in info["counts"]}
    four = _counts(info["labels"], info["support"], False)
    shifted = dict(info)
    shifted["labels"] = info["labels"].copy()
    shifted["labels"][info["support"]] = np.roll(info["labels"][info["support"]], 1)
    return {
        "area": not np.array_equal(output, _render_output(grid, info, area)),
        "four": not np.array_equal(output, _render_output(grid, info, four)),
        "full": not np.array_equal(output, _render_output(grid, info, full=True)),
        "rows": not np.array_equal(output, _render_output(grid, info, geometric=True)),
        "uniform": not np.array_equal(output, _render_output(grid, info, uniform=True)),
        "correspondence": not np.array_equal(output, _render_output(grid, shifted)),
    }


def _support(rng, h, w, kind):
    if h * w < 6:
        return None
    if kind == "full":
        mask = np.ones((h, w), dtype=bool)
    elif kind == "ring":
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
    elif kind == "split":
        mask = rng.random((h, w)) < 0.9
        if w >= 3 and (h < 3 or rng.random() < 0.5):
            mask[:, int(rng.integers(1, w - 1))] = False
        else:
            mask[int(rng.integers(1, h - 1)), :] = False
    else:
        mask = rng.random((h, w)) < rng.uniform(0.53, 0.93)
    if (int(mask.sum()) < 6 or not mask[0].any() or not mask[-1].any()
            or not mask[:, 0].any() or not mask[:, -1].any()):
        return None
    return mask


def _sample_seed(rng, target, dimensions, strong=False, extrapolate=False):
    """Rejection is on small 2..5 cell stencils, never on a whole puzzle."""
    for _ in range(16000):
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        kind = str(rng.choice(("random", "random", "random", "full", "ring", "split")))
        support = _support(rng, h, w, kind)
        if support is None:
            continue
        if extrapolate and int(support.sum(axis=0).max()) <= target:
            continue
        n = int(support.sum())
        distinct = int(rng.integers(2, min(5, n - 1) + 1))
        values = rng.integers(0, distinct, size=n)
        values[:distinct] = np.arange(distinct)
        rng.shuffle(values)
        labels = np.full((h, w), -1, dtype=np.int8)
        labels[support] = values
        counts = _counts(labels, support)
        if max(counts.values()) != target:
            continue
        # These requirements make the two mandatory demonstrations explanatory.
        if strong:
            if min(counts.values()) != 1:
                continue
            four = _counts(labels, support, False)
            if not any(not np.array_equal(_filled(support, counts[c]),
                                          _filled(support, four[c])) for c in counts):
                continue
            if not any(not np.array_equal(_filled(support, counts[c]),
                                          _filled(support, counts[c], True)) for c in counts):
                continue
        # At least one reservoir remains partly carrier-colored in every pair.
        if not any(np.count_nonzero(_filled(support, k)) < n for k in counts.values()):
            continue
        # Pixel area must not accidentally induce the same visible fill depths.
        if not any(not np.array_equal(_filled(support, counts[c]),
                                      _filled(support, int(np.sum(labels == c)))) for c in counts):
            continue
        return support, labels
    raise RuntimeError("could not sample an explanatory labeled stencil")


def _layout(rng, support):
    h, w = support.shape
    options = []
    for gy in range(1, 5):
        for gx in range(1, 5):
            eh, ew = (h - 1) * (h + gy) + h, (w - 1) * (w + gx) + w
            for horizontal in (False, True):
                base_h = max(h, eh) if horizontal else h + 1 + eh
                base_w = w + 1 + ew if horizontal else max(w, ew)
                if base_h <= 30 and base_w <= 30:
                    options.append((gy, gx, eh, ew, horizontal))
    gy, gx, eh, ew, horizontal = options[int(rng.integers(len(options)))]
    room = 30 - (w + ew if horizontal else h + eh)
    gap = int(rng.integers(1, min(5, room) + 1))
    base_h = max(h, eh) if horizontal else h + gap + eh
    base_w = w + gap + ew if horizontal else max(w, ew)
    height = base_h + int(rng.integers(0, min(5, 30 - base_h) + 1))
    width = base_w + int(rng.integers(0, min(5, 30 - base_w) + 1))
    ink = int(support.sum()) ** 2 + int(support.sum())
    while height * width <= 2 * ink:
        axes = [i for i, x in enumerate((height, width)) if x < 30]
        axis = axes[int(rng.integers(len(axes)))]
        if axis == 0:
            height += 1
        else:
            width += 1
    reverse = bool(rng.integers(2))
    if horizontal:
        leading = int(rng.integers(0, width - base_w + 1))
        if reverse:
            ec, sc = leading, leading + ew + gap
        else:
            sc, ec = leading, leading + w + gap
        sr = int(rng.integers(0, height - h + 1))
        er = int(rng.integers(0, height - eh + 1))
    else:
        leading = int(rng.integers(0, height - base_h + 1))
        if reverse:
            er, sr = leading, leading + eh + gap
        else:
            sr, er = leading, leading + h + gap
        sc = int(rng.integers(0, width - w + 1))
        ec = int(rng.integers(0, width - ew + 1))
    return (height, width), (sr, sc), (er, ec), (h + gy, w + gx)


def _make_pair(rng, target, dimensions, strong=False, extrapolate=False):
    support, abstract_labels = _sample_seed(rng, target, dimensions, strong, extrapolate)
    palette = [int(x) for x in rng.permutation(10)]
    background, carrier = palette[:2]
    labels = np.full(support.shape, background, dtype=np.int8)
    for index in np.unique(abstract_labels[support]):
        labels[abstract_labels == index] = palette[2 + int(index)]
    canvas, seed_origin, expansion_origin, pitch = _layout(rng, support)
    grid = np.full(canvas, background, dtype=np.int8)
    sr, sc = seed_origin
    h, w = support.shape
    grid[sr:sr + h, sc:sc + w][support] = labels[support]
    points = [tuple(map(int, p)) for p in np.argwhere(support)]
    er, ec = expansion_origin
    py, px = pitch
    for mr, mc in points:
        for r, c in points:
            grid[er + mr * py + r, ec + mc * px + c] = carrier
    info = {"support": support, "labels": labels, "counts": _counts(labels, support),
            "expansion_origin": expansion_origin, "pitch": pitch}
    # Ground truth is constructed from the sampled stencil and explicit sites;
    # the input-only solver subsequently recovers those sites from pixels.
    return {"input": grid, "output": _render_output(grid, info)}


def _signature(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    high = int(rng.integers(2, 4))
    train_dims = [(3, 3), (3, 4), (4, 3), (4, 4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            strong = split == "train" and index < 2
            extrapolate = split == "test" and (index == 0 or rng.random() < 0.5)
            if split == "train":
                target = 2 if index == 0 else high if index == 1 else int(rng.integers(1, high + 1))
                dims = train_dims if strong else [x for x in DIMENSIONS if max(x) <= 4]
                if strong and target == 3:
                    dims = [(4, 3), (4, 4)]
            elif extrapolate:
                target = high + 1
                dims = [(h, w) for h in range(target + 1, 6) for w in (3, 4)]
            else:
                target = int(rng.integers(1, high + 1))
                dims = list(DIMENSIONS)
            for _ in range(20):
                pair = _make_pair(rng, target, dims, strong, extrapolate)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create a distinct input")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    seen, infos, witnesses = set(), {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            grid, info = _recover(grid)
            _require(grid.shape == output.shape, "output canvas changed")
            expected = _render_output(grid, info)
            _require(np.array_equal(output, expected), "wrong reservoir fill")
            outside = grid != info["carrier"]
            _require(np.array_equal(output[outside], grid[outside]), "seed or background changed")
            _require(np.any(output != grid) and np.any(output == info["carrier"]),
                     "each pair must fill some cells and retain some unfilled cells")
            _require(set(np.unique(output)) <= set(np.unique(grid)), "new color invented")
            # Reconstruct each reservoir separately, without component-grouping tiles.
            h, w = info["support"].shape
            er, ec = info["expansion_origin"]
            py, px = info["pitch"]
            for mr, mc in np.argwhere(info["support"]):
                top, left = er + int(mr) * py, ec + int(mc) * px
                tile = output[top:top + h, left:left + w]
                color = int(info["labels"][mr, mc])
                for col in range(w):
                    rows = np.flatnonzero(info["support"][:, col])
                    k = min(info["counts"][color], len(rows))
                    _require(np.all(tile[rows[:len(rows) - k], col] == info["carrier"]),
                             "a reservoir filled above its count")
                    if k:
                        _require(np.all(tile[rows[-k:], col] == color), "a reservoir is not bottom packed")
            witness = _witnesses(grid, info)
            _require(witness["area"] and witness["full"] and witness["correspondence"],
                     "a parent ablation is invisible")
            if split == "train":
                witnesses.append(witness)
            infos[split].append(info)
    for name in ("area", "four", "full", "rows", "uniform", "correspondence"):
        _require(any(w[name] for w in witnesses), f"training does not refute {name} shortcut")
    train_max = max(max(x["counts"].values()) for x in infos["train"])
    shifted = [x for x in infos["test"] if max(x["counts"].values()) > train_max]
    _require(shifted, "test must extrapolate component count")
    _require(any(int(x["support"].sum(axis=0).max()) > max(x["counts"].values()) for x in shifted),
             "extrapolated depth must remain visible below capacity")


def _puzzle_signature(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed semantic checks, API checks, and descriptive diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "seed diversity failed")
    _require(_puzzle_signature(puzzles[17]) == _puzzle_signature(generate_puzzle(17)), "not deterministic")
    metrics = Counter()
    supports, shapes, palettes, directions, depths, gutters = set(), set(), set(), set(), set(), set()
    normalized = set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, info = _recover(pair["input"])
                before = grid.copy()
                predicted = solve(grid)
                _require(np.array_equal(predicted, pair["output"]), "input-only solve failed")
                # solve must not mutate the caller's grid.
                _require(np.array_equal(grid, before), "input mutation")
                mask = info["support"]
                supports.add((mask.shape, mask.tobytes()))
                shapes.add(grid.shape)
                palettes.add((info["background"], info["carrier"]))
                directions.add(info["direction"])
                depths.update(info["counts"].values())
                gutters.add((info["pitch"][0] - mask.shape[0], info["pitch"][1] - mask.shape[1]))
                roles = {}
                normalized.add((grid.shape, tuple(roles.setdefault(int(c), len(roles)) for c in grid.flat)))
                metrics["pairs"] += 1
                metrics["disconnected8"] += len(_components(mask)) > 1
                metrics["disconnected4"] += len(_components(mask, False)) > 1
                metrics["full_support"] += bool(mask.all())
                metrics["ring"] += bool(min(mask.shape) >= 3 and mask[0].all() and mask[-1].all()
                                        and mask[:, 0].all() and mask[:, -1].all()
                                        and not mask[1:-1, 1:-1].any())
                metrics["saturated_column"] += any(k >= int(mask[:, c].sum()) > 0
                                                   for k in info["counts"].values()
                                                   for c in range(mask.shape[1]))
                metrics["retained_carrier"] += bool(np.any(predicted == info["carrier"]))
    _require(len(supports) >= 65 and len(shapes) >= 40 and len(normalized) >= 150,
             "structure or geometry diversity failed")
    _require(len(palettes) >= 35 and directions == {"left", "right", "above", "below"},
             "palette or placement diversity failed")
    _require(depths >= {1, 2, 3, 4} and len(gutters) >= 8, "depth or gutter diversity failed")
    _require(all(metrics[k] > 0 for k in ("disconnected8", "disconnected4", "full_support", "ring",
                                         "saturated_column")), "support edge case missing")
    # A valid unsampled extreme: six isolated objects exceed a five-cell column.
    # Saturation must be ordinary min(count, capacity), with no special count cap.
    extreme = np.zeros((30, 30), dtype=np.int8)
    stencil = np.full((5, 4), 3, dtype=np.int8)
    stencil[::2, ::2] = 2
    extreme[:5, :4] = stencil
    for mr in range(5):
        for mc in range(4):
            extreme[mr * 6:mr * 6 + 5, 6 + mc * 5:6 + mc * 5 + 4] = 1
    _, info = _recover(extreme)
    _require(info["counts"] == {2: 6, 3: 1}, "extreme component count is wrong")
    output = solve(extreme)
    _require(np.all(output[:5, 6:10] == 2), "overflow failed to saturate a reservoir")
    _require(np.all(output[:4, 11:15] == 1) and np.all(output[4, 11:15] == 3),
             "one-island reservoir did not retain its upper carrier")
    # Palette relabeling and horizontal reflection preserve the fixed downward rule.
    example = puzzles[0]["train"][0]
    permutation = np.array([8, 3, 0, 6, 9, 2, 5, 1, 7, 4], dtype=np.int8)
    _require(np.array_equal(solve(permutation[example["input"]]), permutation[example["output"]]),
             "literal color dependence")
    _require(np.array_equal(solve(example["input"][:, ::-1]), example["output"][:, ::-1]),
             "horizontal reflection changed the rule")
    # Every legal explicit count combination, with smaller and maximum requests.
    calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + calls, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count API failed")
            calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    # Explicitly test falsification: wrong outputs must fail validation.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    grid, info = _recover(bad["train"][0]["input"])
    bad["train"][0]["output"] = _render_output(grid, info, full=True)
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted whole-tile recoloring")
    return {"seeds": 42, **dict(metrics), "distinct_supports": len(supports),
            "input_shapes": len(shapes), "color_normalized_inputs": len(normalized),
            "palette_role_pairs": len(palettes), "directions": sorted(directions),
            "depths": sorted(depths), "gutters": sorted(gutters), "legal_count_calls": calls}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
