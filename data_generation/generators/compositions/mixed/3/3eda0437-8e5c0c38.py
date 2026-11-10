"""Reflection pruning exposes the unique largest empty rectangle.

Fixed, input-only conventions: background 0, output highlight 6, vertical axes.
Each other color denotes one figure, including its disconnected pieces.
Generation never imports or calls a parent generator. No files are written here.
"""
from __future__ import annotations

import hashlib
import numpy as np


BACKGROUND = 0
HIGHLIGHT = 6
FIGURE_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid


def _axis(mask):
    """Score every integral/half-integral vertical axis by supported cell count.

    axis2 = twice the axis column. Convolution counts both members of a pair
    and counts an on-axis cell once, exactly as reflection support requires.
    """
    width = mask.shape[1]
    scores = np.zeros(2 * width - 1, dtype=np.int32)
    for row in mask.astype(np.int16):
        scores += np.convolve(row, row)
    winners = np.flatnonzero(scores == scores.max())
    if len(winners) != 1 or scores[winners[0]] == 0:
        raise ValueError("figure has no unique maximum-support vertical axis")
    axis2 = int(winners[0])
    reflected = axis2 - np.arange(width)
    valid = (reflected >= 0) & (reflected < width)
    keep = np.zeros_like(mask, dtype=bool)
    keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
    return axis2, keep


def _prune(grid):
    if np.any(grid == HIGHLIGHT):
        raise ValueError("highlight 6 is reserved for output")
    colors = [int(c) for c in np.unique(grid) if c != BACKGROUND]
    if not colors or not np.any(grid == BACKGROUND):
        raise ValueError("input needs background and at least one figure")
    clean = np.zeros_like(grid)
    axes = {}
    for color in colors:
        axis2, keep = _axis(grid == color)
        clean[keep] = color
        axes[color] = axis2
    return clean, axes


def _runs(row):
    start = None
    for column, value in enumerate(row):
        if value and start is None:
            start = column
        elif not value and start is not None:
            yield start, column
            start = None
    if start is not None:
        yield start, len(row)


def _rectangles(empty):
    """Row spans and their maximal common column runs; sufficient for maxima."""
    height, width = empty.shape
    candidates = []
    for top in range(height - 1):
        common = empty[top].copy()
        for bottom in range(top + 1, height):
            common &= empty[bottom]
            if not np.any(common):
                break
            for left, right in _runs(common):
                if right - left >= 2:
                    candidates.append((top, left, bottom - top + 1, right - left))
    return candidates


def _area(rect):
    return rect[2] * rect[3]


def _largest(empty):
    candidates = _rectangles(empty)
    if not candidates:
        raise ValueError("no empty rectangle with both dimensions at least two")
    maximum = max(map(_area, candidates))
    winners = [rect for rect in candidates if _area(rect) == maximum]
    if len(winners) != 1:
        raise ValueError("largest eligible empty rectangle is tied")
    return winners[0], candidates


def _region(shape, rect):
    top, left, height, width = rect
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + height, left:left + width] = True
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Prune simultaneously from the original input, then optimize blank area."""
    grid = _grid(input_grid)
    clean, _ = _prune(grid)
    rect, _ = _largest(clean == BACKGROUND)
    output = clean.copy()
    output[_region(grid.shape, rect)] = HIGHLIGHT
    return output


def _features(clean, rect, candidates):
    top, left, height, width = rect
    # Ignore rectangles entirely contained in the winner when measuring rivals.
    others = [r for r in candidates if not (
        top <= r[0] and left <= r[1] and r[0] + r[2] <= top + height
        and r[1] + r[3] <= left + width)]
    target = _region(clean.shape, rect)
    # Background flood fill distinguishes a rectangle from a connected region.
    remaining = clean == BACKGROUND
    visited = target.copy()
    frontier = list(map(tuple, np.argwhere(target)))
    while frontier:
        row, column = frontier.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, column + dc
            if (0 <= r < clean.shape[0] and 0 <= c < clean.shape[1]
                    and remaining[r, c] and not visited[r, c]):
                visited[r, c] = True
                frontier.append((r, c))
    outside = (clean == BACKGROUND) & ~target
    longest = max([b - a for line in list(outside) + list(outside.T)
                   for a, b in _runs(line)] + [0])
    return {
        "area": _area(rect), "rect": tuple(map(int, rect)),
        "rival_area": max(map(_area, others), default=0),
        "wider_rival": any(r[3] > width for r in others),
        "taller_rival": any(r[2] > height for r in others),
        "component_extra": int(np.count_nonzero(visited & ~target)),
        "longest_unselected_line": longest,
        "interior": top > 0 and left > 0 and top + height < clean.shape[0]
                    and left + width < clean.shape[1],
    }


def _symmetric_mask(rng, height, width, kind=None):
    """Six compositional geometries, including contours and supported islands."""
    kind = int(rng.integers(6)) if kind is None else kind
    mask = np.zeros((height, width), dtype=bool)
    radius = width // 2
    if kind == 0:
        # A contour; variable caps and sides leave a genuine enclosed void.
        cap = int(rng.integers(1, max(2, height // 4 + 1)))
        side = int(rng.integers(1, max(2, width // 5 + 1)))
        mask[:cap] = mask[-cap:] = True
        mask[:, :side] = mask[:, -side:] = True
    else:
        for row in range(height):
            if kind == 1:  # oval or diamond profile
                center = (height - 1) * .5
                level = max(0, radius - int(abs(row - center) * radius / max(1, center)))
            elif kind == 2:  # independently sized necks and lobes
                level = int(rng.integers(0, radius + 1))
            elif kind == 3:  # hollow, varying row contours
                level = max(1, radius - int(rng.integers(0, max(1, radius))))
            elif kind == 4:  # disjoint paired branches and occasional empty rows
                level = radius if rng.random() < .65 else 0
            else:  # filled bars with mirrored internal holes
                level = radius
            if width % 2 and (level == 0 or rng.random() < .50):
                mask[row, radius] = True
            for distance in range(level):
                column = radius - 1 - distance
                present = (kind in (1, 2) or distance == level - 1
                           or (kind == 3 and distance >= level - 2)
                           or (kind in (4, 5) and rng.random() < .45))
                if present:
                    mask[row, column] = mask[row, width - 1 - column] = True
        # Width is attained, but no middle-row or occupied-row pair quota.
        row = int(rng.integers(height))
        mask[row, 0] = mask[row, -1] = True
    return mask


def _shape(rng, split, shift):
    if shift == "scale":
        low, high = (10, 20) if split == "train" else (22, 30)
        return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
    if shift == "aspect":
        if split == "train":
            return int(rng.integers(10, 18)), int(rng.integers(18, 31))
        return int(rng.integers(20, 31)), int(rng.integers(10, 18))
    return int(rng.integers(10, 31)), int(rng.integers(10, 31))


def _scene(rng, shape, colors, parity):
    clean = np.zeros(shape, dtype=np.int8)
    height, width = shape
    enclosure = rng.random() < .28
    for index, color in enumerate(colors):
        placed = False
        for attempt in range(70):
            if index == 0 and enclosure:
                h = height - int(rng.integers(0, 4))
                widths = [w for w in range(max(3, width - 4), width + 1)
                          if parity is None or (w - 1) % 2 == parity]
                w = int(rng.choice(widths))
                mask = _symmetric_mask(rng, h, w, kind=0)
            else:
                limit_h = min(height, 13 if attempt < 35 else 6)
                limit_w = min(width, 13 if attempt < 35 else 6)
                h = int(rng.integers(3, limit_h + 1))
                widths = [w for w in range(3, limit_w + 1)
                          if index != 0 or parity is None or (w - 1) % 2 == parity]
                w = int(rng.choice(widths))
                mask = _symmetric_mask(rng, h, w)
            if np.count_nonzero(mask) < 6:
                continue
            top = int(rng.integers(height - h + 1))
            left = int(rng.integers(width - w + 1))
            if np.any(clean[top:top + h, left:left + w][mask] != BACKGROUND):
                continue
            clean[top:top + h, left:left + w][mask] = color
            placed = True
            break
        if not placed:
            return None
    return clean


def _add_one(rng, grid, clean, color, allowed):
    """Only add a cell if the unique recovered subset is still exactly the core."""
    positions = np.argwhere(allowed & (grid == BACKGROUND))
    if not len(positions):
        return False
    core = clean == color
    axis2, _ = _axis(core)
    for index in rng.permutation(len(positions))[:64]:
        row, column = map(int, positions[index])
        reflected = axis2 - column
        if 2 * column == axis2 or (0 <= reflected < grid.shape[1]
                                  and grid[row, reflected] == color):
            continue
        grid[row, column] = color
        try:
            selected_axis, retained = _axis(grid == color)
            if selected_axis == axis2 and np.array_equal(retained, core):
                return True
        except ValueError:
            pass
        grid[row, column] = BACKGROUND
    return False


def _make_pair(rng, split, shift, count, parity=None, contrast=False, avoid_shape=None):
    # Bounded rejection: no retries are logged or retained, and no payloads spill.
    for _ in range(220):
        shape = _shape(rng, split, shift)
        colors = list(map(int, rng.choice(FIGURE_COLORS, count, replace=False)))
        clean = _scene(rng, shape, colors, parity)
        if clean is None:
            continue
        try:
            rect, candidates = _largest(clean == BACKGROUND)
        except ValueError:
            continue
        if avoid_shape is not None and rect[2:] == avoid_shape:
            continue
        if contrast:
            features = _features(clean, rect, candidates)
            if (features["component_extra"] == 0
                    or features["rival_area"] < .55 * features["area"]
                    or not features["wider_rival" if parity == 0 else "taller_rival"]):
                continue
        inp = clean.copy()
        target = _region(shape, rect)
        # A real blocker inside the eventual winner establishes the dependency.
        if not _add_one(rng, inp, clean, colors[0], target):
            continue
        # A second color's deletion outside it rules out simply painting over noise.
        if contrast and not _add_one(rng, inp, clean, colors[1], ~target):
            continue
        if shift == "noise":
            amount = int(rng.integers(2, 6) if split == "train" else rng.integers(7, 13))
        else:
            amount = int(rng.integers(2, 9))
        all_cells = np.ones(shape, dtype=bool)
        added = int(np.count_nonzero(inp != clean))
        for _ in range(amount - added):
            rng.shuffle(colors)
            for color in colors:
                if _add_one(rng, inp, clean, color, all_cells):
                    break
        if shift == "noise" and split == "test" and np.count_nonzero(inp != clean) < 7:
            continue
        reconstructed, axes = _prune(inp)
        if not np.array_equal(reconstructed, clean):
            raise AssertionError("additive corruption changed the supported core")
        if contrast and len(set(axes.values())) < 2:
            continue
        output = clean.copy()
        output[target] = HIGHLIGHT
        return {"input": inp, "output": output}
    raise RuntimeError("bounded scene construction exhausted; no ambiguous fallback is used")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(("scale", "aspect", "noise")))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_count = int(rng.integers(2, 5))
    second_count = int(rng.choice([n for n in range(2, 5) if n != first_count]))
    first_target_shape = None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            anchor = split == "train" and index < 2
            figures = (first_count if index == 0 else second_count) if anchor else int(rng.integers(1, 6))
            for _ in range(12):
                pair = _make_pair(rng, split, shift, figures,
                                  parity=index if anchor else None,
                                  contrast=anchor,
                                  avoid_shape=first_target_shape if anchor and index == 1 else None)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if anchor and index == 0:
                        clean, _ = _prune(pair["input"])
                        first_target_shape = _largest(clean == BACKGROUND)[0][2:]
                    break
            else:
                raise RuntimeError("bounded input deduplication exhausted")
    return puzzle


def _analyze_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "input/output dimensions differ")
    clean, axes = _prune(inp)
    rect, candidates = _largest(clean == BACKGROUND)
    target = _region(inp.shape, rect)
    erased = inp != clean
    expected = clean.copy()
    expected[target] = HIGHLIGHT
    _require(np.array_equal(out, expected), "incorrect pruning or rectangle highlight")
    _require(np.any(erased & target), "pruning must expose a previously blocked target")
    raw = _rectangles(inp == BACKGROUND)
    _require(max(map(_area, raw), default=0) < _area(rect),
             "pruning did not increase the maximum blank area")
    for color, axis2 in axes.items():
        core = clean == color
        _require(np.count_nonzero(core) >= 1, "entire color erased")
        for row, column in np.argwhere(core):
            _require(clean[row, axis2 - column] == color, "retained figure is asymmetric")
        _require(np.all(out[core] == color), "supported foreground changed")
    info = _features(clean, rect, candidates)
    info.update({"shape": tuple(map(int, inp.shape)), "figures": len(axes),
                 "axes": axes, "deleted": int(np.count_nonzero(erased)),
                 "deleted_outside": int(np.count_nonzero(erased & ~target)),
                 "noisy_parities": {axis % 2 for c, axis in axes.items()
                                    if np.any(erased & (inp == c))}})
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp = _grid(pair["input"])
            key = (inp.shape, inp.astype(np.uint8).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze_pair(pair)
            (training if split == "train" else testing).append(info)
    _require(set.union(*(i["noisy_parities"] for i in training)) == {0, 1},
             "training must demonstrate both axis kinds with deletion")
    _require(len({i["rect"][2:] for i in training}) >= 2, "training lacks target-shape contrast")
    _require(len({i["figures"] for i in training}) >= 2, "training lacks figure-count contrast")
    for index, info in enumerate(training[:2]):
        _require(info["deleted_outside"] >= 1, "training needs deletion outside the answer")
        _require(len(set(info["axes"].values())) >= 2, "training needs independent axes")
        _require(info["component_extra"] > 0, "training must distinguish component filling")
        _require(info["rival_area"] >= .55 * info["area"]
                 and info["wider_rival" if index == 0 else "taller_rival"],
                 "training lacks width/height versus area comparison")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 puzzles in memory, geometric coverage, equivariance, counts and mutations."""
    # Literal small cases exercise the written two-dimensional rule outside the
    # sampler's size range. In the first, an eight-cell row loses to a 2x2 void
    # exposed by deleting the unmatched pixel at (2,3).
    small = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 1, 1, 1]], dtype=np.int8)
    small_expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 6, 6, 1, 1, 1],
                               [1, 1, 1, 6, 6, 1, 1, 1]], dtype=np.int8)
    _require(np.array_equal(solve(small), small_expected), "one-dimensional eligibility error")
    _require(np.array_equal(solve([[1, 0, 0], [0, 0, 0]]),
                           [[1, 6, 6], [0, 6, 6]]), "small-grid/axis-self-support error")
    for invalid in ([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous axis/rectangle was accepted")
    shapes, target_shapes, palettes, counts, fingerprints = set(), set(), set(), set(), set()
    positions = set()
    interior = boundary = noisy_outside = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 3:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                _require(np.array_equal(solve(inp), out), "solve mismatch")
                _require(np.array_equal(inp, original), "solve mutates its input")
                _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "mirror equivariance")
                # Renaming foreground colors cannot change the blank-space answer.
                lut = np.arange(10, dtype=np.int8)
                lut[list(FIGURE_COLORS)] = np.roll(np.array(FIGURE_COLORS), 1)
                _require(np.array_equal(solve(lut[inp]), lut[out]), "palette equivariance")
                info = _analyze_pair(pair)
                shapes.add(info["shape"])
                target_shapes.add(info["rect"][2:])
                positions.add(info["rect"])
                counts.add(info["figures"])
                palettes.add(tuple(map(int, np.unique(inp))))
                interior += int(info["interior"])
                boundary += int(not info["interior"])
                noisy_outside += int(info["deleted_outside"] > 0)
    _require(len(fingerprints) == 42, "seeds collapsed")
    _require(len(shapes) >= 55 and len(target_shapes) >= 25 and len(positions) >= 100,
             "insufficient geometric variation")
    _require(counts == {1, 2, 3, 4, 5} and len(palettes) >= 35, "insufficient count/palette variation")
    _require(interior >= 5 and boundary >= 20 and noisy_outside >= 84, "insufficient interaction coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(9000 + a * 10 + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "wrong requested counts")
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(8, 2, 1)
    out = bad["test"][0]["output"]
    row, col = np.argwhere(out == HIGHLIGHT)[0]
    out[row, col] = BACKGROUND
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incomplete highlight")
    return {"seeds": 42, "pairs": 210, "literal_cases": 4, "shapes": len(shapes),
            "target_shapes": len(target_shapes), "target_positions": len(positions),
            "palettes": len(palettes), "figure_counts": sorted(counts),
            "interior_targets": interior, "boundary_targets": boundary,
            "deletions_outside_target": noisy_outside}
