"""Largest empty window seeds a two-ink, gap-center cellular cascade.

No files, examples, RNG, or puzzle-level state are read by solve().
Generation and self_test() retain only one puzzle at a time and write no files.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


HOST_SEEDS = (0, 521044164, 588862828, 426962063, 482524932, 908524834,
              830750254, 467552898, 122693658, 222100249, 542300142,
              635928216, 128178033, 69747405, 601647046, 740836617,
              733013355, 60142358, 478345770, 524353598, 351907542,
              217131630, 922029793, 64568912, 351673734, 81010789,
              824393108, 893262575, 581271472, 955143569, 827674466,
              260555719, 470010732, 406088980, 895105819, 771217319,
              259858419, 260877395, 907807869, 38147006, 917000947,
              487528111)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a


def _largest_rectangles(grid, color):
    """All globally largest all-color rectangles, including thin rectangles.

    Histogram/stack search; bounds are (top, bottom, left, right), exclusive.
    A maximum-area rectangle must be horizontally maximal at its height.
    Equal histogram bars are coalesced, so each maximal rectangle is emitted once.
    """
    h, w = grid.shape
    heights = [0] * w
    best_area, best = 0, set()
    for row in range(h):
        for col in range(w):
            heights[col] = heights[col] + 1 if grid[row, col] == color else 0
        stack = []
        for col in range(w + 1):
            height = heights[col] if col < w else 0
            start = col
            while stack and stack[-1][1] > height:
                left, old_height = stack.pop()
                area = old_height * (col - left)
                bounds = (row + 1 - old_height, row + 1, left, col)
                if area > best_area:
                    best_area, best = area, {bounds}
                elif area == best_area:
                    best.add(bounds)
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return best_area, sorted(best)


def _window(grid):
    best_area, candidates = 0, []
    for color in np.unique(grid):
        area, rectangles = _largest_rectangles(grid, int(color))
        if area > best_area:
            best_area, candidates = area, [(int(color), r) for r in rectangles]
        elif area == best_area:
            candidates.extend((int(color), r) for r in rectangles)
    if len(candidates) != 1:
        raise ValueError("largest solid rectangle must be unique across colors")
    return candidates[0]


def _inks(grid, background):
    counts = Counter(int(c) for c in grid.ravel() if c != background)
    ranked = counts.most_common()
    if len(ranked) < 2 or ranked[0][1] == ranked[1][1]:
        raise ValueError("need two nonbackground inks with distinct frequency ranks")
    if len(ranked) > 2 and ranked[1][1] == ranked[2][1]:
        raise ValueError("derived ink must be uniquely second most frequent")
    return ranked[0][0], ranked[1][0]


def _next(mask):
    following = np.zeros(len(mask), dtype=bool)
    following[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return following


def _layers(width):
    mask = np.zeros(width, dtype=bool)
    mask[::2] = True
    result = [mask]
    while True:
        mask = _next(mask)
        if not mask.any():
            return result
        result.append(mask)


def _interpret(grid):
    background, rect = _window(grid)
    top, bottom, left, right = rect
    iw = right - left - 2
    if iw < 3:
        raise ValueError("window must support at least one derived generation")
    layers = _layers(iw)
    if bottom - top - 2 < 2 * (len(layers) - 1) + 1:
        raise ValueError("window must fit the entire cascade without clipping")
    seed_ink, derived_ink = _inks(grid, background)
    return background, rect, seed_ink, derived_ink, layers


def _paint(grid, rect, seed_ink, derived_ink, layers):
    output = grid.copy()
    top, _, left, right = rect
    for generation, mask in enumerate(layers):
        output[top + 1 + 2 * generation, left + 1:right - 1][mask] = (
            derived_ink if generation % 2 else seed_ink)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select the unique largest blank window, then seed and evolve inside it."""
    grid = _grid(input_grid)
    _, rect, seed_ink, derived_ink, layers = _interpret(grid)
    return _paint(grid, rect, seed_ink, derived_ink, layers)


def _signature(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                grid = np.asarray(pair[key], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _masks(shape, rect):
    top, bottom, left, right = rect
    whole = np.zeros(shape, dtype=bool)
    interior = whole.copy()
    whole[top:bottom, left:right] = True
    interior[top + 1:bottom - 1, left + 1:right - 1] = True
    return whole, interior


def _ring(shape, rect):
    top, bottom, left, right = rect
    ring = np.zeros(shape, dtype=bool)
    ring[max(0, top - 1):min(shape[0], bottom + 1),
         max(0, left - 1):min(shape[1], right + 1)] = True
    whole, _ = _masks(shape, rect)
    return ring & ~whole


def _placement(rng, rh, rw, dh, dw):
    """Choose arbitrary disjoint locations, with a one-cell separation."""
    shapes = [(h, w) for h in range(max(12, rh), 31)
              for w in range(max(12, rw), 31)
              if h >= rh + dh + 1 or w >= rw + dw + 1]
    h, w = shapes[int(rng.integers(0, len(shapes)))]
    starts = [(r, c) for r in range(h - rh + 1) for c in range(w - rw + 1)
              if (r >= dh + 1 or h - r - rh >= dh + 1
                  or c >= dw + 1 or w - c - rw >= dw + 1)]
    top, left = starts[int(rng.integers(0, len(starts)))]
    positions = [(r, c) for r in range(h - dh + 1) for c in range(w - dw + 1)
                 if (r + dh < top or r > top + rh
                     or c + dw < left or c > left + rw)]
    dr, dc = positions[int(rng.integers(0, len(positions)))]
    return (h, w), (top, top + rh, left, left + rw), (dr, dr + dh, dc, dc + dw)


def _texture(rng, shape, background, seed_ink, derived_ink, rare):
    density = float(rng.uniform(0.17, 0.48))
    blank = rng.random(shape) < density
    style = int(rng.integers(0, 3))
    for _ in range(int(rng.integers(2, 8)) if style else 0):
        r = int(rng.integers(0, shape[0] - 1))
        c = int(rng.integers(0, shape[1] - 1))
        if style == 1:
            blank[r:r + 2, c:c + 2] = bool(rng.integers(0, 2))
        elif rng.random() < 0.5:
            blank[r, c:c + int(rng.integers(2, 6))] = True
        else:
            blank[r:r + int(rng.integers(2, 6)), c] = True
    colors = [seed_ink, derived_ink, *rare]
    dominant = float(rng.uniform(0.57, 0.68))
    rare_total = float(rng.uniform(0.03, 0.11)) if rare else 0.0
    weights = [dominant, 1.0 - dominant - rare_total]
    if rare:
        weights += [rare_total / len(rare)] * len(rare)
    grid = rng.choice(colors, size=shape, p=weights).astype(np.int8)
    grid[blank] = background
    return grid


def _make_pair(rng, depth, palette, flip, min_slack=0):
    background, ink_a, ink_b, *rare_pool = palette
    seed_ink, derived_ink = (ink_b, ink_a) if flip else (ink_a, ink_b)
    # Both width parities realize the same depth but different rightmost seed cell.
    rw = 2 * depth + 3 + int(rng.integers(0, 2))
    required_h = 2 * depth + 3
    slack = int(rng.integers(min_slack, min(9, 28 - required_h) + 1))
    rh = required_h + slack
    # Multi-generation examples have a viable but smaller recurrence window.
    if depth >= 2:
        choices = [(dh, dw) for dh in range(5, 10) for dw in range(5, 10)
                   if dh >= 2 * ((dw - 3) // 2) + 3
                   and dh * dw <= 0.65 * rh * rw
                   and (rh + dh + 1 <= 30 or rw + dw + 1 <= 30)]
    else:
        choices = [(dh, dw) for dh in range(3, 6) for dw in range(3, 6)
                   if dh * dw <= 0.65 * rh * rw]
    dh, dw = choices[int(rng.integers(0, len(choices)))]
    shape, rect, decoy = _placement(rng, rh, rw, dh, dw)
    nrare = int(rng.integers(0, 3))
    rare = list(rng.choice(rare_pool, size=nrare, replace=False))
    whole, _ = _masks(shape, rect)
    dt, db, dl, dr = decoy
    for _ in range(100):
        grid = _texture(rng, shape, background, seed_ink, derived_ink, rare)
        grid[whole] = background
        grid[dt:db, dl:dr] = background
        ring = _ring(shape, rect)
        if not np.any(grid[ring] == background):
            r, c = np.argwhere(ring)[int(rng.integers(0, int(ring.sum())))]
            grid[r, c] = background
        try:
            found = _interpret(grid)
        except ValueError:
            continue
        if found[:4] != (background, rect, seed_ink, derived_ink):
            continue
        outside = grid.copy()
        outside[whole] = -1
        competitor_area, _ = _largest_rectangles(outside, background)
        if competitor_area > 0.8 * rh * rw:
            continue
        output = _paint(grid, rect, seed_ink, derived_ink, found[4])
        return {"input": grid, "output": output}
    raise RuntimeError("could not generate an unambiguous window in bounded retries")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one reproducible puzzle; all randomness comes from default_rng(seed)."""
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.choice(10, 5, replace=False)]
    ceiling = int(rng.integers(2, 6))
    train_depths = [1, ceiling]
    train_depths += [int(rng.integers(1, ceiling + 1)) for _ in range(num_train - 2)]
    # The depth shift is always supported by a shallow example and a parity demo.
    test_depths = [ceiling + int(rng.integers(1, 5))]
    test_depths += [int(rng.integers(1, min(9, ceiling + 3) + 1))
                    for _ in range(num_test - 1)]
    initial_flip = bool(rng.integers(0, 2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, depths in (("train", train_depths), ("test", test_depths)):
        for index, depth in enumerate(depths):
            if split == "train" and index < 2:
                flip = initial_flip if index == 0 else not initial_flip
            else:
                flip = bool(rng.integers(0, 2))
            for _ in range(20):
                pair = _make_pair(rng, depth, palette, flip,
                                  min_slack=2 if split == "train" and index == 0 else 0)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct input")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair needs exactly input/output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "dimensions must be preserved")
    background, rect, seed_ink, derived_ink, layers = _interpret(inp)
    whole, interior = _masks(inp.shape, rect)
    _require(np.all(inp[whole] == background), "window must be entirely blank in INPUT")
    _require(np.array_equal(out[~interior], inp[~interior]), "perimeter and outside must be unchanged")
    top, bottom, left, right = rect
    # Cellwise expected color, without relying on _paint or solve.
    expected = inp.copy()
    current = set(range(0, right - left - 2, 2))
    generation = 0
    while current:
        row = top + 1 + 2 * generation
        _require(row < bottom - 1, "nonempty generation clipped by bottom perimeter")
        for col in current:
            expected[row, left + 1 + col] = derived_ink if generation % 2 else seed_ink
        current = {col for col in range(1, right - left - 3)
                   if col not in current and col - 1 in current and col + 1 in current}
        generation += 1
    _require(np.array_equal(out, expected), "incorrect seeding, recurrence, color, spacing, or stopping")
    changed = inp != out
    _require(np.count_nonzero(changed) == sum(int(m.sum()) for m in layers), "incorrect active count")
    _require(set(np.unique(out[changed])) == {seed_ink, derived_ink}, "both phases must be used")
    for row in range(top + 2, bottom - 1, 2):
        _require(np.all(out[row, left + 1:right - 1] == background), "spacer row changed")
    outside = inp.astype(np.int16)
    outside[whole] = -1
    competitor_area, competitors = _largest_rectangles(outside, background)
    _require(competitor_area >= 9, "need a smaller solid blank competitor")
    _require(competitor_area <= 0.8 * int(whole.sum()), "competitor area margin too small")
    _require(np.any(inp[_ring(inp.shape, rect)] == background), "avoid an artificial clean halo")
    return {"background": background, "rect": rect, "seed": seed_ink, "derived": derived_ink,
            "depth": len(layers) - 1, "slack": bottom - top - (2 * len(layers) + 1),
            "competitor_area": competitor_area, "competitors": competitors,
            "active_counts": [int(m.sum()) for m in layers]}


def validate_generated_puzzle(puzzle):
    """Validate schema, complete recurrence, parent necessity, and teaching coverage."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "need exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {}
    for split in ("train", "test"):
        analyses[split] = []
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _signature(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(info)
    all_info = analyses["train"] + analyses["test"]
    _require(len({i["background"] for i in all_info}) == 1, "one background per puzzle")
    _require(len({frozenset((i["seed"], i["derived"])) for i in all_info}) == 1,
             "one pair of primary inks per puzzle")
    train, tests = analyses["train"], analyses["test"]
    _require(any(i["depth"] == 1 and i["slack"] >= 2 for i in train), "show extinction before boundary")
    _require(max(i["depth"] for i in train) >= 2, "training must expose return to seed ink")
    _require(len({i["seed"] for i in train}) == 2, "training must demonstrate frequency-role swapping")
    _require(max(i["depth"] for i in tests) > max(i["depth"] for i in train), "need a depth shift")
    # A-free ablation: seed the same window, but perform no recurrence.
    # B-free ablation: also seed a viable smaller blank rectangle in the noisy field.
    a_witness = b_witness = False
    for pair, info in zip(puzzle["train"], train):
        inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        top, _, left, right = info["rect"]
        seed_only = inp.copy()
        seed_only[top + 1, left + 1:right - 1:2] = info["seed"]
        a_witness |= not np.array_equal(seed_only, out)
        whole, _ = _masks(inp.shape, info["rect"])
        blanks = (inp == info["background"]) & ~whole
        # Any outside all-blank 5x5 patch is a viable depth-one false window.
        for r in range(inp.shape[0] - 4):
            for c in range(inp.shape[1] - 4):
                if np.all(blanks[r:r + 5, c:c + 5]):
                    extra = out.copy()
                    extra[r + 1, c + 1:c + 4:2] = info["seed"]
                    extra[r + 3, c + 2] = info["derived"]
                    b_witness |= not np.array_equal(extra, out)
    _require(a_witness and b_witness, "training must witness both parental contributions")


def self_test():
    """42 host seeds, one puzzle at a time; bounded computation and zero disk I/O."""
    shapes, rectangles, locations, palettes, signatures = set(), set(), set(), set(), set()
    depths, slacks, parities = set(), set(), set()
    edges = square_canvas = even_width = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                info = _analyze_pair(pair)
                _require(np.array_equal(solve(inp), pair["output"]), "input-only solve mismatch")
                top, bottom, left, right = info["rect"]
                rh, rw = bottom - top, right - left
                shapes.add(inp.shape)
                rectangles.add((rh, rw))
                locations.add((top < (inp.shape[0] - rh) / 2, left < (inp.shape[1] - rw) / 2))
                palettes.add((info["background"], info["seed"], info["derived"]))
                depths.add(info["depth"])
                slacks.add(info["slack"])
                parities.add((rh % 2, rw % 2))
                edges += top == 0 or left == 0 or bottom == inp.shape[0] or right == inp.shape[1]
                square_canvas += inp.shape[0] == inp.shape[1]
                even_width += rw % 2 == 0
    _require(len(signatures) == 42, "different seeds produced repeated puzzles")
    _require(len(shapes) >= 35 and len(rectangles) >= 35, "geometry variation too narrow")
    _require(len(locations) == 4 and len(palettes) >= 25, "location or palette variation too narrow")
    _require(min(depths) == 1 and max(depths) >= 8, "missing shallow/deep behavior")
    _require(len(slacks) >= 6 and len(parities) == 4, "need varied unused space and parity")
    _require(edges >= 10 and square_canvas >= 2 and even_width >= 20, "boundary variation missing")
    _require(_digest(generate_puzzle(9081, 7, 3)) == _digest(generate_puzzle(9081, 7, 3)), "nondeterminism")
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(1000 + ntrain * 10 + ntest, ntrain, ntest)
        _require((len(p["train"]), len(p["test"])) == (ntrain, ntest), "explicit counts ignored")
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                          (2.0, 1), (True, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A tied area or color rank has no hidden tie-breaking order.
    for invalid in (np.array([[1, 0, 2, 2, 0, 1]]), np.zeros((6, 6), dtype=int)):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous/inkless input accepted")
