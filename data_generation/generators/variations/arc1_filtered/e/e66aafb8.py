"""Generate missing-rectangle puzzles from fully square-symmetric designs."""

import hashlib
from collections import Counter

import numpy as np


# SHA-256 of shape and contents for the six staged source input/output pairs.
SOURCE_PAIR_DIGESTS = frozenset({
    "4ba162834ac3e009e193677f7fd08ec856bb8520d951d69e31a126520de6ef2d",
    "c209d3df832d6a264f633753d8e81de58cf861db34065ea8b5b6a72379a9c426",
    "3a67c8584b1840ccfe9366e1d1fb4026e4542a49b4bd31f4f9682e6df82027a9",
    "1ec8421c5cb703f4247ad0d6c4011f2ca39aaf03902eca947e734aa7a073fe6a",
    "4aa47ad93d55e6dd0d1ca0557731c3fc488e54a3565e11ce897f7239c49270d6",
    "49f63901563913ef210e5def86a92ff3f896a5a8d807a6a896bb928e8aadab05",
})


def _orbit(n: int, r: int, c: int) -> set[tuple[int, int]]:
    m = n - 1
    return {
        (r, c), (r, m - c), (m - r, c), (m - r, m - c),
        (c, r), (c, m - r), (m - c, r), (m - c, m - r),
    }


def _all_orbits_visible(n: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, h, w = box
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if all(r0 <= rr < r0 + h and c0 <= cc < c0 + w
                   for rr, cc in _orbit(n, r, c)):
                return False
    return True


def _marker_candidates(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    candidates = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        if len(cells) < 4:
            continue
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        r, c = int(lo[0]), int(lo[1])
        h, w = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
        if len(cells) == h * w:
            candidates.append((int(color), (r, c, h, w)))
    return candidates


def _locate_marker(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates = _marker_candidates(grid)
    assert len(candidates) == 1
    return candidates[0]


def _recover(grid: np.ndarray, marker: int,
             box: tuple[int, int, int, int]) -> np.ndarray:
    n = grid.shape[0]
    r0, c0, h, w = box
    missing = grid == marker
    output = np.empty((h, w), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            seen = {int(grid[r, c]) for r, c in _orbit(n, r0 + i, c0 + j)
                    if not missing[r, c]}
            assert len(seen) == 1
            output[i, j] = seen.pop()
    return output


def _make_design(rng: np.random.Generator, n: int,
                 palette: tuple[int, ...], mode: int) -> np.ndarray:
    """Color a triangular fundamental domain and expand its D4 orbits."""
    side, k = (n + 1) // 2, len(palette)
    weights = rng.dirichlet(np.full(k, float(rng.uniform(0.45, 2.3))))
    tri = np.zeros((side, side), dtype=np.int8)
    if mode == 0:  # Intricate source-like fields.
        p, q = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        tile = rng.integers(0, k, size=(p, q))
        row = rng.integers(0, k, size=side)
        for a in range(side):
            for b in range(a, side):
                tri[a, b] = (row[a] + row[b] + tile[a % p, b % q]) % k
    elif mode == 1:  # Irregular orbit colors with varying frequencies.
        for a in range(side):
            for b in range(a, side):
                tri[a, b] = int(rng.choice(k, p=weights))
    elif mode == 2:  # Coherent nested bands and crossing stripes.
        ring = rng.integers(0, k, size=side)
        stripe = rng.integers(0, k, size=side)
        period = int(rng.integers(2, 6))
        for a in range(side):
            for b in range(a, side):
                tri[a, b] = (ring[a] + stripe[b] + (b - a) // period) % k
    elif mode == 3:  # Overlapping nested motifs on a dominant color.
        base = int(rng.choice(k, p=weights))
        tri.fill(base)
        for _ in range(int(rng.integers(3, 10))):
            a0 = int(rng.integers(0, side))
            b0 = int(rng.integers(a0, side))
            ah = int(rng.integers(1, min(5, side - a0) + 1))
            bw = int(rng.integers(1, min(5, side - b0) + 1))
            color = int(rng.choice(k, p=weights))
            for a in range(a0, a0 + ah):
                for b in range(max(a, b0), b0 + bw):
                    tri[a, b] = color
    else:  # Sparse repeated marks on a plain field.
        base = int(rng.choice(k, p=weights))
        tri.fill(base)
        accents = [i for i in range(k) if i != base]
        for _ in range(int(rng.integers(5, 19))):
            a = int(rng.integers(0, side))
            b = int(rng.integers(a, side))
            color = int(rng.choice(accents))
            length = int(rng.integers(1, 4))
            slope = int(rng.integers(2))
            for step in range(length):
                aa, bb = a + step * slope, b + step
                if aa < side and bb < side and aa <= bb:
                    tri[aa, bb] = color

    colors = np.asarray(palette, dtype=np.int8)
    design = np.empty((n, n), dtype=np.int8)
    for r in range(n):
        dr = min(r, n - 1 - r)
        for c in range(n):
            dc = min(c, n - 1 - c)
            design[r, c] = colors[tri[min(dr, dc), max(dr, dc)]]
    return design


def _category(box: tuple[int, int, int, int], n: int) -> str:
    r, c, h, w = box
    low, high = (n - 1) // 2, n // 2
    vertical = c <= low and c + w - 1 >= high
    horizontal = r <= low and r + h - 1 >= high
    if vertical and horizontal:
        return "invalid"
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    return "quadrant"


def _draw_shape(rng: np.random.Generator, n: int,
                regime: str, train_shapes: list[tuple[int, int]],
                informative: bool = False) -> tuple[int, int]:
    if regime == "target_test":
        choice = rng.random()
        if choice < 0.16:
            return 3, 3
        if choice < 0.37 and train_shapes:
            shape = train_shapes[int(rng.integers(len(train_shapes)))]
            if (shape[0] <= n and shape[1] <= n and
                    shape[0] * shape[1] <= n * n // 3 and
                    (shape[0] < n // 2 or shape[1] < n // 2)):
                return shape
    maximum = min(10, n - 2)
    for _ in range(30):
        lower = 2 if informative else 1
        h = int(rng.integers(lower, maximum + 1))
        w = int(rng.integers(lower, maximum + 1))
        if 4 <= h * w <= n * n // 3 and (h < n // 2 or w < n // 2):
            return h, w
    return 3, 3


def _draw_box(rng: np.random.Generator, n: int, shape: tuple[int, int],
              category: str) -> tuple[int, int, int, int] | None:
    h, w = shape
    if h > n or w > n or h * w < 4 or h * w > n * n // 3:
        return None
    options = []
    for r in range(n - h + 1):
        for c in range(n - w + 1):
            box = (r, c, h, w)
            kind = _category(box, n)
            if kind == "invalid":
                continue
            if category in ("vertical", "horizontal", "quadrant") and kind != category:
                continue
            if category == "edge" and not (r == 0 or c == 0 or r + h == n or c + w == n):
                continue
            if category == "diagonal" and not any(
                r <= d < r + h and c <= d < c + w for d in range(n)
            ):
                continue
            if category == "near_center" and not (
                abs(2 * r + h - n) <= 6 or abs(2 * c + w - n) <= 6
            ):
                continue
            options.append(box)
    if not options:
        return None
    box = options[int(rng.integers(len(options)))]
    assert _all_orbits_visible(n, box)
    return box


def _pair_digest(pair: dict) -> str:
    a, b = pair["input"], pair["output"]
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8).tobytes()
                          + bytes(b.shape) + b.astype(np.int8).tobytes()).hexdigest()


def _make_pair(rng: np.random.Generator, n: int, marker: int,
               box: tuple[int, int, int, int], palette_size: int,
               informative: bool) -> dict:
    available = [color for color in range(10) if color != marker]
    for _ in range(100):
        palette = tuple(int(x) for x in rng.choice(available, size=palette_size,
                                                  replace=False))
        mode = int(rng.integers(5))
        design = _make_design(rng, n, palette, mode)
        if len(np.unique(design)) < 2:
            continue
        input_grid = design.copy()
        r, c, h, w = box
        input_grid[r:r + h, c:c + w] = marker
        if _marker_candidates(input_grid) != [(marker, box)]:
            continue
        output_grid = _recover(input_grid, marker, box)
        if informative and (len(np.unique(output_grid)) < 2 or
                            np.array_equal(output_grid, np.flipud(output_grid)) or
                            np.array_equal(output_grid, np.fliplr(output_grid))):
            continue
        pair = {"input": input_grid, "output": output_grid}
        if _pair_digest(pair) not in SOURCE_PAIR_DIGESTS:
            return pair
    raise RuntimeError("Could not construct a uniquely marked, recoverable pair")


def _validate_count(value: int | None, lower: int, upper: int, name: str) -> None:
    if value is not None and (isinstance(value, (bool, np.bool_)) or
                              not isinstance(value, (int, np.integer)) or
                              not lower <= value <= upper):
        raise ValueError(f"{name} must be an integer from {lower} to {upper}")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _validate_count(num_train, 2, 10, "num_train")
    _validate_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The reserved marker color is the only convention shared by every pair.
    marker = int(rng.integers(10))
    base_n = int(rng.integers(12, 31))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    train = []
    train_boxes = []
    train_shapes = []
    categories = list(rng.choice(
        ["quadrant", "vertical", "horizontal", "edge", "diagonal", "near_center"],
        size=num_train, p=[.28, .23, .23, .10, .08, .08]))
    # Every training set shows both an ordinary patch and an axis crossing.
    informative_index, crossing_index = (int(x) for x in
                                         rng.choice(num_train, size=2, replace=False))
    categories[informative_index] = "quadrant"
    categories[crossing_index] = str(rng.choice(["vertical", "horizontal"]))

    for i in range(num_train):
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        for _ in range(40):
            n = base_n if pair_rng.random() < .58 else int(pair_rng.integers(12, 31))
            shape = _draw_shape(pair_rng, n, "source_train", train_shapes,
                                i == informative_index)
            box = _draw_box(pair_rng, n, shape, categories[i])
            if box is not None and (i == 0 or ((n, box) not in train_boxes and
                                              (i != 1 or shape != train_shapes[0]))):
                break
        else:
            raise RuntimeError("Could not place training patch")
        palette_size = int(pair_rng.integers(2, 10))
        pair = _make_pair(pair_rng, n, marker, box, palette_size,
                          i == informative_index)
        train.append(pair)
        train_boxes.append((n, box))
        train_shapes.append(shape)

    # Keep palette size visibly pair-specific even in the smallest training set.
    def observed_size(pair: dict) -> int:
        return len(set(int(x) for x in np.unique(pair["input"])) - {marker})

    if len({observed_size(pair) for pair in train}) == 1:
        desired = 9 if observed_size(train[0]) <= 5 else 2
        for _ in range(20):
            train[-1] = _make_pair(pair_rng, n, marker, box, desired,
                                   informative_index == len(train) - 1)
            if observed_size(train[-1]) != observed_size(train[0]):
                break
        else:
            raise RuntimeError("Could not vary the training palettes")

    test = []
    for i in range(num_test):
        pair_rng = np.random.default_rng(int(pair_seeds[num_train + i]))
        for _ in range(40):
            if pair_rng.random() < .20:
                n, box = train_boxes[int(pair_rng.integers(len(train_boxes)))]
            else:
                n = base_n if pair_rng.random() < .48 else int(pair_rng.integers(12, 31))
                shape = _draw_shape(pair_rng, n, "target_test", train_shapes)
                category = str(pair_rng.choice(
                    ["quadrant", "vertical", "horizontal", "edge", "near_center", "diagonal"],
                    p=[.16, .27, .27, .10, .14, .06]))
                box = _draw_box(pair_rng, n, shape, category)
            if box is not None:
                break
        else:
            raise RuntimeError("Could not place test patch")
        palette_size = int(pair_rng.integers(2, 10))
        test.append(_make_pair(pair_rng, n, marker, box, palette_size, False))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair: dict) -> tuple[int, frozenset[int], tuple[int, int, int, int]]:
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= d <= 30 for d in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.min(grid) >= 0 and np.max(grid) <= 9
    input_grid, output_grid = pair["input"], pair["output"]
    n = input_grid.shape[0]
    assert 12 <= n <= 30 and input_grid.shape == (n, n)
    marker, box = _locate_marker(input_grid)
    r, c, h, w = box
    assert 4 <= h * w <= n * n // 3 and _all_orbits_visible(n, box)
    assert output_grid.shape == (h, w)
    assert np.array_equal(output_grid, _recover(input_grid, marker, box))
    restored = input_grid.copy()
    restored[r:r + h, c:c + w] = output_grid
    assert not np.any(restored == marker)
    assert len(np.unique(restored)) >= 2
    assert np.array_equal(restored, np.flipud(restored))
    assert np.array_equal(restored, np.fliplr(restored))
    assert np.array_equal(restored, restored.T)
    assert _pair_digest(pair) not in SOURCE_PAIR_DIGESTS
    return marker, frozenset(int(x) for x in np.unique(restored)), box


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    markers, palettes, signatures = set(), set(), set()
    train_kinds = set()
    train_shapes = set()
    train_palette_sizes = set()
    informative = False
    for regime, pairs in (("source_train", train), ("target_test", test)):
        for pair in pairs:
            marker, palette, box = _validate_pair(pair)
            markers.add(marker)
            palettes.add(palette)
            n = pair["input"].shape[0]
            signature = (pair["input"].shape, pair["input"].tobytes())
            assert signature not in signatures
            signatures.add(signature)
            if regime == "source_train":
                train_kinds.add(_category(box, n))
                train_shapes.add(box[2:])
                train_palette_sizes.add(len(palette))
                out = pair["output"]
                informative |= (len(np.unique(out)) >= 2 and
                                not np.array_equal(out, np.flipud(out)) and
                                not np.array_equal(out, np.fliplr(out)))
    assert len(markers) == 1
    assert "quadrant" in train_kinds
    assert "vertical" in train_kinds or "horizontal" in train_kinds
    assert len(train_shapes) >= 2 and len(train_palette_sizes) >= 2
    assert informative


def _same_puzzle(a: dict, b: dict) -> bool:
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x[g], y[g]) for x, y in zip(a[k], b[k])
        for g in ("input", "output")) for k in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, shapes, locations = set(), set(), set(), set(), set()
    markers, palette_sizes = set(), set()
    vertical = horizontal = edge = uniform = self_symmetric = 0
    review = Counter()
    first_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training = puzzle["train"]
        training_boxes = [(p["input"].shape[0], _locate_marker(p["input"])[1])
                          for p in training]
        training_shapes = [p["output"].shape for p in training]
        review["size_diverse_puzzles"] += len({p["input"].shape[0] for p in training}) > 1
        repeated_shapes = {}
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_inputs.add(puzzle["train"][0]["input"].tobytes())
        for split, pair in ([('train', p) for p in puzzle["train"]] +
                            [('test', p) for p in puzzle["test"]]):
            marker, palette, box = _validate_pair(pair)
            n = pair["input"].shape[0]
            r, c, h, w = box
            markers.add(marker)
            palette_sizes.add(len(palette))
            sizes.add(n)
            shapes.add(box[2:])
            locations.add((n, box[0], box[1]))
            kind = _category(box, n)
            vertical += kind == "vertical"
            horizontal += kind == "horizontal"
            is_edge = r == 0 or c == 0 or r + h == n or c + w == n
            edge += is_edge
            out = pair["output"]
            uniform += len(np.unique(out)) == 1
            self_symmetric += (np.array_equal(out, np.flipud(out)) or
                               np.array_equal(out, np.fliplr(out)))
            review[(split, "square")] += h == w
            review[(split, "strip")] += min(h, w) == 1
            review[(split, "large")] += h * w >= 40
            review[(split, "under9")] += h * w < 9
            review[(split, "edge")] += is_edge
            review[(split, "diagonal")] += any(
                r <= d < r + h and c <= d < c + w for d in range(n))
            review[(split, "near_center")] += (
                abs(2 * r + h - n) <= 6 or abs(2 * c + w - n) <= 6)
            review[(split, "low_color")] += len(palette) <= 3
            review[(split, "high_color")] += len(palette) >= 7
            review[(split, "odd")] += n % 2 == 1
            review[(split, "even")] += n % 2 == 0
            repeated_shapes.setdefault((h, w), set()).add(out.tobytes())
            if split == "test":
                review["test_3x3"] += (h, w) == (3, 3)
                review["reuse_shape"] += (h, w) in training_shapes
                review["reuse_box"] += (n, box) in training_boxes
                review["smaller_test"] += h * w < min(a * b for a, b in training_shapes)
                review["larger_test"] += h * w > max(a * b for a, b in training_shapes)
                review["test_axis_crossing"] += kind in ("vertical", "horizontal")
        review["repeated_shape_new_content_puzzles"] += any(
            len(values) > 1 for values in repeated_shapes.values())
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 12 and min(sizes) <= 14 and max(sizes) >= 28
    assert len(shapes) >= 30 and len(locations) >= 80
    assert len(markers) >= 6 and len(palette_sizes) >= 6
    assert vertical >= 15 and horizontal >= 15 and edge >= 5
    assert uniform >= 1 and self_symmetric >= 5
    assert len(first_inputs) >= 40
    assert review["size_diverse_puzzles"] >= 20
    assert review["repeated_shape_new_content_puzzles"] >= 12
    for split in ("train", "test"):
        for feature in ("square", "strip", "large", "under9", "edge", "diagonal",
                        "near_center", "low_color", "high_color", "odd", "even"):
            assert review[(split, feature)] >= 5
    for feature in ("test_3x3", "reuse_shape", "reuse_box", "smaller_test",
                    "larger_test", "test_axis_crossing"):
        assert review[feature] >= 3
    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4),
                                            (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(143, requested_train, requested_test))
    assert _same_puzzle(generate_puzzle(917, 7, 3), generate_puzzle(917, 7, 3))
    assert _same_puzzle(generate_puzzle(918), generate_puzzle(918))
    assert not _same_puzzle(generate_puzzle(917), generate_puzzle(918))
    for requested_train, requested_test in ((1, None), (11, None), (None, 0),
                                            (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(5, requested_train, requested_test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
    altered = generate_puzzle(911, 2, 1)
    output = altered["test"][0]["output"]
    output[0, 0] = (int(output[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(altered)
    except AssertionError:
        pass
    else:
        raise AssertionError("An incorrect output was accepted")
