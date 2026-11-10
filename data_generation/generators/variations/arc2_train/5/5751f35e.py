"""Generate concentric-square erasure puzzles in the source ARC family."""

import numpy as np


_SOURCE_PAIRS = (
    (
        ("3030033303", "0222002020", "0220000200", "0208088020", "3220880220",
         "3008080200", "0028088220", "0220222220", "0022202023", "3330333303"),
        ("3333333333", "3222222223", "3222222223", "3228888223", "3228888223",
         "3228888223", "3228888223", "3222222223", "3222222223", "3333333333"),
    ),
    (
        ("0000000000", "0880888880", "0820002200", "0002002080", "0802200000",
         "0820002080", "0802000000", "0800000080", "0088808800", "0000000000"),
        ("0000000000", "0888888880", "0822222280", "0822222280", "0822222280",
         "0822222280", "0822222280", "0822222280", "0888888880", "0000000000"),
    ),
    (
        ("400004044040", "040000440444", "440330333300", "403030303044",
         "043301103304", "040310100340", "003311110304", "043001013304",
         "443333000300", "003033003344", "444444440040", "404404004404"),
        ("444444444444", "444444444444", "443333333344", "443333333344",
         "443311113344", "443311113344", "443311113344", "443311113344",
         "443333333344", "443333333344", "444444444444", "444444444444"),
    ),
)


def _seed_key(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, int(num_test) + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _radius(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    rows, cols = np.indices(shape)
    return np.maximum(np.abs(2 * rows - (height - 1)),
                      np.abs(2 * cols - (width - 1)))


def _paint(shape: tuple[int, int], background: int,
           colors: tuple[int, ...], sides: tuple[int, ...]) -> np.ndarray:
    height, width = shape
    result = np.full(shape, background, dtype=np.int8)
    for color, side in zip(colors, sides):
        top, left = (height - side) // 2, (width - side) // 2
        result[top:top + side, left:left + side] = color
    return result


def _boundary(shape: tuple[int, int], side: int, direction: int) -> np.ndarray:
    height, width = shape
    top, left = (height - side) // 2, (width - side) // 2
    bottom, right = top + side - 1, left + side - 1
    if direction == 0:
        return np.array([(top, c) for c in range(left, right + 1)], dtype=int)
    if direction == 1:
        return np.array([(bottom, c) for c in range(left, right + 1)], dtype=int)
    if direction == 2:
        return np.array([(r, left) for r in range(top, bottom + 1)], dtype=int)
    return np.array([(r, right) for r in range(top, bottom + 1)], dtype=int)


def _geometry(rng: np.random.Generator, layers: int, *,
              parity: int | None = None, rectangle: bool | None = None,
              full: bool | None = None,
              target_shape: tuple[int, int] | None = None) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Choose a centered square stack; canvas margins may differ by axis."""
    if target_shape is None:
        if parity is None:
            parity = int(rng.integers(0, 2))
        if rectangle is None:
            rectangle = bool(rng.random() < 0.30)
        if full is None:
            full = bool(rng.random() < 0.30) and not rectangle
        minimum = (4 if parity == 0 else 3) + 2 * (layers - 1)
        outer = int(rng.choice(np.arange(minimum, 23, 2)))
        if full:
            row_margin = col_margin = 0
        elif rectangle:
            row_margin = int(rng.integers(0, 5))
            choices = [m for m in range(5) if m != row_margin]
            col_margin = int(rng.choice(choices))
        else:
            row_margin = col_margin = int(rng.integers(1, 5))
        shape = (outer + 2 * row_margin, outer + 2 * col_margin)
    else:
        shape = target_shape
        height, width = shape
        assert height % 2 == width % 2
        minimum = (4 if height % 2 == 0 else 3) + 2 * (layers - 1)
        choices = [s for s in range(minimum, min(height, width, 22) + 1, 2)
                   if (height - s) // 2 <= 4 and (width - s) // 2 <= 4]
        if not choices:
            return _geometry(rng, layers, parity=height % 2,
                             rectangle=height != width)
        outer = int(rng.choice(choices))
    inner_options = np.arange(4 if outer % 2 == 0 else 3, outer, 2)
    selected = rng.choice(inner_options, size=layers - 1, replace=False)
    sides = (outer,) + tuple(sorted((int(s) for s in selected), reverse=True))
    return shape, sides


def _damage(rng: np.random.Generator, output: np.ndarray, background: int,
            colors: tuple[int, ...], sides: tuple[int, ...], regime: str) -> np.ndarray:
    shape = output.shape
    structural = output != background
    score = rng.random(shape)
    modes = ("scatter", "patch", "run", "corner", "mixed")
    mode = str(rng.choice(modes if regime == "source_train" else modes[1:]))
    tier = str(rng.choice(("light", "moderate", "heavy"),
                          p=(.30, .43, .27) if regime == "source_train"
                          else (.22, .42, .36)))
    rate = {"light": (.14, .29), "moderate": (.30, .49),
            "heavy": (.50, .68)}[tier]
    fraction = float(rng.uniform(*rate))

    if mode in ("patch", "mixed"):
        h = int(rng.integers(2, min(7, shape[0]) + 1))
        w = int(rng.integers(2, min(7, shape[1]) + 1))
        r = int(rng.integers(0, shape[0] - h + 1))
        c = int(rng.integers(0, shape[1] - w + 1))
        score[r:r + h, c:c + w] -= .85
    if mode in ("run", "mixed"):
        index = int(rng.integers(0, len(sides)))
        side = int(rng.integers(0, 4))
        points = _boundary(shape, sides[index], side)
        length = int(rng.integers(max(2, len(points) // 3), len(points) + 1))
        start = int(rng.integers(0, len(points) - length + 1))
        chosen = points[start:start + length]
        score[chosen[:, 0], chosen[:, 1]] -= .9
    if mode in ("corner", "mixed"):
        index = int(rng.integers(0, len(sides)))
        side = sides[index]
        top, left = (shape[0] - side) // 2, (shape[1] - side) // 2
        extent = int(rng.integers(2, min(5, side) + 1))
        r = top if rng.random() < .5 else top + side - extent
        c = left if rng.random() < .5 else left + side - extent
        score[r:r + extent, c:c + extent] -= .9

    # Different layer densities make damage visibly uneven, independently of mode.
    selected_layer = int(rng.integers(0, len(colors)))
    score[output == colors[selected_layer]] -= float(rng.uniform(.10, .36))

    forced = np.zeros(shape, dtype=bool)
    missing_index = None
    missing_direction = None
    chance = .36 if regime == "source_train" else .57
    if rng.random() < chance:
        possible = list(range(len(sides)))
        if sides[0] < max(shape):
            possible = possible[1:]
        missing_index = int(rng.choice(possible))
        missing_direction = int(rng.integers(0, 4))
        points = _boundary(shape, sides[missing_index], missing_direction)
        forced[points[:, 0], points[:, 1]] = True

    protected = np.zeros(shape, dtype=bool)
    for index, (color, side) in enumerate(zip(colors, sides)):
        for direction in range(4):
            if index == missing_index and direction == missing_direction:
                continue
            points = _boundary(shape, side, direction)
            valid = points[(output[points[:, 0], points[:, 1]] == color)
                           & ~forced[points[:, 0], points[:, 1]]]
            assert len(valid) > 0
            # Prefer side interiors, leaving corners free to form missing patches.
            if len(valid) > 2:
                interior = valid[1:-1]
                if len(interior):
                    valid = interior
            point = valid[int(rng.integers(0, len(valid)))]
            protected[point[0], point[1]] = True

    for color in colors:
        if not np.any(forced & (output == color)):
            points = np.argwhere((output == color) & ~protected)
            point = points[int(rng.integers(0, len(points)))]
            forced[point[0], point[1]] = True
    assert not np.any(forced & protected)
    eligible = structural & ~protected
    total = int(np.count_nonzero(structural))
    goal = min(int(np.count_nonzero(eligible)),
               max(int(round(fraction * total)), int(np.count_nonzero(forced))))
    mask = forced.copy()
    remaining = goal - int(np.count_nonzero(mask))
    if remaining:
        points = np.argwhere(eligible & ~mask)
        priorities = score[points[:, 0], points[:, 1]]
        chosen = points[np.argsort(priorities)[:remaining]]
        mask[chosen[:, 0], chosen[:, 1]] = True
    result = output.copy()
    result[mask] = background
    return result


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _source_copy(pair: dict) -> bool:
    return (_rows(pair["input"]), _rows(pair["output"])) in _SOURCE_PAIRS


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _new_pair(rng: np.random.Generator, background: int, colors: tuple[int, ...],
              regime: str, seen: set, *, parity: int | None = None,
              rectangle: bool | None = None, full: bool | None = None,
              target_shape: tuple[int, int] | None = None) -> dict:
    for _ in range(32):
        shape, sides = _geometry(rng, len(colors), parity=parity,
                                 rectangle=rectangle, full=full,
                                 target_shape=target_shape)
        output = _paint(shape, background, colors, sides)
        input_grid = _damage(rng, output, background, colors, sides, regime)
        pair = {"input": input_grid, "output": output}
        key = _key(input_grid)
        if key not in seen and not _source_copy(pair):
            seen.add(key)
            return pair
    raise RuntimeError("unable to construct a distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_key = _seed_key(seed)
    train_count, test_count = _counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)

    # Background is the sole puzzle-level convention. Layer colors are roles
    # read from each input; a shared training color explicitly changes depth.
    background = int(rng.integers(0, 10))
    palette = np.array([c for c in range(10) if c != background], dtype=int)
    rng.shuffle(palette)
    train_pool = tuple(int(c) for c in palette[:5])
    novel_pool = tuple(int(c) for c in palette[5:])
    shared = train_pool[0]
    counts = list(rng.choice((2, 3, 4), size=2, replace=False))
    counts += [int(rng.choice((2, 3, 4), p=(.40, .42, .18)))
               for _ in range(train_count - 2)]
    first_depth = int(rng.integers(0, counts[0]))
    second_depth = int(rng.choice([d for d in range(counts[1])
                                   if d != first_depth]))
    parities = list(rng.permutation((0, 1)))
    full_index = int(rng.integers(0, 2))
    rectangle_index = 1 - full_index if rng.random() < .63 else None
    puzzle = {"train": [], "test": []}
    seen = set()

    def train_colors(count: int, shared_depth: int | None) -> tuple[int, ...]:
        chosen = list(int(v) for v in rng.choice(train_pool[1:],
                                                 size=count - (shared_depth is not None),
                                                 replace=False))
        if shared_depth is not None:
            chosen.insert(shared_depth, shared)
        return tuple(chosen)

    for index in range(train_count):
        count = counts[index]
        depth = first_depth if index == 0 else second_depth if index == 1 else (
            int(rng.integers(0, count)) if rng.random() < .55 else None)
        colors = train_colors(count, depth)
        if index < 2:
            full = index == full_index
            rectangle = index == rectangle_index
            parity = parities[index]
        else:
            rectangle = bool(rng.random() < .30)
            full = bool(rng.random() < .30) and not rectangle
            parity = None
        puzzle["train"].append(_new_pair(rng, background, colors, "source_train",
                                         seen, parity=parity, rectangle=rectangle,
                                         full=full))

    train_shapes = [p["output"].shape for p in puzzle["train"]]
    trained_rectangle = any(h != w for h, w in train_shapes)
    for _ in range(test_count):
        count = int(rng.choice(counts))
        color_mode = str(rng.choice(("new", "mixed", "reused"), p=(.25, .45, .30)))
        if color_mode == "new":
            colors = tuple(int(v) for v in rng.choice(novel_pool, size=count,
                                                      replace=False))
        elif color_mode == "reused":
            colors = tuple(int(v) for v in rng.choice(train_pool, size=count,
                                                      replace=False))
        else:
            new_count = int(rng.integers(1, count))
            colors_list = list(int(v) for v in rng.choice(novel_pool, size=new_count,
                                                         replace=False))
            colors_list += list(int(v) for v in rng.choice(train_pool,
                                                           size=count - new_count,
                                                           replace=False))
            rng.shuffle(colors_list)
            colors = tuple(colors_list)
        target_shape = None
        if rng.random() < .28:
            # Reuse a demonstrated canvas size while redrawing bands and damage.
            compatible = [shape for shape in train_shapes
                          if min(shape) >= (4 if shape[0] % 2 == 0 else 3)
                          + 2 * (count - 1)]
            if compatible:
                target_shape = compatible[int(rng.integers(0, len(compatible)))]
        rectangle = bool(rng.random() < .35) and trained_rectangle
        full = bool(rng.random() < .22) and not rectangle
        puzzle["test"].append(_new_pair(rng, background, colors, "target_test",
                                        seen, rectangle=rectangle, full=full,
                                        target_shape=target_shape))
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer(input_grid: np.ndarray, background: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    height, width = input_grid.shape
    assert height % 2 == width % 2
    radius = _radius(input_grid.shape)
    layers = []
    for value in np.unique(input_grid):
        color = int(value)
        if color != background:
            side = int(np.max(radius[input_grid == color])) + 1
            layers.append((side, color))
    assert 2 <= len(layers) <= 4
    assert len({side for side, _ in layers}) == len(layers)
    layers.sort(reverse=True)
    sides = tuple(side for side, _ in layers)
    colors = tuple(color for _, color in layers)
    assert all(side >= (4 if height % 2 == 0 else 3)
               and side <= min(height, width)
               and side % 2 == height % 2 for side in sides)
    clean = _paint(input_grid.shape, background, colors, sides)
    visible = input_grid != background
    assert np.array_equal(clean[visible], input_grid[visible])
    return clean, colors, sides


def _pair_info(pair: dict, background: int) -> dict:
    source, target = pair["input"], pair["output"]
    clean, colors, sides = _infer(source, background)
    assert np.array_equal(clean, target)
    assert not np.array_equal(source, target)
    assert np.all((source == target) | ((source == background) & (target != background)))
    boundary_counts = []
    for color, side in zip(colors, sides):
        assert np.any((source == background) & (target == color))
        visible_sides = sum(
            np.any(source[tuple(_boundary(source.shape, side, direction).T)] == color)
            for direction in range(4))
        assert visible_sides >= 3
        boundary_counts.append(visible_sides)
    outer = sides[0]
    margins = ((source.shape[0] - outer) // 2, (source.shape[1] - outer) // 2)
    if margins != (0, 0):
        assert boundary_counts[0] == 4
    erased = (source == background) & (target != background)
    missing_sides = sum(
        not np.any(source[tuple(_boundary(source.shape, side, direction).T)] == color)
        for color, side in zip(colors, sides) for direction in range(4))
    return {"shape": source.shape, "sides": sides, "colors": colors,
            "margins": margins, "layers": len(colors),
            "erasure": float(np.count_nonzero(erased) / np.count_nonzero(target != background)),
            "missing_sides": missing_sides}


def _analyze_puzzle_task(puzzle: dict) -> dict:
    # The changed input color is the learned blank/erasure convention.
    changed = [pair["input"][pair["input"] != pair["output"]]
               for pair in puzzle["train"]]
    assert all(len(values) > 0 for values in changed)
    backgrounds = set(int(v) for values in changed for v in np.unique(values))
    assert len(backgrounds) == 1
    background = next(iter(backgrounds))
    train = [_pair_info(pair, background) for pair in puzzle["train"]]
    test = [_pair_info(pair, background) for pair in puzzle["test"]]
    assert len({info["layers"] for info in train}) >= 2
    assert any(info["margins"] == (0, 0) for info in train)
    assert any(info["margins"] != (0, 0) for info in train)
    assert {shape[0] % 2 for shape in (info["shape"] for info in train)} == {0, 1}
    train_depths = {}
    for info in train:
        for depth, color in enumerate(info["colors"]):
            train_depths.setdefault(color, set()).add(depth)
    assert any(len(depths) >= 2 for depths in train_depths.values())
    if any(info["shape"][0] != info["shape"][1] for info in test):
        assert any(info["shape"][0] != info["shape"][1] for info in train)
    for info in test:
        assert info["layers"] in {entry["layers"] for entry in train}
    return {"background": background, "train_info": train, "test_info": test}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= dimension <= 30 for dimension in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        input_key = _key(pair["input"])
        pair_key = (input_key, _key(pair["output"]))
        assert input_key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        assert not _source_copy(pair)
    _analyze_puzzle_task(puzzle)


def _signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    analyses = []
    signatures = set()
    layouts = set()
    train_counts, test_counts = set(), set()
    coverage = {name: 0 for name in (
        "test_margin", "test_same_shape", "test_fewer_layers", "test_rect",
        "test_four", "test_all_new", "test_all_reused", "test_mixed",
        "train_band_variety", "test_light", "test_moderate", "test_heavy",
        "test_missing_side", "test_outer_reused", "test_middle_reused",
        "test_inner_reused", "combined_rect_four_heavy")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        analysis = _analyze_puzzle_task(puzzle)
        analyses.append(analysis)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info, test_info = analysis["train_info"], analysis["test_info"]
        training_colors = {color for info in train_info for color in info["colors"]}
        coverage["test_margin"] += any(info["margins"] != (0, 0) for info in test_info)
        coverage["test_same_shape"] += any(info["shape"] in {t["shape"] for t in train_info}
                                           for info in test_info)
        coverage["test_fewer_layers"] += any(info["layers"] < max(t["layers"] for t in train_info)
                                              for info in test_info)
        coverage["test_rect"] += any(info["shape"][0] != info["shape"][1] for info in test_info)
        coverage["test_four"] += any(info["layers"] == 4 for info in test_info)
        coverage["test_all_new"] += any(set(info["colors"]).isdisjoint(training_colors)
                                        for info in test_info)
        coverage["test_all_reused"] += any(set(info["colors"]).issubset(training_colors)
                                           for info in test_info)
        coverage["test_mixed"] += any(set(info["colors"]) & training_colors
                                      and set(info["colors"]) - training_colors
                                      for info in test_info)
        coverage["train_band_variety"] += len({(a - b) // 2 for info in train_info
                                                for a, b in zip(info["sides"], info["sides"][1:])}) > 1
        coverage["test_light"] += any(info["erasure"] < .30 for info in test_info)
        coverage["test_moderate"] += any(.30 <= info["erasure"] < .50 for info in test_info)
        coverage["test_heavy"] += any(info["erasure"] >= .50 for info in test_info)
        coverage["test_missing_side"] += any(info["missing_sides"] > 0 for info in test_info)
        coverage["test_outer_reused"] += any(info["colors"][0] in training_colors for info in test_info)
        coverage["test_middle_reused"] += any(len(info["colors"]) >= 3
                                              and info["colors"][1] in training_colors for info in test_info)
        coverage["test_inner_reused"] += any(info["colors"][-1] in training_colors for info in test_info)
        coverage["combined_rect_four_heavy"] += any(info["shape"][0] != info["shape"][1]
                                                    and info["layers"] == 4
                                                    and info["erasure"] >= .50
                                                    for info in train_info + test_info)
        for pair, info in zip(puzzle["train"] + puzzle["test"],
                              analysis["train_info"] + analysis["test_info"]):
            layouts.add((info["shape"], info["sides"]))
            inferred, _, _ = _infer(pair["input"], analysis["background"])
            assert np.array_equal(inferred, pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(layouts) >= 80
    assert all(value >= 1 for value in coverage.values()), coverage
    assert coverage["test_margin"] >= 15 and coverage["test_same_shape"] >= 5
    assert coverage["test_all_new"] >= 5 and coverage["test_middle_reused"] >= 5
    infos = [info for analysis in analyses
             for info in analysis["train_info"] + analysis["test_info"]]
    assert {info["layers"] for info in infos} == {2, 3, 4}
    assert any(info["shape"][0] != info["shape"][1] for info in infos)
    assert any(min(info["margins"]) >= 2 for info in infos)
    assert any(info["missing_sides"] for info in infos)
    assert min(info["erasure"] for info in infos) < .30
    assert max(info["erasure"] for info in infos) > .50
    assert any(info["sides"][-1] >= 7 for info in infos)
    assert any(info["sides"][-1] <= 4 for info in infos)
    assert any(info["shape"] in {entry["shape"] for entry in analysis["train_info"]}
               for analysis in analyses for info in analysis["test_info"])
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                              (103, 7, 3), (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train, test))
    assert _signature(generate_puzzle(999, 5, 2)) != _signature(generate_puzzle(1000, 5, 2))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (2, 4), (4, 4)):
        try:
            generate_puzzle(1234, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
