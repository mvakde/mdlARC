"""Solved ARC puzzles: crop the sole left-right symmetric colored figure."""

import numpy as np


_SOURCE_ROWS = (
    ("0000000000", "0044000000", "0044008880", "0000008088",
     "0000000000", "0000000000", "0000000000", "0022220000",
     "0222000000", "0000000000"),
    ("0000000000", "0220000000", "0022200770", "0000007070",
     "0000000000", "0000000000", "0006666000", "0000660000",
     "0000000000", "0000000000"),
    ("0000000000", "0330000000", "0030050050", "0030055550",
     "0000000000", "0000000000", "0000000000", "0008880000",
     "8888088000", "0000000000"),
    ("0000000000", "0000030030", "0990033330", "0990000030",
     "9999000000", "0000000000", "0000000000", "0000444440",
     "0000400440", "0000000000"),
)
_SOURCE_INPUTS = tuple(
    np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    for rows in _SOURCE_ROWS
)


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.flat)


def _same_puzzle(a: dict, b: dict) -> bool:
    return all(
        len(a[part]) == len(b[part])
        and all(
            np.array_equal(x[name], y[name])
            for x, y in zip(a[part], b[part])
            for name in ("input", "output")
        )
        for part in ("train", "test")
    )


def _count_arg(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(seed: int, rng: np.random.Generator,
            num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count_arg("num_train", num_train, 2, 10)
    test = _count_arg("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _tight(mask: np.ndarray) -> bool:
    return bool(mask.any() and mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any())


def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    seen = {(int(cells[0, 0]), int(cells[0, 1]))}
    frontier = list(seen)
    height, width = mask.shape
    while frontier:
        row, col = frontier.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = row + dr, col + dc
                if (0 <= nr < height and 0 <= nc < width
                        and mask[nr, nc] and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
    return len(seen) == len(cells)


def _left_right(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _target_mask(rng: np.random.Generator, height: int, width: int,
                 orientation_example: bool = False) -> np.ndarray:
    if not orientation_example and (width == 2 or rng.random() < 0.11):
        return np.ones((height, width), dtype=bool)
    half = (width + 1) // 2
    for _ in range(700):
        density = float(rng.uniform(0.43, 0.83))
        left = rng.random((height, half)) < density
        if width % 2:
            mask = np.concatenate((left, left[:, :-1][:, ::-1]), axis=1)
        else:
            mask = np.concatenate((left, left[:, ::-1]), axis=1)
        if not (_tight(mask) and _connected(mask)):
            continue
        if mask.all():
            continue
        if orientation_example and np.array_equal(mask, mask[::-1]):
            continue
        return mask
    # Deterministic connected fallback, with a visible row difference.
    mask = np.ones((height, width), dtype=bool)
    mask[0, width // 2 - 1] = False
    mask[0, width - (width // 2)] = False
    if width % 2:
        mask[0, width // 2] = True
    return mask


def _asym_random(rng: np.random.Generator, height: int, width: int,
                 area: int | None = None) -> np.ndarray | None:
    for _ in range(350):
        n = area if area is not None else int(rng.integers(
            max(3, int(height * width * 0.35)),
            min(height * width - 1, max(5, int(height * width * 0.82))) + 1))
        if not 3 <= n < height * width:
            return None
        mask = np.zeros((height, width), dtype=bool)
        mask.flat[rng.choice(height * width, size=n, replace=False)] = True
        if _tight(mask) and _connected(mask) and not _left_right(mask):
            return mask
    return None


def _near_mirror(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    for _ in range(45):
        base = _target_mask(rng, height, width)
        options = [(r, c) for r in range(height) for c in range(width)
                   if c != width - 1 - c]
        rng.shuffle(options)
        for row, col in options:
            mask = base.copy()
            mask[row, col] = ~mask[row, col]
            if _tight(mask) and _connected(mask) and not _left_right(mask):
                return mask
    return None


def _horizontal_only(rng: np.random.Generator, height: int,
                     width: int) -> np.ndarray | None:
    half = (height + 1) // 2
    for _ in range(400):
        top = rng.random((half, width)) < float(rng.uniform(0.46, 0.81))
        if height % 2:
            mask = np.concatenate((top, top[:-1][::-1]), axis=0)
        else:
            mask = np.concatenate((top, top[::-1]), axis=0)
        if _tight(mask) and _connected(mask) and not _left_right(mask):
            return mask
    return None


def _distractor(rng: np.random.Generator, target: np.ndarray,
                mode: str, compact: bool) -> np.ndarray:
    th, tw = target.shape
    target_area = int(target.sum())
    max_h, max_w = (4, 6) if compact else (8, 10)
    for _ in range(120):
        if mode == "box":
            h, w = th, tw
        else:
            h = int(rng.integers(2, max_h + 1))
            w = int(rng.integers(2, max_w + 1))
        if mode == "tie":
            if not max(h, w) <= target_area < h * w:
                continue
            if target_area < int(h * w * 0.32):
                continue
            mask = _asym_random(rng, h, w, target_area)
        elif mode == "near":
            mask = _near_mirror(rng, h, w)
        elif mode == "horizontal":
            mask = _horizontal_only(rng, max(3, h), max(3, w))
        elif mode in ("small", "large"):
            if mode == "small":
                choices = [n for n in range(3, min(target_area - 2, h * w - 1) + 1)
                           if max(h, w) <= n and n >= int(h * w * 0.32)]
            else:
                choices = [n for n in range(target_area + 3, h * w)
                           if n >= int(h * w * 0.32)]
            if not choices:
                continue
            mask = _asym_random(rng, h, w,
                                int(choices[int(rng.integers(len(choices)))]))
        else:
            mask = _asym_random(rng, h, w)
        if mask is not None:
            return mask
    if mode != "random":
        return _distractor(rng, target, "random", compact)
    raise RuntimeError("could not construct an asymmetric distractor")


def _rects_separate(a: tuple[int, int, int, int],
                    b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah < br or br + bh < ar
            or ac + aw < bc or bc + bw < ac)


def _place(rng: np.random.Generator, masks: list[np.ndarray],
           colors: list[int], background: int, compact: bool) -> tuple[np.ndarray, np.ndarray]:
    largest_h = max(mask.shape[0] for mask in masks)
    largest_w = max(mask.shape[1] for mask in masks)
    if compact:
        base_h = int(rng.integers(10, 15))
        base_w = int(rng.integers(10, 15))
    else:
        floor = 19 if len(masks) <= 3 else 23
        base_h = int(rng.integers(floor, 29))
        base_w = int(rng.integers(floor, 29))
    for growth in (0, 2, 4, 6, 8, 10, 12):
        height = min(30, max(base_h + growth, largest_h + 2))
        width = min(30, max(base_w + growth, largest_w + 2))
        placements: dict[int, tuple[int, int, int, int]] = {}
        order = sorted(range(len(masks)),
                       key=lambda i: int(masks[i].size), reverse=True)
        for index in order:
            mh, mw = masks[index].shape
            positions = [
                (r, c, mh, mw)
                for r in range(height - mh + 1)
                for c in range(width - mw + 1)
                if all(_rects_separate((r, c, mh, mw), prior)
                       for prior in placements.values())
            ]
            if not positions:
                break
            placements[index] = positions[int(rng.integers(len(positions)))]
        if len(placements) != len(masks):
            continue
        canvas = np.full((height, width), background, dtype=np.int8)
        for index, mask in enumerate(masks):
            row, col, mh, mw = placements[index]
            view = canvas[row:row + mh, col:col + mw]
            view[mask] = colors[index]
        row, col, mh, mw = placements[0]
        return canvas, canvas[row:row + mh, col:col + mw].copy()
    raise RuntimeError("could not place separated figures")


def _make_pair(rng: np.random.Generator, regime: str, index: int,
               prior_target: int | None, prior_distractor: int | None,
               first_background: int | None) -> dict:
    compact = (index == 1 if regime == "source_train"
               else bool(rng.random() < (0.38 if regime == "source_train" else 0.22)))
    if regime == "source_train" and index == 0:
        compact = False
        height = int(rng.integers(3, 6))
        width = int(rng.choice([4, 6]))
    elif regime == "source_train" and index == 1:
        height = 2
        width = int(rng.choice([3, 5]))
    elif compact:
        if rng.random() < 0.13:
            height, width = 2, 2
        else:
            height = int(rng.integers(2, 4))
            width = int(rng.integers(3, 6))
    else:
        height = int(rng.integers(2, 7 if regime == "source_train" else 8))
        width = int(rng.integers(3, 9 if regime == "source_train" else 10))
    target = _target_mask(rng, height, width,
                          orientation_example=regime == "source_train" and index == 0)
    n_others = (2 if regime == "source_train" and index < 2
                else int(rng.choice([1, 2, 2, 2, 3, 3, 4])))
    if compact:
        n_others = min(n_others, 2)
    if regime == "target_test":
        primary = str(rng.choice(["tie", "box", "near", "horizontal",
                                  "small", "large", "random"]))
    else:
        primary = str(rng.choice(["tie", "box", "near", "horizontal",
                                  "small", "large", "random", "random"]))
    modes = [primary] + [
        str(rng.choice(["tie", "box", "near", "horizontal",
                        "small", "large", "random"]))
        for _ in range(n_others - 1)
    ]
    masks = [target]
    for mode in modes:
        masks.append(_distractor(rng, target, mode, compact))

    backgrounds = [c for c in range(10) if c != prior_target]
    if first_background is not None:
        backgrounds = [c for c in backgrounds if c != first_background]
    background = int(rng.choice(backgrounds))
    available = [c for c in range(10) if c != background]
    target_choices = ([c for c in available if c != prior_target]
                      if regime == "source_train" and index == 1 else available)
    if prior_distractor is not None and prior_distractor in target_choices and rng.random() < 0.55:
        target_color = prior_distractor
    else:
        target_color = int(rng.choice(target_choices))
    available.remove(target_color)
    if prior_target is not None and prior_target in available:
        distractor_colors = [prior_target]
        available.remove(prior_target)
    else:
        distractor_colors = []
    rng.shuffle(available)
    distractor_colors.extend(available[:n_others - len(distractor_colors)])
    colors = [target_color] + distractor_colors
    canvas, output = _place(rng, masks, colors, background, compact)
    return {"input": canvas, "output": output}


def _analyze(pair: dict) -> dict:
    grid, output = pair["input"], pair["output"]
    values, counts = np.unique(grid, return_counts=True)
    maximum = int(np.argmax(counts))
    assert int(counts[maximum]) > grid.size // 2
    background = int(values[maximum])
    colors = [int(c) for c in values if int(c) != background]
    assert 2 <= len(colors) <= 5
    figures = {}
    for color in colors:
        locations = np.argwhere(grid == color)
        top, left = locations.min(axis=0)
        bottom, right = locations.max(axis=0) + 1
        top, left, bottom, right = map(int, (top, left, bottom, right))
        mask = grid[top:bottom, left:right] == color
        assert _tight(mask) and _connected(mask)
        figures[color] = {
            "box": (top, left, bottom, right),
            "mask": mask,
            "area": int(mask.sum()),
            "symmetric": _left_right(mask),
        }
    for row, col in np.argwhere(grid != background):
        row, col = int(row), int(col)
        block = grid[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
        assert np.all((block == background) | (block == grid[row, col]))
    selected = [c for c in colors if figures[c]["symmetric"]]
    assert len(selected) == 1
    target_color = selected[0]
    top, left, bottom, right = figures[target_color]["box"]
    assert np.array_equal(output, grid[top:bottom, left:right])
    assert set(int(x) for x in np.unique(output)) <= {background, target_color}
    return {
        "background": background,
        "target_color": target_color,
        "figures": figures,
        "target": figures[target_color],
        "distractors": [figures[c] for c in colors if c != target_color],
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        nt, ns = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
        seen_inputs, seen_pairs = set(), set()
        analyses = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert int(grid.min()) >= 0 and int(grid.max()) <= 9
                key = _grid_key(pair["input"])
                pair_key = (key, _grid_key(pair["output"]))
                assert key not in seen_inputs and pair_key not in seen_pairs
                assert not any(np.array_equal(pair["input"], source)
                               for source in _SOURCE_INPUTS)
                seen_inputs.add(key)
                seen_pairs.add(pair_key)
                analysis = _analyze(pair)
                analyses[section].append(analysis)
                assert pair["input"].shape != pair["output"].shape
        train = analyses["train"]
        assert len({a["target_color"] for a in train}) >= 2
        assert len({a["background"] for a in train}) >= 2
        assert any(not np.array_equal(a["target"]["mask"],
                                      a["target"]["mask"][::-1])
                   for a in train)
        assert {a["target"]["mask"].shape[1] % 2 for a in train} == {0, 1}
        selected_colors = {a["target_color"] for a in train}
        distractor_colors = {
            color for section in ("train", "test") for a in analyses[section]
            for color in a["figures"] if color != a["target_color"]
        }
        assert selected_colors & distractor_colors
        # The only learned convention is left-right mask symmetry. Every test
        # has the same unique selection and exact crop operation as training.
        for analysis in analyses["test"]:
            assert len(analysis["distractors"]) >= 1
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f"invalid puzzle: {error}") from error


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _counts(seed_value, rng, num_train, num_test)
    # Puzzle-level rule: reflect each color's tight mask left to right and
    # crop the sole match. Colors, background, dimensions and positions are
    # pair-level properties, visible in the input rather than hidden codes.
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_target = None
    previous_distractor = None
    first_background = None
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(30):
                child_rng = np.random.default_rng(
                    int(rng.integers(0, 1 << 63, dtype=np.uint64)))
                try:
                    pair = _make_pair(child_rng,
                                      "source_train" if section == "train" else "target_test",
                                      index, previous_target, previous_distractor,
                                      first_background if section == "train" and index == 1 else None)
                    analysis = _analyze(pair)
                except (RuntimeError, AssertionError):
                    continue
                key = _grid_key(pair["input"])
                if key in seen or any(np.array_equal(pair["input"], x)
                                      for x in _SOURCE_INPUTS):
                    continue
                previous_target = analysis["target_color"]
                previous_distractor = next(
                    c for c in analysis["figures"] if c != previous_target)
                if section == "train" and index == 0:
                    first_background = analysis["background"]
                puzzle[section].append(pair)
                seen.add(key)
                break
            else:
                raise RuntimeError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, target_masks, layouts, counts, backgrounds = set(), set(), set(), set(), set()
    odd = even = ties = box_ties = small = large = horizontal = 0
    near = edge = midpoint = source_scale = 0
    solid = gapped = wide = tall = sparse = dense = 0
    train_test_overlap = background_variety = role_reuse = 0
    first_test_same_slot = 0
    test_area_ranks = set()
    area_ranks = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        per_section = {
            section: [_analyze(pair) for pair in puzzle[section]]
            for section in ("train", "test")
        }
        train_shapes = {a["target"]["mask"].shape for a in per_section["train"]}
        test_shapes = {a["target"]["mask"].shape for a in per_section["test"]}
        train_test_overlap += bool(train_shapes & test_shapes)
        background_variety += len({a["background"] for section in per_section
                                   for a in per_section[section]}) > 1
        train_colors = {a["target_color"] for a in per_section["train"]}
        other_colors = {c for section in per_section for a in per_section[section]
                        for c in a["figures"] if c != a["target_color"]}
        role_reuse += bool(train_colors & other_colors)
        def coarse_slot(pair, analysis):
            top, left, bottom, right = analysis["target"]["box"]
            height, width = pair["input"].shape
            return ("bottom" if (top + bottom) / (2 * height) > 0.55
                    else "upper_left" if (left + right) / (2 * width) < 0.5
                    else "upper_right")
        first_test_same_slot += (
            coarse_slot(puzzle["train"][0], per_section["train"][0])
            == coarse_slot(puzzle["test"][0], per_section["test"][0])
        )
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _analyze(pair)
                grid = pair["input"]
                t = a["target"]
                shapes.add((grid.shape, pair["output"].shape))
                counts.add(len(a["figures"]))
                backgrounds.add(a["background"])
                target_masks.add((t["mask"].shape, tuple(t["mask"].flat)))
                solid += bool(t["mask"].all())
                gapped += not t["mask"].all()
                layouts.add((grid.shape, t["box"][:2],
                             tuple(sorted(d["box"][:2] for d in a["distractors"]))))
                odd += t["mask"].shape[1] % 2
                even += 1 - t["mask"].shape[1] % 2
                areas = [d["area"] for d in a["distractors"]]
                ties += any(x == t["area"] for x in areas)
                box_ties += any(d["mask"].shape == t["mask"].shape
                                for d in a["distractors"])
                small += any(x < t["area"] - 2 for x in areas)
                large += any(x > t["area"] + 2 for x in areas)
                horizontal += any(np.array_equal(d["mask"], d["mask"][::-1])
                                  for d in a["distractors"])
                near += any(np.count_nonzero(d["mask"] != d["mask"][:, ::-1]) == 2
                            for d in a["distractors"])
                for d in a["distractors"]:
                    dh, dw = d["mask"].shape
                    wide += dw >= 2 * dh
                    tall += dh >= 2 * dw
                    sparse += float(d["mask"].mean()) < 0.5
                    dense += float(d["mask"].mean()) > 0.75
                top, left, bottom, right = t["box"]
                edge += (top == 0 or left == 0 or bottom == grid.shape[0]
                         or right == grid.shape[1])
                midpoint += np.any(grid[grid.shape[0] // 2] != a["background"])
                source_scale += max(grid.shape) <= 14
                rank = ("smallest" if t["area"] < min(areas)
                        else "largest" if t["area"] > max(areas)
                        else "middle" if min(areas) < t["area"] < max(areas)
                        else "tied")
                area_ranks.add(rank)
                if section == "test":
                    test_area_ranks.add(rank)
    assert len(shapes) >= 45 and len(target_masks) >= 80 and len(layouts) >= 100
    assert len(backgrounds) >= 8 and counts >= {2, 3, 4, 5}
    assert odd >= 30 and even >= 30
    assert ties >= 12 and box_ties >= 12 and small >= 12 and large >= 12
    assert horizontal >= 8
    assert area_ranks == {"smallest", "largest", "middle", "tied"}
    assert test_area_ranks == area_ranks
    assert near >= 8 and edge >= 8 and midpoint >= 8 and source_scale >= 8
    assert solid >= 2 and gapped >= 20
    assert wide >= 8 and tall >= 8 and sparse >= 8 and dense >= 8
    assert train_test_overlap >= 1 and background_variety == 42 and role_reuse == 42
    assert first_test_same_slot < 35
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        p = generate_puzzle(917, train, test)
        validate_generated_puzzle(p)
        assert _same_puzzle(p, generate_puzzle(917, train, test))
    assert not _same_puzzle(generate_puzzle(22), generate_puzzle(23))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
