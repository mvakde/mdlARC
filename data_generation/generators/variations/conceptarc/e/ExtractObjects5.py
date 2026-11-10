"""Compose complete square motifs from a noisy canvas, ordered by left edge."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = {
    "05b6cccc60c2678fc2820f4960e88409d12d0353963952b79dbeabbbc5704bee",
    "6290b1777d98dfda8f165fe6a3ac4c4dc7c7e40f0341366df9fd9fa4181896de",
    "5595ae2b056252ec6ea4515ac6ae63d762fbbf550e33dd482a54444d56e71709",
    "a18d4ce341976734afdb61c945f3aa257173e22a25a8c612c5f813126f7addcf",
    "9e1e6df7c93f66354778a8ff0ef396e12e82e4f8c68651be2d7de01bc2cb6cbe",
}


def _check_count(value, name, low, high):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _counts(rng, num_train, num_test):
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    choices = [
        (a, b)
        for a in range(2, 11)
        for b in range(1, 5)
        if a > b
        and (num_train is None or a == num_train)
        and (num_test is None or b == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _source_digest(grid):
    data = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(bytes(data.shape) + data.tobytes()).hexdigest()


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if 0 in colors:
        return 0
    highest = np.max(counts)
    if np.count_nonzero(counts == highest) != 1:
        return None
    return int(colors[int(np.argmax(counts))])


def _all_complete_squares(grid, background):
    """List complete squares, discarding sub-squares of larger solid squares."""
    height, width = grid.shape
    candidates = []
    for side in range(3, min(10, height, width) + 1):
        windows = np.lib.stride_tricks.sliding_window_view(grid, (side, side))
        color = windows[..., 0, 0]
        foreground = color != background
        equal = windows == color[..., None, None]
        solids = foreground & np.all(equal, axis=(-2, -1))
        outline = np.ones((side, side), dtype=bool)
        outline[1:-1, 1:-1] = False
        outlines = foreground & np.all(equal | ~outline, axis=(-2, -1))
        outlines &= np.all((windows == background) | outline, axis=(-2, -1))
        for template, mask in (("solid", solids), ("outline", outlines)):
            for y, x in np.argwhere(mask):
                candidates.append((int(x), int(y), side, template, int(color[y, x])))

    maximal = []
    for x, y, side, template, color in candidates:
        if template == "solid" and any(
            other_template == "solid"
            and other_color == color
            and other_side > side
            and other_x <= x
            and other_y <= y
            and x + side <= other_x + other_side
            and y + side <= other_y + other_side
            for other_x, other_y, other_side, other_template, other_color in candidates
        ):
            continue
        maximal.append((x, y, side, template, color))
    return sorted(maximal)


def _extract(grid):
    """Recognize motifs using only the visible grid and the background convention."""
    background = _background(grid)
    if background is None:
        return None
    motifs = _all_complete_squares(grid, background)
    if not 2 <= len(motifs) <= 6:
        return None
    sides = {item[2] for item in motifs}
    templates = {item[3] for item in motifs}
    if len(sides) != 1 or len(templates) != 1:
        return None
    side = motifs[0][2]
    if side * len(motifs) > 30 or len({item[0] for item in motifs}) != len(motifs):
        return None
    covered = np.zeros(grid.shape, dtype=bool)
    for x, y, _, _, _ in motifs:
        if np.any(covered[y:y + side, x:x + side]):
            return None
        covered[y:y + side, x:x + side] = True
    return background, motifs, covered


def _geometry(rng, side, count, mode):
    """Return distinct ordered footprints; horizontally overlapping ones are vertically separate."""
    if mode == "row":
        steps = [side + int(rng.integers(1, 4)) for _ in range(count - 1)]
        xs = [0]
        for step in steps:
            xs.append(xs[-1] + step)
        if xs[-1] + side > 30:
            return None
        if rng.random() < 0.50:
            height = side
            y = 0
        else:
            height = int(rng.integers(side + 1, min(30, side + 16) + 1))
            y = int(rng.integers(height - side + 1))
        ys = [y] * count
    elif mode in ("stagger", "stair"):
        xs = [0]
        for _ in range(count - 1):
            xs.append(xs[-1] + int(rng.integers((side + 1) // 2, side)))
        if xs[-1] + side > 30:
            return None
        if mode == "stagger":
            gap = int(rng.integers(0, 5))
            low = int(rng.integers(0, 4))
            high = low + side + gap
            ys = [low if index % 2 == 0 else high for index in range(count)]
            if rng.random() < 0.5:
                ys = [high if index % 2 == 0 else low for index in range(count)]
        else:
            ys = [0]
            for _ in range(count - 1):
                ys.append(ys[-1] + side + int(rng.integers(0, 2)))
            if rng.random() < 0.5:
                ys.reverse()
        height = max(ys) + side + int(rng.integers(0, 4))
        if height > 30:
            return None
    else:
        width = int(rng.integers(min(30, side + count), 31))
        height = int(rng.integers(max(side + 1, 8), 31))
        if width < count:
            return None
        xs = sorted(int(x) for x in rng.choice(width - side + 1, count, replace=False))
        ys = []
        for x in xs:
            legal = [
                y for y in range(height - side + 1)
                if all(
                    x + side <= old_x or old_x + side <= x
                    or y + side <= old_y or old_y + side <= y
                    for old_x, old_y in zip(xs[:len(ys)], ys)
                )
            ]
            if not legal:
                return None
            ys.append(int(rng.choice(legal)))
        return height, width, list(zip(xs, ys))

    base_width = xs[-1] + side
    left = int(rng.integers(0, min(4, 31 - base_width)))
    xs = [x + left for x in xs]
    width = base_width + left + int(rng.integers(0, min(4, 31 - base_width - left)))
    return height, width, list(zip(xs, ys))


def _colors(rng, count, background, repetition):
    allowed = [color for color in range(1, 10) if color != background]
    if repetition == "same":
        color = int(rng.choice(allowed))
        return [color] * count
    if repetition == "distinct":
        return [int(x) for x in rng.choice(allowed, count, replace=False)]
    distinct_count = int(rng.integers(2, min(count, len(allowed)) + 1))
    if distinct_count == count:
        distinct_count -= 1
    palette = [int(x) for x in rng.choice(allowed, distinct_count, replace=False)]
    colors = palette + [int(rng.choice(palette)) for _ in range(count - distinct_count)]
    rng.shuffle(colors)
    return colors


def _add_noise(rng, grid, footprints, colors, intensity):
    height, width = grid.shape
    available = np.argwhere(~footprints)
    if not len(available):
        return
    if intensity == "sparse":
        target = int(rng.integers(1, min(8, len(available)) + 1))
    elif intensity == "moderate":
        target = int(rng.integers(8, min(36, len(available)) + 1)) if len(available) >= 8 else len(available)
    else:
        target = min(80, max(1, int(len(available) * rng.uniform(0.35, 0.72))))
    positions = set()
    if rng.random() < 0.72:
        touching = []
        for y, x in available:
            if any(
                0 <= y + dy < height and 0 <= x + dx < width
                and footprints[y + dy, x + dx]
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                touching.append((int(y), int(x)))
        if touching:
            positions.add(touching[int(rng.integers(len(touching)))])
    if rng.random() < 0.60:
        cluster_goal = min(target, int(target * rng.uniform(0.25, 0.70)) + 1)
        for _ in range(cluster_goal * 8):
            if len(positions) >= cluster_goal:
                break
            if positions and rng.random() < 0.8:
                y, x = tuple(positions)[int(rng.integers(len(positions)))]
                dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                y, x = y + dy, x + dx
            else:
                y, x = available[int(rng.integers(len(available)))]
                y, x = int(y), int(x)
            if 0 <= y < height and 0 <= x < width and not footprints[y, x]:
                positions.add((y, x))
    remaining = [(int(y), int(x)) for y, x in available if (int(y), int(x)) not in positions]
    if len(positions) < target:
        chosen = rng.choice(len(remaining), target - len(positions), replace=False)
        positions.update(remaining[int(index)] for index in chosen)
    for y, x in positions:
        grid[y, x] = int(rng.choice(colors))


def _make_pair(rng, regime, template, anchor, recolor):
    for attempt in range(160):
        side = int(rng.integers(3, 6 if regime == "train" else 7))
        maximum = min(6, 30 // side)
        count = int(rng.integers(3 if anchor else 2, maximum + 1))
        mode = (
            ("stagger", "stair")[int(rng.integers(2))]
            if anchor else
            ("row", "stagger", "stair", "scatter")[int(rng.integers(4))]
        )
        geometry = _geometry(rng, side, count, mode)
        if geometry is None:
            continue
        height, width, placements = geometry
        if anchor and not any(
            placements[index][1] > placements[index + 1][1]
            for index in range(count - 1)
        ):
            continue
        background = int(rng.integers(1, 10)) if recolor else 0
        repetition = (
            "distinct" if anchor else
            ("same", "mixed", "distinct")[int(rng.integers(3))]
        )
        if regime == "train" and not anchor and attempt % 3 == 0:
            repetition = "mixed" if count >= 3 else "same"
        colors = _colors(rng, count, background, repetition)
        if anchor and colors == sorted(colors):
            colors = colors[1:] + colors[:1]
        grid = np.full((height, width), background, dtype=np.int8)
        footprints = np.zeros((height, width), dtype=bool)
        for (x, y), color in zip(placements, colors):
            footprints[y:y + side, x:x + side] = True
            if template == "solid":
                grid[y:y + side, x:x + side] = color
            else:
                grid[y, x:x + side] = color
                grid[y + side - 1, x:x + side] = color
                grid[y:y + side, x] = color
                grid[y:y + side, x + side - 1] = color
        if recolor:
            intensity = ("sparse", "moderate")[int(rng.integers(2))]
        else:
            intensity = ("sparse", "moderate", "dense")[int(rng.integers(3))]
        if attempt >= 100:
            intensity = "sparse"
        _add_noise(rng, grid, footprints, colors, intensity)
        if recolor and _background(grid) != background:
            continue
        extracted = _extract(grid)
        if extracted is None:
            continue
        found = extracted[1]
        expected = sorted(
            (x, y, side, template, color)
            for (x, y), color in zip(placements, colors)
        )
        if found != expected:
            continue
        output = np.concatenate(
            [grid[y:y + side, x:x + side] for x, y, _, _, _ in found],
            axis=1,
        )
        if _source_digest(grid) not in _SOURCE_INPUT_DIGESTS:
            return {"input": grid, "output": output}
    raise RuntimeError("could not place unambiguous complete squares")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    recolor_enabled = bool(rng.random() < 0.55)
    anchor_template = ("outline", "solid")[int(rng.integers(2))]
    other_template = "solid" if anchor_template == "outline" else "outline"
    train_plan = [(anchor_template, True), (other_template, False)]
    train_plan.extend(
        (("outline", "solid")[int(rng.integers(2))], False)
        for _ in range(train_count - 2)
    )
    rng.shuffle(train_plan)
    puzzle = {"train": [], "test": []}
    seen = set()
    recolored_train = False
    for template, anchor in train_plan:
        recolor = recolor_enabled and template == "outline" and not recolored_train
        for _ in range(15):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(pair_rng, "train", template, anchor, recolor)
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in seen:
                seen.add(signature)
                puzzle["train"].append(pair)
                recolored_train |= recolor
                break
        else:
            raise RuntimeError("could not make distinct training pair")
    for _ in range(test_count):
        template = ("outline", "solid")[int(rng.integers(2))]
        recolor = recolor_enabled and rng.random() < 0.28
        for _ in range(15):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(pair_rng, "test", template, False, recolor)
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in seen:
                seen.add(signature)
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("could not make distinct test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    extracted = _extract(grid)
    assert extracted is not None
    background, motifs, covered = extracted
    side = motifs[0][2]
    assert output.shape == (side, side * len(motifs))
    expected = np.concatenate(
        [grid[y:y + side, x:x + side] for x, y, _, _, _ in motifs], axis=1
    )
    assert np.array_equal(output, expected)
    assert np.count_nonzero(grid[~covered] != background) >= 1
    for index, (x, y, _, template, color) in enumerate(motifs):
        crop = output[:, index * side:(index + 1) * side]
        if template == "solid":
            assert np.all(crop == color)
        else:
            assert np.all(crop[0] == color) and np.all(crop[-1] == color)
            assert np.all(crop[:, 0] == color) and np.all(crop[:, -1] == color)
            assert np.all(crop[1:-1, 1:-1] == background)
    return background, motifs, covered


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    details = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert not np.array_equal(input_grid, output_grid)
            assert _source_digest(input_grid) not in _SOURCE_INPUT_DIGESTS
            signature = (input_grid.shape, input_grid.tobytes())
            pair_signature = (signature, output_grid.shape, output_grid.tobytes())
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            details[regime].append(_inspect_pair(pair))
    train_templates = {items[0][3] for _, items, _ in details["train"]}
    assert train_templates == {"solid", "outline"}
    assert all(items[0][3] in train_templates for _, items, _ in details["test"])
    assert any(
        len(items) >= 3
        and any(items[i][1] > items[i + 1][1] for i in range(len(items) - 1))
        and [item[4] for item in items] != sorted(item[4] for item in items)
        for _, items, _ in details["train"]
    )
    if any(background != 0 for background, _, _ in details["test"]):
        assert any(
            background != 0 and items[0][3] == "outline"
            for background, items, _ in details["train"]
        )


def self_test() -> None:
    counts, shapes, structures, palettes = set(), set(), set(), set()
    side_values, motif_counts, templates, repetitions = set(), set(), set(), set()
    modes, size_forms, count_repetitions, combinations = set(), set(), set(), set()
    recolored = dense = source_sizes = test_new_colors = 0
    same_row = chains = touching = clusters = high_noise = 0
    increasing = decreasing = irregular = 0
    train_same_outline = test_same_outline = 0
    new_test_sizes = familiar_test_sizes = full_height_solid6 = 0
    sparse_noise = medium_noise = heavy_noise = changed_pair_palettes = 0
    train_sides_seen, test_sides_seen = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train_colors = {
            item[4] for pair in puzzle["train"]
            for item in _inspect_pair(pair)[1]
        }
        train_sides = {
            _inspect_pair(pair)[1][0][2]
            for pair in puzzle["train"]
        }
        per_puzzle_sides, per_puzzle_counts = set(), set()
        pair_palettes = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid = pair["input"]
                background, items, covered = _inspect_pair(pair)
                side, template = items[0][2:4]
                colors = [item[4] for item in items]
                noise = np.argwhere((grid != background) & ~covered)
                side_values.add(side)
                motif_counts.add(len(items))
                templates.add(template)
                repetition = "same" if len(set(colors)) == 1 else "distinct" if len(set(colors)) == len(colors) else "mixed"
                repetitions.add(repetition)
                size_forms.add((side, template))
                count_repetitions.add((len(items), repetition))
                combinations.add((side, len(items), template, repetition, background != 0))
                shapes.add(grid.shape)
                palettes.add(tuple(sorted(int(x) for x in np.unique(grid))))
                structures.add((grid.shape, side, template, tuple((x, y) for x, y, _, _, _ in items)))
                per_puzzle_sides.add(side)
                per_puzzle_counts.add(len(items))
                pair_palettes.add(tuple(sorted(set(colors))))
                if len(noise) <= 8:
                    sparse_noise += 1
                elif len(noise) <= 35:
                    medium_noise += 1
                else:
                    heavy_noise += 1
                source_sizes += grid.shape in {(14,17),(12,21),(17,11),(6,22),(15,20)}
                recolored += background != 0
                dense += np.count_nonzero(grid != background) > grid.size // 2
                high_noise += len(noise) >= 64
                same_row += len({item[1] for item in items}) == 1
                chains += sum(items[i + 1][0] - items[i][0] < side for i in range(len(items)-1)) >= 2
                assert set(int(grid[y, x]) for y, x in noise) <= set(colors)
                heights = [item[1] for item in items]
                increasing += all(heights[i] < heights[i + 1] for i in range(len(heights) - 1))
                decreasing += all(heights[i] > heights[i + 1] for i in range(len(heights) - 1))
                irregular += not (
                    all(heights[i] < heights[i + 1] for i in range(len(heights) - 1))
                    or all(heights[i] > heights[i + 1] for i in range(len(heights) - 1))
                )
                noise_mask = (grid != background) & ~covered
                touching += any(
                    noise_mask[yy, xx]
                    for x, y, _, _, _ in items
                    for yy, xx in (
                        [(row, col) for row in range(y, y + side)
                         for col in (x - 1, x + side)]
                        + [(row, col) for col in range(x, x + side)
                           for row in (y - 1, y + side)]
                    )
                    if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]
                )
                clusters += any(
                    0 <= y + dy < grid.shape[0] and 0 <= x + dx < grid.shape[1]
                    and noise_mask[y + dy, x + dx]
                    for y, x in noise
                    for dy, dx in ((1, 0), (0, 1))
                )
                if template == "outline" and repetition == "same":
                    if regime == "train":
                        train_same_outline += 1
                    else:
                        test_same_outline += 1
                if regime == "test":
                    test_sides_seen.add(side)
                    test_new_colors += bool(set(colors) - train_colors)
                    new_test_sizes += side not in train_sides
                    familiar_test_sizes += side in train_sides
                    full_height_solid6 += (
                        template == "solid" and side == 6
                        and grid.shape[0] == side
                    )
                else:
                    train_sides_seen.add(side)
        modes.add((len(per_puzzle_sides) > 1, len(per_puzzle_counts) > 1))
        changed_pair_palettes += len(pair_palettes) > 1
    assert len({a for a, _ in counts}) >= 4
    assert len({b for _, b in counts}) >= 3
    assert len(shapes) >= 45 and len(structures) >= 150
    assert len(palettes) >= 25
    assert side_values == {3, 4, 5, 6}
    assert {2, 3, 4, 5} <= motif_counts
    assert templates == {"outline", "solid"}
    assert repetitions == {"same", "mixed", "distinct"}
    assert (True, True) in modes
    assert min(recolored, dense, same_row, chains, touching, clusters, test_new_colors) > 0
    assert min(high_noise, increasing, decreasing, irregular) > 0
    assert min(train_same_outline, test_same_outline) > 0
    assert min(new_test_sizes, familiar_test_sizes, full_height_solid6) > 0
    assert len(size_forms) == 8 and len(count_repetitions) >= 10
    assert len(combinations) >= 50
    assert min(sparse_noise, medium_noise, heavy_noise) > 0
    assert changed_pair_palettes == 42
    assert train_sides_seen == {3, 4, 5}
    assert test_sides_seen == {3, 4, 5, 6}
    assert source_sizes > 0

    def identical(a, b):
        return all(
            len(a[regime]) == len(b[regime])
            and all(np.array_equal(pa[key], pb[key])
                    for pa, pb in zip(a[regime], b[regime])
                    for key in ("input", "output"))
            for regime in ("train", "test")
        )

    fixed = generate_puzzle(824, 5, 2)
    assert identical(fixed, generate_puzzle(824, 5, 2))
    assert not identical(fixed, generate_puzzle(825, 5, 2))
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        result = generate_puzzle(900 + (a or 0) + (b or 0), a, b)
        validate_generated_puzzle(result)
        assert a is None or len(result["train"]) == a
        assert b is None or len(result["test"]) == b
    for a, b in ((1,None),(11,None),(None,0),(None,5),(2,2),(2,4),(4,4)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
