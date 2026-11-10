"""Connected objects compete by occupied-cell count; crop the unique winner."""

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be in 2..10")
    if num_test is not None:
        num_test = _integer(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be in 1..4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _tight_connected(mask):
    if mask.ndim != 2 or not mask.any():
        return False
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    seen = {tuple(np.argwhere(mask)[0])}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and mask[p] and p not in seen:
                seen.add(p)
                stack.append(p)
    return len(seen) == int(mask.sum())


def _turn(mask, rng):
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.integers(2):
        mask = np.fliplr(mask)
    return np.array(mask, dtype=bool, copy=True)


def _shape(rng, h, w, count, style):
    """Grow a connected spanning skeleton, or peel a nearly solid rectangle."""
    assert h + w - 1 <= count <= h * w
    if style == "dense":
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > count:
            choices = rng.permutation(np.argwhere(mask))
            for r, c in choices:
                trial = mask.copy()
                trial[int(r), int(c)] = False
                if _tight_connected(trial):
                    mask = trial
                    break
            else:
                style = "bend"
                break
        if style == "dense":
            return _turn(mask, rng)
    mask = np.zeros((h, w), dtype=bool)
    if style == "ring" and h >= 3 and w >= 3 and count >= 2 * (h + w) - 4:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif style == "branch":
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    else:
        mask[0, :] = True
        mask[:, -1] = True
    while int(mask.sum()) < count:
        choices = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if ((r and mask[r - 1, c]) or (r + 1 < h and mask[r + 1, c])
                        or (c and mask[r, c - 1]) or (c + 1 < w and mask[r, c + 1])):
                    choices.append((r, c))
        r, c = choices[int(rng.integers(len(choices)))]
        mask[r, c] = True
    assert _tight_connected(mask)
    return _turn(mask, rng)


def _random_mask(rng, count):
    styles = ("growth", "dense", "line", "ring", "branch", "bend")
    style = styles[int(rng.integers(len(styles)))]
    if style == "line":
        return _turn(np.ones((1, count), dtype=bool), rng)
    dimensions = [(h, w) for h in range(2, 9) for w in range(2, 9)
                  if h + w - 1 <= count <= h * w and h * w <= max(count + 11, 13)]
    if style == "ring":
        ring_dimensions = [(h, w) for h, w in dimensions
                           if h >= 3 and w >= 3 and 2 * (h + w) - 4 <= count]
        if ring_dimensions:
            dimensions = ring_dimensions
        else:
            style = "growth"
    if not dimensions:
        return _turn(np.ones((1, count), dtype=bool), rng)
    h, w = dimensions[int(rng.integers(len(dimensions)))]
    return _shape(rng, h, w, count, style)


def _area_challenge(rng):
    """A smaller-box winner and a larger-box loser with fewer occupied cells."""
    if rng.random() < 0.24:
        if rng.integers(2):
            h, w, a, b, perimeter = 3, 5, 4, 4, 12
        else:
            h, w, a, b, perimeter = 4, 6, 5, 5, 16
        winner_count = int(rng.integers(perimeter + 1, min(h * w, perimeter + 4) + 1))
        winner = _shape(rng, h, w, winner_count, "dense")
        loser = _shape(rng, a, b, perimeter, "ring")
        return winner, loser
    winner_dims = ((2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (2, 5))
    h, w = winner_dims[int(rng.integers(len(winner_dims)))]
    area = h * w
    winner_count = area - int(rng.integers(0, min(3, area - 5)))
    winner = _shape(rng, h, w, winner_count, "dense")
    choices = [(a, b) for a in range(3, 7) for b in range(3, 7)
               if a * b > area and a + b - 1 < winner_count
               and a * b <= area + 17]
    a, b = choices[int(rng.integers(len(choices)))]
    low = a + b - 1
    loser_count = int(rng.integers(low, min(winner_count - 1, low + 4) + 1))
    style = ("bend", "branch", "ring", "growth")[int(rng.integers(4))]
    loser = _shape(rng, a, b, loser_count, style)
    assert loser.size > winner.size and int(loser.sum()) < int(winner.sum())
    return winner, loser


def _separate(box, other):
    r, c, h, w = box
    a, b, x, y = other
    return r + h < a or a + x < r or c + w < b or b + y < c


def _place(rng, objects, background):
    """Sample independent positions; tight, loose and mixed clusters share the rule."""
    order = rng.permutation(len(objects))
    objects = [objects[int(i)] for i in order]
    area = sum(mask.size for mask, _ in objects)
    side = max(9.0, (2.3 * area) ** 0.5)
    ratio = float(rng.uniform(0.7, 1.55))
    initial_h = min(30, max(9, int(side / ratio ** 0.5)))
    initial_w = min(30, max(9, int(side * ratio ** 0.5)))
    mode = int(rng.integers(3))
    for growth in range(0, 23, 2):
        height = min(30, initial_h + growth)
        width = min(30, initial_w + growth)
        for _ in range(5):
            boxes = []
            for mask, _color in objects:
                h, w = mask.shape
                if h > height or w > width:
                    break
                for _attempt in range(240):
                    if boxes and (mode == 0 or mode == 2 and rng.integers(2)):
                        anchor = boxes[int(rng.integers(len(boxes)))]
                        r = int(np.clip(anchor[0] + rng.integers(-h - 2, anchor[2] + 4), 0, height - h))
                        c = int(np.clip(anchor[1] + rng.integers(-w - 2, anchor[3] + 4), 0, width - w))
                    else:
                        r = int(rng.integers(height - h + 1))
                        c = int(rng.integers(width - w + 1))
                    box = (r, c, h, w)
                    if all(_separate(box, old) for old in boxes):
                        boxes.append(box)
                        break
                else:
                    break
            if len(boxes) == len(objects):
                grid = np.full((height, width), background, dtype=np.int8)
                for (mask, color), (r, c, h, w) in zip(objects, boxes):
                    region = grid[r:r + h, c:c + w]
                    region[mask] = color
                coords = np.argwhere(grid != background)
                lo = coords.min(axis=0)
                hi = coords.max(axis=0)
                margins = rng.integers(0, 4, size=4)
                top = min(int(margins[0]), int(lo[0]))
                bottom = min(int(margins[1]), height - int(hi[0]) - 1)
                left = min(int(margins[2]), int(lo[1]))
                right = min(int(margins[3]), width - int(hi[1]) - 1)
                grid = grid[int(lo[0]) - top:int(hi[0]) + bottom + 1,
                            int(lo[1]) - left:int(hi[1]) + right + 1]
                foreground = int(np.sum(grid != background))
                while 2 * foreground >= grid.size:
                    if grid.shape[0] < 30:
                        grid = np.pad(grid, ((0, 1), (0, 0)), constant_values=background)
                    elif grid.shape[1] < 30:
                        grid = np.pad(grid, ((0, 0), (0, 1)), constant_values=background)
                    else:
                        raise AssertionError("cannot make background modal")
                return grid
    raise AssertionError("bounded placement exhausted")


def _components(grid, background):
    active = grid != background
    unseen = set(map(tuple, np.argwhere(active)))
    result = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        cells = [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
                    cells.append(p)
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        colors = {int(grid[p]) for p in cells}
        result.append({"cells": cells, "count": len(cells), "colors": colors,
                       "color": int(grid[start]), "bbox": (min(rows), max(rows), min(cols), max(cols))})
    return result


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert 2 * int(counts[index]) > grid.size
    return int(colors[index])


def _winner(components):
    counts = [item["count"] for item in components]
    assert len(counts) >= 2 and counts.count(max(counts)) == 1
    return components[int(np.argmax(counts))]


def _solve(grid):
    background = _background(grid)
    winner = _winner(_components(grid, background))
    r0, r1, c0, c1 = winner["bbox"]
    output = np.full((r1 - r0 + 1, c1 - c0 + 1), background, dtype=np.int8)
    for r, c in winner["cells"]:
        output[r - r0, c - c0] = winner["color"]
    return output


def _pair(rng, object_count, winner_color, background, area_challenge, same_color):
    if area_challenge:
        winner_mask, challenge_mask = _area_challenge(rng)
    else:
        winner_count = 4 if rng.random() < 0.12 else int(rng.integers(5, 20))
        winner_mask = _random_mask(rng, winner_count)
        challenge_mask = None
    winner_count = int(winner_mask.sum())
    objects = [(winner_mask, winner_color)]
    if challenge_mask is not None:
        objects.append((challenge_mask, None))
    while len(objects) < object_count:
        gap = int(rng.integers(1, min(9, winner_count - 1) + 1))
        count = max(2, winner_count - gap)
        objects.append((_random_mask(rng, count), None))
    same_index = int(rng.integers(1, object_count)) if same_color else -1
    palette = [c for c in range(10) if c not in (background, winner_color)]
    for index in range(1, object_count):
        if index == same_index or rng.random() < 0.08:
            color = winner_color
        else:
            color = palette[int(rng.integers(len(palette)))]
        objects[index] = (objects[index][0], color)
    grid = _place(rng, objects, background)
    return {"input": grid, "output": _solve(grid)}


def _signature(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_signatures():
    raw = [
        (("00000000000", "08880000770", "00800020070", "08800220070", "08800020070", "00000000000"),
         ("888", "080", "880", "880")),
        (("0000000660", "0300440060", "3330440000", "0300440000", "0000000000"),
         ("44", "44", "44")),
        (("000000000", "000700222", "000770020", "000070222", "888000000", "080000000", "000000000"),
         ("222", "020", "222")),
        (("0000000000000", "0220033000000", "0020003001000", "0022000011000", "0222000011100",
          "0000000000000", "0000000000000"),
         ("220", "020", "022", "222")),
        (("000000000", "400000000", "440333000", "040333000", "000303000", "000303000",
          "000000066", "055500666", "055000660"),
         ("333", "333", "303", "303")),
    ]
    return {(_signature(np.array([[int(x) for x in row] for row in inp], dtype=np.int8)),
             _signature(np.array([[int(x) for x in row] for row in out], dtype=np.int8)))
            for inp, out in raw}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The only puzzle-wide convention is the invariant cell-count rule. Palette,
    # canvas, geometry and candidate count are observable pair-level choices.
    area_index = int(rng.integers(train_count))
    same_index = int(rng.integers(train_count))
    train_min_objects = (2, 2, 3, 3, 4)[int(rng.integers(5))]
    train_object_counts = list(map(int, rng.integers(train_min_objects, 6, size=train_count)))
    if len(set(train_object_counts)) == 1:
        alternatives = [n for n in range(train_min_objects, 6) if n != train_object_counts[-1]]
        train_object_counts[-1] = alternatives[int(rng.integers(len(alternatives)))]
    backgrounds = list(map(int, rng.permutation(10)[:2]))
    puzzle = {"train": [], "test": []}
    used = set()
    train_winners = set()
    train_losers = set()
    seen_colors = set()
    for split, length in (("train", train_count), ("test", test_count)):
        for index in range(length):
            for _attempt in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if split == "train":
                    background = backgrounds[index] if index < 2 else int(pair_rng.integers(10))
                    available = [c for c in range(10) if c != background
                                 and (index != 1 or c not in train_winners)]
                    winner_color = available[int(pair_rng.integers(len(available)))]
                    object_count = train_object_counts[index]
                    area = index == area_index or (index != area_index and pair_rng.random() < 0.19)
                    same = index == same_index or pair_rng.random() < 0.16
                else:
                    categories = [list(train_winners),
                                  list(train_losers - train_winners),
                                  [c for c in range(10) if c not in seen_colors]]
                    available_categories = [pool for pool in categories if pool]
                    pool = available_categories[int(pair_rng.integers(len(available_categories)))]
                    winner_color = pool[int(pair_rng.integers(len(pool)))]
                    background = [c for c in range(10) if c != winner_color]
                    background = background[int(pair_rng.integers(len(background)))]
                    object_count = int(pair_rng.integers(2, 7))
                    area = pair_rng.random() < 0.38
                    same = pair_rng.random() < 0.44
                if winner_color == background:
                    continue
                pair = _pair(pair_rng, object_count, winner_color, background, area, same)
                sig = _signature(pair["input"])
                if sig in used:
                    continue
                used.add(sig)
                puzzle[split].append(pair)
                if split == "train":
                    components = _components(pair["input"], background)
                    winner = _winner(components)
                    train_winners.add(winner["color"])
                    train_losers.update(x["color"] for x in components if x is not winner)
                    seen_colors.update(int(x) for x in np.unique(pair["input"]))
                break
            else:
                raise AssertionError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    source = _source_signatures()
    inputs, pairs = set(), set()
    train_winners, train_backgrounds = set(), set()
    area_train = same_train = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            isig, osig = _signature(inp), _signature(out)
            assert isig not in inputs and (isig, osig) not in pairs
            assert (isig, osig) not in source
            inputs.add(isig)
            pairs.add((isig, osig))
            assert isig != osig
            bg = _background(inp)
            components = _components(inp, bg)
            assert 2 <= len(components) <= (5 if split == "train" else 6)
            assert all(len(x["colors"]) == 1 and x["count"] >= 2 for x in components)
            winner = _winner(components)
            r0, r1, c0, c1 = winner["bbox"]
            expected = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=out.dtype)
            for r, c in winner["cells"]:
                expected[r - r0, c - c0] = winner["color"]
            assert np.array_equal(out, expected)
            assert all(not (r0 <= r <= r1 and c0 <= c <= c1)
                       for item in components if item is not winner for r, c in item["cells"])
            if split == "train":
                train_winners.add(winner["color"])
                train_backgrounds.add(bg)
                winner_area = out.size
                area_train |= any((x["bbox"][1] - x["bbox"][0] + 1)
                                  * (x["bbox"][3] - x["bbox"][2] + 1) > winner_area
                                  for x in components if x is not winner)
                same_train |= any(x["color"] == winner["color"] for x in components if x is not winner)
    assert len(train_winners) >= 2
    assert len(train_backgrounds) >= 2
    assert area_train and same_train


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, puzzle_ids = set(), set(), set()
    input_shapes, output_shapes, output_masks = set(), set(), set()
    backgrounds, object_counts, advantages = set(), set(), set()
    area_indices, test_relations, palette_relations = set(), set(), set()
    line_winners = small_winners = hollow_winners = area_tests = same_tests = 0
    arrangements = set()
    all_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_components = []
        test_components = []
        train_colors = set()
        winner_colors = set()
        loser_colors = set()
        train_object_counts = []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                comps = _components(inp, bg)
                win = _winner(comps)
                expected = _solve(inp)
                assert np.array_equal(out, expected)
                assert win["count"] > max(x["count"] for x in comps if x is not win)
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                output_masks.add((out.shape, (out == win["color"]).tobytes()))
                backgrounds.add(bg)
                object_counts.add(len(comps))
                advantages.add(win["count"] - max(x["count"] for x in comps if x is not win))
                all_inputs.add(_signature(inp))
                boxes = [x["bbox"] for x in comps]
                rows_overlap = any(a[0] <= b[1] and b[0] <= a[1] for i, a in enumerate(boxes) for b in boxes[i + 1:])
                cols_overlap = any(a[2] <= b[3] and b[2] <= a[3] for i, a in enumerate(boxes) for b in boxes[i + 1:])
                arrangements.add((rows_overlap, cols_overlap))
                if split == "train":
                    train_components.append(comps)
                    train_object_counts.append(len(comps))
                    train_colors.update(int(x) for x in np.unique(inp))
                    winner_colors.add(win["color"])
                    loser_colors.update(x["color"] for x in comps if x is not win)
                    if any((x["bbox"][1] - x["bbox"][0] + 1) * (x["bbox"][3] - x["bbox"][2] + 1) > out.size
                           for x in comps if x is not win):
                        area_indices.add(index)
                else:
                    test_components.append(comps)
                    if out.shape[0] == 1 or out.shape[1] == 1:
                        line_winners += 1
                    if out.shape == (2, 2):
                        small_winners += 1
                    if np.any(out == bg):
                        hollow_winners += 1
                    if any((x["bbox"][1] - x["bbox"][0] + 1) * (x["bbox"][3] - x["bbox"][2] + 1) > out.size
                           for x in comps if x is not win):
                        area_tests += 1
                    if any(x["color"] == win["color"] for x in comps if x is not win):
                        same_tests += 1
        for comps in test_components:
            win = _winner(comps)
            count = len(comps)
            test_relations.add("equal" if count in train_object_counts else
                               "fewer" if count < min(train_object_counts) else
                               "more" if count > max(train_object_counts) else "intermediate")
            palette_relations.add("winner" if win["color"] in winner_colors else
                                  "loser" if win["color"] in loser_colors else
                                  "new" if win["color"] not in train_colors else "background")
        puzzle_ids.add(tuple((_signature(p["input"]), _signature(p["output"]))
                             for s in ("train", "test") for p in puzzle[s]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_ids) == 42 and len(all_inputs) > 280
    assert len(input_shapes) >= 35 and len(output_shapes) >= 25 and len(output_masks) >= 100
    assert len(backgrounds) >= 8 and len(object_counts) >= 5 and len(advantages) >= 5
    assert len(area_indices) >= 3
    assert {"equal", "fewer", "more"} <= test_relations
    assert {"winner", "loser", "new"} <= palette_relations
    assert line_winners > 0 and small_winners > 0 and hollow_winners > 0
    assert area_tests > 0 and same_tests > 0 and len(arrangements) >= 3
    assert _equal(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert not _equal(generate_puzzle(900, 5, 2), generate_puzzle(901, 5, 2))
    for n, m in ((2, 1), (10, 1), (10, 4), (6, 2), (None, 1),
                 (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1200 + (n or 0) * 10 + (m or 0), n, m)
        assert n is None or len(p["train"]) == n
        assert m is None or len(p["test"]) == m
        validate_generated_puzzle(p)
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(123, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
