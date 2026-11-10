"""Generate colored-copy vertical alignment puzzles."""

import hashlib
import numpy as np

_SOURCE_HASHES = {
    "062cb358c9f0bf04725852f5d8a53e1a8d1c9cd3b409ec555b327835159758d4",
    "7ee023309219ab16241a06897a73e000175922fc78f86d2b690b0595555fa2cc",
    "6026875526e97888b655441cca6735fdd3bc887720b2dc062d7dc918ae6d3b38",
    "572ba20f92e7b06247a81419570d8e4f5d30b9651bc8e488ac0b47932a099cc3",
}


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _signature(mask):
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _connected(mask):
    cells = np.argwhere(mask)
    if not len(cells):
        return False
    start = tuple(int(v) for v in cells[0])
    seen, stack = {start}, [start]
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + dr, col + dc)
            if (0 <= nxt[0] < height and 0 <= nxt[1] < width
                    and mask[nxt] and nxt not in seen):
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _exact_box(mask):
    return (bool(mask[0].any()) and bool(mask[-1].any())
            and bool(mask[:, 0].any()) and bool(mask[:, -1].any()))


def _irregular(rng, height, width):
    mask = np.ones((height, width), dtype=bool)
    target = max(height + width - 1, int(rng.uniform(0.38, 0.78) * height * width))
    for _ in range(height * width):
        if int(mask.sum()) <= target:
            break
        cells = np.argwhere(mask)
        for index in rng.permutation(len(cells)):
            row, col = (int(x) for x in cells[int(index)])
            mask[row, col] = False
            if _exact_box(mask) and _connected(mask):
                break
            mask[row, col] = True
        else:
            break
    return mask


def _new_mask(rng, height, width):
    family = str(rng.choice(
        ["irregular", "symmetric", "striped", "open", "hollow", "sparse", "solid"],
        p=[0.22, 0.18, 0.15, 0.12, 0.09, 0.15, 0.09]))
    if family == "irregular":
        mask = _irregular(rng, height, width)
    elif family == "solid":
        mask = np.ones((height, width), dtype=bool)
    elif family in ("symmetric", "striped"):
        mask = np.zeros((height, width), dtype=bool)
        widths = list(range(1 if width % 2 else 2, width + 1, 2))
        full_row = int(rng.integers(0, height))
        for row in range(height):
            if row == full_row or (family == "striped" and row % 2 == full_row % 2):
                span = width
            elif family == "striped":
                span = widths[0]
            else:
                span = int(rng.choice(widths))
            left = (width - span) // 2
            mask[row, left:left + span] = True
    elif family == "hollow":
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif family == "open":
        mask = np.zeros((height, width), dtype=bool)
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
        if width > 2 and rng.random() < 0.5:
            mask[0, :] = True
            mask[0, int(rng.integers(1, width - 1))] = False
    else:
        mask = np.zeros((height, width), dtype=bool)
        spine = int(rng.integers(0, width))
        mask[:, spine] = True
        mask[int(rng.integers(0, height)), :spine + 1] = True
        mask[int(rng.integers(0, height)), spine:] = True
        for row in rng.choice(height, size=max(1, height // 3), replace=False):
            endpoint = int(rng.integers(0, width))
            mask[int(row), min(spine, endpoint):max(spine, endpoint) + 1] = True
    assert _exact_box(mask) and _connected(mask)
    return mask


def _spec(rng, regime):
    if regime == "target_test" and rng.random() < 0.10:
        return 4, 5, 5
    count = int(rng.choice(np.arange(2, 8), p=[0.09, 0.21, 0.27, 0.23, 0.14, 0.06]))
    if regime == "target_test" and rng.random() < 0.18:
        count = 4
    width = int(rng.integers(2, min(10, 29 // count) + 1))
    height = int(rng.integers(2, 10))
    return count, height, width


def _columns(rng, count, width):
    budget = 30 - (count * width + 1)
    gaps = []
    for _ in range(count - 1):
        sampled = int(rng.choice([0, 1, 2, 3, 4], p=[0.38, 0.27, 0.20, 0.10, 0.05]))
        gap = min(sampled, budget)
        gaps.append(gap)
        budget -= gap
    extra = int(rng.integers(0, min(8, budget) + 1))
    left_margin = 1 + int(rng.integers(0, extra + 1))
    right_margin = extra - (left_margin - 1)
    lefts, cursor = [], left_margin
    for index in range(count):
        lefts.append(cursor)
        cursor += width
        if index < count - 1:
            cursor += gaps[index]
    canvas_width = cursor + right_margin
    assert canvas_width <= 30
    return canvas_width, lefts


def _tops(rng, count, anchor_index, max_top, regime, index):
    followers = [i for i in range(count) if i != anchor_index]
    if regime == "source_train" and index == 0:
        mode = "above"
    elif regime == "source_train" and index == 1:
        mode = "below"
    else:
        mode = str(rng.choice(["above", "below", "mixed"], p=[0.32, 0.32, 0.36]))
    if mode == "mixed" and count == 2:
        mode = "above" if rng.random() < 0.5 else "below"
    aligned = index >= 2 and count >= 3 and rng.random() < 0.22
    if aligned and count == 3 and mode == "mixed":
        mode = "above" if rng.random() < 0.5 else "below"
    if mode == "above":
        anchor_top = int(rng.integers(1, max_top + 1))
    elif mode == "below":
        anchor_top = int(rng.integers(0, max_top))
    else:
        anchor_top = int(rng.integers(1, max_top))
    tops = [anchor_top] * count
    rng.shuffle(followers)
    if aligned:
        followers.pop()
    def choose(side):
        if side == "above":
            return int(rng.integers(0, anchor_top))
        return int(rng.integers(anchor_top + 1, max_top + 1))
    if mode == "mixed":
        tops[followers.pop()] = choose("above")
        tops[followers.pop()] = choose("below")
    else:
        tops[followers.pop()] = choose(mode)
    for follower in followers:
        side = mode if mode != "mixed" else ("above" if rng.random() < 0.5 else "below")
        tops[follower] = choose(side)
    return tops


def _paint(grid, mask, top, left, color):
    height, width = mask.shape
    grid[top:top + height, left:left + width][mask] = color


def _pair_hash(pair):
    inp = pair["input"].astype(np.int8, copy=False)
    out = pair["output"].astype(np.int8, copy=False)
    return hashlib.sha256(str(inp.shape).encode() + inp.tobytes() + out.tobytes()).hexdigest()


def _make_pair(rng, regime, index, anchor_color, reserved_color, previous_masks):
    count, height, width = _spec(rng, regime)
    if previous_masks and rng.random() < 0.24:
        mask = previous_masks[int(rng.integers(0, len(previous_masks)))].copy()
        height, width = mask.shape
        count = min(count, 29 // width)
    else:
        mask = _new_mask(rng, height, width)
    canvas_width, lefts = _columns(rng, count, width)
    canvas_height = int(rng.integers(height + 4, min(30, height + 12) + 1))
    if regime == "target_test" and rng.random() < 0.20:
        canvas_height = min(30, height + int(rng.integers(9, 13)))
    anchor_index = int(rng.integers(0, count))
    if regime == "target_test" and rng.random() < 0.28:
        anchor_index = count - 1
    tops = _tops(rng, count, anchor_index, canvas_height - height, regime, index)
    backgrounds = [c for c in range(10) if c not in (anchor_color, reserved_color)]
    background = 0 if rng.random() < 0.5 else int(rng.choice([c for c in backgrounds if c != 0]))
    palette = [c for c in range(1, 10) if c not in (anchor_color, background)]
    if regime == "source_train":
        palette.remove(reserved_color)
    if regime == "target_test" and reserved_color in palette and rng.random() < 0.55:
        companions = [reserved_color] + [int(v) for v in rng.choice(
            [c for c in palette if c != reserved_color], size=count - 2, replace=False)]
        rng.shuffle(companions)
    else:
        companions = [int(v) for v in rng.choice(palette, size=count - 1, replace=False)]
    colors, cursor = [], 0
    for object_index in range(count):
        if object_index == anchor_index:
            colors.append(anchor_color)
        else:
            colors.append(companions[cursor])
            cursor += 1
    inp = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    out = np.full_like(inp, background)
    for color, top, left in zip(colors, tops, lefts):
        _paint(inp, mask, top, left, color)
        _paint(out, mask, tops[anchor_index], left, color)
    return {"input": inp, "output": out}, mask


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    anchor_color = int(rng.integers(1, 10))
    reserved_color = int(rng.choice([c for c in range(1, 10) if c != anchor_color]))
    puzzle = {"train": [], "test": []}
    masks, seen = [], set()
    for split, regime, number in (("train", "source_train", num_train),
                                  ("test", "target_test", num_test)):
        for index in range(number):
            for _ in range(20):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair, mask = _make_pair(np.random.default_rng(pair_seed), regime, index,
                                        anchor_color, reserved_color, masks)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and _pair_hash(pair) not in _SOURCE_HASHES:
                    break
            else:
                raise ValueError("could not construct a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
            masks.append(mask)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, background):
    result = {}
    for color in (int(c) for c in np.unique(grid) if c != background):
        rows, cols = np.where(grid == color)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        mask = grid[top:bottom + 1, left:right + 1] == color
        assert _exact_box(mask) and _connected(mask)
        result[color] = (top, left, mask)
    return result


def _analyze_pair(pair, anchor_color=None):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    background = int(inp[0, 0])
    assert int(out[0, 0]) == background
    assert set(np.unique(inp)) == set(np.unique(out))
    before, after = _components(inp, background), _components(out, background)
    assert set(before) == set(after) and len(before) >= 2
    mask = next(iter(before.values()))[2]
    for color, (top, left, current_mask) in before.items():
        assert np.array_equal(current_mask, mask)
        after_top, after_left, after_mask = after[color]
        assert left == after_left and np.array_equal(after_mask, mask)
    ordered = sorted((left, left + mask.shape[1] - 1, color)
                     for color, (_, left, _) in before.items())
    gaps = [b[0] - a[1] - 1 for a, b in zip(ordered, ordered[1:])]
    assert all(gap >= 0 for gap in gaps)
    stationary = {color for color in before if before[color][0] == after[color][0]}
    if anchor_color is not None:
        assert anchor_color in stationary
        anchor_top = before[anchor_color][0]
        assert all(after[color][0] == anchor_top for color in before)
        assert len(stationary) < len(before)
        expected = np.full_like(out, background)
        for color, (_, left, _) in before.items():
            _paint(expected, mask, anchor_top, left, color)
        assert np.array_equal(out, expected)
    else:
        assert stationary
    return {
        "background": background, "colors": frozenset(before),
        "stationary": frozenset(stationary), "count": len(before),
        "height": inp.shape[0], "width": inp.shape[1],
        "mask_height": mask.shape[0], "mask_width": mask.shape[1],
        "mask": _signature(mask), "gaps": tuple(gaps),
        "symmetric": bool(np.array_equal(mask, mask[:, ::-1])),
        "solid": bool(mask.all()),
        "anchor_rank": (None if anchor_color is None else
                        [item[2] for item in ordered].index(anchor_color)),
        "anchor_top": (None if anchor_color is None else before[anchor_color][0]),
        "max_top": inp.shape[0] - mask.shape[0],
        "moves": (None if anchor_color is None else tuple(
            before[c][0] - before[anchor_color][0] for c in before if c != anchor_color)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            inp, out = pair["input"], pair["output"]
            input_key = (inp.shape, inp.astype(np.int8).tobytes())
            pair_key = (input_key, out.shape, out.astype(np.int8).tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _pair_hash(pair) not in _SOURCE_HASHES
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    first = _analyze_pair(puzzle["train"][0])
    second = _analyze_pair(puzzle["train"][1])
    assert len(first["stationary"]) == len(second["stationary"]) == 1
    anchor_color = next(iter(first["stationary"]))
    assert second["stationary"] == frozenset({anchor_color})
    train = [_analyze_pair(pair, anchor_color) for pair in puzzle["train"]]
    tests = [_analyze_pair(pair, anchor_color) for pair in puzzle["test"]]
    assert all(anchor_color in record["colors"] and
               anchor_color != record["background"] for record in train + tests)
    assert any(move < 0 for record in train for move in record["moves"])
    assert any(move > 0 for record in train for move in record["moves"])
    assert all(move < 0 for move in train[0]["moves"])
    assert all(move > 0 for move in train[1]["moves"])


def _same_puzzle(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[split], second[split])
        for k in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    records = []
    bg_varied = mask_reused = unseen_color = 0
    count_varied = dimension_varied = rank_varied = gap_varied = 0
    familiar_test_count = new_test_combination = source_like_tests = 0
    for puzzle in automatic:
        anchor = next(iter(_analyze_pair(puzzle["train"][0])["stationary"]))
        train = [_analyze_pair(p, anchor) for p in puzzle["train"]]
        tests = [_analyze_pair(p, anchor) for p in puzzle["test"]]
        records.extend(train + tests)
        bg_varied += len({r["background"] for r in train + tests}) > 1
        mask_reused += len({r["mask"] for r in train + tests}) < len(train + tests)
        count_varied += len({r["count"] for r in train + tests}) > 1
        dimension_varied += len({(r["mask_height"], r["mask_width"]) for r in train + tests}) > 1
        rank_varied += len({r["anchor_rank"] for r in train + tests}) > 1
        gap_varied += len({r["gaps"] for r in train + tests}) > 1
        familiar_test_count += any(r["count"] in {t["count"] for t in train} for r in tests)
        new_test_combination += any(
            (r["count"], r["mask_height"], r["mask_width"])
            not in {(t["count"], t["mask_height"], t["mask_width"]) for t in train}
            for r in tests)
        source_like_tests += sum(r["count"] == 4 and r["mask_height"] == 5
                                 and r["mask_width"] == 5 for r in tests)
        trained_colors = set().union(*(r["colors"] for r in train))
        unseen_color += any(r["colors"] - trained_colors for r in tests)
    assert len({r["count"] for r in records}) >= 5
    assert len({(r["mask_height"], r["mask_width"]) for r in records}) >= 20
    assert len({r["mask"] for r in records}) >= 80
    assert any(0 in r["gaps"] for r in records)
    assert any(g >= 3 for r in records for g in r["gaps"])
    assert any(r["symmetric"] for r in records) and any(not r["symmetric"] for r in records)
    assert any(r["solid"] for r in records)
    assert any(r["anchor_rank"] == r["count"] - 1 for r in records)
    assert any(r["anchor_top"] == 0 for r in records)
    assert any(r["anchor_top"] == r["max_top"] for r in records)
    assert any(0 in r["moves"] for r in records)
    assert any(abs(move) >= 6 for r in records for move in r["moves"])
    assert bg_varied >= 30 and mask_reused >= 12 and unseen_color >= 8
    assert count_varied >= 30 and dimension_varied >= 35
    assert rank_varied >= 30 and gap_varied >= 30
    assert familiar_test_count >= 25 and new_test_combination >= 25
    assert source_like_tests >= 1
    assert sum(r["count"] == 2 for r in records) >= 10
    assert sum(r["count"] >= 6 for r in records) >= 10
    assert sum(r["anchor_rank"] == r["count"] - 1 for r in records) >= 15
    assert sum(r["anchor_rank"] != r["count"] - 1 for r in records) >= 15
    assert sum(all(move < 0 for move in r["moves"]) for r in records) >= 15
    assert sum(all(move > 0 for move in r["moves"]) for r in records) >= 15
    assert sum(any(move < 0 for move in r["moves"]) and
               any(move > 0 for move in r["moves"]) for r in records) >= 15
    assert sum(0 in r["moves"] for r in records) >= 10
    assert sum(any(0 < abs(move) <= 2 for move in r["moves"]) and
               any(abs(move) >= 6 for move in r["moves"]) for r in records) >= 10
    assert sum(0 in r["gaps"] and r["background"] != 0 and
               r["symmetric"] for r in records) >= 10
    shapes = [np.asarray([[cell == "1" for cell in row] for row in r["mask"]])
              for r in records]
    assert sum(mask.mean() <= 0.45 for mask in shapes) >= 10
    assert sum(mask.shape[0] >= 3 and mask.shape[1] >= 3 and
               mask[0].all() and mask[-1].all() and mask[:, 0].all() and
               mask[:, -1].all() and not mask[1:-1, 1:-1].all()
               for mask in shapes) >= 5
    assert sum(mask.shape[0] >= 3 and mask.shape[1] >= 3 and
               mask[-1].all() and mask[:, 0].all() and mask[:, -1].all() and
               not mask[0].all() for mask in shapes) >= 5
    assert len({next(iter(_analyze_pair(p["train"][0])["stationary"])) for p in automatic}) >= 7
    a, b = generate_puzzle(1234, 7, 3), generate_puzzle(1234, 7, 3)
    assert _same_puzzle(a, b) and not _same_puzzle(a, generate_puzzle(1235, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4), (2, None),
                                           (10, None), (7, 3))):
        puzzle = generate_puzzle(2000 + index, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for index, (train, test) in enumerate(((1, None), (11, None), (None, 0),
                                           (None, 5), (2, 2), (3, 4), (4, 4))):
        try:
            generate_puzzle(3000 + index, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
