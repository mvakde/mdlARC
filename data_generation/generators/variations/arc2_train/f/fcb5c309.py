"""Largest-frame crop and recolor ARC puzzle generator."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    "13e544a1072b6e5bd85bfa651cb9763518d2bec531d306cc45467c7bd8533750",
    "2449ae53e0e03a834ba1f717b17fb389e12d6544539b153bddb137859255b3f8",
    "25e7a5a62b47a29ef329b3ca2a908668811cc2db2bec2de548816e62bf5dfdb5",
    "e58be202c2c8b5a60517107a7133bc7c37d3baddcd1ffb643f7bc57a2d01ddad",
}
_SOURCE_SHAPES = ((13, 16), (13, 13), (15, 16), (17, 17))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return tuple(grid.shape), bytes(np.asarray(grid, dtype=np.uint8).ravel())


def _source_hash(grid):
    h, w = grid.shape
    return hashlib.sha256(bytes((h, w)) + np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for row in range(h):
        for col in range(w):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells = set()
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(cells)
    return result


def _frames(mask):
    frames = []
    for cells in _components(mask):
        rows, cols = [r for r, _ in cells], [c for _, c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "short structural component"
        border = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)}
        assert cells == border, "structural component is not a complete thin frame"
        frames.append((r0, c0, r1, c1))
    return sorted(frames)


def _size(frame):
    r0, c0, r1, c1 = frame
    return r1 - r0 + 1, c1 - c0 + 1


def _largest(frames):
    assert len(frames) >= 2
    winners = [frame for frame in frames if all(
        frame == other or (_size(frame)[0] > _size(other)[0]
                           and _size(frame)[1] > _size(other)[1]) for other in frames)]
    assert len(winners) == 1, "largest frame must dominate both dimensions"
    return winners[0]


def _infer(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    dominant = int(np.argmax(counts))
    assert int(counts[dominant]) > int(np.max(np.delete(counts, dominant)))
    background = int(values[dominant])
    remaining = [int(v) for v in values if v != background]
    candidates = []
    for structural in remaining:
        try:
            frames = _frames(grid == structural)
            target = _largest(frames)
        except AssertionError:
            continue
        candidates.append((structural, frames, target))
    assert len(candidates) == 1, "frame color must be identifiable from the input"
    structural, frames, target = candidates[0]
    accent = next(v for v in remaining if v != structural)
    return (background, structural, accent), frames, target


def _predict(grid):
    roles, _, target = _infer(grid)
    r0, c0, r1, c1 = target
    output = grid[r0:r1 + 1, c0:c1 + 1].copy()
    output[output == roles[1]] = roles[2]
    return output


def _separated(a, b):
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _place(rng, canvas, dimensions):
    height, width = canvas
    for _ in range(45):
        placed = []
        for index, (h, w) in enumerate(dimensions):
            positions = [(r, c, r + h - 1, c + w - 1)
                         for r in range(height - h + 1) for c in range(width - w + 1)
                         if all(_separated((r, c, r + h - 1, c + w - 1), old)
                                for old in placed)]
            if not positions:
                break
            if index == 0 and rng.random() < 0.25:
                edge = [p for p in positions if p[0] == 0 or p[1] == 0
                        or p[2] == height - 1 or p[3] == width - 1]
                if edge:
                    positions = edge
            placed.append(positions[int(rng.integers(len(positions)))])
        if len(placed) == len(dimensions):
            return placed
    return None


def _canvas_and_frames(rng, dimensions, split):
    largest_h, largest_w = dimensions[0]
    for attempt in range(18):
        if attempt == 0 and split == "train" and rng.random() < 0.32:
            shape = _SOURCE_SHAPES[int(rng.integers(len(_SOURCE_SHAPES)))]
        elif attempt < 12:
            floor_h, floor_w = max(13, largest_h + 4), max(13, largest_w + 4)
            if len(dimensions) >= 4:
                floor_h, floor_w = max(floor_h, 18), max(floor_w, 18)
            shape = (int(rng.integers(floor_h, 31)), int(rng.integers(floor_w, 31)))
        else:
            shape = (30, 30)
        frames = _place(rng, shape, dimensions)
        if frames is not None:
            return shape, frames
    # Every declared size fits this packing. It makes the fallback total even
    # if all randomly chosen layouts happen to be obstructed.
    first_h, first_w = dimensions[0]
    second_h, second_w = dimensions[1]
    frames = [(0, 0, first_h - 1, first_w - 1),
              (0, first_w + 1, second_h - 1, first_w + second_w)]
    lower_row = max(first_h, second_h) + 1
    for index, (h, w) in enumerate(dimensions[2:]):
        col = index * 7
        frames.append((lower_row, col, lower_row + h - 1, col + w - 1))
    assert all(f[2] < 30 and f[3] < 30 for f in frames)
    assert all(_separated(a, b) for i, a in enumerate(frames)
               for b in frames[i + 1:])
    return (30, 30), frames


def _draw_frame(grid, frame, color):
    r0, c0, r1, c1 = frame
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _interior(frame):
    r0, c0, r1, c1 = frame
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _choose_cells(rng, cells, count):
    if count == 0:
        return []
    assert count <= len(cells)
    if rng.random() < 0.44:
        center = cells[int(rng.integers(len(cells)))]
        priority = sorted(range(len(cells)), key=lambda i:
                          abs(cells[i][0] - center[0]) + abs(cells[i][1] - center[1])
                          + rng.random() * 2.5)
        clustered = int(rng.integers(max(1, count // 2), count + 1))
        selected = priority[:clustered]
        selected_set = set(selected)
        others = [i for i in range(len(cells)) if i not in selected_set]
        if count > clustered:
            selected.extend(int(i) for i in rng.choice(others, count - clustered, replace=False))
    else:
        selected = [int(i) for i in np.atleast_1d(rng.choice(len(cells), count, replace=False))]
    return [cells[i] for i in selected]


def _palette(rng, train_palettes, split, index):
    if split == "train" and index == 1 and rng.random() < 0.45:
        first = train_palettes[0]
        background = int(rng.choice([c for c in range(10) if c not in (first[1], first[2])]))
        return background, first[2], first[1]
    colors = [int(c) for c in rng.choice(10, 3, replace=False)]
    if split == "train" and index == 1 and colors[1:] == list(train_palettes[0][1:]):
        colors[1], colors[2] = colors[2], colors[1]
    if split == "test" and rng.random() < 0.52:
        used_accents = {p[2] for p in train_palettes}
        unseen = [c for c in range(10) if c not in used_accents]
        if unseen:
            accent = int(rng.choice(unseen))
            colors = [c for c in colors if c != accent][:2] + [accent]
    return tuple(colors)


def _target_size(rng, split, source_shift, index):
    low, high = ((5, 8) if split == "train" else (9, 12)) if source_shift else (5, 11)
    if split == "train" and index == 1:
        low = max(6, low)
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _dimensions(rng, target, frame_count, close, rich):
    h, w = target
    dimensions = [target]
    if close:
        dimensions.append((h - 1, w - 1))
    elif rich:
        dimensions.append((max(4, h - 2), max(4, w - 2)))
    else:
        dimensions.append((int(rng.integers(3, min(6, h - 1) + 1)),
                           int(rng.integers(3, min(6, w - 1) + 1))))
    while len(dimensions) < frame_count:
        dimensions.append((int(rng.integers(3, min(6, h - 1) + 1)),
                           int(rng.integers(3, min(6, w - 1) + 1))))
    return dimensions


def _generate_pair(rng, split, palette, target_size, frame_count, close, rich, source_shift):
    background, structural, accent = palette
    dimensions = _dimensions(rng, target_size, frame_count, close, rich)
    canvas, frames = _canvas_and_frames(rng, dimensions, split)
    grid = np.full(canvas, background, dtype=np.int8)
    for frame in frames:
        _draw_frame(grid, frame, structural)
    target_cells = _interior(frames[0])
    inside_max = min(10, len(target_cells) - 1)
    inside_count = int(rng.integers(1, min(3, inside_max) + 1)) if rich else int(rng.integers(1, inside_max + 1))
    for r, c in _choose_cells(rng, target_cells, inside_count):
        grid[r, c] = accent

    if source_shift:
        low, high = (2, 13) if split == "train" else (14, 25)
    else:
        low, high = 1, 24
    if rich:
        low = max(low, inside_count + 2)
    outside_count = int(rng.integers(low, high + 1))
    remaining = outside_count
    for index, frame in enumerate(frames[1:]):
        cells = _interior(frame)
        if index == 0 and rich:
            amount = int(rng.integers(inside_count + 1,
                                      min(len(cells), remaining - 1, inside_count + 6) + 1))
        else:
            amount = int(rng.integers(0, min(len(cells), 7, max(0, remaining - 1)) + 1))
        for r, c in _choose_cells(rng, cells, amount):
            grid[r, c] = accent
        remaining -= amount

    r0, c0, r1, c1 = frames[0]
    open_cells = [(r, c) for r in range(canvas[0]) for c in range(canvas[1])
                  if grid[r, c] == background and not (r0 <= r <= r1 and c0 <= c <= c1)
                  and all(not (f[0] <= r <= f[2] and f[1] <= c <= f[3]) for f in frames[1:])]
    assert remaining <= len(open_cells)
    for r, c in _choose_cells(rng, open_cells, remaining):
        grid[r, c] = accent
    return {"input": grid, "output": _predict(grid)}


def _analyze(pair):
    grid, output = pair["input"], pair["output"]
    roles, frames, target = _infer(grid)
    assert np.array_equal(_predict(grid), output), "wrong crop or recoloring"
    assert output.shape == _size(target)
    assert np.all(output[0] == roles[2]) and np.all(output[-1] == roles[2])
    assert np.all(output[:, 0] == roles[2]) and np.all(output[:, -1] == roles[2])
    interior = output[1:-1, 1:-1]
    inside = int(np.count_nonzero(interior == roles[2]))
    assert inside >= 1 and np.any(interior == roles[0])
    r0, c0, r1, c1 = target
    outside_mask = grid == roles[2]
    outside_mask[r0:r1 + 1, c0:c1 + 1] = False
    outside = int(np.count_nonzero(outside_mask))
    assert outside >= 1
    smaller_counts = [int(np.count_nonzero(grid[f[0] + 1:f[2], f[1] + 1:f[3]] == roles[2]))
                      for f in frames if f != target]
    close = any(_size(f) == (_size(target)[0] - 1, _size(target)[1] - 1)
                for f in frames if f != target)
    edge = any(a == 0 or b == 0 or c == grid.shape[0] - 1 or d == grid.shape[1] - 1
               for a, b, c, d in frames)
    return {"roles": roles, "frames": frames, "target": target,
            "size": _size(target), "inside": inside, "outside": outside,
            "rich": max(smaller_counts) > inside, "close": close, "edge": edge}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The seed chooses a difficulty regime, not a different transformation.
    source_shift = bool(rng.random() < 0.4)
    puzzle = {"train": [], "test": []}
    palettes, train_sizes, seen = [], [], set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            palette = _palette(rng, palettes, split, index)
            if split == "train":
                palettes.append(palette)
            frame_count = 2 if split == "train" and index == 0 else (
                int(rng.integers(3, 6)) if split == "train" and index == 1
                else int(rng.integers(2, 6)))
            close = (split == "train" and index == 0) or bool(rng.random() < 0.48)
            rich = (split == "train" and index == 1) or bool(rng.random() < 0.24)
            for _ in range(30):
                size = _target_size(rng, split, source_shift, index)
                if split == "train" and index == 1 and size == train_sizes[0]:
                    continue
                pair_seed = int(rng.integers(0, 2**63, dtype=np.int64))
                pair = _generate_pair(np.random.default_rng(pair_seed), split, palette,
                                      size, frame_count, close, rich, source_shift)
                key = _grid_key(pair["input"])
                if key not in seen and _source_hash(pair["input"]) not in _SOURCE_HASHES:
                    break
            else:
                raise RuntimeError("failed to produce a unique pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_sizes.append(size)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _source_hash(pair["input"]) not in _SOURCE_HASHES
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            result = _analyze(pair)
            assert 2 <= len(result["frames"]) <= 5
            info[split].append(result)
    train = info["train"]
    assert len({item["roles"] for item in train}) >= 2
    assert len({item["size"] for item in train}) >= 2
    assert any(item["close"] for item in train)
    assert any(item["rich"] for item in train)
    assert any(len(item["frames"]) >= 3 for item in train)
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(pair["input"]), pair["output"])


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    palettes, canvas_shapes, target_shapes, layouts = set(), set(), set(), set()
    frame_counts, source_shapes = set(), set()
    tall = wide = near_train = near_test = rich_train = edge = 0
    sparse = crowded = reused = unseen_test = lower_test_scatter = overlap_test_size = 0
    shifts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_analyze(p) for p in puzzle["train"]]
        test_infos = [_analyze(p) for p in puzzle["test"]]
        accents = {item["roles"][2] for item in train_infos}
        max_out = max(item["outside"] for item in train_infos)
        max_h = max(item["size"][0] for item in train_infos)
        max_w = max(item["size"][1] for item in train_infos)
        shifts += int(all(item["size"][0] > max_h and item["size"][1] > max_w
                          and item["outside"] > max_out for item in test_infos))
        reused += int(any(a["roles"][1] == b["roles"][2] or
                          a["roles"][2] == b["roles"][1]
                          for a in train_infos for b in train_infos if a is not b))
        unseen_test += sum(item["roles"][2] not in accents for item in test_infos)
        lower_test_scatter += sum(item["outside"] <= max_out for item in test_infos)
        overlap_test_size += sum(item["size"][0] <= max_h or item["size"][1] <= max_w
                                 for item in test_infos)
        for split, infos in (("train", train_infos), ("test", test_infos)):
            for pair, item in zip(puzzle[split], infos):
                if split == "test":
                    copied_train = [{k: p[k].copy() for k in ("input", "output")}
                                    for p in puzzle["train"]]
                    for demonstration in copied_train:
                        assert np.array_equal(_predict(demonstration["input"]),
                                              demonstration["output"])
                    assert np.array_equal(_predict(pair["input"].copy()), pair["output"])
                palettes.add(item["roles"])
                canvas_shapes.add(tuple(pair["input"].shape))
                target_shapes.add(item["size"])
                layouts.add((split, tuple(pair["input"].shape), tuple(item["frames"])))
                frame_counts.add(len(item["frames"]))
                source_shapes.add(tuple(pair["input"].shape) in _SOURCE_SHAPES)
                tall += item["size"][0] - item["size"][1] >= 3
                wide += item["size"][1] - item["size"][0] >= 3
                edge += item["edge"]
                sparse += item["outside"] <= 4
                crowded += item["outside"] >= 15
                if split == "train":
                    near_train += item["close"]
                    rich_train += item["rich"]
                else:
                    near_test += item["close"]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 45 and len(canvas_shapes) >= 35
    assert len(target_shapes) >= 30 and len(layouts) >= 120
    assert frame_counts == {2, 3, 4, 5} and source_shapes == {False, True}
    assert min(tall, wide) >= 20 and edge >= 20
    assert near_train >= 42 and near_test >= 10 and rich_train >= 42
    assert sparse >= 15 and crowded >= 25
    assert reused >= 7 and unseen_test >= 10
    assert lower_test_scatter >= 8 and overlap_test_size >= 8 and shifts >= 5
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        p = generate_puzzle(500 + i, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(600 + i, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    assert _equal_puzzles(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _equal_puzzles(generate_puzzle(314159, 7, 3), generate_puzzle(314160, 7, 3))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(700, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
