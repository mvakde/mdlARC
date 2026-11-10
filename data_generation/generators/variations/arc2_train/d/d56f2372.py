"""Select the only left-right symmetric motif and return its unchanged crop."""

import hashlib
import numpy as np


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    seen = {tuple(int(v) for v in points[0])}
    stack = list(seen)
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and q not in seen:
                    seen.add(q)
                    stack.append(q)
    return len(seen) == len(points)


def _valid_mask(mask):
    return (int(mask.sum()) >= 4 and not bool(mask.all())
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any()
            and _connected(mask))


def _mirror(left, width):
    return np.concatenate((left, left[:, :width // 2][:, ::-1]), axis=1)


def _symmetric(rng, h, w, kind=None):
    kind = kind or str(rng.choice(("irregular", "thin", "thin", "arms",
                                   "outline", "bands", "cavity")))
    half = (w + 1) // 2
    for _ in range(160):
        if kind == "irregular":
            mask = _mirror(rng.random((h, half)) < rng.uniform(0.35, 0.68), w)
        elif kind == "thin":
            mask = np.zeros((h, w), dtype=bool)
            center = (w - 1) // 2
            mask[:, center:(w + 2) // 2] = True
            if h >= half:
                start = int(rng.integers(0, h - half + 1))
                for offset in range(1, center + 1):
                    row = start + offset
                    mask[row, center - offset] = True
                    mask[row, w - 1 - center + offset] = True
                if rng.random() < 0.5:
                    row = int(rng.integers(h))
                    mask[row, center - 1] = True
                    mask[row, w - center] = True
            else:
                row = int(rng.integers(h))
                mask[row] = True
                if rng.random() < 0.5:
                    neighbor = min(h - 1, row + 1)
                    mask[neighbor, center - 1] = True
                    mask[neighbor, w - center] = True
        elif kind == "arms":
            mask = np.zeros((h, w), dtype=bool)
            mask[:, (w - 1) // 2:(w + 2) // 2] = True
            for r in range(h):
                extent = int(rng.integers(0, half))
                if rng.random() < 0.25:
                    mask[r, extent] = mask[r, w - 1 - extent] = True
                else:
                    mask[r, extent:w - extent] = True
            mask[int(rng.integers(h))] = True
        elif kind == "outline":
            mask = _mirror(rng.random((h, half)) < rng.uniform(0.06, 0.34), w)
            mask[0] = mask[-1] = True
            mask[:, 0] = mask[:, -1] = True
        elif kind == "bands":
            mask = np.zeros((h, w), dtype=bool)
            mask[:, (w - 1) // 2:(w + 2) // 2] = True
            rows = rng.choice(h, size=int(rng.integers(1, min(4, h) + 1)),
                              replace=False)
            mask[rows] = True
            for r in range(h):
                if rng.random() < 0.55:
                    extent = int(rng.integers(0, half))
                    mask[r, extent:w - extent] = True
        else:
            mask = np.ones((h, w), dtype=bool)
            rows = rng.choice(np.arange(1, h - 1),
                              size=int(rng.integers(1, h - 1)), replace=False)
            for r in rows:
                extent = int(rng.integers(1, half + 1))
                mask[r, extent:w - extent] = False
        if (_valid_mask(mask) and np.array_equal(mask, mask[:, ::-1])
                and 0.16 <= mask.mean() <= 0.88):
            return mask
    mask = np.zeros((h, w), dtype=bool)
    mask[:, (w - 1) // 2:(w + 2) // 2] = True
    mask[0] = mask[-1] = True
    return mask


def _asymmetric(rng, h, w, kind):
    for _ in range(240):
        if kind == "horizontal":
            upper = rng.random(((h + 1) // 2, w)) < rng.uniform(0.35, 0.66)
            mask = np.concatenate((upper, upper[:h // 2][::-1]), axis=0)
        elif kind == "gross":
            mask = rng.random((h, w)) < rng.uniform(0.37, 0.68)
        else:
            mask = _symmetric(rng, h, w).copy()
            defects = 1 if kind == "near" else int(rng.integers(2, 6))
            for _ in range(defects):
                r = int(rng.integers(h))
                c = int(rng.integers(w // 2))
                if rng.random() < 0.5:
                    c = w - 1 - c
                mask[r, c] = ~mask[r, c]
        if _valid_mask(mask) and not np.array_equal(mask, mask[:, ::-1]):
            return mask
    mask = _symmetric(rng, h, w, "arms").copy()
    mask[h // 2, 0] = True
    mask[h // 2, -1] = False
    if _valid_mask(mask) and not np.array_equal(mask, mask[:, ::-1]):
        return mask
    raise RuntimeError("Could not construct asymmetric motif")


def _separate(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah + 1 <= br or br + bh + 1 <= ar
            or ac + aw + 1 <= bc or bc + bw + 1 <= ac)


def _pack(rng, masks, colors, background):
    min_h = max(11, max(m.shape[0] for m in masks) + 2)
    min_w = max(11, max(m.shape[1] for m in masks) + 2)
    for _ in range(90):
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
        if h * w < sum(m.size for m in masks) * 2.05:
            continue
        boxes = [None] * len(masks)
        order = sorted(range(len(masks)), key=lambda i: -masks[i].size + rng.random())
        for i in order:
            mh, mw = masks[i].shape
            for flat in rng.permutation((h - mh + 1) * (w - mw + 1)):
                r, c = divmod(int(flat), w - mw + 1)
                box = (r, c, mh, mw)
                if all(_separate(box, old) for old in boxes if old is not None):
                    boxes[i] = box
                    break
            if boxes[i] is None:
                break
        if all(box is not None for box in boxes):
            grid = np.full((h, w), background, dtype=np.int8)
            for box, mask, color in zip(boxes, masks, colors):
                r, c, mh, mw = box
                grid[r:r + mh, c:c + mw][mask] = color
            return grid
    grid = np.full((30, 30), background, dtype=np.int8)
    slots = [(r, c) for r in (0, 10, 20) for c in (0, 15)]
    for mask, color, slot in zip(masks, colors, rng.permutation(6)):
        top, left = slots[int(slot)]
        mh, mw = mask.shape
        r = top + int(rng.integers(0, 10 - mh))
        c = left + int(rng.integers(0, 15 - mw))
        grid[r:r + mh, c:c + mw][mask] = color
    return grid


def _components(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    visited = np.zeros(grid.shape, dtype=bool)
    candidates = []
    h, w = grid.shape
    for row in range(h):
        for col in range(w):
            color = int(grid[row, col])
            if color == background or visited[row, col]:
                continue
            visited[row, col] = True
            stack = [(row, col)]
            points = []
            while stack:
                r, c = stack.pop()
                points.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        q = (r + dr, c + dc)
                        if (0 <= q[0] < h and 0 <= q[1] < w
                                and not visited[q] and grid[q] == color):
                            visited[q] = True
                            stack.append(q)
            r0 = min(p[0] for p in points)
            c0 = min(p[1] for p in points)
            r1 = max(p[0] for p in points) + 1
            c1 = max(p[1] for p in points) + 1
            mask = grid[r0:r1, c0:c1] == color
            candidates.append((color, (r0, c0, r1 - r0, c1 - c0), mask))
    return background, candidates


def _select(grid):
    _, candidates = _components(grid)
    matches = [item for item in candidates
               if np.array_equal(item[2], item[2][:, ::-1])]
    assert len(matches) == 1
    return matches[0]


def _make_pair(rng, regime, target_color, palette, force_horizontal,
               target_shape=None):
    count = int(rng.integers(2, 7))
    if target_shape is None:
        h = int(rng.integers(3, 8 if regime == "source_train" else 9))
        w = int(rng.integers(3, 10 if regime == "source_train" else 11))
    else:
        h, w = target_shape
    background = (0 if palette is not None or rng.random() < 0.82 else int(rng.choice(
        [c for c in range(1, 10) if c != target_color])))
    pool = [c for c in (palette if palette is not None else range(1, 10))
            if c not in (background, target_color)]
    if len(pool) < count - 1:
        pool = [c for c in range(1, 10) if c not in (background, target_color)]
    distractor_colors = [int(c) for c in rng.choice(pool, count - 1, replace=False)]
    if count >= 3 and rng.random() < 0.20:
        distractor_colors[-1] = distractor_colors[0]
    masks = [_symmetric(rng, h, w)]
    for i in range(count - 1):
        dh = int(rng.integers(3, 9))
        dw = int(rng.integers(3, 11))
        kind = ("horizontal" if i == 0 and force_horizontal
                else str(rng.choice(("near", "several", "gross", "horizontal"))))
        masks.append(_asymmetric(rng, dh, dw, kind))
    grid = _pack(rng, masks, [target_color] + distractor_colors, background)
    output = np.where(masks[0], target_color, background).astype(np.int8)
    return {"input": grid, "output": output}


def _counts(rng, num_train, num_test):
    if num_train is not None and (type(num_train) is not int or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 to 10")
    if num_test is not None and (type(num_test) is not int or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 to 4")
    if num_test is None:
        maximum = 4 if num_train is None else min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    palette = (tuple(int(c) for c in rng.choice(np.arange(1, 10),
               size=int(rng.integers(6, 9)), replace=False))
               if rng.random() < 0.38 else None)
    base = list(palette) if palette is not None else list(range(1, 10))
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count)
    first_two = rng.choice(base, 2, replace=False)
    puzzle = {"train": [], "test": []}
    seen = set()
    targets, distractors, all_train_colors = set(), set(), set()
    for i in range(train_count + test_count):
        regime = "source_train" if i < train_count else "target_test"
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        if regime == "source_train":
            target = int(first_two[i]) if i < 2 else int(rng.choice(base))
        else:
            category = str(rng.choice(("repeat", "former_distractor", "new")))
            options = (targets if category == "repeat" else
                       distractors - targets if category == "former_distractor" else
                       set(base) - all_train_colors)
            if not options:
                options = targets if category != "former_distractor" else distractors
            target = int(rng.choice(sorted(options)))
        target_shape = None
        if regime == "target_test":
            train_shapes = [pair["output"].shape for pair in puzzle["train"]]
            mode = str(rng.choice(("same", "smaller", "larger", "free")))
            if mode == "same":
                target_shape = train_shapes[int(rng.integers(len(train_shapes)))]
            elif mode in ("smaller", "larger"):
                areas = [h * w for h, w in train_shapes]
                possible = [(h, w) for h in range(3, 9) for w in range(3, 11)
                            if (h * w < min(areas) if mode == "smaller"
                                else h * w > max(areas))]
                if possible:
                    target_shape = possible[int(rng.integers(len(possible)))]
        for _ in range(12):
            pair = _make_pair(pair_rng, regime, target, palette, i == 0,
                              target_shape)
            grid = pair["input"]
            key = (grid.shape, grid.tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("Could not generate unique pair")
        seen.add(key)
        puzzle["train" if regime == "source_train" else "test"].append(pair)
        if regime == "source_train":
            _, candidates = _components(grid)
            targets.add(target)
            pair_colors = {item[0] for item in candidates}
            distractors.update(pair_colors - {target})
            all_train_colors.update(pair_colors)
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_key(grid, output):
    digest = hashlib.sha256()
    digest.update(bytes(grid.shape))
    digest.update(grid.astype(np.int8).tobytes())
    digest.update(bytes(output.shape))
    digest.update(output.astype(np.int8).tobytes())
    return digest.hexdigest()


_SOURCE_KEYS = frozenset({
    "cc9d2bd6030e963d7a63d3431b44e057f019d8d409480efd8657597caf590191",
    "82c3ed17915d90a1e3681688decf8ec6c8a6e5342fca10c3e4eaece24cffc6d3",
    "dacda982fbef88b69e42414abc87bda083c90177b4da62bef7b80b7fe45699b8",
    "8efec292f43cf5f3d6dc58cad0c53632a42bb81d211dfafa9f81faed7b3aa01b",
})


def _check_pair(pair, regime):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = pair["input"], pair["output"]
    for array in (grid, output):
        assert isinstance(array, np.ndarray) and array.ndim == 2
        assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
        assert np.issubdtype(array.dtype, np.integer)
        assert np.all((0 <= array) & (array <= 9))
    background, candidates = _components(grid)
    assert np.count_nonzero(grid == background) > grid.size // 2
    assert 2 <= len(candidates) <= 6
    for i, (color, box, mask) in enumerate(candidates):
        assert _valid_mask(mask)
        r, c, h, w = box
        region = grid[r:r + h, c:c + w]
        assert np.all((region == background) | (region == color))
        for other in candidates[i + 1:]:
            assert _separate(box, other[1])
    color, box, mask = _select(grid)
    r, c, h, w = box
    assert np.array_equal(output, grid[r:r + h, c:c + w])
    assert np.array_equal(output, np.where(mask, color, background))
    assert background in output and color in output
    assert output.shape != grid.shape
    assert _source_key(grid, output) not in _SOURCE_KEYS
    if regime == "source_train":
        assert 3 <= h <= 7 and 3 <= w <= 9
    else:
        assert 3 <= h <= 8 and 3 <= w <= 10
    return color, box, candidates, background


def _solve_from_training(train, test_input):
    for pair in train:
        _, box, _ = _select(pair["input"])
        r, c, h, w = box
        assert np.array_equal(pair["output"], pair["input"][r:r + h, c:c + w])
    _, box, _ = _select(test_input)
    r, c, h, w = box
    return test_input[r:r + h, c:c + w].copy()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs, train_records = set(), set(), []
    for regime, group in (("source_train", train), ("target_test", test)):
        for pair in group:
            record = _check_pair(pair, regime)
            if regime == "source_train":
                train_records.append(record)
            grid, output = pair["input"], pair["output"]
            input_key = (grid.shape, grid.tobytes())
            pair_key = (input_key, output.shape, output.tobytes())
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
    assert len({rec[0] for rec in train_records}) >= 2
    assert any(any(np.array_equal(mask, mask[::-1])
                   and not np.array_equal(mask, mask[:, ::-1])
                   for _, _, mask in candidates)
               for _, _, candidates, _ in train_records)
    for pair in test:
        assert np.array_equal(_solve_from_training(train, pair["input"]),
                              pair["output"])


def _same_puzzle(a, b):
    return (all(len(a[k]) == len(b[k]) for k in ("train", "test"))
            and all(np.array_equal(x[key], y[key])
                    for regime in ("train", "test")
                    for x, y in zip(a[regime], b[regime])
                    for key in ("input", "output")))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    canvases, output_sizes, colors, shapes = set(), set(), set(), set()
    candidate_counts, backgrounds = set(), set()
    ranks = {"smaller": 0, "equal": 0, "larger": 0}
    coverage = {name: 0 for name in (
        "same_size", "repeat_color", "former_distractor_color", "new_color",
        "source_scale", "small_canvas", "wide_canvas", "tall_canvas",
        "edge_target", "target_largest", "larger_distractor",
        "horizontal_distractor", "near_defect", "gross_defect",
        "thin_target", "dense_target", "even_target",
        "varying_candidate_count")}
    repeated_colors = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 3:
            assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_areas = [pair["output"].size for pair in puzzle["train"]]
        train_shapes = {pair["output"].shape for pair in puzzle["train"]}
        train_targets = {_select(pair["input"])[0] for pair in puzzle["train"]}
        train_distractors = {item[0] for pair in puzzle["train"]
                             for item in _components(pair["input"])[1]
                             if item[1] != _select(pair["input"])[1]}
        pair_counts = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid, output = pair["input"], pair["output"]
                background, candidates = _components(grid)
                target = _select(grid)
                distractors = [item for item in candidates if item[1] != target[1]]
                assert np.array_equal(_solve_from_training(puzzle["train"], grid), output)
                canvases.add(grid.shape)
                output_sizes.add(output.shape)
                colors.add(int(np.unique(output[output != background])[0]))
                shapes.add((output.shape, (output != background).tobytes()))
                candidate_counts.add(len(candidates))
                pair_counts.add(len(candidates))
                backgrounds.add(background)
                repeated_colors += len({item[0] for item in candidates}) < len(candidates)
                coverage["source_scale"] += grid.shape[0] in (21, 22) and 16 <= grid.shape[1] <= 20
                coverage["small_canvas"] += grid.shape[0] < 21 and grid.shape[1] < 16
                coverage["wide_canvas"] += grid.shape[1] >= grid.shape[0] + 5
                coverage["tall_canvas"] += grid.shape[0] >= grid.shape[1] + 5
                r, c, h, w = target[1]
                coverage["edge_target"] += (r == 0 or c == 0 or r + h == grid.shape[0]
                                            or c + w == grid.shape[1])
                coverage["target_largest"] += all(output.size > item[2].size for item in distractors)
                coverage["larger_distractor"] += any(item[2].size > output.size for item in distractors)
                coverage["horizontal_distractor"] += any(
                    np.array_equal(item[2], item[2][::-1]) for item in distractors)
                defects = [int(np.count_nonzero(item[2] != item[2][:, ::-1]))
                           for item in distractors]
                coverage["near_defect"] += 2 in defects
                coverage["gross_defect"] += any(n >= 6 for n in defects)
                coverage["thin_target"] += target[2].mean() < 0.43
                coverage["dense_target"] += target[2].mean() > 0.70
                coverage["even_target"] += output.shape[1] % 2 == 0
                if regime == "test":
                    area = output.size
                    ranks["smaller" if area < min(train_areas) else
                          "larger" if area > max(train_areas) else "equal"] += 1
                    coverage["same_size"] += output.shape in train_shapes
                    coverage["repeat_color"] += target[0] in train_targets
                    coverage["former_distractor_color"] += (
                        target[0] not in train_targets and target[0] in train_distractors)
                    coverage["new_color"] += (
                        target[0] not in train_targets and target[0] not in train_distractors)
        coverage["varying_candidate_count"] += len(pair_counts) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(canvases) >= 30 and len(output_sizes) >= 20
    assert len(shapes) >= 80 and len(colors) >= 7
    assert len(candidate_counts) >= 4 and len(backgrounds) >= 3
    assert repeated_colors >= 5 and all(ranks.values())
    assert all(value > 0 for value in coverage.values())
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        first = generate_puzzle(901, *counts)
        assert _same_puzzle(first, generate_puzzle(901, *counts))
        if counts[0] is not None:
            assert len(first["train"]) == counts[0]
        if counts[1] is not None:
            assert len(first["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(902, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {counts}")
    damaged = generate_puzzle(903, 2, 1)
    damaged["test"][0]["output"] = np.zeros_like(damaged["test"][0]["output"])
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted an incorrect crop")
