import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 2),
                (6, 3), (7, 3), (8, 4), (10, 4))

_SOURCE_INPUTS = {
    tuple(s.split()) for s in (
        """000000000800000 000000000000000 078777700000000
        077777700044800 077777700044400 077787700044400
        077777708000000 000000000000000 000080000000000
        000008006666600 800000006666600 000008006866600
        000000006666600""",
        """000000000000303 003333333300330 003333333300333
        003033333300000 003333333322200 000000000022000
        000000000020200 022220000022200 022200000022200
        002220000000000 022220033333030 000000033333330
        000000033333330""",
        """0000000000 0044444000 0024344000 0044444002
        0044444000 0000002000 0000000000 8888800777
        8388800777 8888300727""",
        """333333333333333 333333333333333 333333333333233
        333322333333333 333333333333333 333333333333333
        332333333333332 333333333333333 333333333333333
        333333333333333 333333332333333 333233333333333
        333333333333333""",
        """55555555555555 55007055000055 55000057000075
        55000055000055 75000055000055 55555555070055
        55755555000055 55555755000055 55555555000057""",
    )
}


def _resolve_counts(rng, seed, num_train, num_test):
    for value in (num_train, num_test):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("Pair counts must be integers.")
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[int(seed) % len(_AUTO_COUNTS)]
    if num_train is None:
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4.")
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10.")
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("Pair counts are outside the permitted ranges.")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.tobytes()


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _background(grid):
    return int(np.bincount(grid.reshape(-1), minlength=10).argmax())


def _rectangles(grid, background):
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or grid[row, col] == background:
                continue
            color = int(grid[row, col])
            cells = []
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= rr < height and 0 <= cc < width
                            and not visited[rr, cc] and grid[rr, cc] == color):
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells) + 1
            left = min(c for _, c in cells)
            right = max(c for _, c in cells) + 1
            assert len(cells) == (bottom-top) * (right-left)
            result.append((top, bottom, left, right, color))
    return result


def _touches(a, b):
    t, d, l, r = a[:4]
    u, v, x, y = b[:4]
    return ((d == u or v == t) and max(l, x) < min(r, y)
            or (r == x or y == l) and max(t, u) < min(d, v))


def _place_rectangles(rng, height, width, colors, force_edge, want_touch):
    boxes = []
    used_area = 0
    for index, color in enumerate(colors):
        accepted = None
        for phase in (0, 1):
            for _ in range(400):
                box_h = int(rng.integers(3, min(10, height-1) + 1))
                box_w = int(rng.integers(3, min(10, width-1) + 1))
                if box_h * box_w + used_area > height * width * 0.46:
                    continue
                top = int(rng.integers(0, height-box_h+1))
                left = int(rng.integers(0, width-box_w+1))
                if force_edge and index == 0:
                    side = int(rng.integers(4))
                    if side == 0:
                        top = 0
                    elif side == 1:
                        top = height-box_h
                    elif side == 2:
                        left = 0
                    else:
                        left = width-box_w
                candidate = (top, top+box_h, left, left+box_w, color)
                if any(not (candidate[1] <= b[0] or b[1] <= candidate[0]
                            or candidate[3] <= b[2] or b[3] <= candidate[2])
                       for b in boxes):
                    continue
                if any(color == b[4] and _touches(candidate, b) for b in boxes):
                    continue
                if phase == 0 and index == 1 and want_touch:
                    if not any(color != b[4] and _touches(candidate, b)
                               for b in boxes):
                        continue
                accepted = candidate
                break
            if accepted is not None:
                break
        if accepted is None:
            raise RuntimeError("Rectangle placement exhausted.")
        boxes.append(accepted)
        used_area += (accepted[1]-accepted[0]) * (accepted[3]-accepted[2])
    return boxes


def _rectangle_evidence(noisy, clean, box):
    top, bottom, left, right, color = box
    part = noisy[top:bottom, left:right] == color
    h, w = part.shape
    return (int(part.sum()) >= max(6, int(np.ceil(0.72*h*w)))
            and np.all(part.sum(axis=0) >= max(2, int(np.ceil(0.60*h))))
            and np.all(part.sum(axis=1) >= max(2, int(np.ceil(0.60*w)))))


def _corrupt(clean, boxes, background, mode, anchor, rng, stable_noise=None):
    noisy = clean.copy()
    height, width = clean.shape
    rect_mask = clean != background
    used = set()
    absent = [c for c in range(10) if c not in set(int(x) for x in np.unique(clean))]
    single_color = (stable_noise if stable_noise is not None else
                    int(rng.choice(absent))) if mode == "one_color" else None

    def wrong(true_color, first_rectangle=False):
        if mode == "all_hole":
            return 0
        if mode == "one_color":
            return single_color
        if anchor and first_rectangle:
            return background
        if stable_noise is not None:
            return (background if true_color != background
                    and int(rng.integers(3)) == 0 else stable_noise)
        options = [c for c in range(10) if c != true_color]
        if int(rng.integers(3)) == 0:
            seen_options = [c for c in options
                            if c in set(int(x) for x in np.unique(clean))]
            if seen_options:
                options = seen_options
        return int(rng.choice(options))

    for index, box in enumerate(boxes):
        top, bottom, left, right, color = box
        if mode == "all_hole" and color == 0:
            continue
        max_edits = min(3, max(1, (bottom-top)*(right-left)//7))
        count = int(rng.integers(0 if not anchor or index else 1, max_edits+1))
        if anchor and index == 0:
            count = max(1, count)
        candidates = [(r, c) for r in range(top, bottom)
                      for c in range(left, right)]
        rng.shuffle(candidates)
        if count >= 2 and int(rng.integers(4)) == 0:
            r, c = candidates[0]
            neighbors = [(rr, cc) for rr, cc in ((r-1, c), (r+1, c),
                                                  (r, c-1), (r, c+1))
                         if top <= rr < bottom and left <= cc < right]
            rng.shuffle(neighbors)
            candidates = candidates[:1] + neighbors + candidates[1:]
        placed = 0
        for r, c in candidates:
            if placed == count:
                break
            if (r, c) in used:
                continue
            noisy[r, c] = wrong(color, anchor and index == 0 and placed == 0)
            if _rectangle_evidence(noisy, clean, box):
                used.add((r, c))
                placed += 1
            else:
                noisy[r, c] = color

    base_candidates = []
    for r in range(height):
        for c in range(width):
            if rect_mask[r, c]:
                continue
            t, d = max(0, r-1), min(height, r+2)
            l, e = max(0, c-1), min(width, c+2)
            if int(np.count_nonzero(clean[t:d, l:e] == background)) >= 6:
                base_candidates.append((r, c))
    rng.shuffle(base_candidates)
    if mode == "all_hole" and background == 0:
        base_count = 0
    else:
        base_count = int(rng.integers(0 if not anchor else 1, 6))
    base_count = min(base_count, len(base_candidates))
    for r, c in base_candidates[:base_count]:
        noisy[r, c] = wrong(background)
    if not np.any(noisy != clean):
        if boxes:
            top, bottom, left, right, color = boxes[0]
            noisy[top+1, left+1] = wrong(color, anchor)
        else:
            r, c = base_candidates[0]
            noisy[r, c] = wrong(background)
    return noisy


def _make_pair(rng, regime, index, stable_colors):
    anchor = regime == "source_train" and index == 0
    if anchor:
        count = int(rng.integers(2, 5))
    else:
        weights = ([.12, .16, .25, .27, .16, .04] if regime == "source_train"
                   else [.19, .13, .27, .24, .13, .04])
        count = int(rng.choice(6, p=weights))
    low_h = 14 if count >= 4 else 9
    low_w = 15 if count >= 4 else 9
    high = 23 if regime == "source_train" else 26
    height = int(rng.integers(low_h, high))
    width = int(rng.integers(low_w, high+1))
    if anchor or stable_colors is not None:
        background = 0
    else:
        background = (0 if rng.random() < (.57 if regime == "source_train" else .38)
                      else int(rng.integers(1, 10)))
    if stable_colors is None:
        pool = [c for c in range(10) if c != background]
    else:
        pool = [c for c in stable_colors[:3] if c != background]
        if not pool:
            pool = [c for c in range(10) if c != background]
    colors = []
    zero_rectangle = (background != 0 and count > 0 and
                      rng.random() < (.55 if regime == "source_train" else .72))
    for j in range(count):
        if j == 0 and zero_rectangle:
            color = 0
        elif colors and rng.random() < .36:
            color = int(rng.choice(colors))
        else:
            color = int(rng.choice(pool))
        colors.append(color)
    force_edge = count > 0 and rng.random() < (
        .23 if regime == "source_train" else .45)
    want_touch = count >= 2 and colors[0] != colors[1] and rng.random() < .35
    boxes = _place_rectangles(rng, height, width, colors, force_edge, want_touch)
    clean = np.full((height, width), background, dtype=np.int8)
    for top, bottom, left, right, color in boxes:
        clean[top:bottom, left:right] = color
    if int(np.count_nonzero(clean == background)) <= clean.size // 2:
        raise RuntimeError("Base is not dominant.")
    if anchor:
        mode = "mixed"
    else:
        choices = ["one_color", "mixed"]
        if all(color != 0 for color in colors) and (background != 0 or count > 0):
            choices.append("all_hole")
        mode = str(rng.choice(choices))
    noisy = _corrupt(clean, boxes, background, mode, anchor, rng,
                     None if stable_colors is None else stable_colors[3])
    return {"input": noisy, "output": clean}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    stable_colors = (tuple(int(c) for c in rng.choice(np.arange(1, 10),
                                                      size=4, replace=False))
                     if rng.random() < .20 else None)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for regime, total in (("source_train", train_count),
                          ("target_test", test_count)):
        for index in range(total):
            for _ in range(60):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                try:
                    pair = _make_pair(pair_rng, regime, index, stable_colors)
                except RuntimeError:
                    continue
                input_key = _key(pair["input"])
                if (input_key in seen_inputs
                        or _rows(pair["input"]) in _SOURCE_INPUTS):
                    continue
                seen_inputs.add(input_key)
                puzzle["train" if regime == "source_train" else "test"].append(pair)
                break
            else:
                raise RuntimeError("Distinct pair construction exhausted.")
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    noisy, clean = pair["input"], pair["output"]
    for grid in (noisy, clean):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
    assert noisy.shape == clean.shape
    assert _rows(noisy) not in _SOURCE_INPUTS
    changes = noisy != clean
    assert 1 <= int(changes.sum()) <= 15
    background = _background(clean)
    assert int(np.count_nonzero(clean == background)) > clean.size // 2
    boxes = _rectangles(clean, background)
    assert len(boxes) <= 5
    rect_mask = clean != background
    assert int(np.count_nonzero(changes & ~rect_mask)) <= 5
    for box in boxes:
        top, bottom, left, right, color = box
        assert 3 <= bottom-top <= 10 and 3 <= right-left <= 10
        assert _rectangle_evidence(noisy, clean, box)
        assert int(np.count_nonzero(noisy[top:bottom, left:right] == color)) >= 6
    assert int(np.count_nonzero(changes & (clean == background))) < (
        int(np.count_nonzero(clean == background)) // 10 + 1)
    return background, boxes, changes


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    pairs = puzzle["train"] + puzzle["test"]
    inspected = []
    for pair in pairs:
        inspected.append(_inspect_pair(pair))
        key = _key(pair["input"])
        assert key not in seen
        seen.add(key)
    background, boxes, changes = inspected[0]
    assert background == 0 and 2 <= len(boxes) <= 4
    assert int(np.count_nonzero(changes & (pairs[0]["output"] == background))) >= 1
    assert int(np.count_nonzero(changes & (pairs[0]["output"] != background))) >= 1


def _same_puzzle(first, second):
    return all(
        len(first[regime]) == len(second[regime])
        and all(np.array_equal(a[k], b[k])
                for a, b in zip(first[regime], second[regime])
                for k in ("input", "output"))
        for regime in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, counts, edit_counts, colors = set(), set(), set(), set()
    all_hole = one_color = mixed = touching = repeated = uniform = zero_rect = 0
    puzzle_keys = set()
    within_count = within_shape = within_edits = role_switch = 0
    stable_palettes = train_uniform = train_zero = 0
    test_uniform = test_zero = test_new = test_known = 0
    adjacent = boundary = large = combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_counts, local_shapes, local_edits = set(), set(), set()
        roles = {}
        output_nonzero, noise_nonzero = set(), set()
        training_colors = {int(c) for p in puzzle["train"]
                           for c in np.unique(p["output"])}
        puzzle_keys.add(tuple((_key(p["input"]), _key(p["output"]))
                              for p in puzzle["train"] + puzzle["test"]))
        for pair_index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            is_train = pair_index < len(puzzle["train"])
            base, boxes, changes = _inspect_pair(pair)
            clean, noisy = pair["output"], pair["input"]
            local_counts.add(len(boxes))
            local_shapes.add(clean.shape)
            local_edits.add(int(changes.sum()))
            shapes.add(clean.shape)
            counts.add(len(boxes))
            edit_counts.add(int(changes.sum()))
            colors.update(int(c) for c in np.unique(clean))
            if not boxes:
                uniform += 1
                train_uniform += is_train
                test_uniform += not is_train
            if base != 0 and any(b[4] == 0 for b in boxes):
                zero_rect += 1
                train_zero += is_train
                test_zero += not is_train
            if len({b[4] for b in boxes}) < len(boxes):
                repeated += 1
            has_touch = any(a[4] != b[4] and _touches(a, b)
                            for i, a in enumerate(boxes) for b in boxes[i+1:])
            if has_touch:
                touching += 1
            old = {int(c) for c in np.unique(noisy[changes])}
            combined += has_touch and len({b[4] for b in boxes}) < len(boxes) and len(old) > 1
            large += any((d-t >= 8 and r-l >= 4)
                         or (r-l >= 8 and d-t >= 4)
                         for t, d, l, r, _ in boxes)
            boundary += any(changes[t:d, l:r][0].any()
                            or changes[t:d, l:r][-1].any()
                            or changes[t:d, l:r][:, 0].any()
                            or changes[t:d, l:r][:, -1].any()
                            for t, d, l, r, _ in boxes)
            points = np.argwhere(changes)
            adjacent += any(abs(int(a[0])-int(b[0])) +
                            abs(int(a[1])-int(b[1])) == 1
                            for i, a in enumerate(points) for b in points[i+1:])
            if old == {0}:
                all_hole += 1
            elif len(old) == 1:
                one_color += 1
            else:
                mixed += 1
            for c in np.unique(clean):
                role = "base" if c == base else "rect"
                roles.setdefault(int(c), set()).add(role)
                if int(c) != 0:
                    output_nonzero.add(int(c))
            for c in np.unique(noisy[changes]):
                roles.setdefault(int(c), set()).add("noise")
                if int(c) != 0:
                    noise_nonzero.add(int(c))
            if not is_train:
                output_colors = {int(c) for c in np.unique(clean)}
                test_new += bool(output_colors - training_colors)
                test_known += output_colors <= training_colors
        within_count += len(local_counts) > 1
        within_shape += len(local_shapes) > 1
        within_edits += len(local_edits) > 1
        role_switch += any({"rect", "noise"} <= v or {"base", "rect"} <= v
                           for v in roles.values())
        stable_palettes += (len(output_nonzero) <= 3
                            and len(noise_nonzero) <= 1
                            and not (output_nonzero & noise_nonzero))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(counts) == 6
    assert len(edit_counts) >= 8 and len(colors) == 10
    assert len(puzzle_keys) == 42
    assert uniform >= 10 and zero_rect >= 10
    assert repeated >= 10 and touching >= 5
    assert all_hole >= 10 and one_color >= 10 and mixed >= 10
    assert within_count >= 20 and within_shape == 42 and within_edits >= 20
    assert role_switch >= 20
    assert stable_palettes >= 3
    assert train_uniform >= 5 and train_zero >= 5
    assert test_uniform >= 5 and test_zero >= 5
    assert test_new >= 10 and test_known >= 10
    assert adjacent >= 10 and boundary >= 20 and large >= 20
    assert combined >= 5
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(100 + train + test, train, test)
        validate_generated_puzzle(p)
        assert len(p["train"]) == train and len(p["test"]) == test
    for request in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(211, *request))
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (True, None), (None, 1.5)):
        try:
            generate_puzzle(0, *request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {request}")
