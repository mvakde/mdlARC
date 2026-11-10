"""Two colored frames count markers strictly inside their borders."""

import numpy as np


_SOURCE_INPUTS = {
    ("111111103333300", "100000103040300", "104000143000304", "100400103004300", "100000103400340", "111111103333300"),
    ("011111002333300", "012221000302300", "010001202300300", "012001020320302", "010201000300300", "011111002333300"),
    ("001110003333300", "001010003070300", "071010703000300", "001010003070307", "001010703000300", "001110003333300"),
    ("001111100333330", "801008100300030", "001800100308030", "801000108300030", "001080100300030", "001111100333330"),
    ("111111113333333", "100000013000003", "150500013000053", "100005013050003", "105000013005053", "111111113333333"),
    ("001111100033330", "001000100030036", "601000106030030", "001060100030030", "001000100030030", "001111100033336"),
}


def _counts(rng, train, test):
    for value, low, high in ((train, 2, 10), (test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("Illegal pair count")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        test = int(test)
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        train = int(train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(train), int(test)
    if train <= test:
        raise ValueError("Training must contain more pairs than testing")
    return train, test


def _border(shape, box):
    y0, y1, x0, x1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[y0, x0:x1 + 1] = True
    mask[y1, x0:x1 + 1] = True
    mask[y0:y1 + 1, x0] = True
    mask[y0:y1 + 1, x1] = True
    return mask


def _analyze(pair):
    grid, output = pair["input"], pair["output"]
    frames = []
    for color in np.unique(grid):
        ys, xs = np.where(grid == color)
        box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        if box[1] - box[0] >= 2 and box[3] - box[2] >= 2:
            if np.array_equal(grid == color, _border(grid.shape, box)):
                frames.append((box, int(color)))
    frames.sort(key=lambda frame: frame[0][2])
    assert len(frames) == 2
    (left, left_color), (right, right_color) = frames
    assert left[3] < right[2]
    other = list(set(int(c) for c in np.unique(grid)) - {left_color, right_color})
    assert len(other) == 2
    other.sort(key=lambda c: np.count_nonzero(grid == c), reverse=True)
    background, marker = other
    assert np.count_nonzero(grid == background) > np.count_nonzero(grid == marker)
    counts = tuple(
        int(np.count_nonzero(grid[y0 + 1:y1, x0 + 1:x1] == marker))
        for (y0, y1, x0, x1), _ in frames
    )
    assert 1 <= sum(counts) <= 30
    expected = np.array([[left_color] * counts[0] + [right_color] * counts[1]], dtype=np.int8)
    assert np.array_equal(output, expected)
    exterior = int(np.count_nonzero(grid == marker) - sum(counts))
    return {
        "frames": tuple(frames), "counts": counts, "background": background,
        "marker": marker, "exterior": exterior,
        "gap": right[2] - left[3] - 1,
        "offset": left[:2] != (0, grid.shape[0] - 1) or right[:2] != (0, grid.shape[0] - 1),
    }


def _geometry(rng, regime, gap_kind, full, offset_required):
    for _ in range(80):
        height = int(rng.integers(5, 12 if regime == "source_train" else 14))
        if offset_required:
            height = max(7, height)
        lw = int(rng.integers(3, 12 if regime == "source_train" else 13))
        rw = int(rng.integers(3, 12 if regime == "source_train" else 13))
        gap = 0 if gap_kind == "touch" else int(rng.integers(1, 6))
        if gap_kind is None:
            gap = 0 if rng.random() < 0.35 else int(rng.integers(1, 6))
        if rng.random() < 0.12:
            lm = rm = 0
        else:
            lm, rm = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        width = lm + lw + gap + rw + rm
        if width > 30:
            continue
        if full:
            ly, lh, ry, rh = 0, height, 0, height
        elif offset_required:
            lh = int(rng.integers(3, height - 1))
            ly = int(rng.integers(1, height - lh + 1))
            rh = int(rng.integers(3, height + 1))
            ry = int(rng.integers(0, height - rh + 1))
        else:
            lh, rh = int(rng.integers(3, height + 1)), int(rng.integers(3, height + 1))
            ly = int(rng.integers(0, height - lh + 1))
            ry = int(rng.integers(0, height - rh + 1))
        left = (ly, ly + lh - 1, lm, lm + lw - 1)
        rx = lm + lw + gap
        right = (ry, ry + rh - 1, rx, rx + rw - 1)
        return (height, width), (left, right)
    raise RuntimeError("Could not fit frame geometry")


def _one_count(rng, capacity, regime):
    top = min(capacity, 15 if regime == "source_train" else 20)
    if top >= 9 and rng.random() < 0.30:
        return int(rng.integers(9, top + 1))
    return int(rng.integers(1, min(8, top) + 1))


def _sample_counts(rng, caps, mode, regime):
    if mode == "zero_left":
        return (0, _one_count(rng, caps[1], regime))
    if mode == "zero_right":
        return (_one_count(rng, caps[0], regime), 0)
    if mode == "equal":
        n = _one_count(rng, min(caps), regime)
        return (n, n)
    if mode == "both_unequal":
        if max(caps) < 2:
            return None
        for _ in range(12):
            result = tuple(_one_count(rng, cap, regime) for cap in caps)
            if result[0] != result[1] and sum(result) <= 30:
                return result
        return (2, 1) if caps[0] >= 2 else (1, 2)
    raise ValueError("Unknown count mode")


def _palette(rng, frames=None, forbidden_background=None):
    if frames is None:
        frames = tuple(int(c) for c in rng.choice(10, size=2, replace=False))
    available = [c for c in range(10) if c not in frames]
    background = int(rng.choice([c for c in available if c != forbidden_background]))
    marker = int(rng.choice([c for c in available if c != background]))
    return background, tuple(frames), marker


def _make_pair(rng, regime, mode, frames=None, forbidden_background=None,
               gap_kind=None, full=False, offset_required=False, force_exterior=False):
    background, colors, marker = _palette(rng, frames, forbidden_background)
    for _ in range(120):
        shape, boxes = _geometry(rng, regime, gap_kind, full, offset_required)
        grid = np.full(shape, background, dtype=np.int8)
        for box, color in zip(boxes, colors):
            grid[_border(shape, box)] = color
        caps = tuple((box[1] - box[0] - 1) * (box[3] - box[2] - 1) for box in boxes)
        counts = _sample_counts(rng, caps, mode, regime)
        if counts is None or sum(counts) > 30:
            continue
        inside = []
        for box, count in zip(boxes, counts):
            y0, y1, x0, x1 = box
            cells = [(y, x) for y in range(y0 + 1, y1) for x in range(x0 + 1, x1)]
            for index in rng.choice(len(cells), size=count, replace=False):
                grid[cells[int(index)]] = marker
            inside.extend(cells)
        inside_set = set(inside)
        exterior_cells = [(int(y), int(x)) for y, x in zip(*np.where(grid == background))
                          if (int(y), int(x)) not in inside_set]
        blanks = int(np.count_nonzero(grid == background) + sum(counts))
        max_exterior = min(len(exterior_cells), 15, (blanks - 1) // 2 - sum(counts))
        if max_exterior < int(force_exterior):
            continue
        roll = rng.random()
        if not force_exterior and roll < 0.10:
            exterior_count = 0
        elif max_exterior >= 6 and roll < 0.48:
            exterior_count = int(rng.integers(6, max_exterior + 1))
        else:
            exterior_count = int(rng.integers(1, min(5, max_exterior) + 1)) if max_exterior else 0
        chosen = []
        if exterior_count:
            left, right = boxes
            gap_cells = [p for p in exterior_cells if left[3] < p[1] < right[2]]
            vertical_cells = [p for p in exterior_cells if
                              (left[2] <= p[1] <= left[3] and not left[0] <= p[0] <= left[1])
                              or (right[2] <= p[1] <= right[3] and not right[0] <= p[0] <= right[1])]
            if gap_cells and rng.random() < 0.65:
                chosen.append(gap_cells[int(rng.integers(len(gap_cells)))])
            if len(chosen) < exterior_count and vertical_cells and rng.random() < 0.70:
                options = [p for p in vertical_cells if p not in chosen]
                if options:
                    chosen.append(options[int(rng.integers(len(options)))])
            rest = [p for p in exterior_cells if p not in chosen]
            chosen.extend(rest[int(i)] for i in rng.choice(len(rest), size=exterior_count - len(chosen), replace=False))
            for p in chosen:
                grid[p] = marker
        output = np.array([[colors[0]] * counts[0] + [colors[1]] * counts[1]], dtype=np.int8)
        pair = {"input": grid, "output": output}
        try:
            _analyze(pair)
        except AssertionError:
            continue
        return pair
    raise RuntimeError("Could not construct a valid pair")


def _contrast(pair, rng):
    facts = _analyze(pair)
    grid = pair["input"]
    marker, background = facts["marker"], facts["background"]
    (left, left_color), (right, right_color) = facts["frames"]
    interiors = []
    for y0, y1, x0, x1 in (left, right):
        interiors.append([(y, x) for y in range(y0 + 1, y1) for x in range(x0 + 1, x1)])
    all_inside = set(interiors[0] + interiors[1])
    outside = [p for p in zip(*np.where(grid == marker)) if p not in all_inside]
    moves = []
    for target in interiors:
        vacancies = [p for p in target if grid[p] == background]
        if outside and vacancies:
            moves.append((outside, vacancies))
    for source, target in ((interiors[0], interiors[1]), (interiors[1], interiors[0])):
        occupied = [p for p in source if grid[p] == marker]
        vacancies = [p for p in target if grid[p] == background]
        if occupied and vacancies:
            moves.append((occupied, vacancies))
    if not moves:
        return None
    source, target = moves[int(rng.integers(len(moves)))]
    origin, destination = source[int(rng.integers(len(source)))], target[int(rng.integers(len(target)))]
    changed = grid.copy()
    changed[origin], changed[destination] = background, marker
    counts = tuple(sum(changed[p] == marker for p in cells) for cells in interiors)
    result = {"input": changed, "output": np.array([[left_color] * counts[0] + [right_color] * counts[1]], dtype=np.int8)}
    _analyze(result)
    return result


def _mode(rng):
    return ("both_unequal", "equal", "zero_left", "zero_right")[
        int(rng.choice(4, p=[0.38, 0.18, 0.22, 0.22]))
    ]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    anchor = tuple(int(c) for c in rng.choice(10, size=2, replace=False))
    first_gap = "touch" if rng.random() < 0.5 else "separated"
    second_gap = "separated" if first_gap == "touch" else "touch"
    first = _make_pair(rng, "source_train", "both_unequal", frames=anchor,
                       gap_kind=first_gap, full=True, force_exterior=True)
    second = _make_pair(rng, "source_train", "zero_left" if rng.random() < 0.5 else "zero_right",
                        frames=anchor[::-1], forbidden_background=_analyze(first)["background"],
                        gap_kind=second_gap, offset_required=True, force_exterior=True)
    train = [first, second]
    if train_count >= 4:
        for _ in range(40):
            base = _make_pair(rng, "source_train", _mode(rng), force_exterior=True)
            mate = _contrast(base, rng)
            if mate is not None:
                train.extend((base, mate))
                break
        else:
            raise RuntimeError("Could not make a counting contrast")
    while len(train) < train_count:
        train.append(_make_pair(rng, "source_train", _mode(rng)))
    rng.shuffle(train)
    test = [_make_pair(rng, "target_test", _mode(rng), gap_kind="separated", offset_required=True)]
    while len(test) < test_count:
        test.append(_make_pair(rng, "target_test", _mode(rng),
                               gap_kind="touch" if len(test) == 1 else None,
                               offset_required=rng.random() < 0.65))
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_contrast(facts):
    return any(
        a["frames"] == b["frames"] and a["background"] == b["background"]
        and a["marker"] == b["marker"]
        and sum(a["counts"]) + a["exterior"] == sum(b["counts"]) + b["exterior"]
        and a["counts"] != b["counts"]
        for a in facts for b in facts if a is not b
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, facts = set(), set(), []
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= size <= 30 for size in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        inp, out = pair["input"], pair["output"]
        signature = (inp.shape, inp.tobytes())
        pair_signature = (signature, out.shape, out.tobytes())
        assert signature not in seen_inputs and pair_signature not in seen_pairs
        seen_inputs.add(signature)
        seen_pairs.add(pair_signature)
        rows = tuple("".join(str(int(v)) for v in row) for row in inp)
        assert rows not in _SOURCE_INPUTS
        facts.append(_analyze(pair))
    train_facts, test_facts = facts[:len(train)], facts[len(train):]
    assert any(a > 0 and b > 0 and a != b for a, b in (f["counts"] for f in train_facts))
    assert any(0 in f["counts"] for f in train_facts)
    assert {f["gap"] == 0 for f in train_facts} == {True, False}
    assert any(f["offset"] for f in train_facts)
    assert any(f["gap"] > 0 for f in test_facts)
    assert any(f["offset"] for f in test_facts)
    assert any(
        a["frames"][0][1] == b["frames"][1][1]
        and a["frames"][1][1] == b["frames"][0][1]
        and a["background"] != b["background"]
        for a in train_facts for b in train_facts if a is not b
    )
    if len(train) >= 4:
        assert _has_contrast(train_facts)


def _same_puzzle(a, b):
    return all(len(a[split]) == len(b[split]) for split in ("train", "test")) and all(
        np.array_equal(pa[key], pb[key])
        for split in ("train", "test")
        for pa, pb in zip(a[split], b[split])
        for key in ("input", "output")
    )


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes, modes = (set() for _ in range(6))
    split_gaps = {"train": set(), "test": set()}
    exterior_counts, max_counts, first_inputs = set(), set(), set()
    review = {
        split: {name: 0 for name in (
            "zero_exterior", "crowded", "outside_outnumbers", "gap_marker",
            "vertical_marker", "flush", "unequal_widths", "unequal_heights",
        )}
        for split in ("train", "test")
    }
    first_modes = {"train": set(), "test": set()}
    swapped = contrasts = offsets = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_inputs.add((puzzle["train"][0]["input"].shape, puzzle["train"][0]["input"].tobytes()))
        train_facts = [_analyze(pair) for pair in puzzle["train"]]
        swapped += any(
            a["frames"][0][1] == b["frames"][1][1]
            and a["frames"][1][1] == b["frames"][0][1]
            and a["background"] != b["background"]
            for a in train_facts for b in train_facts if a is not b
        )
        contrasts += _has_contrast(train_facts)
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                fact = _analyze(pair)
                grid = pair["input"]
                (left, _), (right, _) = fact["frames"]
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple(box for box, _ in fact["frames"])))
                palettes.add((fact["background"], fact["frames"][0][1], fact["frames"][1][1], fact["marker"]))
                split_gaps[split].add(fact["gap"])
                exterior_counts.add(fact["exterior"])
                max_counts.add(max(fact["counts"]))
                offsets += fact["offset"]
                a, b = fact["counts"]
                mode = "zero_left" if a == 0 else "zero_right" if b == 0 else "equal" if a == b else "unequal"
                modes.add(mode)
                if index == 0:
                    first_modes[split].add(mode)
                outside = [
                    (y, x) for y, x in zip(*np.where(grid == fact["marker"]))
                    if not (left[0] < y < left[1] and left[2] < x < left[3])
                    and not (right[0] < y < right[1] and right[2] < x < right[3])
                ]
                metrics = review[split]
                metrics["zero_exterior"] += fact["exterior"] == 0
                metrics["crowded"] += fact["exterior"] > 5
                metrics["outside_outnumbers"] += fact["exterior"] > a + b
                metrics["gap_marker"] += any(left[3] < x < right[2] for y, x in outside)
                metrics["vertical_marker"] += any(
                    (left[2] <= x <= left[3] and not left[0] <= y <= left[1])
                    or (right[2] <= x <= right[3] and not right[0] <= y <= right[1])
                    for y, x in outside
                )
                metrics["flush"] += left[2] == 0 and right[3] == grid.shape[1] - 1
                metrics["unequal_widths"] += abs((left[3] - left[2]) - (right[3] - right[2])) >= 3
                metrics["unequal_heights"] += left[1] - left[0] != right[1] - right[0]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(first_inputs) >= 40 and len(shapes) >= 35 and len(layouts) >= 100 and len(palettes) >= 80
    assert modes == {"zero_left", "zero_right", "equal", "unequal"}
    assert 0 in exterior_counts and max(exterior_counts) >= 6 and max(max_counts) >= 10
    assert 0 in split_gaps["train"] and 0 in split_gaps["test"]
    assert max(split_gaps["train"]) >= 3 and max(split_gaps["test"]) >= 3
    assert swapped == 42 and contrasts >= 10 and offsets >= 84
    assert first_modes["train"] == first_modes["test"] == modes
    for split in ("train", "test"):
        assert all(review[split][name] > 0 for name in review[split])
    legal = ((2, 1), (10, 1), (10, 4), (4, 2), (7, 3),
             (None, 1), (None, 4), (2, None), (10, None))
    for nt, ns in legal:
        puzzle = generate_puzzle(4711, nt, ns)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(4711, nt, ns))
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
