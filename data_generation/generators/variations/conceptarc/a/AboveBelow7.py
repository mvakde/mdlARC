"""Hollow frames set the landing rows for colored motifs above them."""

import numpy as np


_SOURCE_INPUTS = (
    ("000000000000000", "000000000000000", "000000000000000",
     "000044444444000", "000033333333000", "000030000003000",
     "000030000003000", "000030000003000", "000033333333000",
     "000000000000000"),
    ("000880000000", "008888000800", "088888808880", "022222202220",
     "020000202020", "020000202020", "020000202020", "020000202220",
     "022222200000", "000000000000", "000000000000", "000000000000",
     "000000000000", "000000000000"),
    ("0000000000", "0333033330", "0030003300", "0444044440",
     "0404040040", "0444040040", "0000040040", "0000040040",
     "0000044440", "0000000000", "0000000000", "0000000000"),
    ("02000200000000", "00222000000000", "01111110200020",
     "01000010022200", "01000010111110", "01000010100010",
     "01000010100010", "01111110100010", "00000000111110",
     "00000000000000", "00000000000000", "00000000000000",
     "00000000000000"),
    ("222222222", "000020000", "000020000", "077777770",
     "070000070", "070000070", "070000070", "077777770",
     "000000000", "000000000", "000000000"),
)


def _rand(rng, low, high):
    return int(rng.integers(low, high + 1))


def _count(value, name, low, high):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _palette(rng, previous):
    for _ in range(8):
        background = 0 if rng.random() < 0.68 else _rand(rng, 1, 9)
        others = [c for c in range(10) if c != background]
        roles = rng.choice(others, size=2, replace=False)
        palette = (background, int(roles[0]), int(roles[1]))
        if palette != previous:
            return palette
    background = (previous[0] + 1) % 10
    others = [c for c in range(10) if c != background]
    return background, others[0], others[1]


def _mask(rng, height, width):
    """Build a visible silhouette; broad bars and connected forms are common."""
    kinds = ("bar", "bar_stem", "step", "block", "arch", "irregular", "separated")
    kind = kinds[int(rng.choice(len(kinds), p=(.17, .23, .18, .13, .10, .14, .05)))]
    if height == 1 and kind == "bar_stem":
        kind = "bar"
    if kind == "bar":
        height = 1
    a = np.zeros((height, width), dtype=bool)
    if kind == "bar":
        a[0, :] = True
    elif kind == "bar_stem":
        a[0, :] = True
        stem_width = _rand(rng, 1, min(2, width))
        start = _rand(rng, 0, width - stem_width)
        a[1:, start:start + stem_width] = True
    elif kind == "step":
        a[0, :max(2, width // 2)] = True
        for r in range(1, height):
            start = min(width - 2, r * max(1, width // (height + 1)))
            a[r, start:min(width, start + max(2, width // 2))] = True
        a[height // 2, :] = True
    elif kind == "block":
        a[:, :] = True
        if height > 1 and width > 3:
            a[-1, :_rand(rng, 1, width // 3)] = False
    elif kind == "arch":
        a[0, :] = True
        a[:, 0] = True
        a[:, -1] = True
    elif kind == "irregular":
        a[0, :] = True
        for r in range(1, height):
            left = _rand(rng, 0, max(0, width // 3))
            right = _rand(rng, min(width - 1, 2 * width // 3), width - 1)
            a[r, left:right + 1] = True
    else:
        # Small separated crests above a bridging row, as in source train 2.
        a[-1, :] = True
        for r in range(height - 1):
            for c in range(width):
                if (c + 2 * r) % 3 == 0:
                    a[r, c] = True
    return a


def _layout(rng, frame_count):
    """Place horizontally separate frames with a neutral column between territories."""
    for _ in range(80):
        if frame_count == 1:
            widths = [_rand(rng, 3, 12)]
            gaps = []
            margins = (_rand(rng, 1, 9), _rand(rng, 1, 9))
        else:
            cap = 8 if frame_count == 2 else 6
            widths = [_rand(rng, 3, cap) for _ in range(frame_count)]
            gaps = [_rand(rng, 3, 7 if frame_count == 2 else 6)
                    for _ in range(frame_count - 1)]
            margins = (_rand(rng, 0, 4), _rand(rng, 0, 4))
        width = sum(widths) + sum(gaps) + sum(margins)
        if 7 <= width <= 30:
            break
    else:
        raise AssertionError("horizontal layout exhausted")

    frames = []
    left = margins[0]
    for index, fw in enumerate(widths):
        frames.append([left, fw])
        if index < len(gaps):
            left += fw + gaps[index]
    territories = []
    for index, (left, fw) in enumerate(frames):
        lower = 0 if index == 0 else left - (gaps[index - 1] - 1) // 2
        upper = width - 1 if index == frame_count - 1 else left + fw - 1 + (gaps[index] - 1) // 2
        territories.append((lower, upper))
    return width, frames, territories


def _motif_bounds(rng, left, frame_width, territory, mode):
    right = left + frame_width - 1
    available_left = left - territory[0]
    available_right = territory[1] - right
    if mode == "full":
        return territory
    if mode == "both" and available_left and available_right:
        return (left - _rand(rng, 1, available_left),
                right + _rand(rng, 1, available_right))
    if mode in ("left", "both") and available_left:
        return left - _rand(rng, 1, available_left), right
    if mode in ("right", "both") and available_right:
        return left, right + _rand(rng, 1, available_right)
    if mode == "equal" or frame_width == 3:
        return left, right
    motif_width = _rand(rng, 2, frame_width - 1)
    start = _rand(rng, left, right - motif_width + 1)
    return start, start + motif_width - 1


def _make_pair(rng, frame_count, require_overhang, aligned, roomy, previous_palette):
    width, columns, territories = _layout(rng, frame_count)
    frame_heights = [_rand(rng, 3, 8) for _ in columns]
    motif_heights = [_rand(rng, 1, 4) for _ in columns]
    vertical_gaps = [_rand(rng, 0, 2) for _ in columns]
    top_margin = _rand(rng, 0, 3)
    if aligned and frame_count > 1:
        shared_top = max(h + gap for h, gap in zip(motif_heights, vertical_gaps)) + top_margin
        tops = [shared_top] * frame_count
    else:
        tops = [h + gap + top_margin + _rand(rng, 0, 4)
                for h, gap in zip(motif_heights, vertical_gaps)]
        if frame_count > 1 and len(set(tops)) == 1:
            tops[-1] += 1
    if frame_count > 1 and len({top + fh for top, fh in zip(tops, frame_heights)}) == 1:
        frame_heights[-1] += 1 if frame_heights[-1] < 8 else -1

    objects = []
    overhang_index = _rand(rng, 0, frame_count - 1) if require_overhang else -1
    for index, ((left, fw), territory, top, fh, mh, gap) in enumerate(
            zip(columns, territories, tops, frame_heights, motif_heights, vertical_gaps)):
        available_left = left - territory[0]
        available_right = territory[1] - (left + fw - 1)
        if index == overhang_index:
            options = []
            if available_left:
                options.append("left")
            if available_right:
                options.append("right")
            if available_left and available_right:
                options.extend(("both", "both"))
                if frame_count == 1:
                    options.extend(("full", "full", "full"))
            mode = options[_rand(rng, 0, len(options) - 1)] if options else "equal"
        else:
            mode = "equal" if rng.random() < .27 else "narrow"
        motif_left, motif_right = _motif_bounds(rng, left, fw, territory, mode)
        mask = _mask(rng, mh, motif_right - motif_left + 1)
        objects.append((top, left, fh, fw, top - gap - mask.shape[0], motif_left, mask))

    content_height = max(top + fh + mask.shape[0] for top, _, fh, _, _, _, mask in objects)
    lower_margin = _rand(rng, 6, 11) if roomy else _rand(rng, 0, 4)
    height = min(30, content_height + lower_margin)
    if height < content_height:
        raise AssertionError("vertical layout exceeds ARC limit")
    background, frame_color, motif_color = _palette(rng, previous_palette)
    inp = np.full((height, width), background, dtype=np.int8)
    for top, left, fh, fw, _, _, _ in objects:
        bottom, right = top + fh - 1, left + fw - 1
        inp[top, left:right + 1] = frame_color
        inp[bottom, left:right + 1] = frame_color
        inp[top:bottom + 1, left] = frame_color
        inp[top:bottom + 1, right] = frame_color
    out = inp.copy()
    for top, left, fh, fw, motif_top, motif_left, mask in objects:
        rr, cc = np.nonzero(mask)
        rr, cc = rr + motif_top, cc + motif_left
        assert np.all(rr >= 0) and np.all(inp[rr, cc] == background)
        inp[rr, cc] = motif_color
        land = rr - motif_top + top + fh
        assert np.all(land < height) and np.all(out[land, cc] == background)
        out[land, cc] = motif_color
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = _rand(rng, 1, 4)
        num_train = _rand(rng, max(2, num_test + 1), 10)
    elif num_train is None:
        num_train = _rand(rng, max(2, num_test + 1), 10)
    elif num_test is None:
        num_test = _rand(rng, 1, min(4, num_train - 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The only learned convention is the fixed downward, frame-anchored move.
    # Palette, geometry, and spacing are visible pair properties.
    train_frame_counts = [1, 2] + [int(rng.choice((1, 2, 3), p=(.35, .45, .20)))
                                   for _ in range(num_train - 2)]
    rng.shuffle(train_frame_counts)
    test_frame_counts = [int(rng.choice((1, 2, 3), p=(.43, .39, .18)))
                         for _ in range(num_test)]
    train_overhang = rng.random() < .53
    train_overhang_index = _rand(rng, 0, num_train - 1) if train_overhang else -1
    contained_tests = rng.random() < .30
    test_overhang = [bool(rng.random() < .52) and not contained_tests
                     for _ in range(num_test)]
    if not contained_tests and not train_overhang and rng.random() < .58 and not any(test_overhang):
        test_overhang[_rand(rng, 0, num_test - 1)] = True
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_palette = None
    for split, counts in (("train", train_frame_counts), ("test", test_frame_counts)):
        for index, frame_count in enumerate(counts):
            forced = (index == train_overhang_index if split == "train"
                      else test_overhang[index])
            for _ in range(60):
                pair_seed = int(rng.integers(0, 2**63, dtype=np.int64))
                pair_rng = np.random.default_rng(pair_seed)
                pair = _make_pair(pair_rng, frame_count, forced,
                                  bool(rng.random() < .48), bool(rng.random() < .10),
                                  previous_palette)
                inp = pair["input"]
                key = (inp.shape, inp.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    colors = _roles(pair)
                    previous_palette = colors
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
    return puzzle


def _roles(pair):
    inp, out = pair["input"], pair["output"]
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    others = [int(c) for c in colors if c != background]
    stable = [c for c in others if np.array_equal(inp == c, out == c)]
    assert len(stable) == 1
    return background, stable[0], next(c for c in others if c != stable[0])


def _frames(grid, color, background):
    mask = grid == color
    visited = np.zeros(grid.shape, dtype=bool)
    frames = []
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells = set()
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            assert bottom - top >= 2 and right - left >= 2
            border = {(r, c) for r in range(top, bottom + 1)
                      for c in range(left, right + 1)
                      if r in (top, bottom) or c in (left, right)}
            assert cells == border
            assert np.all(grid[top + 1:bottom, left + 1:right] == background)
            frames.append((top, bottom, left, right))
    frames.sort(key=lambda f: f[2])
    assert 1 <= len(frames) <= 3
    for first, second in zip(frames, frames[1:]):
        assert first[3] + 1 < second[2]
    return frames


def _analyze(inp, out):
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert len(np.unique(inp)) == 3
    background, frame_color, motif_color = _roles({"input": inp, "output": out})
    candidates = []
    for candidate in (frame_color, motif_color):
        try:
            found = _frames(inp, candidate, background)
        except AssertionError:
            continue
        candidates.append((candidate, found))
    assert len(candidates) == 1 and candidates[0][0] == frame_color
    frames = _frames(inp, frame_color, background)
    coords = np.argwhere(inp == motif_color)
    expected = inp.copy()
    expected[expected == motif_color] = background
    groups = []
    for index, (top, bottom, left, right) in enumerate(frames):
        if index == 0:
            lower = 0
        else:
            gap = left - frames[index - 1][3] - 1
            lower = left - (gap - 1) // 2
        if index == len(frames) - 1:
            upper = inp.shape[1] - 1
        else:
            gap = frames[index + 1][2] - right - 1
            upper = right + (gap - 1) // 2
        group = coords[(coords[:, 1] >= lower) & (coords[:, 1] <= upper)]
        assert len(group) > 0
        assert int(group[:, 0].max()) < top
        assert int(group[:, 1].min()) <= right and int(group[:, 1].max()) >= left
        groups.append(group)
        rr = group[:, 0] - int(group[:, 0].min()) + bottom + 1
        cc = group[:, 1]
        assert np.all(rr < inp.shape[0]) and np.all(expected[rr, cc] == background)
        expected[rr, cc] = motif_color
    assert sum(map(len, groups)) == len(coords)
    assert np.array_equal(out, expected)
    return background, frame_color, motif_color, frames, groups


def _predict_from_training(training, test_input):
    # Read the downward, bottom-anchored move from demonstrations. The test
    # output and generation parameters are unavailable to this reconstruction.
    for pair in training:
        _analyze(pair["input"], pair["output"])
    inp = np.array(test_input, copy=True)
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    candidates = []
    for color in colors:
        if color == background:
            continue
        try:
            found = _frames(inp, int(color), background)
        except AssertionError:
            continue
        candidates.append((int(color), found))
    assert len(candidates) == 1
    frame_color, frames = candidates[0]
    motif_color = next(int(c) for c in colors if c not in (background, frame_color))
    coords = np.argwhere(inp == motif_color)
    out = inp.copy()
    out[out == motif_color] = background
    assigned = 0
    for index, (top, bottom, left, right) in enumerate(frames):
        previous_right = frames[index - 1][3] if index else -1
        next_left = frames[index + 1][2] if index + 1 < len(frames) else inp.shape[1]
        lower = 0 if index == 0 else left - (left - previous_right - 2) // 2
        upper = (inp.shape[1] - 1 if index + 1 == len(frames)
                 else right + (next_left - right - 2) // 2)
        group = coords[(coords[:, 1] >= lower) & (coords[:, 1] <= upper)]
        assert len(group) and int(group[:, 0].max()) < top
        assigned += len(group)
        rr = group[:, 0] - int(group[:, 0].min()) + bottom + 1
        cc = group[:, 1]
        assert np.all(rr < inp.shape[0]) and np.all(out[rr, cc] == background)
        out[rr, cc] = motif_color
    assert assigned == len(coords)
    return out


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, palettes = set(), set(), set()
    train_frame_counts = set()
    independent_landing_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert tuple("".join(str(int(v)) for v in row) for row in inp) not in _SOURCE_INPUTS
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            background, frame_color, motif_color, frames, _ = _analyze(inp, out)
            palettes.add((background, frame_color, motif_color))
            if split == "train":
                train_frame_counts.add(len(frames))
                if len({bottom for _, bottom, _, _ in frames}) > 1:
                    independent_landing_demo = True
    assert {1, 2} <= train_frame_counts
    assert independent_landing_demo
    assert len(palettes) >= 2
    training = [{role: pair[role].copy() for role in ("input", "output")}
                for pair in puzzle["train"]]
    for pair in puzzle["test"]:
        prediction = _predict_from_training(training, pair["input"].copy())
        assert np.array_equal(prediction, pair["output"])


def _same_puzzle(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa[role], pb[role])
        for pa, pb in zip(a[split], b[split]) for role in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, frame_counts, palettes, layouts, first_inputs = set(), set(), set(), set(), set()
    train_overhang_puzzles = 0
    contained_test_puzzles = 0
    three_frame_pairs = 0
    single_row_train = 0
    compact = 0
    spacious = 0
    aligned_train = staggered_train = aligned_test = staggered_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first = puzzle["train"][0]["input"]
        first_inputs.add((first.shape, first.tobytes()))
        has_train_overhang = False
        all_test_contained = True
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, fc, mc, frames, groups = _analyze(inp, out)
                sizes.add(inp.shape)
                palettes.add((bg, fc, mc))
                frame_counts.add(len(frames))
                three_frame_pairs += len(frames) == 3
                layouts.add(tuple((t, b, l, r) for t, b, l, r in frames))
                compact += inp.shape[0] <= 13
                spacious += inp.shape[0] >= 22
                if len(frames) > 1:
                    aligned = len({f[0] for f in frames}) == 1
                    if split == "train":
                        aligned_train += aligned
                        staggered_train += not aligned
                    else:
                        aligned_test += aligned
                        staggered_test += not aligned
                for frame, group in zip(frames, groups):
                    overhang = int(group[:, 1].min()) < frame[2] or int(group[:, 1].max()) > frame[3]
                    if split == "train":
                        has_train_overhang |= overhang
                        single_row_train += int(group[:, 0].min()) == int(group[:, 0].max())
                    else:
                        all_test_contained &= not overhang
        train_overhang_puzzles += has_train_overhang
        contained_test_puzzles += all_test_contained
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 35 and len(layouts) >= 100
    assert len(first_inputs) == 42 and len(palettes) >= 30
    assert frame_counts == {1, 2, 3} and three_frame_pairs >= 10
    assert train_overhang_puzzles >= 8 and contained_test_puzzles >= 4
    assert single_row_train >= 20 and compact >= 15 and spacious >= 15
    assert min(aligned_train, staggered_train, aligned_test, staggered_test) >= 8
    for index, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                      (None, 4), (2, None), (10, None), (5, 2))):
        puzzle = generate_puzzle(1000 + index, nt, ne)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(1000 + index, nt, ne))
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(7, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
