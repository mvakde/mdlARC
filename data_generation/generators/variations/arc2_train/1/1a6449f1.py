import numpy as np


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_optional_count(name: str, value, low: int, high: int):
    if value is None:
        return None
    if not _is_integer(value):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count("num_train", num_train, 2, 10)
    num_test = _validate_optional_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        chosen_test = int(rng.integers(1, 5))
        chosen_train = int(rng.integers(chosen_test + 1, 11))
        return chosen_train, chosen_test

    if num_train is None:
        assert num_test is not None
        chosen_train = int(rng.integers(num_test + 1, 11))
        return chosen_train, num_test

    assert num_train is not None and num_test is None
    largest_test = min(4, num_train - 1)
    if largest_test < 1:
        raise ValueError("no legal test count exists for num_train")
    chosen_test = int(rng.integers(1, largest_test + 1))
    return num_train, chosen_test


def _frame_interior_shape(frame: tuple[int, int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1, _ = frame
    return r1 - r0 - 1, c1 - c0 - 1


def _frame_area(frame: tuple[int, int, int, int, int]) -> int:
    height, width = _frame_interior_shape(frame)
    return height * width


def _crop_frame(
    grid: np.ndarray,
    frame: tuple[int, int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1, _ = frame
    return grid[r0 + 1:r1, c0 + 1:c1].copy()


def _find_frames(
    grid: np.ndarray,
    background: int,
) -> list[tuple[int, int, int, int, int]]:
    """
    Return all complete monochromatic non-background rectangular frames.

    Frame tuples contain inclusive outer bounds:
        (top, left, bottom, right, color)
    """
    height, width = grid.shape
    frames: list[tuple[int, int, int, int, int]] = []

    for color in range(10):
        if color == background:
            continue

        mask = grid == color
        row_prefix = np.concatenate(
            (
                np.zeros((height, 1), dtype=np.int16),
                np.cumsum(mask, axis=1, dtype=np.int16),
            ),
            axis=1,
        )
        col_prefix = np.concatenate(
            (
                np.zeros((1, width), dtype=np.int16),
                np.cumsum(mask, axis=0, dtype=np.int16),
            ),
            axis=0,
        )

        for r0 in range(height - 2):
            for r1 in range(r0 + 2, height):
                common_columns = np.flatnonzero(mask[r0] & mask[r1])
                count = int(common_columns.size)
                if count < 2:
                    continue

                for i in range(count - 1):
                    c0 = int(common_columns[i])
                    for j in range(i + 1, count):
                        c1 = int(common_columns[j])
                        if c1 - c0 < 2:
                            continue

                        frame_width = c1 - c0 + 1
                        frame_height = r1 - r0 + 1

                        top_count = (
                            row_prefix[r0, c1 + 1] - row_prefix[r0, c0]
                        )
                        if int(top_count) != frame_width:
                            continue

                        bottom_count = (
                            row_prefix[r1, c1 + 1] - row_prefix[r1, c0]
                        )
                        if int(bottom_count) != frame_width:
                            continue

                        left_count = (
                            col_prefix[r1 + 1, c0] - col_prefix[r0, c0]
                        )
                        if int(left_count) != frame_height:
                            continue

                        right_count = (
                            col_prefix[r1 + 1, c1] - col_prefix[r0, c1]
                        )
                        if int(right_count) != frame_height:
                            continue

                        frames.append((r0, c0, r1, c1, color))

    frames.sort()
    return frames


def _frames_disjoint(
    first: tuple[int, int, int, int, int],
    second: tuple[int, int, int, int, int],
) -> bool:
    return (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def _has_k_disjoint_frames(
    frames: list[tuple[int, int, int, int, int]],
    required: int,
) -> bool:
    if required <= 0:
        return True
    if len(frames) < required:
        return False

    ordered = sorted(frames, key=_frame_area, reverse=True)
    chosen: list[tuple[int, int, int, int, int]] = []

    def search(start: int) -> bool:
        if len(chosen) == required:
            return True

        still_needed = required - len(chosen)
        if len(ordered) - start < still_needed:
            return False

        for index in range(start, len(ordered)):
            candidate = ordered[index]
            if all(_frames_disjoint(candidate, item) for item in chosen):
                chosen.append(candidate)
                if search(index + 1):
                    return True
                chosen.pop()
        return False

    return search(0)


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    maximum = int(counts.max())
    winners = np.flatnonzero(counts == maximum)
    assert winners.size == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    background: int,
) -> tuple[
    list[tuple[int, int, int, int, int]],
    tuple[int, int, int, int, int],
]:
    frames = _find_frames(input_grid, background)
    assert len(frames) >= 2, "input must contain at least two frames"

    largest_area = max(_frame_area(frame) for frame in frames)
    winners = [
        frame for frame in frames
        if _frame_area(frame) == largest_area
    ]
    assert len(winners) == 1, "largest frame must be unique"

    winner = winners[0]
    expected = _crop_frame(input_grid, winner)
    assert np.array_equal(output_grid, expected), (
        "output must be the unchanged interior of the unique largest frame"
    )
    return frames, winner


def _rectangles_separated(a, b):
    r, c, h, w = a
    s, d, j, k = b
    return r + h < s or s + j < r or c + w < d or d + k < c


def _axis_position(rng, maximum):
    if maximum == 0:
        return 0
    if rng.random() < 0.42:
        return int(rng.choice((0, maximum)))
    return int(rng.integers(0, maximum + 1))


def _sample_target(rng, mode, requested_shape=None):
    if requested_shape == "square":
        side = int(rng.integers(8, 13))
        return side, side
    if mode == "perimeter":
        h = int(rng.integers(6, 10))
        w = int(rng.integers(7, 11))
    elif mode == "slender":
        h = int(rng.integers(3, 6))
        w = int(rng.integers(12, 18))
    else:
        kind = int(rng.integers(0, 4))
        if kind == 0:
            h = int(rng.integers(4, 12))
            w = h
        elif kind == 1:
            h = int(rng.integers(4, 11))
            w = h + int(rng.integers(1, 6))
        elif kind == 2:
            w = int(rng.integers(4, 11))
            h = w + int(rng.integers(1, 6))
        else:
            h = int(rng.integers(3, 6))
            w = int(rng.integers(10, 18))
    if requested_shape == "landscape":
        return min(h, w), max(h, w) + int(h == w)
    if rng.random() < 0.5:
        return w, h
    return h, w


def _sample_challenger(rng, target, mode):
    th, tw = target
    area = th * tw
    shorter = min(target)
    perimeter = 2 * (th + tw)
    candidates = []
    for h in range(2, 20):
        for w in range(2, 20):
            a = h * w
            if a >= area:
                continue
            ratio = a / area
            if mode == "slender":
                valid = min(h, w) > shorter and 0.70 <= ratio <= 0.98
            elif mode == "perimeter":
                valid = 2 * (h + w) > perimeter and 0.30 <= ratio <= 0.90
            elif mode == "near":
                valid = 0.80 <= ratio <= 0.98
            elif mode == "dimension":
                valid = (h > th or w > tw) and 0.35 <= ratio <= 0.90
            else:
                valid = 0.20 <= ratio <= 0.65
            if valid:
                candidates.append((h, w))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _place_frames(rng, sizes, extra_count):
    (h0, w0), (h1, w1) = sizes[:2]
    configurations = []
    for gap in (1, 2, 3):
        if h0 + h1 + gap <= 30 and max(w0, w1) <= 30:
            configurations.append(("vertical", gap))
        if w0 + w1 + gap <= 30 and max(h0, h1) <= 30:
            configurations.append(("horizontal", gap))
    if not configurations:
        return None
    orientation, gap = configurations[int(rng.integers(len(configurations)))]
    if orientation == "vertical":
        min_h, min_w = h0 + h1 + gap, max(w0, w1)
    else:
        min_h, min_w = max(h0, h1), w0 + w1 + gap
    min_h, min_w = max(13, min_h), max(13, min_w)
    if min_h > 30 or min_w > 30:
        return None
    bonus = 4 if extra_count else 0
    canvas_h = int(rng.integers(min(30, min_h + bonus), min(30, min_h + bonus + 8) + 1))
    canvas_w = int(rng.integers(min(30, min_w + bonus), min(30, min_w + bonus + 8) + 1))
    if orientation == "vertical":
        start = _axis_position(rng, canvas_h - h0 - h1 - gap)
        if rng.random() < 0.5:
            rows = (start, start + h0 + gap)
        else:
            rows = (start + h1 + gap, start)
        placed = [(rows[0], _axis_position(rng, canvas_w - w0), h0, w0),
                  (rows[1], _axis_position(rng, canvas_w - w1), h1, w1)]
    else:
        start = _axis_position(rng, canvas_w - w0 - w1 - gap)
        if rng.random() < 0.5:
            cols = (start, start + w0 + gap)
        else:
            cols = (start + w1 + gap, start)
        placed = [(_axis_position(rng, canvas_h - h0), cols[0], h0, w0),
                  (_axis_position(rng, canvas_h - h1), cols[1], h1, w1)]
    for h, w in sizes[2:]:
        positions = [(r, c, h, w)
                     for r in range(canvas_h - h + 1)
                     for c in range(canvas_w - w + 1)]
        rng.shuffle(positions)
        found = next((p for p in positions if all(_rectangles_separated(p, x) for x in placed)), None)
        if found is None:
            return None
        placed.append(found)
    return canvas_h, canvas_w, placed


def _fill_interior(rng, interior, background, palette, mode, density):
    h, w = interior.shape
    interior[:] = background
    color = int(rng.choice(palette))
    if mode == "uniform":
        if rng.random() < 0.55:
            interior[:] = color
    elif mode == "objects":
        for _ in range(int(rng.integers(1, 4))):
            rh = int(rng.integers(1, min(4, h) + 1))
            rw = int(rng.integers(1, min(4, w) + 1))
            r = int(rng.integers(0, h - rh + 1))
            c = int(rng.integers(0, w - rw + 1))
            interior[r:r + rh, c:c + rw] = int(rng.choice(palette))
    elif mode == "stripes":
        step = int(rng.integers(2, 5))
        if rng.random() < 0.5:
            interior[int(rng.integers(step))::step, :] = color
        else:
            interior[:, int(rng.integers(step))::step] = color
    elif mode == "repeat":
        step = int(rng.integers(2, 5))
        for r in range(h):
            for c in range(w):
                if (r + c) % step == 0:
                    interior[r, c] = color
    else:
        subset_size = int(rng.integers(1, min(len(palette), 8) + 1))
        colors = rng.choice(palette, size=subset_size, replace=False)
        mask = rng.random((h, w)) < density
        interior[mask] = rng.choice(colors, size=int(mask.sum()))


def _draw_outline(grid, rectangle, color):
    r, c, h, w = rectangle
    grid[r, c:c + w] = color
    grid[r + h - 1, c:c + w] = color
    grid[r:r + h, c] = color
    grid[r:r + h, c + w - 1] = color


def _try_partial(rng, grid, placed, palette, background):
    h, w = grid.shape
    ph, pw = int(rng.integers(4, 8)), int(rng.integers(4, 9))
    choices = [(r, c, ph, pw) for r in range(h - ph + 1)
               for c in range(w - pw + 1)]
    rng.shuffle(choices)
    rect = next((x for x in choices if all(_rectangles_separated(x, p) for p in placed)), None)
    if rect is None:
        return
    color = int(rng.choice(palette))
    r, c, _, _ = rect
    _draw_outline(grid, rect, color)
    grid[r, c + pw // 2] = background


def _try_nested(rng, grid, target, palette, background):
    r, c, h, w = target
    if h < 9 or w < 9:
        return
    nh = int(rng.integers(3, min(6, h - 4) + 1))
    nw = int(rng.integers(3, min(6, w - 4) + 1))
    nr = int(rng.integers(r + 2, r + h - nh - 1))
    nc = int(rng.integers(c + 2, c + w - nw - 1))
    color = int(rng.choice(palette))
    grid[nr + 1:nr + nh - 1, nc + 1:nc + nw - 1] = background
    _draw_outline(grid, (nr, nc, nh, nw), color)


def _generate_pair(rng, regime, palette_order, mode, requested_shape=None):
    for _ in range(160):
        background = int(rng.choice(palette_order))
        palette = tuple(int(x) for x in palette_order if int(x) != background)
        target = _sample_target(rng, mode, requested_shape)
        challenger_mode = (mode if requested_shape != "square" and mode in ("perimeter", "slender")
                           else str(rng.choice(("near", "dimension", "comfortable"))))
        challenger = _sample_challenger(rng, target, challenger_mode)
        if challenger is None:
            continue
        frame_count = int(rng.choice((2, 2, 3, 3, 4, 5)))
        sizes = [(target[0] + 2, target[1] + 2),
                 (challenger[0] + 2, challenger[1] + 2)]
        for _ in range(frame_count - 2):
            upper = min(6, max(3, int(np.sqrt(target[0] * target[1]))))
            dh = int(rng.integers(2, upper + 1))
            dw = int(rng.integers(2, upper + 1))
            if dh * dw >= target[0] * target[1]:
                dh, dw = 2, 2
            sizes.append((dh + 2, dw + 2))
        placed_result = _place_frames(rng, sizes, frame_count - 2)
        if placed_result is None:
            continue
        canvas_h, canvas_w, placed = placed_result
        density = float(rng.uniform(0.015, 0.41))
        grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
        noise = rng.random(grid.shape) < density
        grid[noise] = rng.choice(palette, size=int(noise.sum()))
        colors = [int(rng.choice(palette))]
        colors.append(colors[0] if rng.random() < 0.34 else int(rng.choice(palette)))
        colors.extend(int(rng.choice(palette)) for _ in range(frame_count - 2))
        modes = ("speckle", "restricted", "uniform", "objects", "stripes", "repeat")
        for index, (rect, color) in enumerate(zip(placed, colors)):
            r, c, h, w = rect
            content_mode = str(rng.choice(modes))
            content_density = float(rng.uniform(0.015, 0.63))
            interior = grid[r + 1:r + h - 1, c + 1:c + w - 1]
            _fill_interior(rng, interior, background, palette, content_mode, content_density)
            _draw_outline(grid, rect, color)
        if rng.random() < 0.35:
            _try_partial(rng, grid, placed, palette, background)
        if rng.random() < 0.30:
            _try_nested(rng, grid, placed[0], palette, background)
        planned = [(r, c, r + h - 1, c + w - 1, col)
                   for (r, c, h, w), col in zip(placed, colors)]
        found = _find_frames(grid, background)
        if not set(planned).issubset(found):
            continue
        biggest = max(_frame_area(f) for f in found)
        winners = [f for f in found if _frame_area(f) == biggest]
        if len(winners) != 1 or winners[0] != planned[0]:
            continue
        counts = np.bincount(grid.ravel(), minlength=10)
        if int(np.argmax(counts)) != background or int(np.sum(counts == counts[background])) != 1:
            continue
        return grid, _crop_frame(grid, planned[0])
    raise RuntimeError("could not construct an unambiguous pair")


_SOURCE_INPUT_HASHES = {
    ((21, 23), "37ea0abb6f7fc156a396b57741e5f42565d7a6b0e0c5b034c65fc99dc5e47f2e"),
    ((19, 13), "9a725e70d7f37d980fcf90f48ec0a6dd9d6c01212d464cdb643dd7e24b0357ee"),
    ((22, 21), "13ad3eeef5944f11c755e4db92d9f7e216d2f73cd372a9f232ed8d9e9186af5c"),
    ((27, 25), "70c6f74b8841893bde5fb8144128aa83a36714f4a992f1fac9906f70752e28d5"),
}


def validate_generated_puzzle(puzzle: dict) -> None:
    import hashlib
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_min_side_evidence = False
    train_perimeter_evidence = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for key in ("input", "output"):
                g = pair[key]
                assert isinstance(g, np.ndarray) and g.ndim == 2
                assert np.issubdtype(g.dtype, np.integer)
                assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert int(g.min()) >= 0 and int(g.max()) <= 9
            a, b = pair["input"], pair["output"]
            key = (a.shape, a.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            assert (a.shape, hashlib.sha256(a.astype(np.int8).tobytes()).hexdigest()) not in _SOURCE_INPUT_HASHES
            assert not np.array_equal(a, b)
            bg = _dominant_color(a)
            frames, winner = _analyze_pair(a, b, bg)
            assert _has_k_disjoint_frames(frames, 2)
            if section == "train":
                wh, ww = _frame_interior_shape(winner)
                for f in frames:
                    if f == winner or _frame_area(f) >= _frame_area(winner):
                        continue
                    h, w = _frame_interior_shape(f)
                    train_min_side_evidence |= min(wh, ww) < min(h, w)
                    train_perimeter_evidence |= wh + ww < h + w
    assert train_min_side_evidence, "training must demonstrate a slender area winner"
    assert train_perimeter_evidence, "training must demonstrate area over perimeter"


def _puzzles_equal(a, b):
    for section in ("train", "test"):
        if len(a[section]) != len(b[section]):
            return False
        for x, y in zip(a[section], b[section]):
            for key in ("input", "output"):
                if not np.array_equal(x[key], y[key]):
                    return False
    return True


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette_order = tuple(int(x) for x in rng.permutation(10))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    source_like_shift = bool(rng.random() < 0.18)
    puzzle = {"train": [], "test": []}
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train" and index == 0:
                mode = "perimeter"
            elif section == "train" and index == 1:
                mode = "slender"
            else:
                mode = str(rng.choice(("general", "general", "slender", "perimeter")))
            requested_shape = ("landscape" if source_like_shift and section == "train"
                               else "square" if source_like_shift and section == "test" and index == 0
                               else None)
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    a, b = _generate_pair(pair_rng, section, palette_order, mode, requested_shape)
                except RuntimeError:
                    continue
                puzzle[section].append({"input": a, "output": b})
                break
            else:
                raise RuntimeError("could not generate puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    a = generate_puzzle(91357, 6, 3)
    assert _puzzles_equal(a, generate_puzzle(91357, 6, 3))
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    assert not _puzzles_equal(generate_puzzle(1001, 4, 1), generate_puzzle(1002, 4, 1))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(50000 + i, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(60000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {args}")
    train_counts, test_counts, canvas, outputs, backgrounds, colors = set(), set(), set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    edge = {"train": 0, "test": 0}
    near = 0
    sparse = 0
    dense = 0
    restricted = 0
    nested = 0
    same_color = 0
    total = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        for section in ("train", "test"):
            for pair in p[section]:
                a, b = pair["input"], pair["output"]
                bg = _dominant_color(a)
                frames, winner = _analyze_pair(a, b, bg)
                backgrounds.add(bg)
                colors.add(winner[4])
                canvas.add(a.shape)
                outputs.add(b.shape)
                orientations[section].add("square" if b.shape[0] == b.shape[1] else "portrait" if b.shape[0] > b.shape[1] else "landscape")
                edge[section] += int(winner[0] == 0 or winner[1] == 0 or winner[2] == a.shape[0] - 1 or winner[3] == a.shape[1] - 1)
                ratios = sorted((_frame_area(f) for f in frames), reverse=True)
                near += int(ratios[1] / ratios[0] >= 0.80)
                density = float(np.mean(a != bg))
                sparse += int(density < 0.25)
                dense += int(density > 0.40)
                restricted += int(np.unique(b).size <= 3)
                nested += int(any(f != winner and winner[0] < f[0] and f[2] < winner[2] and winner[1] < f[1] and f[3] < winner[3] for f in frames))
                same_color += int(any(f != winner and f[4] == winner[4] for f in frames))
                total += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 7 and len(colors) >= 7
    assert len(canvas) >= 30 and len(outputs) >= 20
    assert all(len(orientations[s]) == 3 for s in orientations)
    assert all(edge[s] > 0 for s in edge)
    assert near > 0 and sparse > 0 and dense > 0 and restricted > 0 and nested > 0 and same_color > 0
    assert total > 150
