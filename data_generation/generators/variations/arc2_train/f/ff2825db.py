import numpy as np


_PALETTE = np.array(
    [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    dtype=np.int8,
)
_INTERIOR_HEIGHT = 7
_INTERIOR_WIDTH = 8

_SOURCE_PAIR_DIGITS = frozenset(
    {
        (
            "1122334455"
            "4444444444"
            "4000000004"
            "4000000504"
            "4000550004"
            "4000511004"
            "4050500004"
            "4010000004"
            "4000000004"
            "4444444444",
            "1122334455"
            "5555555555"
            "5000000005"
            "5055555505"
            "5050000505"
            "5050000505"
            "5055555505"
            "5000000005"
            "5000000005"
            "5555555555",
        ),
        (
            "1122334455"
            "2222222222"
            "2003330042"
            "2033033432"
            "2030304332"
            "2034000342"
            "2030333402"
            "2033343332"
            "2000000042"
            "2222222222",
            "1122334455"
            "3333333333"
            "3033333333"
            "3030000033"
            "3030000033"
            "3030000033"
            "3030000033"
            "3033333333"
            "3000000003"
            "3333333333",
        ),
        (
            "1122334455"
            "1111111111"
            "1000000021"
            "1002000001"
            "1000000401"
            "1400040001"
            "1000000001"
            "1004004001"
            "1000000021"
            "1111111111",
            "1122334455"
            "4444444444"
            "4000000004"
            "4000000004"
            "4444444404"
            "4400000404"
            "4400000404"
            "4444444404"
            "4000000004"
            "4444444444",
        ),
        (
            "1122334455"
            "2222222222"
            "2000000002"
            "2000000002"
            "2000100002"
            "2000400002"
            "2000040002"
            "2000000002"
            "2000000002"
            "2222222222",
            "1122334455"
            "4444444444"
            "4000000004"
            "4000000004"
            "4000000004"
            "4000440004"
            "4000440004"
            "4000000004"
            "4000000004"
            "4444444444",
        ),
        (
            "1122334455"
            "5555555555"
            "5000000005"
            "5000000205"
            "5001000005"
            "5000001005"
            "5002002005"
            "5000000005"
            "5000000005"
            "5555555555",
            "1122334455"
            "2222222222"
            "2000000002"
            "2002222202"
            "2002000202"
            "2002000202"
            "2002222202"
            "2000000002"
            "2000000002"
            "2222222222",
        ),
    }
)


def _validate_count_value(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
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
    num_train = _validate_count_value("num_train", num_train, 2, 10)
    num_test = _validate_count_value("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_test is None:
        largest_test_count = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test_count + 1))
    elif num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    if num_train <= num_test:
        raise ValueError("no legal count combination exists")

    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The strip is a copyable legend, not a map from a color to a role.
    if rng.random() < 0.16:
        return {"background": 0, "colors": (1, 2, 3, 4, 5)}
    background = 0 if rng.random() < 0.72 else int(rng.integers(6, 10))
    choices = [c for c in range(1, 10) if c != background]
    colors = tuple(int(c) for c in rng.permutation(choices)[:7])
    return {"background": background, "colors": colors}


def _strip(rng, colors, roles, count):
    selected = list(roles)
    others = [c for c in colors if c not in selected]
    extras = rng.choice(others, size=count - 3, replace=False)
    selected.extend(int(c) for c in np.atleast_1d(extras))
    selected = [c for c in colors if c in selected]
    return np.repeat(np.asarray(selected, dtype=np.int8), 2)


def _box_size(rng, ih, iw, first_dense):
    lo = 4 if first_dense else 2
    for _ in range(12):
        kind = int(rng.integers(3))
        if kind == 0:
            side = int(rng.integers(lo, min(ih, iw) + 1))
            bh, bw = side, side
        elif kind == 1:
            bh = int(rng.integers(lo, ih + 1))
            widths = [w for w in range(lo, iw + 1) if w > bh]
            if not widths:
                continue
            bw = int(rng.choice(widths))
        else:
            bw = int(rng.integers(lo, iw + 1))
            heights = [h for h in range(lo, ih + 1) if h > bw]
            if not heights:
                continue
            bh = int(rng.choice(heights))
        if bh * bw < ih * iw:
            return bh, bw
    return lo, lo


def _signal_points(rng, bh, bw, count, style, force_interior):
    corners = {(0, 0), (bh - 1, bw - 1)}
    points = set(corners)
    if force_interior:
        points.add((int(rng.integers(1, bh - 1)), int(rng.integers(1, bw - 1))))
    if style == "perimeter":
        candidates = [(r, c) for r in range(bh) for c in range(bw)
                      if r in (0, bh - 1) or c in (0, bw - 1)]
        for index in rng.permutation(len(candidates)):
            if len(points) == count:
                break
            points.add(candidates[int(index)])
    elif style == "cluster" and len(points) < count:
        # Grow a connected patch, while the diagonal anchors retain the extrema.
        center = (int(rng.integers(1, max(2, bh - 1))),
                  int(rng.integers(1, max(2, bw - 1))))
        patch = {center}
        while len(points | patch) < count:
            frontier = {(r + dr, c + dc) for r, c in patch
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= r + dr < bh and 0 <= c + dc < bw
                        and (r + dr, c + dc) not in patch}
            if not frontier:
                break
            patch.add(sorted(frontier)[int(rng.integers(len(frontier)))])
        points.update(patch)
    if len(points) < count:
        candidates = [(r, c) for r in range(bh) for c in range(bw)
                      if (r, c) not in points]
        for index in rng.permutation(len(candidates)):
            points.add(candidates[int(index)])
            if len(points) == count:
                break
    assert len(points) == count, (bh, bw, count, style, force_interior, len(points))
    return points


def _distractor_points(rng, ih, iw, box, signals, desired):
    r0, r1, c0, c1 = box
    inside = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
              if (r, c) not in signals]
    outside = [(r, c) for r in range(ih) for c in range(iw)
               if not (r0 <= r <= r1 and c0 <= c <= c1)]
    modes = []
    if inside:
        modes.append("inside")
    if outside:
        modes.append("outside")
    if inside and outside and desired >= 2:
        modes.append("mixed")
    mode = modes[int(rng.integers(len(modes)))]
    cap = len(inside) if mode == "inside" else len(outside) if mode == "outside" else len(inside) + len(outside)
    count = min(desired, cap)
    if mode == "mixed":
        count = max(2, count)
    if mode == "inside":
        picked = rng.choice(len(inside), size=count, replace=False)
        return {inside[int(i)] for i in np.atleast_1d(picked)}
    if mode == "outside":
        # Often force a distractor extremum beyond the signal rectangle.
        first = None
        if rng.random() < 0.55:
            far = [p for p in outside if p[0] < r0 or p[0] > r1 or p[1] < c0 or p[1] > c1]
            first = far[int(rng.integers(len(far)))]
        chosen = {first} if first is not None else set()
        pool = [p for p in outside if p not in chosen]
        if count > len(chosen):
            indices = rng.choice(len(pool), size=count - len(chosen), replace=False)
            chosen.update(pool[int(i)] for i in np.atleast_1d(indices))
        return chosen
    a = int(rng.integers(1, count))
    a = min(a, len(inside))
    b = count - a
    if b > len(outside):
        b = len(outside)
        a = count - b
    chosen_inside = rng.choice(len(inside), size=a, replace=False)
    chosen_outside = rng.choice(len(outside), size=b, replace=False)
    return ({inside[int(i)] for i in np.atleast_1d(chosen_inside)} |
            {outside[int(i)] for i in np.atleast_1d(chosen_outside)})


def _solve_input(input_grid):
    height, width = input_grid.shape
    strip = input_grid[0]
    swatches = set(int(c) for c in strip)
    interior = input_grid[2:height - 1, 1:width - 1]
    bg_colors = set(int(c) for c in np.unique(interior)) - swatches
    if len(bg_colors) != 1:
        raise ValueError("ambiguous background")
    bg = next(iter(bg_colors))
    markers = list(set(int(c) for c in np.unique(interior)) - {bg})
    if len(markers) != 2:
        raise ValueError("expected two markers")
    counts = {c: int(np.count_nonzero(interior == c)) for c in markers}
    if counts[markers[0]] == counts[markers[1]]:
        raise ValueError("tied markers")
    signal = max(markers, key=lambda c: counts[c])
    coords = np.argwhere(interior == signal)
    r0, c0 = coords.min(axis=0) + (2, 1)
    r1, c1 = coords.max(axis=0) + (2, 1)
    output = np.full(input_grid.shape, bg, dtype=np.int8)
    output[0] = strip
    output[1, :] = signal
    output[-1, :] = signal
    output[1:, 0] = signal
    output[1:, -1] = signal
    output[r0, c0:c1 + 1] = signal
    output[r1, c0:c1 + 1] = signal
    output[r0:r1 + 1, c0] = signal
    output[r0:r1 + 1, c1] = signal
    return output


def generate_pair(rng, regime, parameters, roles, index=0):
    frame, signal, distractor = roles
    first_dense = regime == "source_train" and index == 0
    nswatches = int(rng.integers(4, min(7, len(parameters["colors"])) + 1))
    strip = _strip(rng, parameters["colors"], roles, nswatches)
    width = len(strip)
    height = int(rng.integers(8, 18))
    ih, iw = height - 3, width - 2
    bh, bw = _box_size(rng, ih, iw, first_dense)
    r0 = int(rng.integers(ih - bh + 1))
    c0 = int(rng.integers(iw - bw + 1))
    if rng.random() < 0.32:
        if rng.random() < 0.5:
            r0 = 0 if rng.random() < 0.5 else ih - bh
        else:
            c0 = 0 if rng.random() < 0.5 else iw - bw
    box = (r0, r0 + bh - 1, c0, c0 + bw - 1)
    area = bh * bw
    perimeter = 2 * (bh + bw) - 4
    if first_dense:
        if rng.random() < 0.55 and area > perimeter + 1:
            count = int(rng.integers(perimeter + 1, area))
        else:
            count = int(rng.integers(max(3, area // 3), max(4, area - 1)))
        style = "cluster" if rng.random() < 0.75 else "random"
    else:
        if regime == "source_train":
            style = ("perimeter" if index == 1 and rng.random() < 0.65 else
                     ("cluster" if rng.random() < 0.35 else "random"))
            density = int(rng.integers(3))
        else:
            style = "perimeter" if rng.random() < 0.48 else "random"
            density = 0 if index == 0 and rng.random() < 0.8 else int(rng.integers(3))
        if density == 0:
            count = int(rng.integers(2, min(6, area - 1) + 1))
        elif density == 1:
            count = int(rng.integers(3, min(area - 1, max(4, area // 2)) + 1))
        else:
            count = int(rng.integers(max(3, area // 2), area))
        if style == "perimeter":
            count = min(count, perimeter)
    count = min(count, area - 1)
    if first_dense:
        count = max(3, count)
    local = _signal_points(rng, bh, bw, count, style, first_dense)
    signals = {(r0 + r, c0 + c) for r, c in local}
    if count <= 8 and rng.random() < 0.38:
        wanted = count - 1
    else:
        wanted = int(rng.integers(1, min(9, count - 1) + 1))
    wanted = min(wanted, ih * iw - count - 1)
    distractors = _distractor_points(rng, ih, iw, box, signals, wanted)
    background = parameters["background"]
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[0] = strip
    input_grid[1, :] = frame
    input_grid[-1, :] = frame
    input_grid[1:, 0] = frame
    input_grid[1:, -1] = frame
    for r, c in signals:
        input_grid[r + 2, c + 1] = signal
    for r, c in distractors:
        input_grid[r + 2, c + 1] = distractor
    return input_grid, _solve_input(input_grid)


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _pair_digit_key(input_grid, output_grid):
    return ("".join(str(int(x)) for x in input_grid.flat),
            "".join(str(int(x)) for x in output_grid.flat))


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    h, w = inp.shape
    assert out.shape == inp.shape and 8 <= h <= 17 and 8 <= w <= 14 and w % 2 == 0
    strip = inp[0]
    assert np.array_equal(strip, out[0])
    assert np.all(strip[0::2] == strip[1::2])
    swatches = tuple(int(c) for c in strip[0::2])
    assert len(set(swatches)) == len(swatches) and len(swatches) >= 4
    frame = int(inp[1, 0])
    assert frame in swatches
    for boundary in (inp[1], inp[-1], inp[1:, 0], inp[1:, -1]):
        assert np.all(boundary == frame)
    interior = inp[2:-1, 1:-1]
    bg_colors = set(int(c) for c in np.unique(interior)) - set(swatches)
    assert len(bg_colors) == 1
    bg = next(iter(bg_colors))
    assert bg not in swatches
    markers = set(int(c) for c in np.unique(interior)) - {bg}
    assert len(markers) == 2 and frame not in markers and markers <= set(swatches)
    counts = {c: int(np.count_nonzero(interior == c)) for c in markers}
    signal = max(markers, key=lambda c: counts[c])
    distractor = next(c for c in markers if c != signal)
    assert counts[signal] > counts[distractor] >= 1
    coords = np.argwhere(interior == signal)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert r0 < r1 and c0 < c1
    expected = _solve_input(inp)
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    assert np.all(out[1] == signal) and np.all(out[-1] == signal)
    dcoords = np.argwhere(interior == distractor)
    inside = sum(r0 <= r <= r1 and c0 <= c <= c1 for r, c in dcoords)
    signal_interior = sum(r0 < r < r1 and c0 < c < c1 for r, c in coords)
    return {"shape": (h, w), "strip": swatches, "background": bg,
            "frame": frame, "signal": signal, "distractor": distractor,
            "signal_count": counts[signal], "distractor_count": counts[distractor],
            "box": (int(r0), int(r1), int(c0), int(c1)),
            "distractor_inside": inside, "signal_interior": signal_interior}


def _assert_train_regime(info):
    assert info["signal_count"] >= 2
    assert info["distractor_count"] >= 1


def _assert_test_regime(info):
    assert info["signal_count"] >= 2
    assert info["distractor_count"] >= 1


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert (pair["input"].shape != (10, 10) or
                    _pair_digit_key(pair["input"], pair["output"]) not in _SOURCE_PAIR_DIGITS)
            info = _analyze_pair(pair)
            (_assert_train_regime if split == "train" else _assert_test_regime)(info)
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    assert len({x["background"] for x in all_infos}) == 1
    assert len({x["frame"] for x in infos["train"]}) >= 2
    assert len({x["signal"] for x in infos["train"]}) >= 2
    train_roles = {name: {x[name] for x in infos["train"]}
                   for name in ("frame", "signal", "distractor")}
    assert (train_roles["frame"] & train_roles["signal"] or
            train_roles["frame"] & train_roles["distractor"] or
            train_roles["signal"] & train_roles["distractor"])
    assert infos["train"][0]["signal_interior"] > 0
    # The first demonstration makes erasure and frame recoloring observable.
    assert all(x["signal"] != x["frame"] for x in all_infos)
    for test_pair in puzzle["test"]:
        # This uses only the test input and the rule visible in training.
        assert np.array_equal(_solve_input(test_pair["input"]), test_pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    colors = parameters["colors"]
    first = tuple(int(c) for c in rng.choice(colors, size=3, replace=False))
    for _ in range(40):
        second = tuple(int(c) for c in rng.choice(colors, size=3, replace=False))
        changed_role = (second[0] in first[1:] or
                        second[1] in (first[0], first[2]) or
                        second[2] in first[:2])
        if second[0] != first[0] and second[1] != first[1] and changed_role:
            break
    else:
        second = (first[1], first[0], first[2])
    roles = [first, second]
    for _ in range(num_train + num_test - 2):
        roles.append(tuple(int(c) for c in rng.choice(colors, size=3, replace=False)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, role in enumerate(roles):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        for _ in range(40):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            inp, out = generate_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                     parameters, role, local_index)
            key = _grid_key(inp)
            if key in seen or (inp.shape == (10, 10) and _pair_digit_key(inp, out) in _SOURCE_PAIR_DIGITS):
                continue
            seen.add(key)
            puzzle[split].append({"input": inp, "output": out})
            break
        else:
            raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def _independent_prediction(grid):
    h, w = grid.shape
    legend = {int(c) for c in grid[0]}
    body = grid[2:h - 1, 1:w - 1]
    bg, = {int(c) for c in body.flat} - legend
    colors, amounts = np.unique(body, return_counts=True)
    marker_counts = {int(c): int(n) for c, n in zip(colors, amounts) if c != bg}
    assert len(marker_counts) == 2
    winner = max(marker_counts, key=marker_counts.get)
    ys, xs = np.where(body == winner)
    top, bottom = int(ys.min()) + 2, int(ys.max()) + 2
    left, right = int(xs.min()) + 1, int(xs.max()) + 1
    result = np.full((h, w), bg, dtype=np.int8)
    for row in range(h):
        for col in range(w):
            if row == 0:
                result[row, col] = grid[row, col]
            elif row in (1, h - 1) or col in (0, w - 1):
                result[row, col] = winner
            elif top <= row <= bottom and left <= col <= right and (
                row in (top, bottom) or col in (left, right)
            ):
                result[row, col] = winner
    return result


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(123, 6, 2), generate_puzzle(123, 6, 2))
    assert not _puzzles_equal(generate_puzzle(123, 6, 2), generate_puzzle(124, 6, 2))
    for t, q in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None), (7, 3)):
        p = generate_puzzle(1000 + (t or 0) * 10 + (q or 0), t, q)
        assert t is None or len(p["train"]) == t
        assert q is None or len(p["test"]) == q
        validate_generated_puzzle(p)
    for t, q in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 3), (4, 4)):
        try:
            generate_puzzle(77, t, q)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, layouts, sizes, roles = set(), set(), set(), set(), set()
    box_sizes = set()
    review_counts = {name: 0 for name in (
        "nonzero_puzzles", "source_strip", "two_large", "train_sparse",
        "train_dense", "test_dense", "combined", "role_reversals", "test_pairs",
        "square", "wide", "tall")}
    dense, perimeter_only, all_inside, mixed, all_outside, edges, near_ties = (0,) * 7
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        review_counts["nonzero_puzzles"] += _analyze_pair(p["train"][0])["background"] != 0
        train_infos = [_analyze_pair(pair) for pair in p["train"]]
        for split in ("train", "test"):
            for pair_index, pair in enumerate(p[split]):
                x = _analyze_pair(pair)
                assert np.array_equal(_independent_prediction(pair["input"]), pair["output"])
                r0, r1, c0, c1 = x["box"]
                bh, bw = r1 - r0 + 1, c1 - c0 + 1
                ih, iw = x["shape"][0] - 3, x["shape"][1] - 2
                box_sizes.add((bh, bw))
                layouts.add((split, x["shape"], x["box"]))
                sizes.add(x["shape"])
                roles.add((x["background"], x["strip"], x["frame"], x["signal"], x["distractor"]))
                review_counts["source_strip"] += x["strip"] == (1, 2, 3, 4, 5)
                review_counts["square"] += bh == bw
                review_counts["wide"] += bw > bh
                review_counts["tall"] += bh > bw
                review_counts["combined"] += (x["background"] != 0 and bw > bh and
                    x["distractor_inside"] == x["distractor_count"])
                review_counts["two_large"] += (x["signal_count"] == 2 and
                    (bh > 2 or bw > 2))
                if split == "train":
                    review_counts["train_sparse"] += x["signal_count"] <= 5
                    review_counts["train_dense"] += x["signal_count"] > 2 * (bh + bw) - 4
                else:
                    review_counts["test_pairs"] += 1
                    review_counts["test_dense"] += x["signal_count"] > 2 * (bh + bw) - 4
                    if pair_index < len(train_infos):
                        t = train_infos[pair_index]
                        review_counts["role_reversals"] += (
                            x["frame"] == t["signal"] and x["signal"] == t["frame"])
                dense += x["signal_count"] > 2 * (bh + bw) - 4
                perimeter_only += x["signal_interior"] == 0
                all_inside += x["distractor_inside"] == x["distractor_count"]
                all_outside += x["distractor_inside"] == 0
                mixed += 0 < x["distractor_inside"] < x["distractor_count"]
                edges += r0 == 0 or c0 == 0 or r1 == ih - 1 or c1 == iw - 1
                near_ties += x["signal_count"] - x["distractor_count"] == 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 150 and len(sizes) >= 15 and len(roles) >= 50
    assert min(dense, perimeter_only, all_inside, mixed, all_outside, edges, near_ties) >= 3
    assert len(box_sizes) >= 30
    assert review_counts["nonzero_puzzles"] >= 5
    assert review_counts["source_strip"] >= 3
    assert review_counts["two_large"] >= 3
    assert review_counts["train_sparse"] >= 20 and review_counts["train_dense"] >= 20
    assert review_counts["test_dense"] >= 2 and review_counts["combined"] >= 2
    assert min(review_counts[k] for k in ("square", "wide", "tall")) >= 5
    assert review_counts["role_reversals"] < review_counts["test_pairs"] // 3
