import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = {
    "b8532dd664dd0a247f2b0d42df755e6edc433703f6fdca5f148d6e98c2864ea7",
    "4874081528d09a6ebeb75c835a61a3ff67592457731fd8529c977d1d3a2d04eb",
    "b92605c9cd14515ddb1cd5e086e145119cdc5b55a91a33641711aadff0ac440d",
    "04c8e91079654c0a4dd93b85e29f025e2c8ee61f971db50d06925585b7cc9203",
    "d63300ffd3ea6f2c1b3fed7fa2856b7edbb6cbb12c1aa2c657e040dcb0a2ee93",
    "17d7359e6979246ee207b40e8f2bc652c76d08ab2aac3fb8107e28e241f24fc0",
}


def _counts(num_train, num_test, rng):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        train = int(rng.integers(2, 11))
        test = int(rng.integers(1, min(4, train - 1) + 1))
        return train, test
    choices = [
        (tr, te) for tr in range(2, 11) for te in range(1, 5)
        if tr > te and (num_train is None or tr == num_train)
        and (num_test is None or te == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _outline(kind, height, width, orientation=0):
    border = np.zeros((height, width), dtype=bool)
    inside = np.zeros_like(border)
    if kind == "rectangle":
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        inside[1:-1, 1:-1] = True
        return border, inside
    assert height == width
    n = height
    if n % 2:
        middle = n // 2
        for row in range(n):
            depth = min(row, n - 1 - row)
            left, right = middle - depth, middle + depth
            border[row, left] = border[row, right] = True
            inside[row, left + 1:right] = True
    else:
        middle = n // 2
        for row in range(n):
            left = middle - row if row <= middle else row - middle
            right = middle + row if row < middle else 3 * middle - 2 - row
            border[row, left] = border[row, right] = True
            inside[row, left + 1:right] = True
    return np.rot90(border, orientation), np.rot90(inside, orientation)


def _valid_origins(inside, size):
    height, width = inside.shape
    origins = []
    for row in range(1, height - size):
        for col in range(1, width - size):
            if (
                np.all(inside[row:row + size, col - 1:col + size + 1])
                and np.all(inside[row - 1:row + size + 1, col:col + size])
            ):
                origins.append((row, col))
    return origins


def _choose_origin(origins, rng, regime):
    rows = np.array([r for r, _ in origins])
    cols = np.array([c for _, c in origins])
    weights = (0.36, 0.28, 0.36) if regime == "target_test" else (0.3, 0.4, 0.3)
    row_target = (int(rng.choice(3, p=weights)) / 2) * (rows.max() - rows.min()) + rows.min()
    col_target = (int(rng.choice(3, p=weights)) / 2) * (cols.max() - cols.min()) + cols.min()
    distance = (rows - row_target) ** 2 + (cols - col_target) ** 2
    best = np.flatnonzero(distance <= np.min(distance) + 1)
    return origins[int(rng.choice(best))]


def _paint_cross(grid, inside, row, col, size, color):
    result = grid.copy()
    for direction, start in ((-1, col - 1), (1, col + size)):
        current = start
        while 0 <= current < grid.shape[1] and np.all(inside[row:row + size, current]):
            result[row:row + size, current] = color
            current += direction
    for direction, start in ((-1, row - 1), (1, row + size)):
        current = start
        while 0 <= current < grid.shape[0] and np.all(inside[current, col:col + size]):
            result[current, col:col + size] = color
            current += direction
    return result


def _dimensions(rng, kind, size, regime):
    if kind == "diamond":
        minimum = max(8, size + 6)
        maximum = 22 if regime == "source_train" else 20
        n = int(rng.integers(minimum, maximum + 1))
        return n, n
    minimum = size + 4
    aspect = int(rng.integers(3))
    if aspect == 0:
        height = int(rng.integers(minimum, 21))
        width = int(np.clip(height + rng.integers(-2, 3), minimum, 24))
    elif aspect == 1:
        height = int(rng.integers(minimum, 17))
        width = int(rng.integers(max(minimum, height + 3), 25))
    else:
        width = int(rng.integers(minimum, 17))
        height = int(rng.integers(max(minimum, width + 3), 25))
    return height, width


def _make_pair(rng, kind, size, same_color, regime):
    for _ in range(32):
        height, width = _dimensions(rng, kind, size, regime)
        orientation = int(rng.integers(4)) if kind == "diamond" else 0
        border, inside = _outline(kind, height, width, orientation)
        origins = _valid_origins(inside, size)
        if origins:
            break
    else:
        raise RuntimeError("no valid seed position")
    row, col = _choose_origin(origins, rng, regime)
    top, bottom, left, right = [int(rng.integers(1, 4)) for _ in range(4)]
    background = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    outline = int(rng.choice([c for c in range(10) if c != background]))
    seed = outline if same_color else int(rng.choice(
        [c for c in range(10) if c not in (background, outline)]
    ))
    grid = np.full((top + height + bottom, left + width + right), background, dtype=np.int8)
    full_inside = np.zeros_like(grid, dtype=bool)
    region = grid[top:top + height, left:left + width]
    region[border] = outline
    region[row:row + size, col:col + size] = seed
    full_inside[top:top + height, left:left + width] = inside
    output = _paint_cross(grid, full_inside, top + row, left + col, size, seed)
    return {"input": grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(num_train, num_test, rng)
    train_kinds = ["rectangle", "diamond"]
    rng.shuffle(train_kinds)
    train_kinds += [str(rng.choice(("rectangle", "diamond"))) for _ in range(train_count - 2)]
    test_kinds = [str(rng.choice(("rectangle", "diamond"))) for _ in range(test_count)]
    train_sizes = [2, 3]
    rng.shuffle(train_sizes)
    train_sizes += [int(rng.integers(2, 4)) for _ in range(train_count - 2)]
    test_sizes = [int(rng.integers(2, 4)) for _ in range(test_count)]
    train_same = list(rng.random(train_count) < 0.3)
    test_same = list(rng.random(test_count) < 0.35)
    if all(train_same):
        train_same[int(rng.integers(train_count))] = False
    if any(test_same) and not any(train_same):
        train_same[int(rng.integers(train_count))] = True
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, kinds, sizes, flags, destination in (
        ("source_train", train_kinds, train_sizes, train_same, "train"),
        ("target_test", test_kinds, test_sizes, test_same, "test"),
    ):
        for kind, size, same in zip(kinds, sizes, flags):
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _make_pair(pair_rng, kind, size, bool(same), regime)
                inp = pair["input"]
                signature = (inp.shape, inp.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[destination].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
    return puzzle


def _input_hash(grid):
    return hashlib.sha256(str(grid.shape).encode() + grid.astype(np.int8).tobytes()).hexdigest()


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    background = int(inp[0, 0])
    occupied = np.argwhere(inp != background)
    assert len(occupied) > 0
    top, left = occupied.min(axis=0)
    bottom, right = occupied.max(axis=0)
    height, width = int(bottom - top + 1), int(right - left + 1)
    assert top >= 1 and left >= 1 and bottom < inp.shape[0] - 1 and right < inp.shape[1] - 1
    assert 6 <= height <= 24 and 6 <= width <= 24
    for kind in ("rectangle", "diamond"):
        if kind == "diamond" and (height != width or height < 8):
            continue
        for orientation in range(1 if kind == "rectangle" else 4):
            border, inside = _outline(kind, height, width, orientation)
            full_border = np.zeros(inp.shape, dtype=bool)
            full_inside = np.zeros(inp.shape, dtype=bool)
            full_border[top:bottom + 1, left:right + 1] = border
            full_inside[top:bottom + 1, left:right + 1] = inside
            outline_values = inp[full_border]
            outline_color = int(outline_values[0])
            if outline_color == background or not np.all(outline_values == outline_color):
                continue
            seed_cells = np.argwhere((inp != background) & ~full_border)
            if len(seed_cells) not in (4, 9):
                continue
            sr, sc = seed_cells.min(axis=0)
            er, ec = seed_cells.max(axis=0)
            size = int(er - sr + 1)
            if size not in (2, 3) or ec - sc + 1 != size:
                continue
            seed_color = int(inp[sr, sc])
            expected_input = np.full(inp.shape, background, dtype=inp.dtype)
            expected_input[full_border] = outline_color
            expected_input[sr:er + 1, sc:ec + 1] = seed_color
            if seed_color == background or not np.array_equal(inp, expected_input):
                continue
            if not np.all(full_inside[sr:er + 1, sc:ec + 1]):
                continue
            if (int(sr - top), int(sc - left)) not in _valid_origins(inside, size):
                continue
            horizontal = np.all(full_inside[sr:er + 1, :], axis=0)
            vertical = np.all(full_inside[:, sc:ec + 1], axis=1)
            h_left = int(sc)
            while h_left > 0 and horizontal[h_left - 1]:
                h_left -= 1
            h_right = int(ec)
            while h_right + 1 < inp.shape[1] and horizontal[h_right + 1]:
                h_right += 1
            v_top = int(sr)
            while v_top > 0 and vertical[v_top - 1]:
                v_top -= 1
            v_bottom = int(er)
            while v_bottom + 1 < inp.shape[0] and vertical[v_bottom + 1]:
                v_bottom += 1
            expected_output = inp.copy()
            expected_output[sr:er + 1, h_left:h_right + 1] = seed_color
            expected_output[v_top:v_bottom + 1, sc:ec + 1] = seed_color
            if not np.array_equal(out, expected_output):
                continue
            assert h_left < sc and h_right > ec and v_top < sr and v_bottom > er
            assert _input_hash(inp) not in _SOURCE_INPUT_HASHES
            return {
                "shape": kind, "height": height, "width": width,
                "size": size, "background": background, "outline": outline_color,
                "seed": seed_color, "same": seed_color == outline_color,
                "row": int(sr - top), "col": int(sc - left),
                "h_arms": (int(sc - h_left), int(h_right - ec)),
                "v_arms": (int(sr - v_top), int(v_bottom - er)),
                "parity": height % 2 if kind == "diamond" else None,
            }
    raise AssertionError("pair is not a valid constant-width cross")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_sig = (inp.shape, inp.tobytes())
            pair_sig = (input_sig, out.tobytes())
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            infos[regime].append(_analyze_pair(pair))
    assert {info["shape"] for info in infos["train"]} == {"rectangle", "diamond"}
    assert {info["size"] for info in infos["train"]} == {2, 3}
    assert any(not info["same"] for info in infos["train"])
    if any(info["same"] for info in infos["test"]):
        assert any(info["same"] for info in infos["train"])


def _same_puzzle(a, b):
    return all(
        len(a[regime]) == len(b[regime])
        and all(np.array_equal(x[key], y[key]) for x, y in zip(a[regime], b[regime]) for key in ("input", "output"))
        for regime in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, hashes = set(), set(), set()
    features = {"train": [], "test": []}
    layouts, combinations = set(), set()
    within = {"shape": 0, "size": 0, "same": 0, "background": 0, "outline": 0}
    straight_bar_checks = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        hashes.add(_input_hash(puzzle["train"][0]["input"]))
        local = []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                x = _analyze_pair(pair)
                features[regime].append(x)
                local.append(x)
                layouts.add((x["shape"], x["height"], x["width"], x["size"], x["row"], x["col"]))
                combinations.add((regime, x["shape"], x["size"], x["same"], x["background"] == 0))
                if x["shape"] == "diamond":
                    a, b = pair["input"], pair["output"]
                    occupied = np.argwhere(a != x["background"])
                    top = int(occupied[:, 0].min())
                    extra = 0
                    for row in range(top + x["row"], top + x["row"] + x["size"]):
                        boundary = np.flatnonzero(a[row] != x["background"])
                        left, right = int(boundary[0]), int(boundary[-1])
                        extra += right - left - 1 - int(np.count_nonzero(
                            b[row, left + 1:right] == x["seed"]
                        ))
                    assert extra > 0
                    straight_bar_checks += 1
        for key in within:
            within[key] += len({x[key] for x in local}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(hashes) >= 40
    assert len(layouts) >= 250 and len(combinations) >= 28
    assert within["shape"] == 42 and within["outline"] >= 38
    assert within["size"] == 42 and within["same"] >= 30 and within["background"] >= 30
    assert straight_bar_checks >= 150
    for regime in ("train", "test"):
        f = features[regime]
        assert {x["shape"] for x in f} == {"rectangle", "diamond"}
        assert {x["size"] for x in f} == {2, 3}
        assert {x["same"] for x in f} == {False, True}
        assert {x["parity"] for x in f if x["shape"] == "diamond"} == {0, 1}
        assert len({x["background"] for x in f}) >= 4
        assert len({x["outline"] for x in f}) >= 8
        assert any(x["width"] - x["height"] >= 4 for x in f)
        assert any(x["height"] - x["width"] >= 4 for x in f)
        assert any(x["h_arms"][0] != x["h_arms"][1] for x in f if x["shape"] == "diamond")
        assert { (x["shape"], x["size"]) for x in f } == {
            ("rectangle", 2), ("rectangle", 3), ("diamond", 2), ("diamond", 3)
        }
        assert any(x["same"] and x["shape"] == "diamond" for x in f)
        assert any(x["same"] and x["shape"] == "rectangle" and x["width"] - x["height"] >= 4 for x in f)
        assert any(x["background"] != 0 for x in f)
        for shape in ("rectangle", "diamond"):
            subset = [x for x in f if x["shape"] == shape]
            assert any(x["h_arms"][0] + 1 < x["h_arms"][1] for x in subset)
            assert any(x["h_arms"][1] + 1 < x["h_arms"][0] for x in subset)
            assert any(x["v_arms"][0] + 1 < x["v_arms"][1] for x in subset)
            assert any(x["v_arms"][1] + 1 < x["v_arms"][0] for x in subset)
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(101, tr, te)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(101, tr, te))
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
