import numpy as np


FILL = {"H": 7, "V": 3}
NEUTRAL = (1, 2, 4, 5, 6, 8, 9)
SOURCE_INPUTS = {
    ("0222000000", "0202000000", "0202000000", "0202000000",
     "0222000000", "0000000000", "0000000000", "0004444400",
     "0004000400", "0004444400"),
    ("00000000000444", "00000000000404", "00000000000404",
     "02222222200404", "02000000200404", "02000000200444",
     "02222222200000", "00000000000777", "00000000000707",
     "00777777700707", "00700000700707", "00777777700707",
     "00000000000707", "00000000000777"),
    ("000000000000000000", "000000000000000000", "055555555500000000",
     "050000000500000000", "050000000500000000", "055555555500000000",
     "000000000000000000", "000000000000000000", "000444400004444444",
     "000400400004000004", "000400400004444444", "000400400000000000",
     "000400400000000000", "000400400000555000", "000400400000505000",
     "000444400000505000", "000000000000505000", "000000000000555000"),
    ("00000000000", "03333333330", "03000000030", "03333333330",
     "00000000000", "00003330000", "00003030000", "00003030000",
     "00003030000", "00003330000"),
    ("33333066666600000000000", "30003060000604444444444",
     "30003060000604000000004", "30003060000604000000004",
     "30003060000604000000004", "30003060000604444444444",
     "30003060000600000000000", "30003060000607770077700",
     "30003060000607070070700", "30003060000607070070700",
     "33333066666607070070700", "00000000000007770077700"),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _resolve_counts(seed, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= int(value) <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
    if num_train is None and num_test is None:
        num_train = 2 + seed % 9
        num_test = 1 + (seed // 9) % min(4, num_train - 1)
    elif num_train is None:
        num_test = int(num_test)
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_train = int(num_train)
        num_test = 1 + (seed // 3) % min(4, num_train - 1)
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _separated(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _grid_key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _source_key(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _sample_shape(rng, count):
    minimum = (7, 8, 10, 12, 14)[count - 1]
    if rng.random() < 0.24:
        candidates = ((10, 10), (14, 14), (18, 18), (10, 11), (12, 23))
        valid = [shape for shape in candidates if min(shape) >= minimum]
        return valid[int(rng.integers(len(valid)))]
    height = int(rng.integers(minimum, 29))
    width = int(rng.integers(minimum, 29))
    if rng.random() < 0.24:
        width = height
    return height, width


def _place_boxes(rng, height, width, orientations):
    order = [int(v) for v in rng.permutation(len(orientations))]
    for _ in range(85):
        boxes = [None] * len(orientations)
        placed = []
        for index in order:
            orientation = orientations[index]
            axis = width if orientation == "H" else height
            cross = height if orientation == "H" else width
            for _ in range(80):
                short_max = min(6, cross, axis - 1)
                short = int(rng.integers(3, short_max + 1))
                long_max = min(axis, 18)
                long = int(rng.integers(short + 1, long_max + 1))
                bh, bw = (short, long) if orientation == "H" else (long, short)
                row = int(rng.integers(height - bh + 1))
                column = int(rng.integers(width - bw + 1))
                candidate = (row, column, bh, bw)
                if all(_separated(candidate, old) for old in placed):
                    boxes[index] = candidate
                    placed.append(candidate)
                    break
            else:
                break
        else:
            return boxes
    return None


def _sample_colors(rng, count, shared=None):
    if shared is not None:
        return [shared] + [int(rng.integers(1, 10)) for _ in range(count - 1)]
    if count >= 2 and rng.random() < 0.18:
        return [int(rng.integers(1, 10))] * count
    colors = [int(rng.integers(1, 10)) for _ in range(count)]
    if count >= 4 and rng.random() < 0.5:
        choices = rng.choice(np.arange(1, 10), size=min(count, 4), replace=False)
        for i, color in enumerate(choices):
            colors[i] = int(color)
        rng.shuffle(colors)
    return colors


def _draw_pair(height, width, orientations, colors, boxes):
    inp = np.zeros((height, width), dtype=np.int8)
    for color, (row, col, bh, bw) in zip(colors, boxes):
        inp[row, col:col + bw] = color
        inp[row + bh - 1, col:col + bw] = color
        inp[row:row + bh, col] = color
        inp[row:row + bh, col + bw - 1] = color
    out = inp.copy()
    for orientation, (row, col, bh, bw) in zip(orientations, boxes):
        out[row + 1:row + bh - 1, col + 1:col + bw - 1] = FILL[orientation]
    return {"input": inp, "output": out}


def _make_pair(rng, regime, index, shared_color):
    weights = (0.12, 0.33, 0.25, 0.20, 0.10) if regime == "source_train" else (
        0.13, 0.20, 0.22, 0.25, 0.20)
    for _ in range(40):
        count = int(rng.choice(np.arange(1, 6), p=weights))
        height, width = _sample_shape(rng, count)
        orientations = ["H" if rng.random() < 0.5 else "V" for _ in range(count)]
        mandatory = regime == "source_train" and index < 2
        if mandatory:
            orientations[0] = "H" if index == 0 else "V"
        colors = _sample_colors(rng, count, shared_color if mandatory else None)
        boxes = _place_boxes(rng, height, width, orientations)
        if boxes is None:
            continue
        return _draw_pair(height, width, orientations, colors, boxes)
    count = 5
    orientations = ["H", "V", "H", "V", "H"]
    if regime == "source_train" and index == 1:
        orientations[0] = "V"
    colors = _sample_colors(rng, count, shared_color if regime == "source_train" and index < 2 else None)
    slots = ((0, 0), (0, 8), (0, 16), (8, 0), (8, 8))
    boxes = [(r, c, 3, 5) if orientation == "H" else (r, c, 5, 3)
             for (r, c), orientation in zip(slots, orientations)]
    return _draw_pair(28, 28, orientations, colors, boxes)


def _extract_frames(grid):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    frames = []
    for sr in range(height):
        for sc in range(width):
            if grid[sr, sc] == 0 or seen[sr, sc]:
                continue
            color = int(grid[sr, sc])
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] != 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            bh, bw = bottom - top + 1, right - left + 1
            _require(bh >= 3 and bw >= 3 and bh != bw, "frame needs a clear long axis")
            expected = np.zeros((bh, bw), dtype=grid.dtype)
            expected[0, :] = color
            expected[-1, :] = color
            expected[:, 0] = color
            expected[:, -1] = color
            _require(np.array_equal(grid[top:bottom + 1, left:right + 1], expected), "malformed frame")
            _require(len(cells) == 2 * bh + 2 * bw - 4, "joined or incomplete frames")
            frames.append((top, left, bh, bw, color, "V" if bh > bw else "H"))
    for i, first in enumerate(frames):
        for second in frames[i + 1:]:
            _require(_separated(first[:4], second[:4]), "frames must be separate")
    return frames


def _expected_output(inp, frames):
    result = inp.copy()
    for row, col, bh, bw, _, orientation in frames:
        result[row + 1:row + bh - 1, col + 1:col + bw - 1] = FILL[orientation]
    return result


def _signature(pair):
    inp = pair["input"]
    frames = _extract_frames(inp)
    return inp.shape, len(frames), tuple(sorted({f[4] for f in frames}))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen = set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid palette")
            _require(inp.shape == out.shape, "shape changed")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_source_key(inp) not in SOURCE_INPUTS, "source input copied")
            frames = _extract_frames(inp)
            _require(1 <= len(frames) <= 5, "frame count")
            _require(np.array_equal(out, _expected_output(inp, frames)), "orientation fill rule")
            _require(not np.array_equal(inp, out), "unchanged pair")
            if split == "train":
                training.extend(frames)
    for orientation in ("H", "V"):
        _require(any(f[5] == orientation and f[4] != FILL[orientation] for f in training),
                 "training must visibly establish both fills")
    _require(any(h[4] == v[4] and h[4] in NEUTRAL for h in training
                 for v in training if h[5] == "H" and v[5] == "V"),
             "training needs same-color opposite orientations")
    train_signatures = {_signature(p) for p in puzzle["train"]}
    _require(any(_signature(p) not in train_signatures for p in puzzle["test"]),
             "test needs a new configuration")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    shared_color = int(rng.choice(NEUTRAL))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(60):
                pair = _make_pair(pair_rng, regime, index, shared_color)
                key = _grid_key(pair["input"])
                if key in seen or _source_key(pair["input"]) in SOURCE_INPUTS:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not create a distinct pair")
    train_signatures = {_signature(p) for p in puzzle["train"]}
    for _ in range(60):
        if any(_signature(p) not in train_signatures for p in puzzle["test"]):
            break
        pair = _make_pair(np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max))),
                          "target_test", 0, shared_color)
        if _grid_key(pair["input"]) not in seen and _source_key(pair["input"]) not in SOURCE_INPUTS:
            puzzle["test"][0] = pair
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzles(first, second):
    return all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for k in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, counts, shorts, palettes, matches, layouts = set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                frames = _extract_frames(inp)
                shapes.add(inp.shape)
                counts.add(len(frames))
                shorts.update(min(f[2], f[3]) for f in frames)
                palettes.add(tuple(sorted({f[4] for f in frames})))
                layouts.add((inp.shape, tuple(sorted(f[:4] for f in frames))))
                matches.add((split, sum(f[4] == FILL[f[5]] for f in frames), len(frames)))
                assert np.array_equal(pair["output"], _expected_output(inp, frames))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len(shapes) >= 30 and any(h != w for h, w in shapes)
    assert counts == {1, 2, 3, 4, 5}
    assert {3, 4, 5, 6} <= shorts
    assert len(palettes) >= 20 and len(layouts) >= 100
    assert any(s == "test" and m == 0 for s, m, n in matches)
    assert any(s == "test" and 0 < m < n for s, m, n in matches)
    assert any(s == "test" and m == n for s, m, n in matches)
    for seed, nt, ne in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                         (104, None, 1), (105, None, 4), (106, 2, None),
                         (107, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ne is not None:
            assert len(puzzle["test"]) == ne
    assert _same_puzzles(generate_puzzle(17), generate_puzzle(17))
    assert not _same_puzzles(generate_puzzle(17), generate_puzzle(18))
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(777, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(999, 2, 1)
    frame = _extract_frames(bad["train"][0]["input"])[0]
    bad["train"][0]["output"][frame[0] + 1, frame[1] + 1] = 9
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong fill accepted")
