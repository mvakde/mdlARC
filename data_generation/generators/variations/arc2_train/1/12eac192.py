import numpy as np


_AUTO_COUNT_OPTIONS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 4),
    (7, 1),
    (8, 2),
    (9, 3),
    (10, 4),
)

_SOURCE_ROWS = (
    (
        (
            "17710805",
            "17711010",
            "88007777",
            "01000011",
            "50801011",
        ),
        (
            "37710303",
            "37711030",
            "33007777",
            "03000011",
            "30303011",
        ),
    ),
    (
        (
            "00181110",
            "15171100",
            "08077788",
            "08800080",
            "07008550",
            "10000001",
            "10877800",
            "00877088",
            "08808088",
        ),
        (
            "00331110",
            "33371100",
            "08077788",
            "08800080",
            "03003330",
            "30000003",
            "30877300",
            "00877088",
            "08803088",
        ),
    ),
    (
        ("105", "100", "777"),
        ("303", "300", "777"),
    ),
    (
        (
            "00107770",
            "88005500",
            "08800550",
            "01108001",
            "07018000",
            "80001070",
            "08881000",
        ),
        (
            "00307770",
            "88005500",
            "08800550",
            "03303003",
            "03033000",
            "30003030",
            "08883000",
        ),
    ),
    (
        (
            "05015505",
            "11000110",
            "07700005",
            "11050100",
            "01055501",
            "07007007",
            "10100017",
            "00110107",
        ),
        (
            "03033303",
            "33000330",
            "03300003",
            "11050300",
            "01055503",
            "03003007",
            "30100037",
            "00110307",
        ),
    ),
)

def _validated_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return int(seed)


def _resolve_counts(seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                 ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]
    if num_train is None:
        choices = list(range(max(2, int(num_test) + 1), 11))
        num_train = choices[seed % len(choices)]
    if num_test is None:
        choices = list(range(1, min(4, int(num_train) - 1) + 1))
        num_test = choices[seed % len(choices)]
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(grid, background, diagonal=False):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    found = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == background or seen[r, c]:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in steps:
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and int(grid[yy, xx]) == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            found.append((color, tuple(cells)))
    return found


def _apply_rule(grid, background, highlight, diagonal=False):
    result = grid.copy()
    for _, cells in _components(grid, background, diagonal):
        if len(cells) <= 2:
            for r, c in cells:
                result[r, c] = highlight
    return result


def _normal(shape):
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _transform(shape, rng):
    reflected = bool(rng.integers(2))
    cells = [(r, -c if reflected else c) for r, c in shape]
    for _ in range(int(rng.integers(4))):
        cells = [(c, -r) for r, c in cells]
    return _normal(cells)


def _shape(size, style, rng):
    if style == "ring" and size >= 8:
        h = int(rng.integers(3, min(7, size // 2 + 1)))
        w = max(3, min(8, (size + 4 - 2 * h) // 2))
        return _normal([(r, c) for r in range(h) for c in range(w)
                        if r in (0, h - 1) or c in (0, w - 1)])
    if style == "bridge" and size >= 10:
        span = max(2, min(10, size - 7))
        cells = {(0, 0), (0, 1), (1, 0), (1, 1),
                 (0, span + 1), (0, span + 2),
                 (1, span + 1), (1, span + 2)}
        cells.update((0, c) for c in range(2, span + 1))
        return _normal(cells)
    if style == "path":
        cells = [(0, 0)]
        occupied = {(0, 0)}
        direction = int(rng.integers(4))
        steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
        for _ in range(size - 1):
            choices = []
            for k in (direction, (direction + 1) % 4, (direction + 3) % 4,
                      (direction + 2) % 4):
                dy, dx = steps[k]
                p = (cells[-1][0] + dy, cells[-1][1] + dx)
                if p not in occupied:
                    choices.append((k, p))
            if not choices:
                break
            k, p = choices[min(int(rng.geometric(0.55)) - 1, len(choices) - 1)]
            direction = k
            occupied.add(p)
            cells.append(p)
        if len(cells) == size:
            return _normal(cells)
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            frontier.update(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        frontier -= cells
        options = sorted(frontier)
        if style == "blob":
            weights = np.array([1 + 3 * sum((r + dr, c + dc) in cells
                                  for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                                for r, c in options], dtype=float)
            weights /= weights.sum()
            chosen = options[int(rng.choice(len(options), p=weights))]
        else:
            chosen = options[int(rng.integers(len(options)))]
        cells.add(chosen)
    return _normal(cells)


def _neighbors(r, c, h, w):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y, x = r + dr, c + dc
        if 0 <= y < h and 0 <= x < w:
            yield y, x


def _has_different_color_contact(grid, background):
    h, w = grid.shape
    return any(
        grid[r, c] != background
        and ((r + 1 < h and grid[r + 1, c] != background
              and grid[r + 1, c] != grid[r, c])
             or (c + 1 < w and grid[r, c + 1] != background
                 and grid[r, c + 1] != grid[r, c]))
        for r in range(h) for c in range(w)
    )


def _lock(grid, bg, color, shape, forbidden, rng, trials=90):
    h, w = grid.shape
    for _ in range(trials):
        variant = _transform(shape, rng)
        sh = max(r for r, _ in variant) + 1
        sw = max(c for _, c in variant) + 1
        if sh > h or sw > w:
            continue
        top = int(rng.integers(h - sh + 1))
        left = int(rng.integers(w - sw + 1))
        cells = {(top + r, left + c) for r, c in variant}
        if any((r in (0, h - 1) and c in (0, w - 1))
               or grid[r, c] != bg or (r, c) in forbidden[color]
               for r, c in cells):
            continue
        if any(grid[y, x] == color for r, c in cells
               for y, x in _neighbors(r, c, h, w) if (y, x) not in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        for r, c in cells:
            forbidden[color].update(_neighbors(r, c, h, w))
        return True
    return False


def _witness(grid, bg, color, forbidden, rng):
    h, w = grid.shape
    base = ((0, 0), (1, 1), (1, 2), (2, 2))
    for _ in range(120):
        cells = _transform(base, rng)
        sh = max(r for r, _ in cells) + 1
        sw = max(c for _, c in cells) + 1
        if sh > h or sw > w:
            continue
        top = int(rng.integers(h - sh + 1))
        left = int(rng.integers(w - sw + 1))
        placed = [(top + r, left + c) for r, c in cells]
        if any((r in (0, h - 1) and c in (0, w - 1)) for r, c in placed):
            continue
        if any(grid[r, c] != bg for r, c in placed):
            continue
        for r, c in placed:
            grid[r, c] = color
        for r, c in placed:
            forbidden[color].update(_neighbors(r, c, h, w))
        return True
    return False


def _sample_dimensions(rng, regime, first):
    if first:
        lo, hi = [(8, 12), (13, 20), (21, 30)][int(rng.choice(3, p=[.55, .32, .13]))]
    else:
        ranges = [(3, 5), (6, 12), (13, 20), (21, 30)]
        weights = [.09, .47, .30, .14] if regime == "source_train" else [.06, .35, .38, .21]
        lo, hi = ranges[int(rng.choice(4, p=weights))]
    if rng.random() < .09:
        h = w = int(rng.integers(lo, hi + 1))
    else:
        h = int(rng.integers(lo, hi + 1))
        w = int(rng.integers(lo, hi + 1))
        if rng.random() < .26:
            h = int(rng.integers(max(3, lo - 4), min(30, hi + 5) + 1))
    return h, w


def _make_pair(rng, regime, first, bg, highlight, palette):
    for attempt in range(24):
        h, w = _sample_dimensions(rng, regime, first)
        grid = np.full((h, w), bg, dtype=np.int8)
        forbidden = {color: set() for color in palette}
        area = h * w
        if first:
            main, other = palette[:2]
            if not _witness(grid, bg, main, forbidden, rng):
                continue
            if not _lock(grid, bg, other, _shape(2, "path", rng), forbidden, rng):
                continue
            if not _lock(grid, bg, other, _shape(int(rng.integers(3, 7)), "branch", rng), forbidden, rng):
                continue
            mode = "mixed"
        else:
            mode = str(rng.choice(("mixed", "small", "large"), p=(.76, .12, .12)))
            if area <= 25:
                mode = "mixed"
            if mode == "mixed" and area > 25:
                color = palette[int(rng.integers(len(palette)))]
                _lock(grid, bg, color, _shape(int(rng.integers(3, min(14, area // 5) + 1)),
                                              str(rng.choice(("path", "branch", "blob"))), rng),
                      forbidden, rng)
        if mode in ("mixed", "large") and area > 25:
            minimum = max(1, area // 150) if mode == "large" else 1
            maximum = max(minimum + 1, min(12, area // 35 + 2))
            nlarge = int(rng.integers(minimum, maximum + 1))
            for _ in range(nlarge):
                color = palette[int(rng.integers(len(palette)))]
                upper = min(35, max(6, area // 9))
                lower = min(upper, max(3, area // 80)) if mode == "large" else 3
                size = int(rng.integers(lower, upper + 1))
                style = str(rng.choice(("path", "branch", "blob", "ring", "bridge")))
                _lock(grid, bg, color, _shape(size, style, rng), forbidden, rng, 65)
        if mode == "small":
            desired = int(area * rng.uniform(.12, .43))
            for _ in range(min(area * 2, 500)):
                if int(np.count_nonzero(grid != bg)) >= desired:
                    break
                color = palette[int(rng.integers(len(palette)))]
                size = 1 if rng.random() < .55 else 2
                _lock(grid, bg, color, _shape(size, "path", rng), forbidden, rng, 15)
        elif mode == "mixed":
            desired = int(area * rng.uniform(.15, .69))
            desired = max(desired, int(np.count_nonzero(grid != bg)))
            cells = [(r, c) for r in range(h) for c in range(w)
                     if not (r in (0, h - 1) and c in (0, w - 1))]
            rng.shuffle(cells)
            for r, c in cells:
                if int(np.count_nonzero(grid != bg)) >= desired:
                    break
                if grid[r, c] != bg:
                    continue
                choices = [color for color in palette if (r, c) not in forbidden[color]]
                if choices:
                    grid[r, c] = choices[int(rng.integers(len(choices)))]
        components = _components(grid, bg)
        if not components:
            continue
        if first:
            sizes = [len(cells) for _, cells in components]
            if not (1 in sizes and 2 in sizes and any(size >= 3 for size in sizes)):
                continue
            if np.array_equal(_apply_rule(grid, bg, highlight), _apply_rule(grid, bg, highlight, True)):
                continue
            by_color = {color: {len(cells) <= 2 for c, cells in components if c == color}
                        for color, _ in components}
            if sum(len(values) == 2 for values in by_color.values()) < 2:
                continue
            if not _has_different_color_contact(grid, bg):
                continue
        if mode == "small" and any(len(cells) >= 3 for _, cells in components):
            continue
        if mode == "large" and any(len(cells) <= 2 for _, cells in components):
            continue
        return {"input": grid, "output": _apply_rule(grid, bg, highlight)}
    raise RuntimeError("bounded pair construction failed")


def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _is_source_pair(inp, out):
    a = tuple("".join(str(int(v)) for v in row) for row in inp)
    b = tuple("".join(str(int(v)) for v in row) for row in out)
    return (a, b) in _SOURCE_ROWS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validated_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = [int(v) for v in rng.permutation(10)]
    highlight = roles[0]
    variable_bg = bool(rng.random() < .32)
    backgrounds = roles[1:1 + (2 if variable_bg else 1)]
    available = roles[1 + len(backgrounds):]
    pool_size = int(rng.integers(3, min(6, len(available)) + 1))
    pool = available[:pool_size]
    novel = available[pool_size:]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(60):
                bg = backgrounds[(index if section == "train" else index + num_train) % len(backgrounds)]
                if section == "train" and index == 0:
                    palette = [pool[0], pool[1]]
                    if rng.random() < .55:
                        palette.extend(pool[2:])
                elif section == "train":
                    n = int(rng.integers(1, len(pool) + 1))
                    palette = [int(v) for v in rng.choice(pool, n, replace=False)]
                    palette.append(pool[index % len(pool)])
                    palette = list(dict.fromkeys(palette))
                else:
                    seen_colors = sorted({int(v) for p in puzzle["train"]
                                          for v in np.unique(p["input"])
                                          if int(v) not in backgrounds and int(v) != highlight})
                    n = int(rng.integers(1, len(seen_colors) + 1))
                    palette = [int(v) for v in rng.choice(seen_colors, n, replace=False)]
                    if novel and rng.random() < .23:
                        palette.append(novel[int(rng.integers(len(novel)))])
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    pair = _make_pair(child, "source_train" if section == "train" else "target_test",
                                      section == "train" and index == 0, bg, highlight, palette)
                except RuntimeError:
                    continue
                key = _grid_key(pair["input"])
                if key in seen or _is_source_pair(pair["input"], pair["output"]):
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inferred_roles(puzzle):
    backgrounds = []
    highlights = set()
    for pair in puzzle["train"] + puzzle["test"]:
        inp, out = pair["input"], pair["output"]
        corners = (int(inp[0, 0]), int(inp[0, -1]), int(inp[-1, 0]), int(inp[-1, -1]))
        assert len(set(corners)) == 1
        bg = corners[0]
        backgrounds.append(bg)
        assert np.array_equal(inp[inp == bg], out[inp == bg])
    for pair in puzzle["train"]:
        changed = pair["input"] != pair["output"]
        highlights.update(int(v) for v in np.unique(pair["output"][changed]))
    assert len(highlights) == 1
    highlight = next(iter(highlights))
    assert highlight not in backgrounds
    return backgrounds, highlight


def _predict_from_training(train, inp):
    colors = {int(v) for pair in train
              for v in np.unique(pair["output"][pair["input"] != pair["output"]])}
    assert len(colors) == 1
    bg = int(inp[0, 0])
    return _apply_rule(inp, bg, next(iter(colors)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    seen = set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert inp.shape == out.shape
        key = _grid_key(inp)
        assert key not in seen
        seen.add(key)
        assert not _is_source_pair(inp, out)
    backgrounds, highlight = _inferred_roles(puzzle)
    assert set(backgrounds[len(puzzle["train"]):]).issubset(
        backgrounds[:len(puzzle["train"])])
    for pair, bg in zip(pairs, backgrounds):
        inp, out = pair["input"], pair["output"]
        assert highlight not in inp
        assert np.array_equal(out, _apply_rule(inp, bg, highlight))
        assert np.array_equal(out, _predict_from_training(puzzle["train"], inp))
    witness = any(not np.array_equal(pair["output"],
                                     _apply_rule(pair["input"], bg, highlight, True))
                  for pair, bg in zip(puzzle["train"], backgrounds))
    assert witness
    train_components = [_components(p["input"], bg)
                        for p, bg in zip(puzzle["train"], backgrounds)]
    train_sizes = {len(cells) for components in train_components for _, cells in components}
    assert 1 in train_sizes and 2 in train_sizes and any(s >= 3 for s in train_sizes)
    assert any(sum(any(len(cells) <= 2 for col, cells in components if col == color)
                       and any(len(cells) >= 3 for col, cells in components if col == color)
                       for color in {col for col, _ in components}) >= 2
               for components in train_components)
    assert any(_has_different_color_contact(pair["input"], bg)
               for pair, bg in zip(puzzle["train"], backgrounds))
    assert len({color for components in train_components for color, _ in components}) >= 2
    assert all(int(v) != highlight for v in backgrounds)


def _signature(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for pair in puzzle["train"] + puzzle["test"])


def self_test() -> None:
    train_counts, test_counts, shapes, roles, profiles = set(), set(), set(), set(), set()
    densities, component_counts, palette_sizes, large_sizes = set(), set(), set(), set()
    mixed_pairs = both_small = witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        bgs, hi = _inferred_roles(puzzle)
        roles.add((tuple(bgs), hi))
        for j, pair in enumerate(puzzle["train"] + puzzle["test"]):
            inp = pair["input"]
            bg = bgs[j]
            comps = _components(inp, bg)
            sizes = [len(c) for _, c in comps]
            shapes.add(inp.shape)
            profiles.add((inp.shape, tuple(sorted(sizes))))
            densities.add(round(np.count_nonzero(inp != bg) / inp.size, 1))
            component_counts.add(len(comps))
            palette_sizes.add(len({color for color, _ in comps}))
            large_sizes.update(s for s in sizes if s >= 3)
            both_small += int(1 in sizes and 2 in sizes)
            mixed_pairs += int(any(any(len(c) <= 2 for k, c in comps if k == color)
                                   and any(len(c) >= 3 for k, c in comps if k == color)
                                   for color, _ in comps))
            if j < len(puzzle["train"]):
                witnesses += int(not np.array_equal(pair["output"],
                                                    _apply_rule(inp, bg, hi, True)))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(profiles) >= 100
    assert len(roles) >= 30 and len(densities) >= 5
    assert len(component_counts) >= 15 and len(palette_sizes) >= 4
    assert max(large_sizes) >= 10 and both_small >= 30
    assert mixed_pairs >= 42 and witnesses >= 42
    assert _signature(generate_puzzle(17)) == _signature(generate_puzzle(17))
    assert _signature(generate_puzzle(17)) != _signature(generate_puzzle(18))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8000 + 10 * train + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9000 + (train or 0) + (test or 0), train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(123, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
