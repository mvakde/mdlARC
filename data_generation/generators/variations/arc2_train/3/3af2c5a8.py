"""Whole-grid, four-quadrant reflection puzzles."""

import numpy as np


def _mirror_compose(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, grid[:, ::-1]), axis=1)
    return np.concatenate((top, top[::-1, :]), axis=0)


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


_SOURCE_INPUTS = (
    ((0, 0, 3, 3), (0, 3, 0, 3), (3, 3, 3, 0)),
    ((0, 0, 8, 0), (0, 8, 0, 8), (0, 0, 8, 0)),
    ((3, 3, 3, 3), (3, 0, 0, 0), (3, 0, 0, 0)),
    ((4, 0, 0, 0), (0, 0, 0, 4), (4, 4, 0, 0)),
)
_SOURCE_PAIRS = frozenset(
    (_grid_key(np.array(rows)), _grid_key(_mirror_compose(np.array(rows))))
    for rows in _SOURCE_INPUTS
)


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _optional_count(num_train, "num_train")
    test = _optional_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        choices = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return choices[seed % len(choices)]
    if train is None:
        choices = list(range(max(2, test + 1), 11))
        train = choices[seed % len(choices)]
    if test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        test = choices[seed % len(choices)]
    return train, test


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.choice(5, p=[0.13, 0.37, 0.19, 0.20, 0.11]))
    if mode == 0:
        return 3, 4
    if mode == 1:
        return int(rng.integers(2, 9)), int(rng.integers(2, 9))
    if mode == 2:
        return int(rng.integers(7, 16)), int(rng.integers(7, 16))
    if mode == 3:
        if rng.random() < 0.5:
            return int(rng.integers(2, 7)), int(rng.integers(8, 16))
        return int(rng.integers(8, 16)), int(rng.integers(2, 7))
    if rng.random() < 0.5:
        return 1, int(rng.integers(3, 16))
    return int(rng.integers(3, 16)), 1


def _random_mask(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    return rng.random(shape) < float(rng.uniform(0.13, 0.87))


def _mask_for_style(rng: np.random.Generator, shape: tuple[int, int], style: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if style == "random":
        return _random_mask(rng, shape)
    if style == "singleton":
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    elif style == "walk":
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(2, max(4, 2 * (h + w))))):
            mask[r, c] = True
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            r, c = min(h - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
    elif style in ("block", "outline", "hole"):
        top = int(rng.integers(0, max(1, h - 1)))
        left = int(rng.integers(0, max(1, w - 1)))
        bottom = int(rng.integers(top + 1, h + 1))
        right = int(rng.integers(left + 1, w + 1))
        if style in ("outline", "hole") and h >= 4 and w >= 4:
            top = min(top, h - 3)
            left = min(left, w - 3)
            bottom = max(bottom, top + 3)
            right = max(right, left + 3)
        mask[top:bottom, left:right] = True
        if style in ("outline", "hole") and bottom - top >= 3 and right - left >= 3:
            if style == "hole" and bottom - top >= 5 and right - left >= 5:
                inset = int(rng.integers(1, min((bottom - top) // 2, (right - left) // 2) + 1))
            else:
                inset = 1
            mask[top + inset:bottom - inset, left + inset:right - inset] = False
    elif style == "stripe":
        period = int(rng.integers(2, min(7, max(h, w) + 1)))
        thick = int(rng.integers(1, period))
        offset = int(rng.integers(period))
        if rng.random() < 0.5:
            mask[:] = ((np.arange(w) + offset) % period < thick)[None, :]
        else:
            mask[:] = ((np.arange(h) + offset) % period < thick)[:, None]
    elif style == "periodic":
        th = int(rng.integers(2, min(5, h + 1))) if h >= 2 else 1
        tw = int(rng.integers(2, min(5, w + 1))) if w >= 2 else 1
        tile = _random_mask(rng, (th, tw))
        mask[:] = np.tile(tile, ((h + th - 1) // th, (w + tw - 1) // tw))[:h, :w]
    elif style == "objects":
        for _ in range(int(rng.integers(2, 7))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            dh = int(rng.integers(1, min(4, h - r + 1)))
            dw = int(rng.integers(1, min(4, w - c + 1)))
            mask[r:r + dh, c:c + dw] = True
    elif style in ("sym_h", "sym_v", "sym_both"):
        hh = (h + 1) // 2 if style != "sym_h" else h
        ww = (w + 1) // 2 if style != "sym_v" else w
        base = _random_mask(rng, (hh, ww))
        if style != "sym_v":
            base = np.concatenate((base, base[:, :w // 2][:, ::-1]), axis=1)
        if style != "sym_h":
            base = np.concatenate((base, base[:h // 2][::-1, :]), axis=0)
        mask[:] = base
    return mask


_STYLES = ("random", "walk", "block", "outline", "hole", "stripe", "periodic",
           "objects", "sym_h", "sym_v", "sym_both", "singleton", "mosaic", "constant")
_STYLE_WEIGHTS = (0.22, 0.10, 0.08, 0.08, 0.05, 0.09, 0.08,
                  0.08, 0.045, 0.045, 0.035, 0.035, 0.07, 0.03)


def _sample_input(rng: np.random.Generator, shape: tuple[int, int], regime: str) -> np.ndarray:
    weights = np.array(_STYLE_WEIGHTS)
    if regime == "target_test":
        weights[[3, 4, 6, 7, 12]] *= 1.5
        weights[0] *= 0.8
    style = str(rng.choice(_STYLES, p=weights / weights.sum()))
    palette_probs = (0.47, 0.22, 0.16, 0.10, 0.05) if regime == "source_train" else (0.34, 0.25, 0.19, 0.14, 0.08)
    palette_size = 1 if style == "constant" else int(rng.choice((2, 3, 4, 5, 6), p=palette_probs))
    colors = rng.permutation(10)[:palette_size].astype(np.int8)
    if style == "constant":
        return np.full(shape, colors[0], dtype=np.int8)
    if style == "mosaic":
        return colors[rng.integers(palette_size, size=shape)].astype(np.int8)
    mask = _mask_for_style(rng, shape, style)
    grid = np.full(shape, colors[0], dtype=np.int8)
    if style.startswith("sym_"):
        h, w = shape
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    rr = min(r, h - 1 - r) if style != "sym_h" else r
                    cc = min(c, w - 1 - c) if style != "sym_v" else c
                    if (rr, cc) == (r, c):
                        grid[r, c] = colors[int(rng.integers(1, palette_size))]
                    else:
                        grid[r, c] = grid[rr, cc]
    else:
        grid[mask] = colors[rng.integers(1, palette_size, size=int(mask.sum()))]
    return grid


def _source_like_input(rng: np.random.Generator, field: int) -> np.ndarray:
    shape = (3, 4)
    style = str(rng.choice(_STYLES[:12]))
    mask = _mask_for_style(rng, shape, style)
    if not np.any(mask) or np.all(mask):
        mask[:] = False
        mask[int(rng.integers(3)), int(rng.integers(4))] = True
    motif = int(rng.choice([color for color in range(10) if color != field]))
    grid = np.full(shape, field, dtype=np.int8)
    grid[mask] = motif
    return grid


def _transforms(grid: np.ndarray) -> tuple[np.ndarray, ...]:
    return (grid, grid[:, ::-1], grid[::-1, :], grid[::-1, ::-1],
            grid.T, grid.T[:, ::-1], grid.T[::-1, :], grid.T[::-1, ::-1])


def _training_resolves_tests(train: list, test: list) -> bool:
    """Reject quadrant alternatives consistent with training but different on a test."""
    for pair in test:
        test_options = _transforms(pair["input"])
        for intended in (0, 1, 2, 3):
            target = test_options[intended]
            for alternative in range(8):
                candidate = test_options[alternative]
                if candidate.shape != target.shape or np.array_equal(candidate, target):
                    continue
                if all(
                    (options := _transforms(item["input"]))[alternative].shape == options[intended].shape
                    and np.array_equal(options[alternative], options[intended])
                    for item in train
                ):
                    return False
    return True


def _axis_witnesses(train: list) -> bool:
    inputs = [pair["input"] for pair in train]
    return (
        any(not np.array_equal(grid, grid[:, ::-1]) for grid in inputs)
        and any(not np.array_equal(grid, grid[::-1, :]) for grid in inputs)
    )


def _component_count(mask: np.ndarray) -> int:
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        pending = [remaining.pop()]
        while pending:
            r, c = pending.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    pending.append(point)
    return count


def _new_pair(rng: np.random.Generator, shape: tuple[int, int], used: set, regime: str,
              source_field: int | None = None) -> dict:
    for _ in range(128):
        grid = (_source_like_input(rng, source_field) if source_field is not None
                else _sample_input(rng, shape, regime))
        key = _grid_key(grid)
        if key in used:
            continue
        pair = {"input": grid, "output": _mirror_compose(grid)}
        if (key, _grid_key(pair["output"])) in _SOURCE_PAIRS:
            continue
        used.add(key)
        return pair
    for field in range(10):
        for motif in range(10):
            if field == motif:
                continue
            for position in range(shape[0] * shape[1]):
                grid = np.full(shape, field, dtype=np.int8)
                grid.flat[position] = motif
                key = _grid_key(grid)
                pair = {"input": grid, "output": _mirror_compose(grid)}
                if key not in used and (key, _grid_key(pair["output"])) not in _SOURCE_PAIRS:
                    used.add(key)
                    return pair
    raise ValueError("exhausted distinct inputs for shape")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    for _ in range(40):
        source_field = int(rng.integers(10)) if rng.random() < 0.10 else None
        fixed_shape = (3, 4) if source_field is not None else (_sample_shape(rng) if rng.random() < 0.20 else None)
        if fixed_shape is not None and 1 in fixed_shape:
            fixed_shape = None
        shapes = [fixed_shape or _sample_shape(rng) for _ in range(train_count + test_count)]
        if fixed_shape is None and len(set(shapes)) == 1:
            shapes[-1] = _sample_shape(rng)
        if fixed_shape is None:
            train_areas = [h * w for h, w in shapes[:train_count]]
            for i in range(train_count, train_count + test_count):
                direction = int(rng.choice((-1, 0, 1), p=(0.23, 0.54, 0.23)))
                if direction:
                    bound = min(train_areas) if direction < 0 else max(train_areas)
                    for _ in range(40):
                        proposal = _sample_shape(rng)
                        if (proposal[0] * proposal[1] - bound) * direction > 0:
                            shapes[i] = proposal
                            break
        if fixed_shape is None and any(h == w for h, w in shapes[train_count:]) and not any(h == w for h, w in shapes[:train_count]):
            side = int(rng.integers(3, 9))
            shapes[int(rng.integers(train_count))] = side, side
        used = set()
        puzzle = {"train": [], "test": []}
        for i, shape in enumerate(shapes):
            regime = "train" if i < train_count else "test"
            sampler_regime = "source_train" if regime == "train" else "target_test"
            puzzle[regime].append(_new_pair(rng, shape, used, sampler_regime, source_field))
        if _axis_witnesses(puzzle["train"]) and _training_resolves_tests(puzzle["train"], puzzle["test"]):
            validate_generated_puzzle(puzzle)
            return puzzle
    used = set()
    anchor = np.arange(12, dtype=np.int8).reshape(3, 4) % 10
    train = [{"input": anchor, "output": _mirror_compose(anchor)}]
    used.add(_grid_key(anchor))
    for _ in range(train_count - 1):
        train.append(_new_pair(rng, _sample_shape(rng), used, "source_train"))
    test = [_new_pair(rng, _sample_shape(rng), used, "target_test") for _ in range(test_count)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= d <= 30 for d in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert bool(np.all((grid >= 0) & (grid <= 9)))
        h, w = inp.shape
        assert out.shape == (2 * h, 2 * w)
        assert np.array_equal(out[:h, :w], inp)
        assert np.array_equal(out[:h, w:], inp[:, ::-1])
        assert np.array_equal(out[h:, :w], inp[::-1, :])
        assert np.array_equal(out[h:, w:], inp[::-1, ::-1])
        assert set(np.unique(inp)) == set(np.unique(out))
        key = _grid_key(inp)
        assert key not in used
        assert (key, _grid_key(out)) not in _SOURCE_PAIRS
        used.add(key)
    assert _axis_witnesses(puzzle["train"])
    assert _training_resolves_tests(puzzle["train"], puzzle["test"])


def self_test() -> None:
    ambiguous_train = [
        {"input": np.array([[3, 3], [3, 8]], dtype=np.int8)},
        {"input": np.array([[3, 5], [5, 5]], dtype=np.int8)},
    ]
    ambiguous_test = [{"input": np.array([[3, 6], [3, 3]], dtype=np.int8)}]
    assert not _training_resolves_tests(ambiguous_train, ambiguous_test)
    validate_generated_puzzle(generate_puzzle(262, 2, 1))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(731 + (counts[0] or 0) * 11 + (counts[1] or 0), *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(17, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    puzzles = [generate_puzzle(1000 + 37 * i) for i in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            assert np.array_equal(pair["output"], _mirror_compose(pair["input"]))
    def signature(p):
        return tuple(tuple((_grid_key(x["input"]), _grid_key(x["output"])) for x in p[s]) for s in ("train", "test"))
    assert signature(puzzles[0]) == signature(generate_puzzle(1000))
    assert len({signature(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    pairs = [x for p in puzzles for x in p["train"] + p["test"]]
    shapes = {x["input"].shape for x in pairs}
    assert len(shapes) >= 30
    assert max(h for h, _ in shapes) >= 12 and max(w for _, w in shapes) >= 12
    assert any(h == 1 or w == 1 for h, w in shapes)
    assert sum(len({x["input"].shape for x in p["train"] + p["test"]}) > 1 for p in puzzles) >= 20
    assert any(len({x["input"].shape for x in p["train"] + p["test"]}) == 1 for p in puzzles)
    assert any(
        all(x["input"].shape == (3, 4) and len(np.unique(x["input"])) == 2
            for x in p["train"] + p["test"])
        for p in puzzles
    )
    assert {len(np.unique(x["input"])) for x in pairs} >= {1, 2, 3, 4}
    assert any(len({int(np.unique(x["input"])[0]) for x in p["train"] + p["test"]}) > 1 for p in puzzles)
    assert any(np.array_equal(x["input"], x["input"][:, ::-1]) for x in pairs)
    assert any(not np.array_equal(x["input"], x["input"][:, ::-1]) for x in pairs)
    assert any(h < w for h, w in shapes) and any(h > w for h, w in shapes)
    assert any(h == w for h, w in shapes)
    assert any(h % 2 == 0 for h, w in shapes) and any(h % 2 == 1 for h, w in shapes)
    assert any(w % 2 == 0 for h, w in shapes) and any(w % 2 == 1 for h, w in shapes)
    assert sum(any(np.array_equal(x["input"], x["input"][:, ::-1]) or np.array_equal(x["input"], x["input"][::-1, :]) for x in p["train"][:2]) for p in puzzles) >= 8
    assert sum(np.array_equal(x["input"], x["input"][:, ::-1]) or np.array_equal(x["input"], x["input"][::-1, :]) for p in puzzles for x in p["test"]) >= 8
    assert sum(not np.array_equal(x["input"], x["input"][:, ::-1]) and not np.array_equal(x["input"], x["input"][::-1, :]) for p in puzzles for x in p["test"]) >= 8
    assert sum(len(np.unique(x["input"])) >= 3 for x in pairs) >= 40
    assert sum(len(np.unique(x["input"])) == 1 for x in pairs) >= 4
    dominant_fractions = [np.unique(x["input"], return_counts=True)[1].max() / x["input"].size for x in pairs]
    assert sum(fraction >= 0.8 for fraction in dominant_fractions) >= 20
    assert sum(fraction <= 0.4 for fraction in dominant_fractions) >= 5
    components = []
    for pair in pairs:
        colors, counts = np.unique(pair["input"], return_counts=True)
        components.append(_component_count(pair["input"] != colors[int(np.argmax(counts))]))
    assert min(components) == 0 and max(components) >= 8
    assert len(set(components)) >= 8
    unseen, smaller, larger, new_color, familiar_color = 0, 0, 0, 0, 0
    for puzzle in puzzles:
        train, test = puzzle["train"], puzzle["test"]
        train_shapes = {p["input"].shape for p in train}
        train_areas = [p["input"].size for p in train]
        train_colors = {int(c) for p in train for c in np.unique(p["input"])}
        for pair in test:
            grid = pair["input"]
            colors = {int(c) for c in np.unique(grid)}
            unseen += grid.shape not in train_shapes
            smaller += grid.size < min(train_areas)
            larger += grid.size > max(train_areas)
            new_color += bool(colors - train_colors)
            familiar_color += bool(colors & train_colors)
    assert unseen >= 15 and smaller >= 3 and larger >= 3
    assert new_color >= 3 and familiar_color >= 10
