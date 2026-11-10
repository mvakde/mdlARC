from __future__ import annotations

import numpy as np


# Automatic counts deliberately cycle through many legal combinations.
_AUTO_COUNTS = (
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (3, 2),
    (4, 2),
    (6, 2),
    (8, 2),
    (4, 3),
    (7, 3),
    (9, 3),
    (5, 4),
    (10, 4),
)

_SOURCE_INPUT_TEXTS = frozenset(
    {
        "\n".join(
            (
                "0000000000",
                "0000000000",
                "0055000000",
                "0055000000",
                "0000000000",
                "0000000000",
                "0000005500",
                "0000005500",
                "0000000000",
                "0000000000",
            )
        ),
        "\n".join(
            (
                "0000000000",
                "0000000000",
                "0000000000",
                "0000550000",
                "0000550000",
                "0000000000",
                "0000000000",
                "0000000000",
                "0000000000",
                "0000000000",
            )
        ),
        "\n".join(
            (
                "000000000000000",
                "000000000000000",
                "000550000000000",
                "000550000000000",
                "000000005500000",
                "000000005500000",
                "000000000000000",
                "000000000000000",
                "000055000000000",
                "000055000000000",
                "000000000000000",
                "000000000550000",
                "000000000550000",
                "000000000000000",
                "000000000000000",
            )
        ),
        "\n".join(
            (
                "000000000000000",
                "055000000005500",
                "055000000005500",
                "000000000000000",
                "000005500000000",
                "000005500000000",
                "000000000000550",
                "000000000000550",
                "000000000000000",
                "000000000000000",
                "005500000000000",
                "005500000550000",
                "000000000550000",
                "000000000000000",
                "000000000000000",
            )
        ),
    }
)

def _normalize_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _normalize_count("num_train", num_train, 2, 10)
    num_test = _normalize_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_train, num_test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _sites(position):
    row, column = position
    return ((row - 1, column - 1), (row - 1, column + 2),
            (row + 2, column - 1), (row + 2, column + 2))


def _body(position):
    row, column = position
    return {(row + dr, column + dc) for dr in (0, 1) for dc in (0, 1)}


def _compatible_position(first, second):
    a, b = _body(first), _body(second)
    sa, sb = set(_sites(first)), set(_sites(second))
    if a & b or a & sb or b & sa or sa & sb:
        return False
    # Touching bodies would be one connected object of ambiguous geometry.
    return all((r + dr, c + dc) not in b for r, c in a
               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))


def _on_border(position, height, width):
    return position[0] in (1, height - 3) or position[1] in (1, width - 3)


def _ordered_candidates(rng, candidates, style, height, width):
    rng.shuffle(candidates)
    if style == "row":
        anchor = int(rng.integers(1, height - 2))
        candidates.sort(key=lambda p: (abs(p[0] - anchor), abs(p[1] - width // 2)))
    elif style == "column":
        anchor = int(rng.integers(1, width - 2))
        candidates.sort(key=lambda p: (abs(p[1] - anchor), abs(p[0] - height // 2)))
    elif style == "cluster":
        center = (int(rng.integers(1, height - 2)),
                  int(rng.integers(1, width - 2)))
        candidates.sort(key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
    return candidates


def _place_objects(rng, height, width, count, border_mode, style):
    candidates = [(r, c) for r in range(1, height - 2)
                  for c in range(1, width - 2)]
    if border_mode == "interior":
        candidates = [p for p in candidates if not _on_border(p, height, width)]
    if not candidates:
        return None
    # Several orderings create distinct alignments and crowded staggered arrangements.
    for _ in range(12):
        order = _ordered_candidates(rng, candidates.copy(), style, height, width)
        if border_mode == "border":
            edge = [p for p in order if _on_border(p, height, width)]
            other = [p for p in order if not _on_border(p, height, width)]
            order = edge + other
        chosen = []
        for position in order:
            if all(_compatible_position(position, prior) for prior in chosen):
                chosen.append(position)
                if len(chosen) == count:
                    return sorted(chosen)
    # A 4-step lattice has disjoint bodies and labels; vary its phase and orientation.
    rows = list(range(2 if border_mode == "interior" else 1, height - 2, 4))
    columns = list(range(2 if border_mode == "interior" else 1, width - 2, 4))
    lattice = [(r, c) for r in rows for c in columns]
    if border_mode == "border":
        lattice = [p for p in lattice if _on_border(p, height, width)] + [
            p for p in lattice if not _on_border(p, height, width)]
    if len(lattice) >= count:
        return sorted(lattice[:count])
    return None


def _canvas(rng, count, border_mode, regime):
    # Canvas class is drawn independently of count; infeasible draws are retried.
    classes = ("small", "medium", "large", "wide", "tall")
    weights = (0.18, 0.39, 0.23, 0.10, 0.10) if regime == "source_train" else (
        0.13, 0.37, 0.28, 0.11, 0.11)
    for _ in range(96):
        kind = str(rng.choice(classes, p=weights))
        if kind == "small":
            height, width = int(rng.integers(4, 10)), int(rng.integers(4, 10))
        elif kind == "medium":
            height, width = int(rng.integers(10, 21)), int(rng.integers(10, 21))
        elif kind == "large":
            height, width = int(rng.integers(21, 31)), int(rng.integers(21, 31))
        elif kind == "wide":
            height, width = int(rng.integers(4, 10)), int(rng.integers(18, 31))
        else:
            height, width = int(rng.integers(18, 31)), int(rng.integers(4, 10))
        start = 2 if border_mode == "interior" else 1
        capacity = len(range(start, height - 2, 4)) * len(range(start, width - 2, 4))
        if capacity >= count:
            return height, width
    # Deterministic, roomy fallback for every legal count.
    return 24, 24


def _count_plan(rng, ntrain, ntest, mixed):
    mode = int(rng.integers(0, 3))
    if mode == 0:  # Source-like count extrapolation.
        train = [int(rng.integers(2 if mixed else 1, 5)),
                 int(rng.integers(1, 5))]
        if train[0] == train[1]:
            train[1] = 1 if train[0] != 1 else 4
        train += [int(rng.integers(1, 5)) for _ in range(ntrain - 2)]
        test = [int(rng.integers(5, 11)) for _ in range(ntest)]
    elif mode == 1:  # Equal and overlapping count ranges.
        train = [int(rng.integers(2 if mixed else 1, 9)),
                 int(rng.integers(1, 10))]
        if train[0] == train[1]:
            train[1] = 1 if train[0] != 1 else 6
        train += [int(rng.integers(1, 12)) for _ in range(ntrain - 2)]
        test = [int(rng.choice(train))] + [
            int(rng.integers(1, 12)) for _ in range(ntest - 1)]
    else:  # Every test has fewer squares than every demonstration.
        train = [int(rng.integers(3, 8)), int(rng.integers(3, 10))]
        if train[0] == train[1]:
            train[1] = min(10, train[1] + 1)
        train += [int(rng.integers(3, 13)) for _ in range(ntrain - 2)]
        test = [int(rng.integers(1, min(train))) for _ in range(ntest)]
    if mixed and max(test) < 2:
        test[-1] = 2
    return mode, train, test


def _input_text(grid):
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(value) for value in grid.flat)


def _make_pair(rng, labels, count, border_mode, regime, mixed, previous_bg=None):
    available = [color for color in range(10) if color not in labels]
    background_options = [color for color in available if color != previous_bg]
    background = int(rng.choice(background_options or available))
    object_options = [color for color in available if color != background]
    ncolors = min(count, int(rng.integers(2, min(3, count) + 1))) if mixed else 1
    object_colors = [int(v) for v in rng.choice(object_options, size=ncolors, replace=False)]
    style = str(rng.choice(("free", "row", "column", "cluster")))
    for attempt in range(20):
        height, width = _canvas(rng, count, border_mode, regime)
        positions = _place_objects(rng, height, width, count, border_mode, style)
        if positions is not None:
            break
        style = str(rng.choice(("free", "row", "column", "cluster")))
    else:
        raise RuntimeError("could not place squares")
    inp = np.full((height, width), background, dtype=np.int8)
    rng.shuffle(positions)
    if mixed:
        colors = object_colors + [int(rng.choice(object_colors)) for _ in range(count - ncolors)]
        rng.shuffle(colors)
    else:
        colors = [object_colors[0]] * count
    for (row, col), color in zip(positions, colors):
        inp[row:row + 2, col:col + 2] = color
    out = inp.copy()
    for position in positions:
        for site, color in zip(_sites(position), labels):
            out[site] = color
    return {"input": inp, "output": out}


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    height, width = inp.shape
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    assert np.all(border == border[0])
    background = int(border[0])
    mask = inp != background
    seen = np.zeros(inp.shape, dtype=bool)
    objects = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            assert len(cells) == 4
            top, left = min(y for y, _ in cells), min(x for _, x in cells)
            assert set(cells) == _body((top, left))
            assert len({int(inp[y, x]) for y, x in cells}) == 1
            objects.append((top, left, int(inp[top, left])))
    assert objects
    positions = [(r, c) for r, c, _ in objects]
    sites = []
    for position in positions:
        for site in _sites(position):
            assert 0 <= site[0] < height and 0 <= site[1] < width
            assert int(inp[site]) == background
            sites.append(site)
    assert len(sites) == len(set(sites)) == 4 * len(objects)
    for i, first in enumerate(positions):
        for second in positions[i + 1:]:
            assert _compatible_position(first, second)
    labels = tuple(int(out[site]) for site in _sites(positions[0]))
    assert len(set(labels)) == 4
    assert background not in labels
    assert not (set(color for _, _, color in objects) & set(labels))
    expected = inp.copy()
    for position in positions:
        for site, color in zip(_sites(position), labels):
            expected[site] = color
    assert np.array_equal(out, expected)
    return {"count": len(objects), "border": sum(_on_border(p, height, width) for p in positions),
            "labels": labels, "background": background,
            "object_colors": frozenset(color for _, _, color in objects),
            "positions": positions, "shape": inp.shape}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, rng, num_train, num_test)
    labels = tuple(int(v) for v in rng.choice(10, size=4, replace=False))
    mixed = bool(rng.integers(0, 3) == 0)
    mode, train_counts, test_counts = _count_plan(rng, ntrain, ntest, mixed)
    specs = []
    for i, count in enumerate(train_counts):
        if mode == 0:
            border_mode = "interior"
        else:
            border_mode = "interior" if i == 0 else "border" if i == 1 else str(
                rng.choice(("interior", "border", "any")))
        specs.append(("train", "source_train", count, border_mode,
                      mixed and count >= 2 and (i == 0 or i > 1 and bool(rng.integers(0, 3) == 0))))
    for i, count in enumerate(test_counts):
        border_mode = "border" if mode == 0 else str(rng.choice(("interior", "border", "any")))
        specs.append(("test", "target_test", count, border_mode,
                      mixed and count >= 2 and (i == 0 or bool(rng.integers(0, 2)))))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_bg = None
    for split, regime, count, border_mode, use_mixed in specs:
        for _ in range(64):
            child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair = _make_pair(np.random.default_rng(child_seed), labels, count,
                              border_mode, regime, use_mixed, previous_bg)
            key = _grid_key(pair["input"])
            if key in seen or _input_text(pair["input"]) in _SOURCE_INPUT_TEXTS:
                continue
            seen.add(key)
            previous_bg = _inspect_pair(pair)["background"]
            puzzle[split].append(pair)
            break
        else:
            raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    metas = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            ik = _grid_key(pair["input"])
            pk = (ik, _grid_key(pair["output"]))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            assert _input_text(pair["input"]) not in _SOURCE_INPUT_TEXTS
            metas[split].append(_inspect_pair(pair))
    labels = metas["train"][0]["labels"]
    assert all(m["labels"] == labels for split in metas for m in metas[split])
    assert len({m["count"] for m in metas["train"]}) >= 2
    # The four colors are seen at every training square. A mixed test has a mixed
    # training example, so color independence is demonstrated, not assumed.
    if any(len(m["object_colors"]) > 1 for m in metas["test"]):
        assert any(len(m["object_colors"]) > 1 for m in metas["train"])
    # Duplicate input grids are already excluded; a shared size, count, and
    # border category can still contain a genuinely new arrangement.


def _puzzle_key(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def _predict_from_training(train_pairs, test_input):
    # This function receives no test output, seed, or construction metadata.
    labels = _inspect_pair(train_pairs[0])["labels"]
    grid = test_input.copy()
    height, width = grid.shape
    background = int(grid[0, 0])
    positions = []
    for row in range(1, height - 2):
        for column in range(1, width - 2):
            color = int(grid[row, column])
            if (color != background and int(grid[row - 1, column]) == background
                    and int(grid[row, column - 1]) == background
                    and np.all(grid[row:row + 2, column:column + 2] == color)):
                positions.append((row, column))
    assert len(positions) * 4 == int(np.count_nonzero(grid != background))
    for position in positions:
        for site, color in zip(_sites(position), labels):
            grid[site] = color
    return grid


def self_test() -> None:
    assert _compatible_position((1, 1), (4, 3))
    assert not _compatible_position((1, 1), (3, 3))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, counts, layouts, palettes = set(), set(), set(), set()
    count_regimes, border_regimes = set(), set()
    small = large = narrow = mixed = near = border_train = interior_test = 0
    color_changes = object_color_changes = adjacent_labels = source_like = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            train_copy = [{name: grid.copy() for name, grid in demonstration.items()}
                          for demonstration in puzzle["train"]]
            predicted = _predict_from_training(train_copy, pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
        train = [_inspect_pair(p) for p in puzzle["train"]]
        test = [_inspect_pair(p) for p in puzzle["test"]]
        count_regimes.add("extrapolate" if min(m["count"] for m in test) > max(m["count"] for m in train)
                          else "reverse" if max(m["count"] for m in test) < min(m["count"] for m in train)
                          else "overlap")
        border_regimes.add((any(m["border"] for m in train), any(m["border"] for m in test)))
        border_train += any(m["border"] for m in train)
        interior_test += any(not m["border"] for m in test)
        color_changes += len({m["background"] for m in train + test}) > 1
        object_color_changes += len(set.union(*(set(m["object_colors"]) for m in train + test))) > 1
        source_like += (not any(m["border"] for m in train)
                        and all(m["border"] for m in test)
                        and min(m["count"] for m in test) > max(m["count"] for m in train))
        palettes.add(train[0]["labels"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _inspect_pair(pair)
                h, w = m["shape"]
                shapes.add(m["shape"])
                counts.add(m["count"])
                small += min(h, w) < 10
                large += max(h, w) >= 25
                narrow += max(h, w) / min(h, w) >= 2
                mixed += len(m["object_colors"]) > 1
                layouts.add((m["shape"], tuple(sorted(m["positions"]))))
                positions = m["positions"]
                near += any(abs(a[0] - b[0]) < 4 and abs(a[1] - b[1]) < 4
                            for i, a in enumerate(positions) for b in positions[i + 1:])
                sites = [site for position in positions for site in _sites(position)]
                adjacent_labels += any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                                       for i, a in enumerate(sites) for b in sites[i + 1:])
    assert len(shapes) >= 50 and len(layouts) >= 150 and len(palettes) >= 20
    assert len(counts) >= 10 and len(count_regimes) == 3
    assert small >= 10 and large >= 15 and narrow >= 8
    assert mixed >= 10 and near >= 3
    assert border_train >= 5 and interior_test >= 5 and color_changes >= 35
    assert object_color_changes >= 35 and adjacent_labels >= 20 and source_like >= 5
    assert _puzzle_key(generate_puzzle(11)) == _puzzle_key(generate_puzzle(11))
    for i, (nt, nv) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        a = generate_puzzle(700 + i, nt, nv)
        b = generate_puzzle(700 + i, nt, nv)
        assert _puzzle_key(a) == _puzzle_key(b)
        assert (len(a["train"]), len(a["test"])) == (nt, nv)
    for i, (nt, nv) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(800 + i, nt, nv)
        assert (nt is None or len(p["train"]) == nt) and (nv is None or len(p["test"]) == nv)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")
