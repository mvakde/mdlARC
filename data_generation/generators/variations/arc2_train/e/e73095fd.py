"""Rectangular chambers in an orthogonal line network."""
import hashlib
import numpy as np

_SOURCE_DIGESTS = {
    "fa7ff6e6d56f6e3cd7b8f3eef9ea6f819cad228f9345cf63c7ebe76dd6325161",
    "65d749681ec2f39af97e07cf43d10118e26e3447a26a33e36d79895c4077d8ca",
    "520fbfd3dc2897610c0d5eab2bfecb96f231ecd2e834042b3390ab5f596518dd",
    "5c60f556d0a61bdd725a3998bdf4d6b6cca5c2c98eefdd96709d6736e976658c",
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} out of range")
    return int(value)


def _counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _regions(grid, background):
    h, w = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    for row in range(h):
        for col in range(w):
            if visited[row, col] or grid[row, col] != background:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and not visited[yy, xx] and grid[yy, xx] == background:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            yield tuple(sorted(cells))


def _arm_length(grid, row, side, structure):
    width = grid.shape[1]
    if side == "left":
        col = 0
        while col < width and grid[row, col] == structure:
            col += 1
        return col
    col = width - 1
    while col >= 0 and grid[row, col] == structure:
        col -= 1
    return width - col - 1


def _find_chambers(grid, background, structure):
    """Internal rectangles and side U frames with equal top/bottom arms."""
    h, w = grid.shape
    found = []
    for cells in _regions(grid, background):
        rows, cols = [p[0] for p in cells], [p[1] for p in cells]
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        if len(cells) != (bottom - top + 1) * (right - left + 1):
            continue
        if top == 0 or bottom == h - 1 or (left == 0 and right == w - 1):
            continue
        if not (np.all(grid[top - 1, left:right + 1] == structure)
                and np.all(grid[bottom + 1, left:right + 1] == structure)):
            continue
        edge = None
        if left == 0:
            edge = "left"
            if not np.all(grid[top:bottom + 1, right + 1] == structure):
                continue
            if (_arm_length(grid, top - 1, edge, structure) != right + 2
                    or _arm_length(grid, bottom + 1, edge, structure) != right + 2):
                continue
        elif right == w - 1:
            edge = "right"
            if not np.all(grid[top:bottom + 1, left - 1] == structure):
                continue
            if (_arm_length(grid, top - 1, edge, structure) != w - left + 1
                    or _arm_length(grid, bottom + 1, edge, structure) != w - left + 1):
                continue
        elif not (np.all(grid[top:bottom + 1, left - 1] == structure)
                  and np.all(grid[top:bottom + 1, right + 1] == structure)):
            continue
        found.append({"cells": cells, "top": top, "bottom": bottom,
                      "left": left, "right": right, "height": bottom - top + 1,
                      "width": right - left + 1, "edge": edge})
    return sorted(found, key=lambda c: (c["top"], c["left"]))


def _negative_pockets(grid, background, structure):
    """Return source-like edge enclosures which the rule leaves unfilled."""
    h, w = grid.shape
    pockets = set()
    for cells in _regions(grid, background):
        rows, cols = [p[0] for p in cells], [p[1] for p in cells]
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        if len(cells) != (bottom - top + 1) * (right - left + 1):
            continue
        if top == 0 and bottom < h - 1 and 0 < left and right < w - 1:
            if (np.all(grid[0:bottom + 2, left - 1] == structure)
                    and np.all(grid[0:bottom + 2, right + 1] == structure)
                    and np.all(grid[bottom + 1, left:right + 1] == structure)):
                pockets.add("top")
        if top > 0 and bottom < h - 1:
            if left == 0 and right < w - 1:
                if (np.all(grid[top - 1, :right + 1] == structure)
                        and np.all(grid[bottom + 1, :right + 1] == structure)
                        and np.all(grid[top:bottom + 1, right + 1] == structure)
                        and _arm_length(grid, top - 1, "left", structure)
                        != _arm_length(grid, bottom + 1, "left", structure)):
                    pockets.add("left")
            if right == w - 1 and left > 0:
                if (np.all(grid[top - 1, left:] == structure)
                        and np.all(grid[bottom + 1, left:] == structure)
                        and np.all(grid[top:bottom + 1, left - 1] == structure)
                        and _arm_length(grid, top - 1, "right", structure)
                        != _arm_length(grid, bottom + 1, "right", structure)):
                    pockets.add("right")
    return pockets


def _signatures(chambers):
    return tuple(sorted(chamber["cells"] for chamber in chambers))


def _apply_rule(grid, background, structure, highlight):
    result = grid.copy()
    for chamber in _find_chambers(grid, background, structure):
        for row, col in chamber["cells"]:
            result[row, col] = highlight
    return result


def _components(grid, structure):
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row, col in np.argwhere(grid == structure):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack, cells = [(row, col)], []
        seen[row, col] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]
                        and not seen[yy, xx] and grid[yy, xx] == structure):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _shape(rng):
    kind = int(rng.choice(5, p=[.16, .20, .20, .34, .10]))
    if kind == 0:
        return 1, 1
    if kind == 1:
        return int(rng.integers(1, 4)), int(rng.integers(4, 8))
    if kind == 2:
        return int(rng.integers(4, 8)), int(rng.integers(1, 4))
    if kind == 3:
        return int(rng.integers(2, 5)), int(rng.integers(2, 6))
    return int(rng.integers(4, 7)), int(rng.integers(4, 7))


def _frame_candidate(grid, background, structure, edge, size, rng, asym=False):
    h, w = grid.shape
    ch, cw = size
    if ch + 2 >= h or cw + 2 >= w:
        return None, None
    if edge in ("left", "right"):
        top = int(rng.integers(1, h - ch))
        left = 0 if edge == "left" else w - cw
    elif edge == "top":
        top = 0
        left = int(rng.integers(1, w - cw))
    else:
        top = int(rng.integers(1, h - ch))
        left = int(rng.integers(1, w - cw))
    bottom, right = top + ch - 1, left + cw - 1
    if not np.all(grid[top:bottom + 1, left:right + 1] == background):
        return None, None
    candidate = grid.copy()
    if edge != "top":
        candidate[top - 1, max(0, left - 1):min(w, right + 2)] = structure
    candidate[bottom + 1, max(0, left - 1):min(w, right + 2)] = structure
    if edge != "left":
        candidate[max(0, top - 1):bottom + 2, left - 1] = structure
    if edge != "right":
        candidate[max(0, top - 1):bottom + 2, right + 1] = structure
    if asym:
        extension = int(rng.integers(2, 5))
        if edge == "left":
            candidate[top - 1, right + 1:min(w, right + 1 + extension)] = structure
        elif edge == "right":
            candidate[bottom + 1, max(0, left - 1 - extension):left - 1] = structure
    cells = tuple((y, x) for y in range(top, bottom + 1) for x in range(left, right + 1))
    return candidate, cells


def _place_frame(grid, background, structure, expected, edge, rng, negative=False):
    for _ in range(220):
        if edge == "top":
            size = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        elif edge is not None:
            size = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        else:
            size = _shape(rng)
        candidate, cells = _frame_candidate(
            grid, background, structure, edge, size, rng,
            asym=negative and edge in ("left", "right"))
        if candidate is None or np.array_equal(candidate, grid):
            continue
        if negative:
            if _signatures(_find_chambers(candidate, background, structure)) == expected:
                return candidate, expected
        else:
            new_expected = tuple(sorted(expected + (cells,)))
            if _signatures(_find_chambers(candidate, background, structure)) == new_expected:
                return candidate, new_expected
    raise RuntimeError("frame placement failed")


def _try_line(grid, positions, protected, background, structure, expected):
    if any(protected[y, x] for y, x in positions):
        return None
    candidate = grid.copy()
    for y, x in positions:
        candidate[y, x] = structure
    if np.array_equal(candidate, grid):
        return None
    if _signatures(_find_chambers(candidate, background, structure)) != expected:
        return None
    return candidate


def _connect(grid, protected, background, structure, expected):
    for _ in range(20):
        components = _components(grid, structure)
        if len(components) <= 1:
            return grid
        pairs = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                a = components[i][::max(1, len(components[i]) // 35)]
                b = components[j][::max(1, len(components[j]) // 35)]
                for p in a:
                    for q in b:
                        pairs.append((abs(p[0] - q[0]) + abs(p[1] - q[1]), p, q))
        pairs.sort(key=lambda item: item[0])
        linked = False
        for _, (y1, x1), (y2, x2) in pairs[:100]:
            for cy, cx in ((y1, x2), (y2, x1)):
                path = ([(y, x1) for y in range(min(y1, cy), max(y1, cy) + 1)]
                        + [(cy, x) for x in range(min(x1, cx), max(x1, cx) + 1)]
                        + [(y, x2) for y in range(min(cy, y2), max(cy, y2) + 1)])
                candidate = _try_line(grid, path, protected, background, structure, expected)
                if candidate is not None:
                    grid = candidate
                    linked = True
                    break
            if linked:
                break
        if not linked:
            break
    return grid


def _extras(grid, protected, background, structure, expected, rng, connected):
    for _ in range(int(rng.integers(0, 7))):
        for _ in range(50):
            h, w = grid.shape
            if connected or bool(rng.integers(2)):
                cells = np.argwhere(grid == structure)
                y, x = map(int, cells[int(rng.integers(len(cells)))])
            else:
                y, x = int(rng.integers(h)), int(rng.integers(w))
            dy, dx = ((0, 1), (0, -1), (1, 0), (-1, 0))[int(rng.integers(4))]
            length = int(rng.integers(2, max(4, min(h, w) + 1)))
            path = [(y + step * dy, x + step * dx) for step in range(length)
                    if 0 <= y + step * dy < h and 0 <= x + step * dx < w]
            candidate = _try_line(grid, path, protected, background, structure, expected)
            if candidate is not None:
                grid = candidate
                break
    if rng.random() < .45:
        for _ in range(45):
            h, w = grid.shape
            ch, cw = int(rng.integers(3, 7)), int(rng.integers(3, 8))
            if ch >= h or cw >= w:
                continue
            y, x = int(rng.integers(0, h - ch)), int(rng.integers(0, w - cw))
            sides = [[(y, xx) for xx in range(x, x + cw)],
                     [(y + ch - 1, xx) for xx in range(x, x + cw)],
                     [(yy, x) for yy in range(y, y + ch)],
                     [(yy, x + cw - 1) for yy in range(y, y + ch)]]
            missing = int(rng.integers(4))
            path = [p for i, side in enumerate(sides) if i != missing for p in side]
            candidate = _try_line(grid, path, protected, background, structure, expected)
            if candidate is not None:
                grid = candidate
                break
    return grid


def _pair(rng, colors, regime, required_edge=None, required_negative=None):
    background, structure, highlight = colors
    for _ in range(100):
        if regime == "source_train":
            h, w = int(rng.integers(13, 28)), int(rng.integers(15, 29))
            probabilities = [.22, .34, .29, .15]
        else:
            h, w = int(rng.integers(12, 31)), int(rng.integers(14, 31))
            probabilities = [.15, .29, .34, .22]
        count = int(rng.choice([2, 3, 4, 5], p=probabilities))
        if required_edge is not None:
            count = max(3, count)
        edge_count = int(rng.choice([0, 1, 2], p=[.33, .49, .18]))
        if required_edge is not None:
            edge_count = max(1, edge_count)
        edge_count = min(edge_count, count - 1)
        edges = [required_edge] if required_edge is not None else []
        while len(edges) < edge_count:
            side = ("left", "right")[int(rng.integers(2))]
            if edges.count(side) < 2:
                edges.append(side)
        grid = np.full((h, w), background, dtype=np.int8)
        expected = ()
        try:
            for edge in edges + [None] * (count - edge_count):
                grid, expected = _place_frame(grid, background, structure, expected, edge, rng)
            if required_negative is not None:
                grid, expected = _place_frame(
                    grid, background, structure, expected, required_negative, rng, negative=True)
            elif rng.random() < .48:
                side = ("left", "right", "top")[int(rng.integers(3))]
                grid, expected = _place_frame(
                    grid, background, structure, expected, side, rng, negative=True)
        except RuntimeError:
            continue
        protected = np.zeros(grid.shape, dtype=bool)
        for cells in expected:
            for y, x in cells:
                protected[y, x] = True
        connected = rng.random() < .55
        if connected:
            grid = _connect(grid, protected, background, structure, expected)
            if len(_components(grid, structure)) != 1:
                continue
        grid = _extras(grid, protected, background, structure, expected, rng, connected)
        if _signatures(_find_chambers(grid, background, structure)) != expected:
            continue
        if required_negative is not None and required_negative not in _negative_pockets(
                grid, background, structure):
            continue
        if connected and len(_components(grid, structure)) != 1:
            continue
        if np.count_nonzero(grid == background) <= np.count_nonzero(grid == structure):
            continue
        return {"input": grid, "output": _apply_rule(grid, background, structure, highlight)}
    raise RuntimeError("pair construction failed")


def _signature(grid):
    return grid.shape, grid.tobytes()


def _source_digest(pair):
    first, second = pair["input"], pair["output"]
    payload = bytes(first.shape) + first.astype(np.uint8).tobytes() + second.astype(np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    highlight = int(rng.integers(10))
    available = [color for color in range(10) if color != highlight]
    rng.shuffle(available)
    palettes = [(available[0], available[1], highlight),
                (available[2], available[3], highlight)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            required_edge = (("left", "right")[index] if split == "train" and index < 2
                             else None)
            required_negative = (("left", "top")[index] if split == "train" and index < 2
                                 else None)
            palette_index = index if split == "train" and index < 2 else int(rng.integers(2))
            colors = palettes[palette_index]
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                regime = "source_train" if split == "train" else "target_test"
                pair = _pair(pair_rng, colors, regime, required_edge, required_negative)
                signature = _signature(pair["input"])
                if signature not in seen and _source_digest(pair) not in _SOURCE_DIGESTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_facts(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    first, second = pair["input"], pair["output"]
    for grid in (first, second):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert 0 <= int(grid.min()) and int(grid.max()) <= 9
    assert first.shape == second.shape
    assert _source_digest(pair) not in _SOURCE_DIGESTS
    changed = first != second
    assert np.any(changed)
    backgrounds = np.unique(first[changed])
    highlights = np.unique(second[changed])
    assert len(backgrounds) == len(highlights) == 1
    background, highlight = int(backgrounds[0]), int(highlights[0])
    input_colors = set(map(int, np.unique(first)))
    assert len(input_colors) == 2 and background in input_colors
    structure = next(iter(input_colors - {background}))
    assert highlight not in input_colors
    assert np.array_equal(second, _apply_rule(first, background, structure, highlight))
    chambers = _find_chambers(first, background, structure)
    assert 2 <= len(chambers) <= 5
    assert all(c["height"] <= 7 and c["width"] <= 7 for c in chambers)
    assert np.count_nonzero(first == background) > np.count_nonzero(first == structure)
    return (background, structure, highlight), chambers


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, highlights = set(), set(), set()
    train_palettes, test_palettes, train_edges = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            colors, chambers = _pair_facts(pair)
            highlights.add(colors[2])
            (train_palettes if split == "train" else test_palettes).add(colors)
            if split == "train":
                train_edges.update(c["edge"] for c in chambers if c["edge"])
            input_signature = _signature(pair["input"])
            pair_signature = input_signature, _signature(pair["output"])
            assert input_signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            assert any(c["edge"] is None for c in chambers)
    assert len(highlights) == 1 and len(train_palettes) >= 2
    assert test_palettes <= train_palettes and {"left", "right"} <= train_edges
    for index, side in enumerate(("left", "right")):
        colors, chambers = _pair_facts(puzzle["train"][index])
        assert any(c["edge"] == side for c in chambers)
    assert "left" in _negative_pockets(
        puzzle["train"][0]["input"], *_pair_facts(puzzle["train"][0])[0][:2])
    assert "top" in _negative_pockets(
        puzzle["train"][1]["input"], *_pair_facts(puzzle["train"][1])[0][:2])


def _equal_puzzles(first, second):
    return (all(len(first[s]) == len(second[s]) for s in ("train", "test"))
            and all(np.array_equal(a[k], b[k])
                    for s in ("train", "test")
                    for a, b in zip(first[s], second[s])
                    for k in ("input", "output")))


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes, chamber_counts = (set() for _ in range(6))
    connected, disconnected = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                colors, chambers = _pair_facts(pair)
                palettes.add(colors)
                shapes.add((split, pair["input"].shape))
                chamber_counts.add((split, len(chambers)))
                layouts.add((split, pair["input"].shape,
                             tuple(sorted((c["height"], c["width"], c["edge"] or "")
                                          for c in chambers))))
                if len(_components(pair["input"], colors[1])) == 1:
                    connected += 1
                else:
                    disconnected += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(layouts) >= 100 and len(palettes) >= 24
    assert connected >= 15 and disconnected >= 15 and len(chamber_counts) >= 6
    assert _equal_puzzles(generate_puzzle(918273, 7, 3), generate_puzzle(918273, 7, 3))
    assert not _equal_puzzles(generate_puzzle(1001, 6, 2), generate_puzzle(1002, 6, 2))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4), (4, 4), (2.0, 1), (3, True)):
        try:
            generate_puzzle(4000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
