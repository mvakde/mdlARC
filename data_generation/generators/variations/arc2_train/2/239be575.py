"""Synthetic examples for the two-terminal connector decision task."""

import heapq
import numpy as np

_SOURCE_ROWS = (
    ("8800220", "0880228", "0008080", "8000000", "0220808", "0228808"),
    ("8000080", "0022080", "8022000", "0080080", "0082208", "8002280"),
    ("800228", "808220", "000080", "228080", "220008", "088080"),
    ("0800000", "2208880", "2288022", "0080022", "0800800"),
    ("8228800", "0220008", "0880080", "0080008", "8088822", "8000022"),
    ("00808", "22800", "22008", "00022", "88022"),
    ("080000", "000822", "088822", "080008", "000800", "822008", "022000", "080880"),
    ("228808", "220800", "880008", "088800", "808008", "008220", "800220", "080008"),
)
_SOURCE_KEYS = frozenset((len(x), len(x[0]), tuple(int(c) for row in x for c in row)) for x in _SOURCE_ROWS)
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _key(a):
    return (a.shape[0], a.shape[1], tuple(int(x) for x in a.flat))


def _rect(origin, size):
    r, c = origin
    h, w = size
    return frozenset((y, x) for y in range(r, r + h) for x in range(c, c + w))


def _distance(a, b):
    return min(max(abs(y - v), abs(x - u)) for y, x in a for v, u in b)


def _adjacent(a, b, diagonal=True):
    return any((max(abs(y - v), abs(x - u)) == 1 if diagonal else abs(y - v) + abs(x - u) == 1)
               for y, x in a for v, u in b)


def _neighbors(p, h, w, diagonal=True):
    y, x = p
    for dy, dx in (_D8 if diagonal else _D4):
        q = (y + dy, x + dx)
        if 0 <= q[0] < h and 0 <= q[1] < w:
            yield q


def _components(cells, h, w, diagonal=True):
    remaining = set(cells)
    result = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            for q in _neighbors(stack.pop(), h, w, diagonal):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(frozenset(component))
    return result


def _bridge(cells, blocks, h, w, diagonal_links=True, diagonal_contact=True):
    return any(_adjacent(comp, blocks[0], diagonal_contact) and
               _adjacent(comp, blocks[1], diagonal_contact)
               for comp in _components(cells, h, w, diagonal_links))


def _shortest_bridge(cells, blocks, h, w):
    """Minimum number of connector cells in a terminal-to-terminal chain."""
    starts = {p for p in cells if _adjacent((p,), blocks[0])}
    goals = {p for p in cells if _adjacent((p,), blocks[1])}
    seen = set(starts)
    frontier = list(starts)
    length = 1
    while frontier:
        if any(p in goals for p in frontier):
            return length
        following = []
        for p in frontier:
            for q in _neighbors(p, h, w):
                if q in cells and q not in seen:
                    seen.add(q)
                    following.append(q)
        frontier = following
        length += 1
    return None


def _rectangles_of_color(grid, color):
    h, w = grid.shape
    cells = {(int(y), int(x)) for y, x in np.argwhere(grid == color)}
    comps = _components(cells, h, w, False)
    if len(comps) != 2:
        return None
    sizes = []
    for comp in comps:
        ys, xs = zip(*comp)
        top, left, bottom, right = min(ys), min(xs), max(ys), max(xs)
        shape = (bottom - top + 1, right - left + 1)
        if min(shape) < 2 or _rect((top, left), shape) != comp:
            return None
        sizes.append(shape)
    if _distance(comps[0], comps[1]) < 2:
        return None
    return tuple(sorted(comps, key=lambda b: (min(y for y, _ in b), min(x for _, x in b)))), tuple(sizes)


def _analyze(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]]
    background = int(colors[order[-1]])
    candidates = []
    for color in colors:
        if int(color) != background:
            found = _rectangles_of_color(grid, int(color))
            if found is not None:
                candidates.append((int(color), found))
    assert len(candidates) == 1
    terminal, (blocks, sizes) = candidates[0]
    connector = next(int(c) for c in colors if int(c) not in (background, terminal))
    cells = frozenset((int(y), int(x)) for y, x in np.argwhere(grid == connector))
    h, w = grid.shape
    return {
        "background": background, "terminal": terminal, "connector": connector,
        "blocks": blocks, "sizes": sizes, "cells": cells,
        "bridge": _bridge(cells, blocks, h, w),
        "shortest": _shortest_bridge(cells, blocks, h, w),
        "components": _components(cells, h, w),
        "touches": tuple(any(_adjacent(comp, block) for comp in _components(cells, h, w)) for block in blocks),
    }


def _counts(rng, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} is outside its legal integer range")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    return int(num_train), int(num_test)


def _dimensions(rng, regime):
    # Both splits cover compact grids; tests more often ask for long chains and shifts.
    bucket = int(rng.choice(4, p=([.35, .38, .22, .05] if regime == "source_train" else [.20, .34, .34, .12])))
    ranges = ((5, 7), (8, 13), (14, 23), (24, 30))
    lo, hi = ranges[bucket]
    h = int(rng.integers(lo, hi + 1))
    w = int(rng.integers(lo, hi + 1))
    if bucket in (1, 2, 3) and rng.random() < .42:
        if rng.random() < .5:
            h = int(rng.integers(5, min(12, h) + 1))
        else:
            w = int(rng.integers(5, min(12, w) + 1))
    if bucket == 3 and rng.random() < .13:
        h = w = 30
    return h, w


def _choose_blocks(rng, h, w, sizes, orientation, close=False):
    sh1, sw1 = sizes[0]
    sh2, sw2 = sizes[1]
    if sh1 > h or sh2 > h or sw1 > w or sw2 > w:
        return None
    for _ in range(160):
        a = (int(rng.integers(0, h - sh1 + 1)), int(rng.integers(0, w - sw1 + 1)))
        if close:
            offsets = ((3, 0), (-3, 0), (0, 3), (0, -3), (3, 1), (-3, -1), (1, 3), (-1, -3))
            dy, dx = offsets[int(rng.integers(len(offsets)))]
            b = (a[0] + dy, a[1] + dx)
            if not (0 <= b[0] <= h - sh2 and 0 <= b[1] <= w - sw2):
                continue
        else:
            b = (int(rng.integers(0, h - sh2 + 1)), int(rng.integers(0, w - sw2 + 1)))
        blocks = (_rect(a, sizes[0]), _rect(b, sizes[1]))
        if _distance(*blocks) < 2:
            continue
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        if orientation == "horizontal" and dc < dr + 2:
            continue
        if orientation == "vertical" and dr < dc + 2:
            continue
        if orientation == "diagonal" and (dr < 2 or dc < 2):
            continue
        return blocks
    return None


def _random_path(rng, free, blocks, h, w):
    starts = [p for p in free if _adjacent((p,), blocks[0])]
    goals = {p for p in free if _adjacent((p,), blocks[1])}
    if not starts or not goals:
        return None
    blocked = {p for p in free if rng.random() < .13}
    blocked.difference_update(starts)
    blocked.difference_update(goals)
    weights = {p: float(rng.uniform(.7, 4.2)) for p in free}
    heap = []
    dist = {}
    prev = {}
    for p in starts:
        dist[p] = weights[p]
        prev[p] = None
        heapq.heappush(heap, (dist[p], p))
    goal = None
    while heap:
        score, p = heapq.heappop(heap)
        if score != dist[p]:
            continue
        if p in goals:
            goal = p
            break
        for q in _neighbors(p, h, w):
            if q not in free or q in blocked:
                continue
            new = score + weights[q]
            if new < dist.get(q, float("inf")):
                dist[q], prev[q] = new, p
                heapq.heappush(heap, (new, q))
    if goal is None:
        return None
    path = []
    while goal is not None:
        path.append(goal)
        goal = prev[goal]
    return path[::-1]


def _add_texture(rng, cells, free, blocks, h, w, target, positive):
    cells = set(cells)
    available = list(free - cells)
    attempts = 0
    while len(cells) < target and available and attempts < max(80, target * 18):
        attempts += 1
        if rng.random() < .68:
            choices = [p for p in available if any(q in cells for q in _neighbors(p, h, w))]
            if not choices:
                choices = available
        else:
            choices = available
        p = choices[int(rng.integers(len(choices)))]
        available.remove(p)
        if positive or not _bridge(cells | {p}, blocks, h, w):
            cells.add(p)
    return frozenset(cells)


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _paint(h, w, blocks, cells, palette):
    background, terminal, connector = palette
    grid = np.full((h, w), background, dtype=np.int8)
    for block in blocks:
        for y, x in block:
            grid[y, x] = terminal
    for y, x in cells:
        grid[y, x] = connector
    return grid


def _witness(rng, palette):
    h = int(rng.integers(9, 14))
    w = int(rng.integers(9, 14))
    a = (int(rng.integers(0, h - 6)), int(rng.integers(0, w - 6)))
    b = (a[0] + 5, a[1] + 5)
    blocks = (_rect(a, (2, 2)), _rect(b, (2, 2)))
    cells = {(a[0] + k, a[1] + k) for k in (2, 3, 4)}
    grid = _paint(h, w, blocks, cells, palette)
    if rng.random() < .5:
        grid = np.fliplr(grid).copy()
    if rng.random() < .5:
        grid = np.flipud(grid).copy()
    if rng.random() < .5:
        grid = grid.T.copy()
    return grid


def _make_pair(rng, regime, positive, palette, expanded=False, sparse=False):
    for _ in range(130):
        h, w = _dimensions(rng, regime)
        if expanded and min(h, w) < 7:
            h = max(h, 7)
            w = max(w, 7)
        if expanded:
            shapes = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
            one = shapes[int(rng.integers(len(shapes)))]
            two = shapes[int(rng.integers(len(shapes)))] if rng.random() < .68 else (2, 2)
            sizes = (one, two)
        else:
            sizes = ((2, 2), (2, 2))
        orientation = ("horizontal", "vertical", "diagonal", "any")[int(rng.integers(4))]
        blocks = _choose_blocks(rng, h, w, sizes, orientation,
                                close=sparse and positive and not expanded and rng.random() < .55)
        if blocks is None:
            continue
        terminal_cells = set(blocks[0]) | set(blocks[1])
        free = {(y, x) for y in range(h) for x in range(w)} - terminal_cells
        path = _random_path(rng, free, blocks, h, w)
        if not path:
            continue
        if positive:
            cells = set(path)
            if sparse and len(path) > 2:
                continue
        else:
            if sparse:
                choices = [p for p in free if _adjacent((p,), blocks[0])]
                if not choices:
                    continue
                cells = {choices[int(rng.integers(len(choices)))]}
                if rng.random() < .45:
                    cells.clear()
                    cells.add(next(iter(free)))
            else:
                cuts = list(range(1, len(path) - 1))
                rng.shuffle(cuts)
                cells = None
                for index in cuts:
                    candidate = set(path)
                    candidate.remove(path[index])
                    if not _bridge(candidate, blocks, h, w):
                        cells = candidate
                        break
                if cells is None:
                    left = [p for p in free if _adjacent((p,), blocks[0])]
                    right = [p for p in free if _adjacent((p,), blocks[1])]
                    if not left or not right:
                        continue
                    cells = {left[int(rng.integers(len(left)))], right[int(rng.integers(len(right)))]}
                    if _bridge(cells, blocks, h, w):
                        continue
        max_count = min(len(free) - 1, (h * w - len(terminal_cells) - 1) // 2)
        max_count = max(1, max_count)
        if len(cells) > max_count:
            continue
        if sparse:
            target = len(cells)
        else:
            density = float(rng.uniform(.08, .34))
            target = min(max_count, max(len(cells), int(h * w * density)))
            if rng.random() < .3:
                target = min(max_count, max(len(cells), int(h * w * rng.uniform(.22, .39))))
        cells = _add_texture(rng, cells, free, blocks, h, w, target, positive)
        grid = _paint(h, w, blocks, cells, palette)
        try:
            analysis = _analyze(grid)
        except AssertionError:
            continue
        if analysis["bridge"] != positive or _key(grid) in _SOURCE_KEYS:
            continue
        if positive and not sparse:
            minimum_length = 7 if max(h, w) >= 15 else 4 if max(h, w) >= 8 else 3
            if analysis["shortest"] < minimum_length:
                continue
        if expanded and all(shape == (2, 2) for shape in analysis["sizes"]):
            continue
        return grid
    raise RuntimeError("Unable to construct pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    # Only the adjacency convention is shared. Palette and geometry are visible pair properties.
    stable_palette = train_count <= 3 or rng.random() < .23
    expanded_family = train_count >= 3 and rng.random() < .63
    base_palette = _palette(rng)
    train_labels = [True, False] + [bool(rng.integers(2)) for _ in range(train_count - 2)]
    test_labels = ([True, False] + [bool(rng.integers(2)) for _ in range(test_count - 2)]) if test_count >= 2 else [bool(rng.integers(2))]
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=train_count + test_count, dtype=np.uint64)
    train, test = [], []
    used = set()
    for index, label in enumerate(train_labels + test_labels):
        prng = np.random.default_rng(int(pair_seeds[index]))
        split = "source_train" if index < train_count else "target_test"
        if stable_palette or index == 0:
            palette = base_palette
        else:
            palette = _palette(prng)
            if index == 1:
                while palette == base_palette or palette[0] == base_palette[2]:
                    palette = _palette(prng)
        expanded = expanded_family and (index == 1 or (index > 1 and prng.random() < .43))
        sparse = index > 1 and prng.random() < .12
        for _ in range(30):
            if index == 0:
                grid = _witness(prng, palette)
            else:
                grid = _make_pair(prng, split, label, palette, expanded, sparse)
            if _key(grid) not in used:
                break
        else:
            raise RuntimeError("Unable to construct unique pair")
        used.add(_key(grid))
        analysis = _analyze(grid)
        output = np.array([[analysis["connector"] if analysis["bridge"] else analysis["background"]]], dtype=np.int8)
        (train if index < train_count else test).append({"input": grid, "output": output})
    train = [train[int(i)] for i in rng.permutation(len(train))]
    test = [test[int(i)] for i in rng.permutation(len(test))]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_analyses = []
    test_analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for a in (grid, output):
                assert isinstance(a, np.ndarray) and a.ndim == 2
                assert 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            assert output.shape == (1, 1)
            key = _key(grid)
            assert key not in seen and key not in _SOURCE_KEYS
            seen.add(key)
            analysis = _analyze(grid)
            expected = analysis["connector"] if analysis["bridge"] else analysis["background"]
            assert int(output[0, 0]) == expected
            (train_analyses if split == "train" else test_analyses).append(analysis)
    assert {x["bridge"] for x in train_analyses} == {False, True}
    # A positive demonstration must separately disprove each mixed four/eight-neighbor rule.
    assert any(x["bridge"] and not _bridge(x["cells"], x["blocks"], *puzzle["train"][i]["input"].shape, False, True)
               for i, x in enumerate(train_analyses))
    assert any(x["bridge"] and not _bridge(x["cells"], x["blocks"], *puzzle["train"][i]["input"].shape, True, False)
               for i, x in enumerate(train_analyses))
    if any(any(size != (2, 2) for size in x["sizes"]) for x in test_analyses):
        assert any(any(size != (2, 2) for size in x["sizes"]) for x in train_analyses)
    train_palettes = {(x["background"], x["terminal"], x["connector"]) for x in train_analyses}
    test_palettes = {(x["background"], x["terminal"], x["connector"]) for x in test_analyses}
    if not test_palettes.issubset(train_palettes):
        assert len(train_analyses) >= 4 and len(train_palettes) >= 2


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) == 42
    shapes, orientations, palettes, paths, densities = set(), set(), set(), set(), set()
    varied_shapes = varied_palettes = 0
    expanded_train = expanded_test = 0
    hard_negatives = sparse_examples = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_shapes, local_palettes = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                x = _analyze(grid)
                shapes.add(grid.shape)
                local_shapes.add(grid.shape)
                palette = (x["background"], x["terminal"], x["connector"])
                palettes.add(palette)
                local_palettes.add(palette)
                b0, b1 = x["blocks"]
                cy0 = sum(y for y, _ in b0) / len(b0)
                cx0 = sum(c for _, c in b0) / len(b0)
                cy1 = sum(y for y, _ in b1) / len(b1)
                cx1 = sum(c for _, c in b1) / len(b1)
                dy, dx = abs(cy0 - cy1), abs(cx0 - cx1)
                orientations.add("horizontal" if dx > dy * 1.5 else "vertical" if dy > dx * 1.5 else "diagonal")
                paths.add(x["shortest"])
                densities.add(round(len(x["cells"]) / grid.size, 2))
                if any(size != (2, 2) for size in x["sizes"]):
                    if split == "train": expanded_train += 1
                    else: expanded_test += 1
                if not x["bridge"] and all(x["touches"]):
                    hard_negatives += 1
                if len(x["cells"]) == 1:
                    sparse_examples += 1
        varied_shapes += len(local_shapes) >= 2
        varied_palettes += len(local_palettes) >= 2
    assert len(shapes) >= 30 and len(orientations) == 3 and len(palettes) >= 25
    assert varied_shapes >= 25 and varied_palettes >= 10
    assert expanded_train >= 8 and expanded_test >= 4
    assert hard_negatives >= 20 and sparse_examples >= 1
    assert len({x for x in paths if x is not None}) >= 8 and len(densities) >= 15
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(9000 + i, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for i, args in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(9100 + i, *args)
        assert (args[0] is None or len(p["train"]) == args[0]) and (args[1] is None or len(p["test"]) == args[1])
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (True, None), (None, 1.5)):
        try:
            generate_puzzle(12345, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {args}")
    assert _signature(generate_puzzle(777, 6, 2)) == _signature(generate_puzzle(777, 6, 2))
    assert _signature(generate_puzzle(777, 6, 2)) != _signature(generate_puzzle(778, 6, 2))
