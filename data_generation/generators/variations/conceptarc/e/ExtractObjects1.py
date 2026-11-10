import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SOURCE_INPUTS = {
    ("0000000030", "0000000000", "0000000000", "0033300000", "0000303000", "0000300000", "0000300000", "0000300000", "0000000000", "0000000030"),
    ("0440000000", "4400000000", "4000000000", "4000000004", "0004000000", "4000000000", "0000000000", "0040000000", "0000000000", "0000000400"),
    ("0006000000", "0000000600", "0000000000", "0666600000", "0600666600", "0600600000", "0666600000", "0000000000", "0000000000", "6006000000"),
    ("0008000000", "0000000000", "8888888000", "0008000000", "0008000000", "0008000000", "0008000000", "0000000000", "0008000000", "0000000000"),
    ("0200000200", "0000000000", "0002222202", "0002000200", "0002000200", "0002000200", "0202222200", "0000000000", "0000220000", "0000000042"),
    ("3000000000", "0007000000", "0007000000", "0007000000", "0037030000", "0007000000", "0037000000", "0007000000", "0000000000", "0000000000"),
}


def _components(grid):
    h, w = grid.shape
    seen = set()
    found = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = set()
            while stack:
                x, y = stack.pop()
                cells.add((x, y))
                for dr, dc in _DIRS:
                    q = (x + dr, y + dc)
                    if (0 <= q[0] < h and 0 <= q[1] < w
                            and q not in seen and int(grid[q]) == color):
                        seen.add(q)
                        stack.append(q)
            found.append((color, cells))
    return found


def _solution(grid):
    comps = _components(grid)
    assert comps
    sizes = [len(cells) for _, cells in comps]
    top = max(sizes)
    assert sizes.count(top) == 1
    color, cells = comps[sizes.index(top)]
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int8)
    for r, c in cells:
        out[r - r0, c - c0] = color
    return out, color, cells, comps


def _shape(rng, family):
    if family == "line":
        n = int(rng.integers(4, 9))
        return np.ones((n, 1) if int(rng.integers(2)) else (1, n), dtype=np.int8)
    if family == "ring":
        if int(rng.integers(4)) == 0:
            h = w = 5
        else:
            h = int(rng.integers(3, 7))
            w = int(rng.integers(3, 8))
        a = np.zeros((h, w), dtype=np.int8)
        a[0, :] = a[-1, :] = 1
        a[:, 0] = a[:, -1] = 1
        return a
    if family == "wide":
        h = int(rng.integers(3, 6))
        w = int(rng.integers(h + 2, 9))
    else:
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
    a = np.zeros((h, w), dtype=np.int8)
    if family == "branch":
        trunk = int(rng.integers(w))
        a[:, trunk] = 1
        rows = rng.choice(np.arange(h), size=int(rng.integers(2, min(h, 5) + 1)), replace=False)
        for row in rows:
            endpoint = int(rng.integers(w))
            a[int(row), min(trunk, endpoint):max(trunk, endpoint) + 1] = 1
        a[int(rows[0]), :] = 1
    else:
        row = int(rng.integers(h))
        col = int(rng.integers(w))
        a[row, :] = 1
        a[:, col] = 1
        extra = int(rng.integers(1, min(7, h * w - h - w + 2)))
        for _ in range(extra):
            frontier = [(r, c) for r in range(h) for c in range(w)
                        if a[r, c] == 0 and any(0 <= r + dr < h and 0 <= c + dc < w
                                                and a[r + dr, c + dc] for dr, dc in _DIRS)]
            if frontier:
                a[frontier[int(rng.integers(len(frontier)))]] = 1
    if int(rng.integers(2)):
        a = np.fliplr(a)
    if int(rng.integers(2)):
        a = np.rot90(a)
    return a


def _group_shape(rng, size):
    if size == 1:
        return {(0, 0)}
    side = 4 if size <= 8 else 5
    cells = {(int(rng.integers(side)), int(rng.integers(side)))}
    while len(cells) < size:
        frontier = {(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS
                    if 0 <= r + dr < side and 0 <= c + dc < side
                    and (r + dr, c + dc) not in cells}
        cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _place_group(rng, grid, color, size, target, relation=None):
    h, w = grid.shape
    for _ in range(14):
        shape = _group_shape(rng, size)
        gh = max(r for r, _ in shape) + 1
        gw = max(c for _, c in shape) + 1
        choices = []
        for r0 in range(h - gh + 1):
            for c0 in range(w - gw + 1):
                cells = {(r0 + r, c0 + c) for r, c in shape}
                if any(int(grid[p]) != 0 for p in cells):
                    continue
                if any((r + dr, c + dc) not in cells
                       and 0 <= r + dr < h and 0 <= c + dc < w
                       and int(grid[r + dr, c + dc]) == color
                       for r, c in cells for dr, dc in _DIRS):
                    continue
                if relation == "touch" and not any(
                        (r + dr, c + dc) in target
                        for r, c in cells for dr, dc in _DIRS):
                    continue
                if relation == "diagonal" and not any(
                        (r + dr, c + dc) in target
                        for r, c in cells for dr in (-1, 1) for dc in (-1, 1)):
                    continue
                if relation == "inside":
                    rmin = min(r for r, _ in target)
                    rmax = max(r for r, _ in target)
                    cmin = min(c for _, c in target)
                    cmax = max(c for _, c in target)
                    if not all(rmin <= r <= rmax and cmin <= c <= cmax
                               for r, c in cells):
                        continue
                choices.append(cells)
        if choices:
            cells = choices[int(rng.integers(len(choices)))]
            for p in cells:
                grid[p] = color
            return True
    return False


def _embed(rng, shape, color, canvas):
    h, w = shape.shape
    if h > canvas or w > canvas:
        return None
    r0 = int(rng.integers(canvas - h + 1))
    c0 = int(rng.integers(canvas - w + 1))
    grid = np.zeros((canvas, canvas), dtype=np.int8)
    target = {(r0 + r, c0 + c) for r, c in zip(*np.nonzero(shape))}
    for p in target:
        grid[p] = color
    return grid, target


def _pair(rng, mode, family, color, others):
    for _ in range(110):
        shape = _shape(rng, family)
        size = int(np.count_nonzero(shape))
        if mode == "paradox" and not 7 <= size <= 12:
            continue
        canvas = int(rng.integers(13, 18)) if mode == "paradox" else int(rng.integers(10, 18))
        embedded = _embed(rng, shape, color, canvas)
        if embedded is None:
            continue
        grid, target = embedded
        ok = True
        if mode == "diagonal":
            ok = _place_group(rng, grid, color, 1, target, "diagonal")
            if ok and int(rng.integers(2)):
                _place_group(rng, grid, color, int(rng.integers(1, 4)), target)
        elif mode == "paradox":
            other = others[0]
            ok = _place_group(rng, grid, other, size - 1, target, "touch")
            ok = ok and _place_group(rng, grid, other, 4, target)
            ok = ok and _place_group(rng, grid, other, 1, target, "inside")
            if ok and int(rng.integers(2)):
                _place_group(rng, grid, color, 1, target)
        else:
            ngroups = int(rng.integers(1, 5))
            close = int(rng.integers(4)) == 0
            palette = [color] + list(others)
            for j in range(ngroups):
                group_color = int(palette[int(rng.integers(len(palette)))])
                max_size = min(size - 1, 7)
                group_size = max(1, min(size - 1, int(rng.integers(1, max_size + 1))))
                if j == 0 and close:
                    group_size = min(size - 1, int(rng.integers(5, 9)))
                relation = None
                if group_color != color and int(rng.integers(3)) == 0:
                    relation = "touch"
                if not _place_group(rng, grid, group_color, group_size, target, relation):
                    ok = False
                    break
        if not ok:
            continue
        try:
            out, winner, cells, comps = _solution(grid)
        except AssertionError:
            continue
        if winner != color or cells != target or len(comps) < 2:
            continue
        if mode == "paradox" and not any(
                int(np.count_nonzero(grid == other)) > int(np.count_nonzero(grid == color))
                for other in others):
            continue
        if mode == "diagonal" and not _has_diagonal_distractor(grid, cells, color):
            continue
        rows = tuple("".join(str(int(x)) for x in row) for row in grid)
        if rows not in _SOURCE_INPUTS:
            return {"input": grid, "output": out}
    raise RuntimeError("Could not construct a valid pair")


def _has_diagonal_distractor(grid, target, color):
    h, w = grid.shape
    for r, c in target:
        for dr in (-1, 1):
            for dc in (-1, 1):
                q = (r + dr, c + dc)
                if (0 <= q[0] < h and 0 <= q[1] < w and q not in target
                        and int(grid[q]) == color):
                    return True
    return False


def _choose_counts(rng, num_train, num_test):
    if num_train is not None and (type(num_train) is not int or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 to 10")
    if num_test is not None and (type(num_test) is not int or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 to 4")
    legal = [(tr, te) for tr in range(2, 11) for te in range(1, 5)
             if tr > te and (num_train is None or tr == num_train)
             and (num_test is None or te == num_test)]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    return legal[int(rng.integers(len(legal)))]


def _mask_key(out):
    return (out.shape, tuple(int(x != 0) for x in out.flat))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _choose_counts(rng, num_train, num_test)
    families = ("irregular", "branch", "wide", "ring", "line")
    train_palette = tuple(int(x) for x in rng.choice(np.arange(1, 10),
                          size=int(rng.integers(3, 6)), replace=False))
    novel = tuple(c for c in range(1, 10) if c not in train_palette)
    test_novel = int(rng.integers(2)) == 0
    train_families = [families[int(rng.integers(len(families)))] for _ in range(ntrain)]
    if train_families[1] in ("ring", "line"):
        train_families[1] = "branch" if int(rng.integers(2)) else "irregular"
    result = {"train": [], "test": []}
    inputs = set()
    masks = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            if split == "train":
                color = train_palette[i] if i < 2 else int(train_palette[int(rng.integers(len(train_palette)))])
                family = train_families[i]
                mode = "diagonal" if i == 0 else "paradox" if i == 1 else "random"
                mixed = mode == "paradox" or (mode == "random" and int(rng.integers(2)) == 0)
            else:
                pool = novel if test_novel and i == 0 else tuple(range(1, 10))
                color = int(pool[int(rng.integers(len(pool)))])
                family = train_families[int(rng.integers(ntrain))] if int(rng.integers(2)) else families[int(rng.integers(len(families)))]
                if int(rng.integers(4)) == 0:
                    mode = "paradox"
                    if family in ("line", "ring"):
                        family = ("irregular", "branch", "wide")[int(rng.integers(3))]
                    mixed = True
                else:
                    mode = "random"
                    if int(rng.integers(5)) == 0:
                        family = "line"
                    mixed = int(rng.integers(3)) != 0
            available = [c for c in range(1, 10) if c != color]
            other_count = int(rng.integers(1, 3)) if mixed else 0
            others = tuple(int(x) for x in rng.choice(available, size=other_count, replace=False)) if mixed else ()
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for attempt in range(180):
                trial_family = family if attempt < 90 else families[int(pair_rng.integers(len(families)))]
                pair = _pair(pair_rng, mode, trial_family, color, others)
                key = (pair["input"].shape, tuple(int(x) for x in pair["input"].flat))
                mask = _mask_key(pair["output"])
                if key not in inputs and mask not in masks:
                    inputs.add(key)
                    masks.add(mask)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not make a distinct pair")
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, masks = set(), set()
    train_colors = set()
    diagonal_witness = paradox_witness = close_witness = interior_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape[0] == inp.shape[1] and 10 <= inp.shape[0] <= 17
            assert out.shape != inp.shape and np.any(inp == 0)
            key = (inp.shape, tuple(int(x) for x in inp.flat))
            assert key not in seen_inputs
            seen_inputs.add(key)
            mask = _mask_key(out)
            assert mask not in masks
            masks.add(mask)
            if inp.shape == (10, 10):
                rows = tuple("".join(str(int(x)) for x in row) for row in inp)
                assert rows not in _SOURCE_INPUTS
            comps = _components(inp)
            assert len(comps) >= 2
            sizes = sorted((len(cells) for _, cells in comps), reverse=True)
            assert sizes[0] > sizes[1]
            expected, winner, cells, _ = _solution(inp)
            assert np.array_equal(out, expected)
            assert set(int(x) for x in np.unique(out)) <= {0, winner}
            assert int(np.count_nonzero(out)) == len(cells)
            if split == "train":
                train_colors.add(winner)
                diagonal_witness |= _has_diagonal_distractor(inp, cells, winner)
                r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
                c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
                interior_witness |= any(int(inp[r, c]) != 0 and (r, c) not in cells
                                        for r in range(r0, r1 + 1)
                                        for c in range(c0, c1 + 1))
                paradox_witness |= any(int(np.count_nonzero(inp == c)) >
                                       int(np.count_nonzero(inp == winner))
                                       for c in np.unique(inp) if c not in (0, winner))
                close_witness |= sizes[1] >= max(5, int(0.65 * sizes[0]))
    assert len(train_colors) >= 2
    assert diagonal_witness and paradox_witness and close_witness and interior_witness


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"])
        for p, q in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, palettes, canvases, positions = set(), set(), set(), set()
    mixed_train = mixed_test = mono_test = novel_test = 0
    rings = lines = wides = ring_5 = line_7 = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(tuple(int(x) for x in p["input"].flat)
                             for s in ("train", "test") for p in puzzle[s]))
        train_used = {int(x) for p in puzzle["train"] for x in np.unique(p["input"])}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                expected, winner, cells, comps = _solution(inp)
                assert np.array_equal(out, expected)
                assert len(comps) > 1 and len({len(c) for _, c in comps}) > 1
                shapes.add(_mask_key(out))
                palettes.add(tuple(int(x) for x in np.unique(inp)))
                canvases.add(inp.shape)
                positions.add((min(r for r, _ in cells), min(c for _, c in cells)))
                multicolor = len(np.unique(inp)) > 2
                if split == "train":
                    mixed_train += multicolor
                else:
                    mixed_test += multicolor
                    mono_test += not multicolor
                    novel_test += winner not in train_used
                if out.shape[0] == 1 or out.shape[1] == 1:
                    lines += 1
                    line_7 += max(out.shape) == 7
                elif out.shape[0] >= 3 and out.shape[1] >= 3 and np.all(out[0]) and np.all(out[-1]) and np.all(out[:, 0]) and np.all(out[:, -1]) and not np.any(out[1:-1, 1:-1]):
                    rings += 1
                    ring_5 += out.shape == (5, 5)
                if max(out.shape) >= min(out.shape) + 2 and min(out.shape) >= 3:
                    wides += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 80
    assert len(palettes) >= 30 and len(canvases) >= 6 and len(positions) >= 35
    assert mixed_train >= 42 and mixed_test >= 20 and mono_test >= 10 and novel_test >= 10
    assert rings >= 20 and lines >= 20 and wides >= 30
    assert ring_5 >= 3 and line_7 >= 3
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(300 + tr * 10 + te, tr, te)
            validate_generated_puzzle(p)
            assert len(p["train"]) == tr and len(p["test"]) == te
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(501 + (tr or 0) + (te or 0), tr, te)
        validate_generated_puzzle(p)
        assert tr is None or len(p["train"]) == tr
        assert te is None or len(p["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4), (4, 4)):
        try:
            generate_puzzle(55, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
