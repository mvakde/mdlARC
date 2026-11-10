import numpy as np


_SOURCE_FIRST_ROWS = {
    "6688164664618866", "4922979669792294", "2777111113317772",
    "1664671661764661", "1828979779798281",
}
_SIDES = ("top", "right", "bottom", "left", "center")
_SHAPES = ("single", "separated", "irregular")


def _orbits(n: int) -> dict:
    groups = {}
    for r in range(n):
        for c in range(n):
            a, b = abs(2 * r - n + 1), abs(2 * c - n + 1)
            groups.setdefault((max(a, b), min(a, b)), []).append((r, c))
    return groups


def _counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("Illegal pair count")
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _pattern(rng, n, colors, complexity):
    m = (n + 1) // 2
    wedge = np.zeros((m, m), dtype=np.int8)
    rings = [int(rng.choice(colors))]
    repeat = {"simple": 0.48, "medium": 0.24, "busy": 0.12}[complexity]
    for _ in range(1, m):
        rings.append(rings[-1] if rng.random() < repeat
                     else int(rng.choice(colors)))
    for outer in range(m):
        wedge[outer, :outer + 1] = rings[outer]

    # Short sectors and bands create the repeated local motifs visible in the source.
    bands = {"simple": 2, "medium": 4, "busy": 7}[complexity]
    for _ in range(bands):
        outer0 = int(rng.integers(m))
        outer1 = min(m, outer0 + int(rng.integers(1, max(3, m // 2 + 1))))
        inner0 = int(rng.integers(m))
        inner1 = min(m, inner0 + int(rng.integers(1, max(3, m // 2 + 1))))
        color = int(rng.choice(colors))
        for outer in range(outer0, outer1):
            for inner in range(inner0, min(inner1, outer + 1)):
                wedge[outer, inner] = color

    if rng.random() < 0.82:
        color = int(rng.choice(colors))
        width = int(rng.integers(1, 3))
        for outer in range(m):
            for inner in range(outer + 1):
                if outer - inner < width:
                    wedge[outer, inner] = color
    if rng.random() < 0.78:
        color = int(rng.choice(colors))
        width = int(rng.integers(1, 3))
        for outer in range(m):
            wedge[outer, :min(width, outer + 1)] = color

    if complexity != "simple":
        color = int(rng.choice(colors))
        parity = int(rng.integers(2))
        lo = int(rng.integers(max(1, m // 3)))
        hi = int(rng.integers(max(lo + 1, m // 2), m + 1))
        for outer in range(lo, hi):
            for inner in range(outer + 1):
                if (outer + inner) % 2 == parity:
                    wedge[outer, inner] = color

    mutation = {"simple": 0.07, "medium": 0.19, "busy": 0.34}[complexity]
    for outer in range(m):
        for inner in range(outer + 1):
            if rng.random() < mutation:
                wedge[outer, inner] = int(rng.choice(colors))

    # Every selected color is represented by a complete orbit, not just metadata.
    positions = [(a, b) for a in range(m) for b in range(a + 1)]
    rng.shuffle(positions)
    ordered = list(colors)
    rng.shuffle(ordered)
    for (a, b), color in zip(positions, ordered):
        wedge[a, b] = color

    output = np.empty((n, n), dtype=np.int8)
    for (a, b), cells in _orbits(n).items():
        color = wedge[a // 2, b // 2]
        for r, c in cells:
            output[r, c] = color
    return output


def _rectangle(rng, n, side):
    height = int(rng.integers(2, max(4, n // 3 + 2)))
    width = int(rng.integers(2, max(4, n // 3 + 2)))
    height = min(height, n // 2)
    width = min(width, n // 2)
    if side == "top":
        r0 = int(rng.integers(0, max(1, n // 5 + 1)))
        c0 = int(rng.integers(0, n - width + 1))
    elif side == "bottom":
        r0 = n - height - int(rng.integers(0, max(1, n // 5 + 1)))
        c0 = int(rng.integers(0, n - width + 1))
    elif side == "left":
        r0 = int(rng.integers(0, n - height + 1))
        c0 = int(rng.integers(0, max(1, n // 5 + 1)))
    elif side == "right":
        r0 = int(rng.integers(0, n - height + 1))
        c0 = n - width - int(rng.integers(0, max(1, n // 5 + 1)))
    else:
        r0 = int(rng.integers(max(0, n // 3 - height // 2),
                              min(n - height, 2 * n // 3) + 1))
        c0 = int(rng.integers(max(0, n // 3 - width // 2),
                              min(n - width, 2 * n // 3) + 1))
    return r0, c0, height, width


def _component_cells(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        stack = [remaining.pop()]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        components.append(cells)
    return components


def _components(mask):
    return sorted(len(cells) for cells in _component_cells(mask))


def _mask(rng, n, shape, side):
    mask = np.zeros((n, n), dtype=bool)
    if shape == "irregular":
        r0, c0, height, width = _rectangle(rng, n, side)
        height = min(n - r0, max(4, height))
        left = c0
        for r in range(r0, r0 + height):
            left = max(0, min(n - 2, left + int(rng.integers(-1, 2))))
            span = max(2, min(n - left, width + int(rng.integers(-1, 2))))
            mask[r, left:left + span] = True
        if rng.random() < 0.45:
            other = side if rng.random() < 0.7 else str(rng.choice(_SIDES))
            r, c, h, w = _rectangle(rng, n, other)
            mask[r:r + h, c:c + w] = True
    else:
        r, c, h, w = _rectangle(rng, n, side)
        mask[r:r + h, c:c + w] = True
        if shape == "separated":
            for _ in range(45):
                other = side if rng.random() < 0.7 else str(rng.choice(_SIDES))
                r2, c2, h2, w2 = _rectangle(rng, n, other)
                candidate = mask.copy()
                candidate[r2:r2 + h2, c2:c2 + w2] = True
                if len(_components(candidate)) == 2:
                    mask = candidate
                    break
            else:
                # The corner pair is a bounded, recoverable fallback.
                mask[:] = False
                mask[1:3, 1:3] = True
                mask[n - 3:n - 1, 1:3] = True
    return mask


def _repair(input_grid, damage):
    n = input_grid.shape[0]
    output = input_grid.copy()
    for cells in _orbits(n).values():
        values = {int(input_grid[r, c]) for r, c in cells
                  if int(input_grid[r, c]) != damage}
        if len(values) != 1:
            raise AssertionError("Ambiguous symmetry orbit")
        color = values.pop()
        for r, c in cells:
            if input_grid[r, c] == damage:
                output[r, c] = color
    return output


def _pair(rng, n, damage, shape, side, complexity):
    available = [c for c in range(10) if c != damage]
    color_count = int(rng.integers(4, 9))
    colors = tuple(int(c) for c in rng.choice(available, color_count,
                                              replace=False))
    rich = rng.random() < 0.62
    for _ in range(180):
        output = _pattern(rng, n, colors, complexity)
        mask = _mask(rng, n, shape, side)
        if not 7 <= int(mask.sum()) <= min(75, n * n // 3):
            continue
        if shape == "separated" and len(_components(mask)) != 2:
            continue
        if n % 2 and mask[n // 2, n // 2]:
            continue
        concealed = len(set(int(c) for c in output[mask]))
        if concealed < (min(4, color_count) if rich else 2):
            continue
        inp = output.copy()
        inp[mask] = damage
        try:
            if not np.array_equal(_repair(inp, damage), output):
                continue
        except AssertionError:
            continue
        if n == 16 and "".join(str(int(v)) for v in inp[0]) in _SOURCE_FIRST_ROWS:
            continue
        return inp, output
    raise RuntimeError("Unable to construct an unambiguous damaged design")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    damage = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    signatures = set()

    # The first two demonstrations cover multiple sides and mask structures,
    # while their order and the association between side and structure vary.
    sides = ["left", str(rng.choice([s for s in _SIDES if s != "left"]))]
    shapes = ["separated", str(rng.choice(("single", "irregular")))]
    rng.shuffle(sides)
    rng.shuffle(shapes)
    first_size = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            side = sides[index] if split == "train" and index < 2 else str(rng.choice(_SIDES))
            shape = shapes[index] if split == "train" and index < 2 else str(
                rng.choice(_SHAPES, p=(0.29, 0.46, 0.25) if split == "train"
                           else (0.22, 0.49, 0.29)))
            complexity = str(rng.choice(("simple", "medium", "busy"),
                                        p=(0.15, 0.29, 0.56)))
            sizes = list(range(10, 25))
            if split == "train" and index == 1:
                sizes = [n for n in sizes if n != first_size]
            n = int(rng.choice(sizes))
            if split == "train" and index == 0:
                first_size = n
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                inp, out = _pair(pair_rng, n, damage, shape, side, complexity)
                signature = (inp.shape, inp.tobytes())
                if signature not in signatures:
                    break
            else:
                raise RuntimeError("Unable to generate a unique pair")
            signatures.add(signature)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    damage_colors = set()
    input_keys = set()
    pair_keys = set()
    train_sizes = set()
    separated_train = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert grid.shape[0] == grid.shape[1]
                assert 1 <= grid.shape[0] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.tobytes())
            assert key not in input_keys and pair_key not in pair_keys
            input_keys.add(key)
            pair_keys.add(pair_key)
            n = inp.shape[0]
            if n == 16:
                assert "".join(str(int(v)) for v in inp[0]) not in _SOURCE_FIRST_ROWS
            extra = set(int(c) for c in np.unique(inp)) - set(int(c) for c in np.unique(out))
            assert len(extra) == 1
            damage = extra.pop()
            damage_colors.add(damage)
            mask = inp == damage
            assert 7 <= int(mask.sum()) <= min(75, n * n // 3)
            assert not (n % 2 and mask[n // 2, n // 2])
            assert np.array_equal(inp[~mask], out[~mask])
            assert np.all(inp[mask] != out[mask])
            assert len(np.unique(out)) >= 4
            assert len(set(int(c) for c in out[mask])) >= 2
            assert np.array_equal(_repair(inp, damage), out)
            for cells in _orbits(n).values():
                assert len({int(out[r, c]) for r, c in cells}) == 1
            if split == "train":
                train_sizes.add(n)
                if len(_components(mask)) >= 2:
                    separated_train = True
    assert len(damage_colors) == 1
    assert len(train_sizes) >= 2
    assert separated_train


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    signatures = set()
    sizes = set()
    odd = even = multi_train = multi_test = single_train = single_test = 0
    irregular_train = irregular_test = rich_train = rich_test = 0
    simple_design = busy_design = near_edge = near_center = 0
    varied_shapes = 0
    train_sides, test_sides = set(), set()
    train_counts, test_counts, damage_colors = set(), set(), set()
    novel_later = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        shapes_this_puzzle = set()
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_colors = set(int(c) for c in np.unique(puzzle["train"][0]["output"]))
        damage = int(next(iter(set(np.unique(puzzle["train"][0]["input"])) - first_colors)))
        damage_colors.add(damage)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                n = inp.shape[0]
                sizes.add(n)
                odd += n % 2
                even += 1 - n % 2
                mask = inp == damage
                cells = _component_cells(mask)
                components = len(cells)
                irregular = any(
                    len(component) !=
                    (max(r for r, _ in component) - min(r for r, _ in component) + 1)
                    * (max(c for _, c in component) - min(c for _, c in component) + 1)
                    for component in cells
                )
                shape = "irregular" if irregular else (
                    "multi" if components >= 2 else "single")
                shapes_this_puzzle.add(shape)
                rows, columns = np.nonzero(mask)
                distances = (rows.min(), n - 1 - rows.max(),
                             columns.min(), n - 1 - columns.max())
                side = ("top", "bottom", "left", "right")[int(np.argmin(distances))]
                near_edge += min(distances) <= 1
                near_center += bool(np.any(mask[n // 3:2 * n // 3 + 1,
                                                n // 3:2 * n // 3 + 1]))
                changes = (np.count_nonzero(out[1:] != out[:-1])
                           + np.count_nonzero(out[:, 1:] != out[:, :-1]))
                detail = changes / (2 * n * (n - 1))
                simple_design += detail < 0.55
                busy_design += detail >= 0.70
                if split == "train":
                    multi_train += components >= 2
                    single_train += components == 1
                    irregular_train += irregular
                    rich_train += len(np.unique(out[mask])) >= 4
                    train_sides.add(side)
                else:
                    multi_test += components >= 2
                    single_test += components == 1
                    irregular_test += irregular
                    rich_test += len(np.unique(out[mask])) >= 4
                    test_sides.add(side)
                assert np.array_equal(_repair(inp, damage), out)
                assert set(int(c) for c in np.unique(out)) <= set(range(10)) - {damage}
                assert n != 16 or "".join(str(int(v)) for v in inp[0]) not in _SOURCE_FIRST_ROWS
        novel_later += any(set(int(c) for c in np.unique(p["output"])) - first_colors
                           for p in puzzle["train"][1:] + puzzle["test"])
        varied_shapes += len(shapes_this_puzzle) >= 2
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for s in ("train", "test") for p in puzzle[s]))
    assert len(signatures) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 10 and odd >= 50 and even >= 50
    assert len(damage_colors) >= 8
    assert multi_train >= 42 and multi_test >= 10
    assert single_train >= 10 and single_test >= 3
    assert irregular_train >= 10 and irregular_test >= 3
    assert rich_train >= 100 and rich_test >= 25
    assert simple_design >= 10 and busy_design >= 70
    assert near_edge >= 80 and near_center >= 70
    assert varied_shapes >= 35
    assert len(train_sides) == 4 and len(test_sides) == 4
    assert novel_later >= 20
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(1000, *counts)
        validate_generated_puzzle(puzzle)
        assert counts[0] is None or len(puzzle["train"]) == counts[0]
        assert counts[1] is None or len(puzzle["test"]) == counts[1]
        assert _same_puzzle(puzzle, generate_puzzle(1000, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
