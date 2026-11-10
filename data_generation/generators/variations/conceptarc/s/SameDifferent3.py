"""Color-occurrence ARC puzzle generator."""
import numpy as np

_SOURCE_INPUTS = {
    ("00000080000", "02220888000", "02220088000", "02220000000", "00000080000", "00111088800", "00111008800", "00111000000", "03000020000", "03000222000", "03000022000"),
    ("00006600", "00006600", "00400000", "04440000", "00400700", "00007770", "77000700", "77000000"),
    ("000000000", "077703330", "007003330", "077703330", "000000000", "033300000", "003000000", "033300000", "000000000"),
    ("0000066600", "0000000600", "0000000000", "0000880440", "0000800400", "0000880440", "0000000000", "0000440000", "0000440000", "0000440000"),
    ("000000000000", "000000000000", "000440000600", "000440700600", "000000700600", "003330777000", "000000000000", "004000400000", "000404000000", "000040000000", "000000000000", "000000000000"),
    ("600000000000", "606660000000", "600000044440", "600000000000", "600000000000", "077703330000", "000000000000", "000000000000", "000000000000", "000000000000", "000000777770", "000000000000"),
}


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(grid, background):
    result = {}
    for value in np.unique(grid):
        color = int(value)
        if color == background:
            continue
        unseen = set(map(tuple, np.argwhere(grid == color)))
        groups = []
        while unseen:
            start = unseen.pop()
            group = {start}
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbor = (r + dr, c + dc)
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            group.add(neighbor)
                            stack.append(neighbor)
            groups.append(frozenset(group))
        result[color] = groups
    return result


def _rule_output(grid):
    background = _background(grid)
    output = grid.copy()
    for color, groups in _components(grid, background).items():
        if len(groups) == 1:
            output[grid == color] = background
    return output


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _shape(rng, size, kind=None):
    if size == 1:
        return ((0, 0),)
    if kind is None:
        kind = str(rng.choice(("solid", "line", "bent", "open", "irregular", "diagonal"),
                              p=(.19, .18, .14, .14, .27, .08)))
    if kind == "diagonal":
        cells = [(i, i) for i in range(size)]
    elif kind == "solid":
        width = 3 if size == 9 else 4 if size == 8 else max(2, int(np.ceil(np.sqrt(size))))
        cells = [(i // width, i % width) for i in range(size)]
    elif kind == "line":
        cells = [(0, i) for i in range(size)]
    elif kind == "bent":
        turn = max(2, size // 2)
        cells = [(0, i) for i in range(turn)] + [(j, turn-1) for j in range(1, size-turn+1)]
    elif kind == "open" and size >= 5:
        cells = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        cells += [(1, 2), (0, 2), (3, 0), (3, 2), (2, 3), (4, 0)][:size-5]
    else:
        cells = {(0, 0)}
        for _ in range(size-1):
            choices = sorted({(r+dr, c+dc) for r, c in cells
                              for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                              if (r+dr, c+dc) not in cells})
            cells.add(choices[int(rng.integers(len(choices)))])
    mode = int(rng.integers(8))
    moved = []
    for r, c in cells:
        if mode & 1:
            r = -r
        if mode & 2:
            c = -c
        if mode & 4:
            r, c = c, r
        moved.append((r, c))
    return _normalize(moved)


def _place_objects(rng, height, width, background, objects, touch_bias):
    objects = sorted(objects, key=lambda item: -len(item[1]))
    for _ in range(30):
        grid = np.full((height, width), background, dtype=np.int8)
        success = True
        for color, cells in objects:
            sh = max(r for r, _ in cells) + 1
            sw = max(c for _, c in cells) + 1
            if sh > height or sw > width:
                success = False
                break
            places = []
            for br in range(height-sh+1):
                for bc in range(width-sw+1):
                    points = [(br+r, bc+c) for r, c in cells]
                    if any(grid[r, c] != background for r, c in points):
                        continue
                    same, contact = False, False
                    for r, c in points:
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                rr, cc = r+dr, c+dc
                                if 0 <= rr < height and 0 <= cc < width:
                                    value = int(grid[rr, cc])
                                    same |= value == color
                                    contact |= value != background and value != color
                    if not same:
                        places.append((contact, br, bc))
            if not places:
                success = False
                break
            choices = [p for p in places if p[0]] if touch_bias and any(p[0] for p in places) else places
            _, br, bc = choices[int(rng.integers(len(choices)))]
            for r, c in cells:
                grid[br+r, bc+c] = color
        if success:
            return grid
    raise RuntimeError("Could not place groups")


def _make_pair(rng, background, kept, removed, recipe, target=False,
               allow_singletons=True):
    for _ in range(45):
        objects = []
        if recipe == "contrast":
            small = int(rng.integers(2, 4))
            objects += [(kept[0], _shape(rng, small)), (kept[0], _shape(rng, small))]
            objects += [(removed[0], _shape(rng, int(rng.choice((8, 9))), "solid"))]
        elif recipe == "diagonal_equal":
            total = int(rng.integers(4, 7))
            objects += [(kept[0], _shape(rng, total//2)),
                        (kept[0], _shape(rng, total-total//2)),
                        (removed[0], _shape(rng, total, "diagonal"))]
        else:
            for color in kept:
                sizes = ([int(rng.integers(2, 4))] * 2 if recipe == "challenge" and color == kept[0]
                         else [int(rng.integers(1 if allow_singletons else 2, 7))
                               for _ in range(int(rng.choice((2, 2, 2, 3, 3, 4))))])
                for size in sizes:
                    objects.append((color, _shape(rng, size)))
            for color in removed:
                size = int(rng.integers(8, 11)) if recipe == "challenge" and color == removed[0] else int(rng.integers(2, 12))
                shared = [cells for k, cells in objects if k in kept]
                shape = (shared[int(rng.integers(len(shared)))]
                         if recipe != "challenge" and shared and rng.random() < .32
                         else _shape(rng, size, "solid" if recipe == "challenge" and color == removed[0] else None))
                objects.append((color, shape))
        total_cells = sum(len(cells) for _, cells in objects)
        min_side = max(8, int(np.ceil(np.sqrt(total_cells * 3))))
        if recipe == "diagonal_equal":
            min_side = max(9, min_side)
        height = min(21, min_side + int(rng.integers(0, 6)) + int(target))
        width = min(22, max(8, min_side + int(rng.integers(-2, 5)) + int(target)))
        if rng.random() < .27:
            height, width = width, height
        try:
            grid = _place_objects(rng, height, width, background, objects,
                                  touch_bias=rng.random() < (.68 if target else .55))
        except RuntimeError:
            continue
        components = _components(grid, background)
        if (all(len(components[c]) >= 2 for c in kept)
                and all(len(components[c]) == 1 for c in removed)
                and np.count_nonzero(grid != background) < .48 * grid.size):
            return {"input": grid, "output": _rule_output(grid)}
    raise RuntimeError("Could not generate pair")


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or isinstance(value, (bool, np.bool_))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10)-1)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    background = 0 if rng.random() < .78 else int(rng.integers(1, 10))
    colors = [c for c in range(10) if c != background]
    rng.shuffle(colors)
    palette = colors[:int(rng.integers(4, 8))]
    first, second = palette[:2]
    seen = set()
    train, test = [], []

    def add(destination, kept, removed, recipe, target=False,
            allow_singletons=True):
        for _ in range(16):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, background, kept, removed, recipe,
                              target, allow_singletons)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                destination.append(pair)
                return
        raise RuntimeError("Could not generate unique pair")

    add(train, (first,), (second,), "contrast")
    add(train, (second,), (first,), "diagonal_equal")
    for index in range(2, num_train):
        available = palette[:min(len(palette), 2+index)]
        size = int(rng.integers(2, min(5, len(available))+1))
        selected = list(map(int, rng.choice(available, size=size, replace=False)))
        rng.shuffle(selected)
        kept_count = int(rng.integers(1, min(3, size-1)+1))
        add(train, tuple(selected[:kept_count]), tuple(selected[kept_count:]), "general")
    taught = {int(v) for pair in train for v in np.unique(pair["input"])} - {background}
    singleton_taught = any(
        len(group) == 1 for pair in train
        for groups in _components(pair["input"], background).values()
        for group in groups
    )
    for index in range(num_test):
        size = int(rng.integers(2, min(5, len(taught))+1))
        selected = list(map(int, rng.choice(sorted(taught), size=size, replace=False)))
        rng.shuffle(selected)
        kept_count = int(rng.integers(1, min(3, size-1)+1))
        add(test, tuple(selected[:kept_count]), tuple(selected[kept_count:]),
            "challenge" if index == 0 else "general", target=True,
            allow_singletons=singleton_taught)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert type(train) is list and type(test) is list
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, backgrounds, taught_colors, roles = set(), set(), set(), []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            bg = _background(inp)
            backgrounds.add(bg)
            if bg == 0:
                rows = tuple("".join(str(int(v)) for v in row) for row in inp)
                assert rows not in _SOURCE_INPUTS
            components = _components(inp, bg)
            assert 2 <= len(components) <= 5
            kept = {c for c, groups in components.items() if len(groups) > 1}
            removed = set(components) - kept
            assert kept and removed
            assert np.array_equal(out, _rule_output(inp))
            assert not np.array_equal(inp, out)
            assert np.count_nonzero(inp != bg) < .5 * inp.size
            if split == "train":
                taught_colors.update(components)
                roles.append((kept, removed))
            else:
                assert set(components) <= taught_colors
    assert len(backgrounds) == 1
    assert roles[0][0] & roles[1][1] and roles[0][1] & roles[1][0]
    first_bg = _background(train[0]["input"])
    first_components = _components(train[0]["input"], first_bg)
    assert max(len(g) for c in roles[0][1] for g in first_components[c]) >= 8
    second_bg = _background(train[1]["input"])
    second_components = _components(train[1]["input"], second_bg)
    assert any(
        len(g) >= 4 and (len({r-c for r, c in g}) == 1
                         or len({r+c for r, c in g}) == 1)
        for color in roles[1][1] for g in second_components[color]
    )
    assert sum(map(len, second_components[next(iter(roles[1][0]))])) == sum(
        map(len, second_components[next(iter(roles[1][1]))])
    )
    challenge = test[0]["input"]
    challenge_groups = _components(challenge, _background(challenge))
    retained_totals = [sum(map(len, groups)) for groups in challenge_groups.values()
                       if len(groups) > 1]
    erased_totals = [len(groups[0]) for groups in challenge_groups.values()
                     if len(groups) == 1]
    assert max(erased_totals) >= min(retained_totals)
    if any(len(g) == 1 for p in test
           for groups in _components(p["input"], _background(p["input"])).values()
           for g in groups):
        assert any(len(g) == 1 for p in train
                   for groups in _components(p["input"], _background(p["input"])).values()
                   for g in groups)


def _freeze(puzzle):
    return tuple(tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    shape_rng = np.random.default_rng(99)
    for kind in ("solid", "line", "bent", "open", "irregular", "diagonal"):
        for size in range(1, 12):
            for _ in range(5):
                cells = _shape(shape_rng, size, kind)
                assert len(cells) == size
                shape_grid = np.zeros((max(r for r, c in cells)+1,
                                       max(c for r, c in cells)+1), dtype=np.int8)
                for r, c in cells:
                    shape_grid[r, c] = 1
                assert len(_components(shape_grid, 0)[1]) == 1
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_freeze(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({pair["input"].shape for p in puzzles for pair in p["train"]+p["test"]}) >= 20
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    for counts in ((2, 1), (10, 4), (10, 1), (None, 4), (2, None), (None, 1), (6, 3)):
        a, b = generate_puzzle(731, *counts), generate_puzzle(731, *counts)
        assert _freeze(a) == _freeze(b)
        if counts[0] is not None:
            assert len(a["train"]) == counts[0]
        if counts[1] is not None:
            assert len(a["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(731, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts accepted")
