"""Generate two-object ARC puzzles whose one-cell gap becomes a full divider."""

import numpy as np


_SOURCE_INPUTS = (
    ("0000000000", "0555055000", "0500005000", "0555055000", "0500005000", "0555055000", "0000000000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "0001110000", "0011011000", "0110001100", "0000000000", "0011011000", "0001110000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0022220000", "0020020000", "0000000000", "0020020000", "0022220000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0006060000", "0066066000", "0066066000", "0006060000", "0000000000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "0222200000", "0222200000", "0222200000", "0000000000", "0222200000", "0000000000", "0000000000", "0000000000", "0000000000"),
)
_SOURCE_KEYS = {(10, 10, tuple(int(ch) for row in rows for ch in row)) for rows in _SOURCE_INPUTS}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(v) for v in grid.flat))


def _counts(seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} outside legal range")
    if num_train is None and num_test is None:
        if seed % 11 == 0:
            return 3, 2
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif num_train is None:
        test = int(num_test)
        train = test + 1 + seed % (10 - test)
    elif num_test is None:
        train = int(num_train)
        test = 1 + seed % min(4, train - 1)
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _extent(rng, limit, bar_chance=0.12):
    if limit == 1 or rng.random() < bar_chance:
        return 1
    if rng.random() < 0.36:
        return int(rng.integers(max(2, (limit * 2) // 3), limit + 1))
    return int(rng.integers(2, limit + 1))


def _canvas(rng):
    if rng.random() < 0.16:
        return 10, 10
    def axis():
        band = int(rng.choice(3, p=[0.38, 0.25, 0.37]))
        low, high = ((6, 12), (13, 18), (19, 30))[band]
        return int(rng.integers(low, high + 1))
    return axis(), axis()


def _mask(rng, height, width, solid):
    if solid or min(height, width) == 1:
        return np.ones((height, width), dtype=bool)
    style = int(rng.integers(4))
    if style == 0 and min(height, width) >= 3:
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        return mask
    if style == 1 and height >= 3 and width >= 5:
        mask = np.zeros((height, width), dtype=bool)
        left = (width - 1) // 2
        right = width // 2
        mask[0, left:right+1] = True
        old_left, old_right = left, right
        for row in range(1, height):
            left = ((width - 1) // 2) * (height - 1 - row) // (height - 1)
            right = width - 1 - ((width - 1) // 2) * (height - 1 - row) // (height - 1)
            mask[row, left:old_left+1] = True
            mask[row, old_right:right+1] = True
            old_left, old_right = left, right
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    mask = np.zeros((height, width), dtype=bool)
    steps = [0] * (height - 1) + [1] * (width - 1)
    rng.shuffle(steps)
    row = col = 0
    mask[row, col] = True
    for step in steps:
        if step == 0:
            row += 1
        else:
            col += 1
        mask[row, col] = True
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    density = float(rng.uniform(0.28, 0.82)) if style == 2 else float(rng.uniform(0.62, 0.96))
    target = min(height * width - 1, max(int(mask.sum()), round(density * height * width)))
    while int(mask.sum()) < target:
        frontier = set()
        for r, c in zip(*np.nonzero(mask)):
            for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            break
        r, c = _choice(rng, sorted(frontier))
        mask[r, c] = True
    return mask


def _make_pair(rng, orientation, background, divider, colors, dual, regime, required_irregular=False):
    for attempt in range(160):
        height, width = _canvas(rng)
        along = width if orientation == "vertical" else height
        across = height if orientation == "vertical" else width
        edge_gap = rng.random() < (0.22 if regime == "target_test" else 0.13)
        gap = _choice(rng, (1, along - 2)) if edge_gap else int(rng.integers(2, along - 2))
        left_limit, right_limit = gap, along - gap - 1
        relation = _choice(rng, ("mirror", "equal_other", "unequal", "unequal"))
        if min(left_limit, right_limit) == 1:
            relation = "unequal"
        if relation == "mirror" or relation == "equal_other":
            a_len = b_len = _extent(rng, min(left_limit, right_limit), 0.04)
            a_span = b_span = _extent(rng, across, 0.06)
        else:
            a_len = _extent(rng, left_limit, 0.23 if regime == "target_test" else 0.18)
            b_len = _extent(rng, right_limit, 0.23 if regime == "target_test" else 0.18)
            a_span = _extent(rng, across, 0.14)
            b_span = _extent(rng, across, 0.14)
            if (a_len, a_span) == (b_len, b_span):
                continue
        a_start = int(rng.integers(0, across - a_span + 1))
        if relation == "mirror":
            b_start = a_start
        else:
            b_min = max(0, a_start - b_span + 1)
            b_max = min(across - b_span, a_start + a_span - 1)
            b_start = int(rng.integers(b_min, b_max + 1))
        solid_probability = 0.40 if regime == "target_test" else 0.32
        a_solid = bool(rng.random() < solid_probability or min(a_span, a_len) == 1)
        b_solid = a_solid if relation == "mirror" else bool(rng.random() < solid_probability or min(b_span, b_len) == 1)
        if required_irregular and a_solid and b_solid:
            continue
        a_mask = _mask(rng, a_span, a_len, a_solid)
        b_mask = np.fliplr(a_mask) if relation == "mirror" else _mask(rng, b_span, b_len, b_solid)
        if relation == "equal_other" and np.array_equal(np.fliplr(a_mask), b_mask):
            continue
        if relation == "unequal" and a_mask.shape == b_mask.shape:
            continue
        a_axis = gap - a_len
        b_axis = gap + 1
        canonical = np.full((across, along), background, dtype=np.int8)
        a_color, b_color = colors
        first = canonical[a_start:a_start+a_span, a_axis:gap]
        second = canonical[b_start:b_start+b_span, b_axis:b_axis+b_len]
        first[a_mask] = a_color
        second[b_mask] = b_color if dual else a_color
        inp = canonical if orientation == "vertical" else canonical.T.copy()
        if any(int(inp[r, c]) != background for r, c in ((0,0), (0,inp.shape[1]-1), (inp.shape[0]-1,0), (inp.shape[0]-1,inp.shape[1]-1))):
            continue
        if int(np.count_nonzero(inp == background)) <= inp.size // 2:
            continue
        if _key(inp) in _SOURCE_KEYS:
            continue
        out = inp.copy()
        if orientation == "vertical":
            out[:, gap] = divider
        else:
            out[gap, :] = divider
        return {"input": inp, "output": out}
    raise RuntimeError("pair construction exhausted")


def _orientations(rng, train_count, test_count, source_like):
    if source_like:
        train = ["vertical", "horizontal", "horizontal"]
        test = ["vertical", "horizontal"]
    elif rng.random() < 0.32:
        chosen = _choice(rng, ("vertical", "horizontal"))
        train = [chosen] * train_count
        test = [chosen] * test_count
    else:
        train = ["vertical", "horizontal"] + [_choice(rng, ("vertical", "horizontal")) for _ in range(train_count - 2)]
        rng.shuffle(train)
        test = [_choice(rng, tuple(set(train))) for _ in range(test_count)]
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation(10)]
    divider = palette[0]
    backgrounds = palette[1:3] if rng.random() < 0.34 else palette[1:2]
    object_palette = [c for c in palette if c != divider and c not in backgrounds]
    train_palette = object_palette[:3]
    train_colors = train_palette[:2] + [_choice(rng, train_palette) for _ in range(train_count - 2)]
    rng.shuffle(train_colors)
    train_backgrounds = ([backgrounds[0], backgrounds[1]] + [_choice(rng, backgrounds) for _ in range(train_count - 2)]) if len(backgrounds) == 2 else [backgrounds[0]] * train_count
    rng.shuffle(train_backgrounds)
    dual_enabled = bool(rng.random() < 0.46)
    train_dual = [True] + [bool(rng.random() < 0.42) for _ in range(train_count - 1)] if dual_enabled else [False] * train_count
    rng.shuffle(train_dual)
    source_like = train_count == 3 and test_count == 2 and seed % 11 == 0
    train_orients, test_orients = _orientations(rng, train_count, test_count, source_like)
    irregular_slot = int(rng.integers(train_count))
    puzzle = {"train": [], "test": []}
    used = set()
    for index in range(train_count):
        color = train_colors[index]
        other = _choice(rng, [c for c in train_palette if c != color])
        for _ in range(10):
            pair_rng = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
            pair = _make_pair(pair_rng, train_orients[index], train_backgrounds[index], divider, (color, other), train_dual[index], "source_train", index == irregular_slot)
            key = _key(pair["input"])
            if key not in used:
                break
        else:
            raise RuntimeError("could not make unique training pair")
        used.add(key)
        puzzle["train"].append(pair)
    seen = list(set(train_colors))
    novel = [c for c in object_palette if c not in train_palette]
    test_modes = [_choice(rng, ("seen", "novel")) for _ in range(test_count)]
    if test_count >= 2 and rng.random() < 0.48:
        test_modes[:2] = ["seen", "novel"]
        rng.shuffle(test_modes)
    for index in range(test_count):
        primary = _choice(rng, seen if test_modes[index] == "seen" else novel)
        color_pool = seen if test_modes[index] == "seen" else novel
        secondary = _choice(rng, [c for c in color_pool if c != primary])
        background = _choice(rng, train_backgrounds)
        dual = dual_enabled and bool(rng.random() < 0.53)
        for _ in range(10):
            pair_rng = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
            pair = _make_pair(pair_rng, test_orients[index], background, divider, (primary, secondary), dual, "target_test")
            key = _key(pair["input"])
            if key not in used:
                break
        else:
            raise RuntimeError("could not make unique test pair")
        used.add(key)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for point in ((row-1,col), (row+1,col), (row,col-1), (row,col+1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    frontier.append(point)
        components.append(component)
    return components


def _inspect(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "grid dimensions changed")
    background = int(inp[0, 0])
    h, w = inp.shape
    _require(all(int(inp[r,c]) == background for r,c in ((0,0),(0,w-1),(h-1,0),(h-1,w-1))), "background corners obscured")
    _require(int(np.count_nonzero(inp == background)) > inp.size // 2, "background not dominant")
    changed = inp != out
    _require(bool(np.any(changed)), "input equals output")
    colors = np.unique(out[changed])
    _require(len(colors) == 1, "multiple divider colors")
    divider = int(colors[0])
    _require(divider not in inp, "divider already appears in input")
    vertical = [c for c in range(w) if np.all(changed[:,c])]
    horizontal = [r for r in range(h) if np.all(changed[r,:])]
    _require(len(vertical) + len(horizontal) == 1, "output does not add one complete line")
    if vertical:
        orientation, line = "vertical", vertical[0]
        expected = np.zeros_like(changed)
        expected[:,line] = True
        _require(np.all(inp[:,line] == background), "gap column not background")
    else:
        orientation, line = "horizontal", horizontal[0]
        expected = np.zeros_like(changed)
        expected[line,:] = True
        _require(np.all(inp[line,:] == background), "gap row not background")
    _require(np.array_equal(changed, expected), "output changes cells outside divider")
    components = _components(inp != background)
    _require(len(components) == 2, "input must have exactly two connected objects")
    boxes = []
    masks = []
    for component in components:
        rows, cols = zip(*component)
        box = (min(rows), max(rows), min(cols), max(cols))
        boxes.append(box)
        mask = np.zeros((box[1]-box[0]+1, box[3]-box[2]+1), dtype=bool)
        for r,c in component:
            mask[r-box[0], c-box[2]] = True
        masks.append(mask)
        _require(len({int(inp[r,c]) for r,c in component}) == 1, "object contains multiple colors")
    candidates = []
    a,b = boxes
    for first,second in ((a,b),(b,a)):
        if first[3] + 2 == second[2] and max(first[0],second[0]) <= min(first[1],second[1]):
            candidates.append(("vertical",first[3]+1))
        if first[1] + 2 == second[0] and max(first[2],second[2]) <= min(first[3],second[3]):
            candidates.append(("horizontal",first[1]+1))
    _require(candidates == [(orientation,line)], "gap is not unique or line is misplaced")
    if orientation == "vertical":
        order = sorted(range(2), key=lambda i: boxes[i][2])
        mirror = boxes[order[0]][0] == boxes[order[1]][0] and np.array_equal(np.fliplr(masks[order[0]]), masks[order[1]])
        overlap = min(a[1], b[1]) - max(a[0], b[0]) + 1
    else:
        order = sorted(range(2), key=lambda i: boxes[i][0])
        mirror = boxes[order[0]][2] == boxes[order[1]][2] and np.array_equal(np.flipud(masks[order[0]]), masks[order[1]])
        overlap = min(a[3], b[3]) - max(a[2], b[2]) + 1
    spans = [mask.shape for mask in masks]
    return {"background": background, "divider": divider, "orientation": orientation, "line": line,
            "colors": tuple(sorted({int(inp[r,c]) for part in components for r,c in part})),
            "bar": any(min(mask.shape) == 1 for mask in masks), "solid": tuple(bool(np.all(mask)) for mask in masks),
            "mirror": bool(mirror), "equal_dims": spans[0] == spans[1],
            "bar_sides": tuple(side for side, index in enumerate(order) if min(masks[index].shape) == 1),
            "max_extent": max(max(mask.shape) for mask in masks), "overlap": overlap,
            "crossless": any(not np.all(mask) and not any(np.all(mask[r,:]) for r in range(mask.shape[0])) and not any(np.all(mask[:,c]) for c in range(mask.shape[1])) for mask in masks)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen_inputs = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), "invalid grid type")
                _require(all(1 <= n <= 30 for n in grid.shape), "invalid grid size")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
            key = _key(pair["input"])
            _require(key not in seen_inputs and key not in _SOURCE_KEYS, "duplicate or source input")
            seen_inputs.add(key)
            info[split].append(_inspect(pair))
    train, test = info["train"], info["test"]
    _require(len({i["divider"] for i in train + test}) == 1, "divider color varies")
    _require(len({c for i in train for c in i["colors"]}) >= 2, "training foreground colors do not vary")
    _require(any(not all(i["solid"]) for i in train), "training lacks irregular objects")
    _require({i["orientation"] for i in test} <= {i["orientation"] for i in train}, "test orientation absent in training")
    _require({i["background"] for i in test} <= {i["background"] for i in train}, "test background absent in training")
    if any(len(i["colors"]) == 2 for i in test):
        _require(any(len(i["colors"]) == 2 for i in train), "dual object colors absent in training")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts insufficient")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts insufficient")
    for seed, train, test in ((101,2,1),(102,10,1),(103,10,4),(201,None,1),(202,None,4),(203,2,None),(204,10,None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(17,*args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    def signature(puzzle):
        return tuple(_key(pair[name]) for split in ("train","test") for pair in puzzle[split] for name in ("input","output"))
    _require(signature(generate_puzzle(314159,7,3)) == signature(generate_puzzle(314159,7,3)), "nondeterministic")
    _require(len({signature(p) for p in puzzles}) >= 40, "too few distinct puzzles")
    rows = [(split, pair, _inspect(pair)) for puzzle in puzzles for split in ("train","test") for pair in puzzle[split]]
    sizes = {pair["input"].shape for _,pair,_ in rows}
    _require(len(sizes) >= 15 and (10,10) in sizes and any(max(s) >= 19 for s in sizes), "canvas diversity insufficient")
    _require({info["orientation"] for _,_,info in rows} == {"vertical","horizontal"}, "orientation diversity insufficient")
    for split in ("train", "test"):
        group = [info for row_split,_,info in rows if row_split == split]
        relations = {"mirror" if i["mirror"] else "equal_nonmirror" if i["equal_dims"] else "unequal" for i in group}
        _require(relations == {"mirror", "equal_nonmirror", "unequal"}, f"{split} object relations insufficient")
        solidity = {"both" if all(i["solid"]) else "mixed" if i["solid"][0] != i["solid"][1] else "irregular" for i in group}
        _require(solidity == {"both", "mixed", "irregular"}, f"{split} solidity modes insufficient")
        _require(any(i["crossless"] for i in group), f"{split} lacks crossless objects")
        _require(any(len(i["colors"]) == 2 for i in group), f"{split} lacks two-color objects")
        for orientation in ("vertical", "horizontal"):
            oriented = [i for i in group if i["orientation"] == orientation]
            _require(oriented and any(i["bar_sides"] == (0,) for i in oriented) and any(i["bar_sides"] == (1,) for i in oriented), f"{split} lacks bars on both {orientation} sides")
            _require(any(i["line"] in (1, pair["input"].shape[1 if orientation == "vertical" else 0]-2) for row_split,pair,i in rows if row_split == split and i["orientation"] == orientation), f"{split} lacks {orientation} near-edge gap")
    _require(sum(info["crossless"] for _,_,info in rows) >= 25, "crossless shapes too rare")
    _require(any(info["max_extent"] >= 10 for _,_,info in rows), "long objects missing")
    _require(any(info["overlap"] == 1 for _,_,info in rows) and any(info["overlap"] >= 6 for _,_,info in rows), "overlap range insufficient")
    _require(any(info["bar"] and max(pair["input"].shape) >= 19 for _,pair,info in rows), "large bar combinations missing")
    _require(any(len(info["colors"]) == 2 and info["solid"][0] != info["solid"][1] for _,_,info in rows), "dual-color mixed-solidity pairs missing")
    train_orientation_sets = [{_inspect(pair)["orientation"] for pair in puzzle["train"]} for puzzle in puzzles]
    _require(sum(len(values) == 1 for values in train_orientation_sets) >= 5 and sum(len(values) == 2 for values in train_orientation_sets) >= 5, "orientation variety across puzzles insufficient")
    _require(sum(len({_inspect(pair)["background"] for split in ("train","test") for pair in puzzle[split]}) >= 2 for puzzle in puzzles) >= 5, "within-puzzle backgrounds do not vary")
    _require(sum(len({len(_inspect(pair)["colors"]) for split in ("train","test") for pair in puzzle[split]}) == 2 for puzzle in puzzles) >= 5, "within-puzzle color relations do not vary")
    test_mode_sets = []
    seen_positions = set()
    for puzzle in puzzles:
        training_colors = [set(_inspect(pair)["colors"]) for pair in puzzle["train"]]
        training_union = set().union(*training_colors)
        modes = set()
        for pair in puzzle["test"]:
            colors = set(_inspect(pair)["colors"])
            _require(colors <= training_union or colors.isdisjoint(training_union), "test pair mixes seen and novel foregrounds")
            if colors <= training_union:
                modes.add("seen")
                seen_positions.update(index for index, train_colors in enumerate(training_colors) if colors & train_colors)
            else:
                modes.add("novel")
        test_mode_sets.append(frozenset(modes))
    _require(set(test_mode_sets) == {frozenset(("seen",)), frozenset(("novel",)), frozenset(("seen","novel"))}, "seen/novel test mixtures missing")
    _require({0,1,2} <= seen_positions, "test colors only repeat one training position")
