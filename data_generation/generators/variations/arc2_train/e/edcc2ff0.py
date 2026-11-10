import numpy as np


# The upper background and the number of swatches are shared within a puzzle.
# All other visible geometry and color roles belong to individual pairs.
_SOURCE_INPUTS = (
    '''0000000000 2000000000 0000000000 8000000000 0000000000 4000000000 0000000000 3333333333 3113332383 3113332383 3333332383 3322333383 3333333333 3388333333 3388344433 3333333333 3223333333 3223333223 3333333333 3333333333''',
    '''0000000000 4000000000 0000000000 2000000000 0000000000 3000000000 0000000000 1111111111 1221111111 1211122131 1111122131 1111111111 1111111111 1181133111 1181133111 1181111111 1111111111 1113333111 1111111111 1111111111''',
    '''0000000000 1000000000 0000000000 2000000000 0000000000 4000000000 0000000000 8888888888 8228888848 8888881848 8684481888 8684488888 8888888888 8482288338 8488888888 8888866888 8818866888 8818888888 8888844448 8888888888''',
    '''0000000000 2000000000 0000000000 8000000000 0000000000 3000000000 0000000000 4444444444 4222443344 4444444444 4774442244 4444442244 4334444444 4334444424 4444554424 4444554424 4444444444 4224444444 4224442244 4444444444''',
)
_SOURCE_GRIDS = tuple(np.array([[int(c) for c in row] for row in text.split()], dtype=np.int8)
                      for text in _SOURCE_INPUTS)

_BASE_SHAPES = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)),),
    3: (((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (1, 1))),
    4: (((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2))),
    5: (((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2))),
    6: (((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),),
}


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _resolve_pair_counts(rng, num_train, num_test, seed):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = num_test + 1
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _shape(rng, size):
    pool = _BASE_SHAPES[size]
    cells = pool[int(rng.integers(len(pool)))]
    turns = int(rng.integers(4))
    for _ in range(turns):
        cells = tuple((c, -r) for r, c in cells)
    if bool(rng.integers(2)):
        cells = tuple((r, -c) for r, c in cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple((r - r0, c - c0) for r, c in cells)


def _components(field, background):
    height, width = field.shape
    seen = np.zeros(field.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            color = int(field[r, c])
            if seen[r, c] or color == background:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < height and 0 <= xx < width and not seen[yy, xx]
                            and int(field[yy, xx]) == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append((color, tuple(cells)))
    return result


def _inspect_input(grid):
    height, width = grid.shape
    upper = int(grid[0, 0])
    # Every lower-field cell differs from the neutral upper background.
    field_start = next((r for r in range(1, height)
                        if np.count_nonzero(grid[r] != upper) >= width // 2), None)
    _check(field_start is not None and field_start < height - 2, "field boundary missing")
    query_rows = []
    query_cols = []
    query_colors = []
    for r in range(field_start):
        positions = np.flatnonzero(grid[r] != upper)
        _check(len(positions) <= 1, "upper row is not blank or a singleton query")
        if len(positions):
            query_rows.append(r)
            query_cols.append(int(positions[0]))
            query_colors.append(int(grid[r, positions[0]]))
    _check(2 <= len(query_rows) <= 4, "query count outside 2..4")
    _check(len(set(query_colors)) == len(query_colors), "query colors repeat")
    _check(query_rows[0] >= 1 and field_start - query_rows[-1] >= 2,
           "query panel needs blank border rows")
    _check(all(b-a >= 2 for a, b in zip(query_rows, query_rows[1:])),
           "query swatches need blank separating rows")
    field = grid[field_start:]
    _check(np.all(field != upper), "upper background enters lower field")
    frequencies = np.bincount(field.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(frequencies))
    _check(np.count_nonzero(frequencies == frequencies[background]) == 1,
           "field background is not uniquely modal")
    _check(background not in query_colors, "field background is queried")
    records = _components(field, background)
    sizes = {}
    for color, cells in records:
        sizes.setdefault(color, []).append(len(cells))
        _check(len(cells) <= 6, "object exceeds supported small-shape range")
    counts = tuple(len(sizes.get(color, ())) for color in query_colors)
    _check(all(col + count <= width for col, count in zip(query_cols, counts)),
           "unary bar does not fit")
    distractors = set(sizes) - set(query_colors)
    return dict(upper=upper, field_start=field_start, background=background,
                query_rows=tuple(query_rows), query_cols=tuple(query_cols),
                query_colors=tuple(query_colors), counts=counts,
                distractors=frozenset(distractors), sizes=sizes,
                field_shape=field.shape, records=records)


def _solve_input(grid):
    info = _inspect_input(grid)
    result = grid.copy()
    upper = info["upper"]
    for row, col, color, count in zip(info["query_rows"], info["query_cols"],
                                       info["query_colors"], info["counts"]):
        result[row] = upper
        result[row, col:col+count] = color
    field = result[info["field_start"]:]
    original = grid[info["field_start"]:]
    keep = np.isin(original, (info["background"],) + info["query_colors"])
    field[~keep] = info["background"]
    return result


def _roles(rng, available, nquery, first_info=None, forced=False, training=False):
    colors = [int(x) for x in rng.permutation(available)]
    if forced:
        background = int(rng.choice(first_info["query_colors"]))
        mandatory = [first_info["background"],
                     int(rng.choice(tuple(sorted(first_info["distractors"]))))]
        queries = mandatory + [c for c in colors if c not in mandatory and c != background][:nquery-2]
        queries = [int(x) for x in rng.permutation(queries)]
    else:
        background = colors[0]
        queries = colors[1:1+nquery]
    remaining = [c for c in colors if c != background and c not in queries]
    ndistractor = int(rng.integers(1 if training and first_info is None else 0,
                                   min(3, len(remaining)) + 1))
    distractors = remaining[:ndistractor]
    return background, tuple(queries), tuple(distractors)


def _query_counts(rng, nquery, regime, index, seen_tests):
    if regime == "source_train" and index == 0:
        values = [2, 2] + [int(rng.integers(1, 5)) for _ in range(nquery-2)]
    elif regime == "source_train" and index == 1:
        values = [0, 3] + [int(rng.integers(1, 5)) for _ in range(nquery-2)]
    else:
        for _ in range(30):
            values = [int(x) for x in rng.integers(0, 5, size=nquery)]
            if regime == "target_test":
                if rng.random() < 0.42:
                    values[int(rng.integers(nquery))] = int(rng.integers(5, 7))
                if rng.random() < 0.38:
                    values[int(rng.integers(nquery))] = 0
            if any(values) and tuple(values) not in seen_tests:
                break
        else:
            values = [1 + (index + j) % 4 for j in range(nquery)]
    values = tuple(int(x) for x in rng.permutation(values))
    if regime == "target_test" and values in seen_tests:
        # Count questions should differ when a puzzle has multiple tests.
        values = tuple((x + 1) % 5 for x in values)
    return values


def _layout(rng, nquery):
    row = int(rng.integers(1, 3))
    rows = []
    for i in range(nquery):
        rows.append(row)
        row += 1 + int(rng.integers(1, 3))
    return tuple(rows), row


def _make_entries(rng, queries, counts, distractors, regime, index):
    entries = []
    special = {}
    if regime == "source_train" and index == 0:
        twos = [i for i, count in enumerate(counts) if count == 2]
        if len(twos) >= 2:
            special[twos[0]] = (1, 3)
            special[twos[1]] = (3, 4)
    if regime == "source_train" and index == 1:
        threes = [i for i, count in enumerate(counts) if count == 3]
        if threes:
            special[threes[0]] = (1, 1, 2)
    for i, (color, count) in enumerate(zip(queries, counts)):
        for j in range(count):
            size = (special[i][j] if i in special else
                    int(rng.choice(np.arange(1, 7), p=(.18, .22, .19, .28, .09, .04))))
            entries.append((color, _shape(rng, size)))
    if distractors:
        total = int(rng.integers(0 if regime == "target_test" or index > 0 else 1, 6))
        if regime == "source_train" and index == 0:
            total = max(1, total)
        for _ in range(total):
            color = distractors[int(rng.integers(len(distractors)))]
            size = int(rng.choice(np.arange(1, 7), p=(.2, .2, .2, .27, .09, .04)))
            entries.append((color, _shape(rng, size)))
    return entries


def _place(rng, height, width, background, entries):
    field = np.full((height, width), background, dtype=np.int8)
    ordered = [entries[i] for i in rng.permutation(len(entries))]
    ordered.sort(key=lambda item: -len(item[1]))
    for color, shape in ordered:
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        choices = [(r, c) for r in range(height-sh+1) for c in range(width-sw+1)]
        order = rng.permutation(len(choices))
        found = False
        for strict in (True, False):
            for k in order:
                r0, c0 = choices[int(k)]
                cells = [(r0+r, c0+c) for r, c in shape]
                cellset = set(cells)
                if any(field[r, c] != background for r, c in cells):
                    continue
                valid = True
                for r, c in cells:
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        rr, cc = r+dr, c+dc
                        if (0 <= rr < height and 0 <= cc < width and (rr,cc) not in cellset
                                and (field[rr,cc] == color or (strict and field[rr,cc] != background))):
                            valid = False
                            break
                    if not valid:
                        break
                    # A corner contact of same-colored cells could obscure the
                    # four-neighbor convention, so keep those separate too.
                    for dr in (-1, 1):
                        for dc in (-1, 1):
                            rr, cc = r+dr, c+dc
                            if (0 <= rr < height and 0 <= cc < width and (rr,cc) not in cellset
                                    and field[rr,cc] == color):
                                valid = False
                    if not valid:
                        break
                if valid:
                    for r, c in cells:
                        field[r, c] = color
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    return field


def _generate_pair(rng, params, regime, index, first_info, seen_tests, source_size):
    nquery = params["nquery"]
    for attempt in range(40):
        rows, field_start = _layout(rng, nquery)
        background, queries, distractors = _roles(
            rng, params["available"], nquery, first_info,
            regime == "source_train" and index == 1,
            regime == "source_train")
        counts = _query_counts(rng, nquery, regime, index, seen_tests)
        if regime == "target_test" and counts in seen_tests:
            continue
        entries = _make_entries(rng, queries, counts, distractors, regime, index)
        if source_size and attempt < 8:
            width, field_height = 10, 20-field_start
        else:
            width = int(rng.integers(8, 31))
            low = max(9, (sum(len(shape) for _, shape in entries)*2 + width-1)//width)
            high = 30-field_start
            if low > high:
                continue
            field_height = int(rng.integers(low, high+1))
        if field_height < 9:
            continue
        column = 0 if regime == "source_train" and index == 0 else (
            int(rng.integers(1, 3)) if regime == "source_train" and index == 1
            else int(rng.integers(0, 4)))
        if column + max(counts) > width:
            continue
        field = _place(rng, field_height, width, background, entries)
        if field is None:
            continue
        grid = np.full((field_start+field_height, width), params["upper"], dtype=np.int8)
        for row, color in zip(rows, queries):
            grid[row, column] = color
        grid[field_start:] = field
        info = _inspect_input(grid)
        if info["counts"] != counts:
            continue
        output = _solve_input(grid)
        if np.array_equal(grid, output):
            continue
        return {"input": grid, "output": output}, info
    raise ValueError("bounded placement could not construct a valid pair")


def _key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_pair_counts(rng, num_train, num_test, seed)
    upper = int(rng.integers(10))
    params = {"upper": upper,
              "available": tuple(c for c in range(10) if c != upper),
              "nquery": 2 + seed % 3}
    puzzle = {"train": [], "test": []}
    seen = set()
    first_info = None
    seen_tests = set()
    for regime, number in (("source_train", ntrain), ("target_test", ntest)):
        split = "train" if regime == "source_train" else "test"
        for i in range(number):
            for retry in range(8):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                pair, info = _generate_pair(pair_rng, params, regime, i, first_info,
                                            seen_tests, seed % 7 == 0 and i == 0
                                            and regime == "source_train")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise ValueError("could not construct unique input")
            seen.add(key)
            puzzle[split].append(pair)
            if first_info is None:
                first_info = info
            if regime == "target_test":
                seen_tests.add(info["counts"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[k], list) for k in ("train", "test")), "split schema")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "pair counts")
    infos = {"train": [], "test": []}
    inputs = set()
    pairs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for name in ("input", "output"):
                g = pair[name]
                _check(isinstance(g, np.ndarray) and g.ndim == 2, "grid array")
                _check(np.issubdtype(g.dtype, np.integer), "grid dtype")
                _check(all(1 <= d <= 30 for d in g.shape), "grid dimensions")
                _check(np.all((g >= 0) & (g <= 9)), "grid colors")
            x, y = pair["input"], pair["output"]
            _check(x.shape == y.shape, "dimensions changed")
            _check(not any(np.array_equal(x, source) for source in _SOURCE_GRIDS), "source input copied")
            ikey = _key(x)
            pkey = (ikey, _key(y))
            _check(ikey not in inputs and pkey not in pairs, "duplicate pair")
            inputs.add(ikey)
            pairs.add(pkey)
            info = _inspect_input(x)
            _check(np.array_equal(y, _solve_input(x)), "component rule violated")
            _check(not np.array_equal(x, y), "unchanged example")
            if split == "train":
                _check(max(info["counts"]) <= 4, "training exceeds demonstrated count range")
            else:
                _check(max(info["counts"]) <= 6, "test count exceeds supported range")
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    _check(len({i["upper"] for i in all_infos}) == 1, "upper background changed in puzzle")
    _check(len({len(i["query_colors"]) for i in all_infos}) == 1, "query number changed in puzzle")
    _check(any(i["distractors"] for i in infos["train"]), "no training distractor removal")
    _check(any(0 in i["counts"] for i in infos["train"]), "no training absent query")
    _check(len({i["counts"] for i in infos["test"]}) == len(infos["test"]),
           "test pairs repeat the same count question")
    _check(any(any(c > 1 for c in i["counts"]) for i in infos["train"]), "no count demonstration")
    _check(any(any(col > 0 and count > 0 for col, count in zip(i["query_cols"], i["counts"]))
               for i in infos["train"]), "moved swatch not demonstrated")
    # Verify count, not area, is needed in the training evidence.
    observations = [(count, sum(info["sizes"].get(color, ())))
                    for info in infos["train"]
                    for color, count in zip(info["query_colors"], info["counts"])
                    if count > 0]
    _check(any(a[0] == b[0] and a[1] != b[1] for a in observations for b in observations),
           "equal counts never have different areas")
    _check(any(a[0] != b[0] and a[1] == b[1] for a in observations for b in observations),
           "equal areas never have different counts")


def _equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(np.array_equal(pa[x], pb[x])
               for pa, pb in zip(a[k], b[k]) for x in ("input", "output"))
               for k in ("train", "test"))


def _self_test_prediction(input_grid):
    # An independent traversal used only by self_test, separate from the
    # record builder and ground-truth renderer above.
    upper = int(input_grid[0, 0])
    width = input_grid.shape[1]
    start = next(r for r in range(1, input_grid.shape[0])
                 if np.count_nonzero(input_grid[r] != upper) >= width // 2)
    swatches = []
    for r in range(start):
        cols = np.flatnonzero(input_grid[r] != upper)
        if len(cols):
            swatches.append((r, int(cols[0]), int(input_grid[r, cols[0]])))
    field = input_grid[start:]
    colors, frequency = np.unique(field, return_counts=True)
    background = int(colors[np.argmax(frequency)])
    visited = set()
    counts = {color: 0 for _, _, color in swatches}
    for r in range(field.shape[0]):
        for c in range(field.shape[1]):
            color = int(field[r, c])
            if color not in counts or (r, c) in visited:
                continue
            counts[color] += 1
            stack = [(r, c)]
            visited.add((r, c))
            while stack:
                y, x = stack.pop()
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < field.shape[0] and 0 <= xx < field.shape[1]
                            and (yy, xx) not in visited and int(field[yy, xx]) == color):
                        visited.add((yy, xx))
                        stack.append((yy, xx))
    result = input_grid.copy()
    for r, c, color in swatches:
        result[r] = upper
        result[r, c:c+counts[color]] = color
    for r in range(start, result.shape[0]):
        for c in range(width):
            if int(input_grid[r, c]) not in counts and int(input_grid[r, c]) != background:
                result[r, c] = background
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _check(np.array_equal(_self_test_prediction(pair["input"]), pair["output"]),
                       "independent component traversal disagrees")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _check(_equal(puzzles[17], generate_puzzle(17)), "automatic determinism")
    _check(len({_key(p["train"][0]["input"]) for p in puzzles}) == 42, "seed diversity")
    infos = {split: [_inspect_input(pair["input"]) for p in puzzles for pair in p[split]]
             for split in ("train", "test")}
    _check({len(i["query_colors"]) for i in infos["train"]} == {2, 3, 4}, "query-number diversity")
    _check(len({i["upper"] for i in infos["train"]}) >= 6, "upper-background diversity")
    _check(len({i["field_shape"] for i in infos["train"]}) >= 30, "field-size diversity")
    _check(any(i["counts"] and max(i["counts"]) > 4 for i in infos["test"]), "no extrapolation")
    _check(any(all(c > 0 for c in i["counts"]) for i in infos["test"]), "no all-present tests")
    _check(any(max(i["counts"]) <= 4 for i in infos["test"]), "no within-range tests")
    _check({j for i in infos["test"] for j, c in enumerate(i["counts"]) if c == 0}
           >= {0, 1, 2}, "zero is confined to one query slot")
    _check({j for i in infos["test"] for j, c in enumerate(i["counts"]) if c > 4}
           >= {0, 1, 2}, "extrapolation is confined to one query slot")
    all_infos = infos["train"] + infos["test"]
    _check({size for i in all_infos for _, cells in i["records"]
            for size in (len(cells),)} == set(range(1, 7)), "shape-size range collapsed")
    _check(any(not i["distractors"] for i in all_infos), "no distractor-free input")
    _check(any(sum(color in i["distractors"] for color, _ in i["records"]) >= 4
               for i in all_infos), "no strong distractor pressure")
    _check(any(i["background"] == 0 for i in all_infos), "zero never a field background")
    _check(any(0 in i["query_colors"] for i in all_infos), "zero never a query color")
    _check(any(len({r % 3 for _, cells in i["records"] for r, _ in cells}) == 3
               and len({c % 3 for _, cells in i["records"] for _, c in cells}) == 3
               for i in all_infos), "all layouts retain the old slot lattice")
    shapes = [pair["input"].shape for p in puzzles for split in ("train", "test")
              for pair in p[split]]
    _check((20, 10) in shapes and any(w == 30 for _, w in shapes)
           and any(h == 30 for h, _ in shapes), "field scale range collapsed")
    for seed, tr, te in ((1001,2,1),(1002,10,1),(1003,10,4),
                         (1101,None,1),(1102,None,4),(1103,2,None),(1104,10,None)):
        p = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(p)
        if tr is not None:
            _check(len(p["train"]) == tr, "explicit train count")
        if te is not None:
            _check(len(p["test"]) == te, "explicit test count")
    a = generate_puzzle(777, 7, 3)
    _check(_equal(a, generate_puzzle(777, 7, 3)), "explicit determinism")
    for kwargs in ({"num_train":1},{"num_train":11},{"num_test":0},
                   {"num_test":5},{"num_train":2,"num_test":2},
                   {"num_train":3,"num_test":4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
