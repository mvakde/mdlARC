import numpy as np


_SOURCE_INPUTS = {
    ("220200", "002220", "222222", "220202", "200220", "000200"),
    ("002020", "020020", "002222", "220022", "222222", "020020"),
    ("002020", "202002", "202220", "222222", "002200", "202022"),
    (
        "222222", "000200", "020200", "022202", "000200",
        "200202", "002202", "200200", "020202", "200202",
    ),
    (
        "000200020000", "200000200000", "020002000000",
        "002020000000", "000200020002", "002020000000",
        "020002002000", "200000200200", "020000020000",
        "000200002000", "000000200200", "000200000020",
    ),
}


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _inside(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def _through(h, w, r, c, dr, dc):
    while _inside(r - dr, c - dc, h, w):
        r, c = r - dr, c - dc

    cells = []
    while _inside(r, c, h, w):
        cells.append((r, c))
        r, c = r + dr, c + dc
    return tuple(cells)


def _all_diagonal_paths(h, w, dr, dc):
    paths = []
    for r in range(h):
        for c in range(w):
            if not _inside(r - dr, c - dc, h, w):
                paths.append(_through(h, w, r, c, dr, dc))
    return paths


def _mask(h, w, cells):
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _options(grid, fg):
    """Find complete, intersecting line pairs visible in an input."""
    h, w = grid.shape
    options = []

    full_rows = [
        r for r in range(h)
        if np.all(grid[r, :] == fg)
    ]
    full_cols = [
        c for c in range(w)
        if np.all(grid[:, c] == fg)
    ]

    for r in full_rows:
        for c in full_cols:
            mask = np.zeros((h, w), dtype=bool)
            mask[r, :] = True
            mask[:, c] = True
            options.append(("hv", mask, (r, c)))

    if h >= 9 and w >= 9:
        down_right = [
            path for path in _all_diagonal_paths(h, w, 1, 1)
            if len(path) >= 8
            and all(grid[r, c] == fg for r, c in path)
        ]
        down_left = [
            path for path in _all_diagonal_paths(h, w, 1, -1)
            if len(path) >= 8
            and all(grid[r, c] == fg for r, c in path)
        ]

        for a in down_right:
            a_set = set(a)
            for b in down_left:
                shared = a_set.intersection(b)
                if len(shared) != 1:
                    continue

                cell = next(iter(shared))
                ia = a.index(cell)
                ib = b.index(cell)
                if min(
                    ia, len(a) - ia - 1,
                    ib, len(b) - ib - 1,
                ) < 3:
                    continue

                options.append(
                    ("diag", _mask(h, w, a_set.union(b)), cell)
                )

    return options


def _hv_shape(rng, regime, first):
    mode = int(rng.integers(0, 5))
    if first and regime == "source_train" and rng.random() < 0.65:
        mode = 0
    if mode == 0:
        return int(rng.integers(5, 9)), int(rng.integers(5, 9))
    if mode == 1:
        h = int(rng.integers(7, 14))
        return h, int(np.clip(h + rng.integers(-2, 3), 5, 17))
    if mode == 2:
        return int(rng.integers(9, 18)), int(rng.integers(5, 11))
    if mode == 3:
        return int(rng.integers(5, 11)), int(rng.integers(9, 18))
    return int(rng.integers(10, 18)), int(rng.integers(10, 18))


def _geometry(rng, kind, regime, first=False, source_slot=None):
    if kind == "hv":
        if source_slot == 0:
            h = w = 6
            r, c = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        elif source_slot == 1:
            h, w = 10, 6
            r, c = 0, int(rng.integers(1, 5))
        else:
            h, w = _hv_shape(rng, regime, first)
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            edge = int(rng.integers(0, 6))
            if edge == 1:
                r = 0
            elif edge == 2:
                r = h - 1
            elif edge == 3:
                c = 0
            elif edge == 4:
                c = w - 1
            elif edge == 5:
                if rng.random() < 0.5:
                    r = int(rng.choice((0, h - 1)))
                else:
                    c = int(rng.choice((0, w - 1)))
        mask = np.zeros((h, w), dtype=bool)
        mask[r, :] = True
        mask[:, c] = True
        return mask

    for _ in range(200):
        h, w = int(rng.integers(10, 19)), int(rng.integers(10, 19))
        r, c = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))
        a, b = _through(h, w, r, c, 1, 1), _through(h, w, r, c, 1, -1)
        ia, ib = a.index((r, c)), b.index((r, c))
        if min(len(a), len(b)) >= 8 and min(
            ia, len(a) - ia - 1, ib, len(b) - ib - 1
        ) >= 3:
            return _mask(h, w, set(a).union(b))
    raise RuntimeError("Could not place crossing diagonals")


def _noise(rng, structure, style, kind):
    h, w = structure.shape
    free = np.argwhere(~structure)
    retained = int(np.count_nonzero(structure))
    cap = 0.7 if kind == "diag" else 0.9
    maximum = min(len(free), max(1, int(cap * retained)))
    if style == "sparse":
        target = int(rng.integers(1, min(4, maximum + 1)))
    elif style == "dense":
        target = int(rng.integers(min(max(2, int(0.55 * retained)), maximum),
                                  maximum + 1))
    elif style in ("cluster", "run"):
        target = int(rng.integers(2, min(maximum, 8) + 1))
    else:
        target = int(rng.integers(2, min(maximum, max(4, retained // 2)) + 1))
    noise = np.zeros((h, w), dtype=bool)
    if style == "run":
        choices = []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for r, c in free:
                for length in (2, 3):
                    cells = [(int(r + i * dr), int(c + i * dc)) for i in range(length)]
                    if not all(_inside(y, x, h, w) and not structure[y, x]
                               for y, x in cells):
                        continue
                    if any(_inside(y + dy, x + dx, h, w)
                           and structure[y + dy, x + dx]
                           for y, x in cells
                           for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                        choices.append(cells)
        if choices:
            for r, c in choices[int(rng.integers(0, len(choices)))][:target]:
                noise[r, c] = True
    elif style == "cluster":
        noise[tuple(free[int(rng.integers(0, len(free)))] )] = True
        while int(np.count_nonzero(noise)) < target:
            candidates = set()
            for r, c in np.argwhere(noise):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    y, x = int(r + dr), int(c + dc)
                    if _inside(y, x, h, w) and not structure[y, x] and not noise[y, x]:
                        candidates.add((y, x))
            if not candidates:
                break
            ordered = sorted(candidates)
            noise[ordered[int(rng.integers(0, len(ordered)))]] = True
    remaining = free[~noise[free[:, 0], free[:, 1]]]
    n = target - int(np.count_nonzero(noise))
    if n:
        selected = remaining[rng.choice(len(remaining), size=n, replace=False)]
        noise[selected[:, 0], selected[:, 1]] = True
    return noise


def _pair(rng, kind, colors, regime, first=False, source_slot=None):
    bg, fg = colors
    styles = ("sparse", "scatter", "cluster", "run", "dense")
    for attempt in range(240):
        structure = _geometry(rng, kind, regime, first, source_slot)
        style = styles[int(rng.integers(0, 5))] if attempt < 180 else "sparse"
        noise = _noise(rng, structure, style, kind)
        inp = np.full(structure.shape, bg, dtype=np.int8)
        inp[structure | noise] = fg
        # Each background crossing is pierced by an extra foreground cell.
        # A greedy hitting set usually needs only a few cells, even diagonally.
        cap = int((0.7 if kind == "diag" else 0.9) * np.count_nonzero(structure))
        for _ in range(12):
            inverse = _options(inp, bg)
            if not inverse:
                break
            hits = np.zeros(inp.shape, dtype=np.int16)
            for _, mask, _ in inverse:
                hits += mask & (inp == bg)
            peak = int(hits.max())
            if peak == 0:
                break
            choices = np.argwhere(hits == peak)
            r, c = choices[int(rng.integers(0, len(choices)))]
            inp[r, c] = fg
            noise[r, c] = True
        if np.count_nonzero(noise) > cap:
            continue
        if _rows(inp) in _SOURCE_INPUTS:
            continue
        options = _options(inp, fg)
        if (len(options) != 1 or options[0][0] != kind
                or not np.array_equal(options[0][1], structure)
                or _options(inp, bg)):
            continue
        out = np.full(structure.shape, bg, dtype=np.int8)
        out[structure] = fg
        return inp, out
    raise RuntimeError("Could not sample an unambiguous pair")


def _resolve_counts(rng, num_train, num_test):
    for value, lower, upper in (
        (num_train, 2, 10),
        (num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper
        ):
            raise ValueError("Invalid train or test pair count")

    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(
            rng.integers(1, min(4, num_train - 1) + 1)
        )
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(
        rng, num_train, num_test
    )

    palette = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    swap_roles = bool(rng.integers(0, 2))
    diagonal_training = bool(rng.integers(0, 2))
    source_episode = not diagonal_training and num_test >= 3 and rng.random() < 0.65
    train_kinds = ["hv", "diag" if diagonal_training else "hv"]
    train_kinds += [
        "diag" if diagonal_training and rng.random() < 0.35 else "hv"
        for _ in range(num_train - 2)
    ]
    if source_episode:
        test_kinds = ["hv", "hv", "diag"] + [
            "diag" if rng.random() < 0.5 else "hv"
            for _ in range(num_test - 3)
        ]
        test_slots = [0, 1] + [None] * (num_test - 2)
    elif diagonal_training:
        test_kinds = ["diag" if rng.random() < 0.45 else "hv"
                      for _ in range(num_test)]
        if num_test > 1 and len(set(test_kinds)) == 1:
            test_kinds[-1] = "hv" if test_kinds[0] == "diag" else "diag"
        test_slots = [None] * num_test
    else:
        test_kinds = ["diag"] + ["hv"] * (num_test - 1)
        test_slots = [None] * num_test
    if num_test > 1 and not source_episode:
        rng.shuffle(test_kinds)

    train_colors = [palette, palette[::-1] if swap_roles else palette]
    train_colors += [
        palette[::-1] if swap_roles and rng.random() < 0.5 else palette
        for _ in range(num_train - 2)
    ]
    test_colors = [
        palette[::-1] if swap_roles and rng.random() < 0.5 else palette
        for _ in range(num_test)
    ]

    puzzle = {"train": [], "test": []}
    seen = set()
    seen_train_layouts = set()

    for regime, kinds, assignments, slots in (
        ("source_train", train_kinds, train_colors, [None] * num_train),
        ("target_test", test_kinds, test_colors, test_slots),
    ):
        split = "train" if regime == "source_train" else "test"

        for i, (kind, colors, slot) in enumerate(zip(kinds, assignments, slots)):
            for _ in range(60):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, 2**63 - 1))
                )
                inp, out = _pair(pair_rng, kind, colors, regime, i == 0, slot)

                signature = (
                    inp.shape,
                    inp.astype(np.uint8).tobytes(),
                )
                layout = (out.shape, (out == colors[1]).tobytes())
                if signature not in seen and (split != "train" or layout not in seen_train_layouts):
                    seen.add(signature)
                    if split == "train":
                        seen_train_layouts.add(layout)
                    puzzle[split].append({
                        "input": inp,
                        "output": out,
                    })
                    break
            else:
                raise RuntimeError(
                    "Could not sample a unique input"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen_inputs, seen_pairs = set(), set()
    palette = None
    train_roles, test_roles = set(), set()
    train_kinds, test_kinds = [], []
    train_layouts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            colors = np.unique(inp)
            assert len(colors) == 2 and np.array_equal(colors, np.unique(out))
            values = tuple(int(x) for x in colors)
            if palette is None:
                palette = values
            assert values == palette
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _rows(inp) not in _SOURCE_INPUTS
            candidates = []
            for fg in values:
                for kind, mask, center in _options(inp, fg):
                    bg = values[0] if fg == values[1] else values[1]
                    candidates.append((kind, mask, center, bg, fg))
            assert len(candidates) == 1
            kind, mask, center, bg, fg = candidates[0]
            assert np.array_equal(out, np.where(mask, fg, bg))
            assert np.all(inp[mask] == fg)
            assert int(np.count_nonzero((inp == fg) & ~mask)) >= 1
            if split == "train":
                train_roles.add((bg, fg))
                train_kinds.append(kind)
                train_layouts.add((out.shape, mask.tobytes()))
            else:
                test_roles.add((bg, fg))
                test_kinds.append(kind)
    assert train_kinds[0] == "hv"
    assert "hv" in train_kinds
    assert len(train_layouts) >= 2
    assert test_roles <= train_roles
    if "diag" not in train_kinds:
        assert "diag" in test_kinds
    if nv > 1 and "diag" in train_kinds:
        assert set(test_kinds) == {"hv", "diag"}


def _equal_puzzles(a, b):
    if set(a) != set(b):
        return False
    for key in ("train", "test"):
        if len(a[key]) != len(b[key]):
            return False
        for x, y in zip(a[key], b[key]):
            if not np.array_equal(
                x["input"], y["input"]
            ):
                return False
            if not np.array_equal(
                x["output"], y["output"]
            ):
                return False
    return True


def self_test() -> None:
    train_counts, test_counts, shapes, layouts = set(), set(), set(), set()
    roles, palettes, noise_counts, edges, kinds = set(), set(), set(), set(), set()
    first_test_kinds, aspects, diagonal_lengths = set(), set(), set()
    edge_splits = set()
    inputs = set()
    swapped = source_like = sparse = dense = train_diag = 0
    one_extra = isolated = clustered = touching = run_near = majority = 0
    small_hv_test = large_hv_test = small_multi_test = 0
    diagonal_later = combined_swap_diagonal = varied_shapes = 0
    source_sequences = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_roles, local_kinds, local_shapes = set(), set(), set()
        for split in ("train", "test"):
            for i, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                inputs.add((inp.shape, inp.tobytes()))
                colors = tuple(int(x) for x in np.unique(inp))
                options = [(fg, option) for fg in colors for option in _options(inp, fg)]
                assert len(options) == 1
                fg, (kind, mask, center) = options[0]
                bg = colors[0] if fg == colors[1] else colors[1]
                assert np.array_equal(out, np.where(mask, fg, bg))
                assert _rows(inp) not in _SOURCE_INPUTS
                roles.add((bg, fg))
                palettes.add(colors)
                shapes.add((split, inp.shape))
                layouts.add((kind, inp.shape, center))
                kinds.add((split, kind))
                local_shapes.add(inp.shape)
                aspects.add("square" if inp.shape[0] == inp.shape[1] else
                            "tall" if inp.shape[0] > inp.shape[1] else "wide")
                noise = (inp == fg) & ~mask
                n = int(np.count_nonzero(noise))
                noise_counts.add(n)
                sparse += n <= 3
                dense += n >= 8
                one_extra += n == 1
                majority += np.count_nonzero(inp == fg) > inp.size / 2
                assert n <= int((0.7 if kind == "diag" else 0.9)
                                * np.count_nonzero(mask))
                for r, c in np.argwhere(noise):
                    r, c = int(r), int(c)
                    neighbors = [(r+dr, c+dc) for dr, dc in
                                 ((-1, 0), (1, 0), (0, -1), (0, 1),
                                  (-1, -1), (-1, 1), (1, -1), (1, 1))
                                 if _inside(r+dr, c+dc, *inp.shape)]
                    isolated += not any(noise[y, x] for y, x in neighbors)
                    clustered += any(noise[y, x] for y, x in neighbors)
                    touching += any(
                        _inside(r+dr, c+dc, *inp.shape)
                        and mask[r+dr, c+dc]
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    )
                    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                        y, x = r+dr, c+dc
                        if _inside(y, x, *inp.shape) and noise[y, x]:
                            near = any(
                                _inside(z+dy, q+dx, *inp.shape)
                                and mask[z+dy, q+dx]
                                for z, q in ((r, c), (y, x))
                                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                            )
                            run_near += near
                if split == "train":
                    local_roles.add((bg, fg))
                    local_kinds.add(kind)
                else:
                    if i == 0:
                        first_test_kinds.add(kind)
                    if i > 0:
                        diagonal_later += kind == "diag"
                    small_hv_test += kind == "hv" and max(inp.shape) <= 7
                    large_hv_test += kind == "hv" and max(inp.shape) >= 10
                if kind == "hv":
                    r, c = center
                    if r == 0:
                        edges.add("top")
                        edge_splits.add((split, "top"))
                    if r == inp.shape[0] - 1:
                        edges.add("bottom")
                        edge_splits.add((split, "bottom"))
                    if c == 0:
                        edges.add("left")
                        edge_splits.add((split, "left"))
                    if c == inp.shape[1] - 1:
                        edges.add("right")
                        edge_splits.add((split, "right"))
                else:
                    a = _through(*inp.shape, *center, 1, 1)
                    b = _through(*inp.shape, *center, 1, -1)
                    diagonal_lengths.add((len(a), len(b)))
        swapped += len(local_roles) > 1
        train_diag += "diag" in local_kinds
        source_like += "diag" not in local_kinds
        combined_swap_diagonal += len(local_roles) > 1 and "diag" in local_kinds
        varied_shapes += len(local_shapes) >= 2
        small_multi_test += len(puzzle["train"]) <= 4 and len(puzzle["test"]) >= 2
        if len(puzzle["test"]) >= 3:
            first, second, third = puzzle["test"][:3]
            first_options = [
                option for color in np.unique(first["input"])
                for option in _options(first["input"], int(color))
            ]
            second_options = [
                option for color in np.unique(second["input"])
                for option in _options(second["input"], int(color))
            ]
            third_options = [
                option for color in np.unique(third["input"])
                for option in _options(third["input"], int(color))
            ]
            source_sequences += (
                first["input"].shape == (6, 6)
                and first_options[0][0] == "hv"
                and second["input"].shape == (10, 6)
                and second_options[0][0] == "hv"
                and second_options[0][2][0] == 0
                and third_options[0][0] == "diag"
            )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(inputs) >= 200 and len(layouts) >= 100
    assert len({shape for _, shape in shapes}) >= 20
    assert len(palettes) >= 10 and len(roles) >= 15
    assert len(noise_counts) >= 12 and sparse >= 15 and dense >= 15
    assert one_extra >= 5 and isolated >= 20 and clustered >= 20
    assert touching >= 20 and run_near >= 20 and majority >= 1
    assert edges == {"top", "bottom", "left", "right"}
    assert len(edge_splits) == 8
    assert aspects == {"square", "tall", "wide"}
    assert len(diagonal_lengths) >= 12
    assert {("train", "hv"), ("train", "diag"),
            ("test", "hv"), ("test", "diag")} <= kinds
    assert first_test_kinds == {"hv", "diag"}
    assert swapped >= 8 and source_like >= 8 and train_diag >= 8
    assert combined_swap_diagonal >= 4 and varied_shapes == 42
    assert small_multi_test >= 2 and diagonal_later >= 8
    assert small_hv_test >= 2 and large_hv_test >= 10
    assert source_sequences >= 2

    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None), (3, 2), (4, 3)):
        puzzle = generate_puzzle(1081, tr, te)
        validate_generated_puzzle(puzzle)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
        assert _equal_puzzles(puzzle, generate_puzzle(1081, tr, te))

    illegal_cases = (
        (1, None),
        (11, None),
        (None, 0),
        (None, 5),
        (2, 2),
        (3, 4),
        (2, 4),
        (-1, 1),
        (True, 1),
    )
    for tr, te in illegal_cases:
        try:
            generate_puzzle(9, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Illegal count accepted"
            )
