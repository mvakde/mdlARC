import numpy as np


def _grid(rows):
    return np.array(
        [[ord(ch) - ord("0") for ch in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _grid(
            (
                "00000000000000",
                "00002222220000",
                "00002035020000",
                "00002068020000",
                "00002000020000",
                "00002000020000",
                "00002222220000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
            )
        ),
        _grid(
            (
                "222222",
                "233552",
                "233552",
                "266882",
                "266882",
                "222222",
            )
        ),
    ),
    (
        _grid(
            (
                "00000000000000",
                "00222222222200",
                "00200000000200",
                "00203344000200",
                "00203344000200",
                "00201188000200",
                "00201188000200",
                "00200000000200",
                "00200000000200",
                "00200000000200",
                "00222222222200",
                "00000000000000",
                "00000000000000",
                "00000000000000",
            )
        ),
        _grid(
            (
                "2222222222",
                "2333344442",
                "2333344442",
                "2333344442",
                "2333344442",
                "2111188882",
                "2111188882",
                "2111188882",
                "2111188882",
                "2222222222",
            )
        ),
    ),
    (
        _grid(
            (
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "02222222222000",
                "02033660002000",
                "02033660002000",
                "02044110002000",
                "02044110002000",
                "02000000002000",
                "02000000002000",
                "02000000002000",
                "02000000002000",
                "02222222222000",
            )
        ),
        _grid(
            (
                "2222222222",
                "2333366662",
                "2333366662",
                "2333366662",
                "2333366662",
                "2444411112",
                "2444411112",
                "2444411112",
                "2444411112",
                "2222222222",
            )
        ),
    ),
)


def _requested_count(value, name, low, high):
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(seed, rng, num_train, num_test):
    n_train = _requested_count(num_train, "num_train", 2, 10)
    n_test = _requested_count(num_test, "num_test", 1, 4)

    if n_train is None and n_test is None:
        seed_value = int(seed)
        n_test = 1 + seed_value % 4
        n_train = n_test + 1 + (seed_value // 4) % (10 - n_test)
    elif n_train is None:
        n_train = int(rng.integers(n_test + 1, 11))
    elif n_test is None:
        largest_test = min(4, n_train - 1)
        n_test = int(rng.integers(1, largest_test + 1))

    if n_train <= n_test:
        raise ValueError("num_train must be greater than num_test")

    return n_train, n_test


def _sample_size(rng):
    return int(rng.choice(np.arange(1, 7), p=(.25, .25, .19, .15, .10, .06)))


def _sample_rectangle(rng, require_large=False):
    low = 2 if require_large else 1
    a = int(rng.integers(low, 6))
    choices = [x for x in range(low, 6) if x != a]
    b = int(rng.choice(choices))
    if bool(rng.integers(0, 2)):
        return a, b
    return b, a


def _plan_shapes(rng, n_train, n_test):
    third_scale_mode = rng.random() < .28
    if third_scale_mode:
        train = [(1, 1), (2, 2)]
        train += [(1, 1) if rng.random() < .5 else (2, 2)
                  for _ in range(n_train - 2)]
    elif rng.random() < .64:
        train = [(1, 1), (2, 2)]
    else:
        a = _sample_size(rng)
        b = _sample_size(rng)
        if a == b:
            b = a % 6 + 1
        train = [(a, a), (b, b)]
    if not third_scale_mode:
        train += [(_sample_size(rng),) * 2 for _ in range(n_train - 2)]

    complex_allowed = n_train >= 3 and rng.random() < .68
    rectangle_allowed = n_train >= 3 and not third_scale_mode and rng.random() < .48
    if complex_allowed and min(train[1]) < 2:
        train[1] = (2, 2)
    if rectangle_allowed:
        train[2] = _sample_rectangle(rng)
    if train[0] == train[1]:
        train[0] = (1, 1) if train[1] != (1, 1) else (2, 2)

    train_patterns = ["quadrant"] * n_train
    if complex_allowed:
        train_patterns[1] = "complex"
        for i in range(2, n_train):
            if min(train[i]) >= 2 and rng.random() < .22:
                train_patterns[i] = "complex"

    test = []
    test_patterns = []
    for index in range(n_test):
        if third_scale_mode and index == 0:
            q = int(rng.integers(3, 7))
            shape = (q, q)
        elif rectangle_allowed and rng.random() < .38:
            shape = _sample_rectangle(rng)
        elif index == 0 and rng.random() < .42:
            unseen = [q for q in range(1, 7) if (q, q) not in train]
            q = int(rng.choice(unseen)) if unseen else _sample_size(rng)
            shape = (q, q)
        else:
            q = _sample_size(rng)
            shape = (q, q)
        pattern = "quadrant"
        if complex_allowed and min(shape) >= 2 and rng.random() < .42:
            pattern = "complex"
        test.append(shape)
        test_patterns.append(pattern)

    return list(zip(train, train_patterns)), list(zip(test, test_patterns))


def _choose_roles(rng, shared, base, index):
    if shared or index == 0:
        return base
    choices = [int(x) for x in rng.choice(10, size=2, replace=False)]
    if index == 1 and tuple(choices) == base:
        choices.reverse()
    return tuple(choices)


def _choose_content(rng, background, frame, previous):
    available = [x for x in range(10) if x not in (background, frame)]
    if previous is None:
        return tuple(int(x) for x in rng.choice(available, size=4, replace=False))
    shared = [x for x in previous if x in available]
    fresh = [x for x in available if x not in previous]
    possible = [k for k in (1, 2, 3) if len(shared) >= k and len(fresh) >= 4 - k]
    k = int(rng.choice(possible))
    chosen = list(rng.choice(shared, size=k, replace=False))
    chosen += list(rng.choice(fresh, size=4-k, replace=False))
    rng.shuffle(chosen)
    return tuple(int(x) for x in chosen)


def _motif(rng, shape, pattern, colors):
    qh, qw = shape
    h, w = 2 * qh, 2 * qw
    order = [int(x) for x in rng.permutation(colors)]
    motif = np.empty((h, w), dtype=np.int8)
    motif[:qh, :qw] = order[0]
    motif[:qh, qw:] = order[1]
    motif[qh:, :qw] = order[2]
    motif[qh:, qw:] = order[3]
    if pattern == "complex":
        # Several local edits turn the source-like template into a cellwise
        # scaling example while retaining the same four content colors.
        count = int(rng.integers(max(2, h*w//8), max(3, h*w//4)+1))
        for _ in range(count):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            alternatives = [x for x in order if x != int(motif[r, c])]
            motif[r, c] = int(rng.choice(alternatives))
        # Force a visible deviation even if random edits cancel one another.
        if _is_quadrant(motif):
            motif[0, 0] = order[1]
        if len(np.unique(motif)) < 4:
            for color, (r, c) in zip(order, ((0, 0), (0, -1), (-1, 0), (-1, -1))):
                motif[r, c] = color
    return motif


def _is_quadrant(motif):
    h, w = motif.shape
    if h % 2 or w % 2:
        return False
    qh, qw = h // 2, w // 2
    blocks = (motif[:qh, :qw], motif[:qh, qw:],
              motif[qh:, :qw], motif[qh:, qw:])
    return all(np.all(block == block[0, 0]) for block in blocks) and len(
        {int(block[0, 0]) for block in blocks}
    ) == 4


def _axis_position(rng, slack):
    if slack == 0:
        return 0
    mode = rng.random()
    if mode < .23:
        return 0
    if mode < .49:
        return slack
    if slack == 1:
        return int(rng.integers(0, 2))
    return int(rng.integers(1, slack))


def _canvas_dimension(rng, outer):
    slack_limit = 30 - outer
    if slack_limit == 0:
        return outer
    if rng.random() < .48:
        slack = int(rng.integers(0, min(4, slack_limit) + 1))
    else:
        slack = int(rng.integers(0, slack_limit + 1))
    return outer + slack


def _generate_pair(rng, shape, pattern, roles, colors, canvas, anchor=False):
    background, frame = roles
    motif = _motif(rng, shape, pattern, colors)
    mh, mw = motif.shape
    outer_h, outer_w = 2 * mh + 2, 2 * mw + 2
    if canvas is None:
        height = _canvas_dimension(rng, outer_h)
        width = _canvas_dimension(rng, outer_w)
    else:
        height, width = canvas
    if anchor and height == outer_h and width == outer_w:
        if height < 30:
            height += 1
        else:
            width += 1
    r0 = _axis_position(rng, height - outer_h)
    c0 = _axis_position(rng, width - outer_w)
    row_offset = int(rng.integers(0, mh + 1))
    col_offset = int(rng.integers(0, mw + 1))
    if anchor and (row_offset, col_offset) == (mh//2, mw//2):
        row_offset = 0

    inp = np.full((height, width), background, dtype=np.int8)
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1
    inp[r0, c0:c1+1] = frame
    inp[r1, c0:c1+1] = frame
    inp[r0:r1+1, c0] = frame
    inp[r0:r1+1, c1] = frame
    mr, mc = r0 + 1 + row_offset, c0 + 1 + col_offset
    inp[mr:mr+mh, mc:mc+mw] = motif

    out = np.full((outer_h, outer_w), frame, dtype=np.int8)
    out[1:-1, 1:-1] = np.repeat(np.repeat(motif, 2, axis=0), 2, axis=1)
    return inp, out


def _array_key(array):
    return array.shape, tuple(int(x) for x in array.flat)


def _pair_key(inp, out):
    return _array_key(inp), _array_key(out)


def _is_source_pair(inp, out):
    return any(inp.shape == source_in.shape and out.shape == source_out.shape
               and np.array_equal(inp, source_in) and np.array_equal(out, source_out)
               for source_in, source_out in _SOURCE_PAIRS)


def _extract_input(inp):
    candidates = []
    for frame in (int(x) for x in np.unique(inp)):
        positions = np.argwhere(inp == frame)
        r0, c0 = positions.min(axis=0)
        r1, c1 = positions.max(axis=0)
        outer_h, outer_w = int(r1-r0+1), int(c1-c0+1)
        if outer_h < 6 or outer_w < 6 or outer_h % 2 or outer_w % 2:
            continue
        border_count = 2*outer_h + 2*outer_w - 4
        if len(positions) != border_count:
            continue
        crop = inp[r0:r1+1, c0:c1+1]
        if not (np.all(crop[0] == frame) and np.all(crop[-1] == frame)
                and np.all(crop[:, 0] == frame) and np.all(crop[:, -1] == frame)
                and not np.any(crop[1:-1, 1:-1] == frame)):
            continue
        for background in (int(x) for x in np.unique(inp) if int(x) != frame):
            content_positions = np.argwhere((inp != background) & (inp != frame))
            if len(content_positions) == 0:
                continue
            mr0, mc0 = content_positions.min(axis=0)
            mr1, mc1 = content_positions.max(axis=0)
            mh, mw = int(mr1-mr0+1), int(mc1-mc0+1)
            if (mh, mw) != ((outer_h-2)//2, (outer_w-2)//2):
                continue
            if not (r0 < mr0 <= mr1 < r1 and c0 < mc0 <= mc1 < c1):
                continue
            motif = inp[mr0:mr1+1, mc0:mc1+1]
            content_colors = {int(x) for x in np.unique(motif)}
            if len(content_colors) != 4 or background in content_colors or frame in content_colors:
                continue
            expected = np.full(inp.shape, background, dtype=inp.dtype)
            expected[r0, c0:c1+1] = frame
            expected[r1, c0:c1+1] = frame
            expected[r0:r1+1, c0] = frame
            expected[r0:r1+1, c1] = frame
            expected[mr0:mr1+1, mc0:mc1+1] = motif
            if np.array_equal(inp, expected):
                candidates.append({
                    "background": background, "frame": frame,
                    "content": frozenset(content_colors),
                    "motif": motif.copy(), "shape": motif.shape,
                    "outer": (outer_h, outer_w),
                    "frame_pos": (int(r0), int(c0)),
                    "offset": (int(mr0-r0-1), int(mc0-c0-1)),
                    "quadrant": _is_quadrant(motif),
                })
    assert len(candidates) == 1, "input must have a unique frame and tight motif"
    return candidates[0]


def _infer_output(inp):
    found = _extract_input(inp)
    out = np.full(found["outer"], found["frame"], dtype=np.int8)
    out[1:-1, 1:-1] = np.repeat(np.repeat(found["motif"], 2, axis=0), 2, axis=1)
    return out


def _analyze_pair(inp, out):
    found = _extract_input(inp)
    assert np.array_equal(out, _infer_output(inp))
    assert {int(x) for x in np.unique(out)} == found["content"] | {found["frame"]}
    return found


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(seed, rng, num_train, num_test)
    train_specs, test_specs = _plan_shapes(rng, n_train, n_test)
    specs = train_specs + test_specs

    shared_roles = rng.random() < .58
    base_roles = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    common_canvas = None
    if rng.random() < .26:
        max_h = max(4*shape[0]+2 for shape, _ in specs)
        max_w = max(4*shape[1]+2 for shape, _ in specs)
        common_canvas = (
            int(rng.integers(max_h, min(30, max_h+8)+1)),
            int(rng.integers(max_w, min(30, max_w+8)+1)),
        )
        if common_canvas == (max_h, max_w):
            common_canvas = (max_h + 1, max_w)

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_content = None
    for index, (shape, pattern) in enumerate(specs):
        section = "train" if index < n_train else "test"
        roles = _choose_roles(rng, shared_roles, base_roles, index)
        colors = _choose_content(rng, *roles, previous_content)
        previous_content = colors
        for attempt in range(32):
            inp, out = _generate_pair(rng, shape, pattern, roles, colors,
                                      common_canvas, anchor=(index == 0))
            if _array_key(inp) in used_inputs or _is_source_pair(inp, out):
                continue
            used_inputs.add(_array_key(inp))
            puzzle[section].append({"input": inp, "output": out})
            break
        else:
            raise RuntimeError("could not construct a unique non-source pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert _array_key(inp) not in seen_inputs
            assert _pair_key(inp, out) not in seen_pairs
            assert not _is_source_pair(inp, out)
            seen_inputs.add(_array_key(inp))
            seen_pairs.add(_pair_key(inp, out))
            analyses[section].append(_analyze_pair(inp, out))

    train = analyses["train"]
    tests = analyses["test"]
    assert len({a["shape"] for a in train}) >= 2
    assert any(a["offset"] != (a["shape"][0]//2, a["shape"][1]//2)
               for a in train)
    assert any(p["input"].shape != a["outer"]
               for p, a in zip(puzzle["train"], train))
    assert any(0 < len(train[0]["content"] & a["content"]) < 4 for a in train[1:])
    if any(not a["quadrant"] for a in tests):
        assert any(not a["quadrant"] for a in train)
    if any(a["shape"][0] != a["shape"][1] for a in tests):
        assert any(a["shape"][0] != a["shape"][1] for a in train)


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k])
        for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    assert _puzzles_equal(a, b)
    for nt, ns in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(1000+10*nt+ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
        validate_generated_puzzle(p)
    for seed, nt, ns in ((2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
        validate_generated_puzzle(p)
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(99, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    all_pairs = [(split, pair, _analyze_pair(pair["input"], pair["output"]))
                 for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert all(np.array_equal(_infer_output(pair["input"].copy()), pair["output"])
               for _, pair, _ in all_pairs)
    assert len({_array_key(pair["input"]) for _, pair, _ in all_pairs}) >= len(all_pairs)*9//10
    assert len({pair["input"].shape for _, pair, _ in all_pairs}) >= 20
    assert len({a["outer"] for _, _, a in all_pairs}) >= 6
    assert len({(pair["input"].shape, a["outer"], a["frame_pos"], a["offset"])
                for _, pair, a in all_pairs}) >= len(all_pairs)//2
    assert any(not a["quadrant"] for split, _, a in all_pairs if split == "train")
    assert any(not a["quadrant"] for split, _, a in all_pairs if split == "test")
    assert any(a["shape"][0] != a["shape"][1] for split, _, a in all_pairs if split == "train")
    assert any(a["shape"][0] != a["shape"][1] for split, _, a in all_pairs if split == "test")
    assert any(a["offset"] == (a["shape"][0]//2, a["shape"][1]//2)
               for _, _, a in all_pairs)
    assert any(a["frame_pos"][0] == 0 for _, _, a in all_pairs)
    assert any(a["frame_pos"][1] == 0 for _, _, a in all_pairs)
    assert any(a["frame_pos"][0] + a["outer"][0] == pair["input"].shape[0]
               for _, pair, a in all_pairs)
    assert any(a["frame_pos"][1] + a["outer"][1] == pair["input"].shape[1]
               for _, pair, a in all_pairs)
    assert len({_array_key(pair["input"]) for _, pair, _ in all_pairs}) > 40
    assert not _puzzles_equal(puzzles[0], puzzles[1])

    measured = {
        "palette_varied": 0, "roles_varied": 0, "roles_shared": 0,
        "common_canvas": 0, "third_test_shape": 0,
        "two_train_shapes_third_test": 0, "complex_and_rectangle": 0,
    }
    for puzzle in puzzles:
        train = [_extract_input(pair["input"]) for pair in puzzle["train"]]
        test = [_extract_input(pair["input"]) for pair in puzzle["test"]]
        all_a = train + test
        role_count = len({(a["background"], a["frame"]) for a in all_a})
        measured["palette_varied"] += len({a["content"] for a in all_a}) > 1
        measured["roles_varied"] += role_count > 1
        measured["roles_shared"] += role_count == 1
        measured["common_canvas"] += len({
            pair["input"].shape for split in ("train", "test")
            for pair in puzzle[split]
        }) == 1
        unseen = any(a["shape"] not in {b["shape"] for b in train} for a in test)
        measured["third_test_shape"] += unseen
        measured["two_train_shapes_third_test"] += (
            len({a["shape"] for a in train}) == 2 and unseen
        )
        measured["complex_and_rectangle"] += (
            any(not a["quadrant"] for a in all_a)
            and any(a["outer"][0] != a["outer"][1] for a in all_a)
        )
    assert measured["palette_varied"] == 42
    assert measured["roles_varied"] >= 5 and measured["roles_shared"] >= 5
    assert measured["common_canvas"] >= 4
    assert measured["third_test_shape"] >= 8
    assert measured["two_train_shapes_third_test"] >= 2
    assert measured["complex_and_rectangle"] >= 3

    def count_pairs(split, predicate):
        return sum(predicate(pair, a) for section, pair, a in all_pairs
                   if section == split)

    for split in ("train", "test"):
        assert count_pairs(split, lambda p, a: a["frame_pos"][0] == 0) >= 5
        assert count_pairs(split, lambda p, a:
                           a["frame_pos"][0] + a["outer"][0] == p["input"].shape[0]) >= 5
        assert count_pairs(split, lambda p, a:
                           a["frame_pos"][1] + a["outer"][1] == p["input"].shape[1]) >= 5
        assert count_pairs(split, lambda p, a:
                           0 < a["frame_pos"][0]
                           and a["frame_pos"][0] + a["outer"][0] < p["input"].shape[0]
                           and 0 < a["frame_pos"][1]
                           and a["frame_pos"][1] + a["outer"][1] < p["input"].shape[1]) >= 5
        assert count_pairs(split, lambda p, a:
                           a["offset"] == (a["shape"][0]//2, a["shape"][1]//2)) >= 1
        assert count_pairs(split, lambda p, a:
                           max(p["input"].shape[0]-a["outer"][0],
                               p["input"].shape[1]-a["outer"][1]) <= 4) >= 10
        assert count_pairs(split, lambda p, a: a["quadrant"]) >= 20
        assert count_pairs(split, lambda p, a: not a["quadrant"]) >= 5
        assert count_pairs(split, lambda p, a:
                           a["outer"][0] != a["outer"][1]) >= 3
    assert count_pairs("train", lambda p, a: a["outer"][1] == p["input"].shape[1]) >= 5
    assert count_pairs("test", lambda p, a: a["outer"] == (6, 6)) >= 5
    assert count_pairs("test", lambda p, a: a["outer"] == (26, 26)) >= 3
