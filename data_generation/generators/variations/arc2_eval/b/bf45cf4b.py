import numpy as np


_SOURCE_INPUT_ENCODINGS = frozenset(
    (
        "444444444444/444444224244/444444242244/444444424244/"
        "444444444444/483844444444/434344444444/483844444444/"
        "444444444444/444444444444/444444444444/444444444444",

        "33333333333333333/33333333333333333/33133333333333333/"
        "33313333333333333/33313333333333333/33111333888333333/"
        "33333333828333333/33333333888333333/33333333333333333/"
        "33333333333333333/33333333333333333/33333333333333333",

        "111111111111/111111111111/166611111111/161611111111/"
        "116111111111/111111111111/111118998111/111119449111/"
        "111119449111/111118998111/111111111111/111111111111",

        "8888888888888/8663668888888/8643468888888/8333338888888/"
        "8643468888888/8663668888888/8888888888888/8888888111888/"
        "8888881888188/8888881111188/8888881818188/8888881111188/"
        "8888888888888/8888888888888",
    )
)


def _validate_optional_count(name, value, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _select_counts(rng, seed, num_train, num_test):
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is None and num_test is None:
        # This mapping guarantees count diversity over consecutive seeds.
        num_test = 1 + (seed % 4)
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = minimum_train + ((seed // 4) % span)
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("pair counts must satisfy num_train > num_test")

    return num_train, num_test


def _build_output(mask, stamp, background):
    mask_height, mask_width = mask.shape
    stamp_height, stamp_width = stamp.shape

    output = np.full(
        (mask_height * stamp_height, mask_width * stamp_width),
        int(background),
        dtype=np.int8,
    )

    for row in range(mask_height):
        for col in range(mask_width):
            if mask[row, col]:
                output[
                    row * stamp_height : (row + 1) * stamp_height,
                    col * stamp_width : (col + 1) * stamp_width,
                ] = stamp

    return output


# The only learned convention is direct, unchanged block substitution. These
# parameters select distributions, not hidden information needed by the solver.
def _sample_puzzle_parameters(rng):
    return {
        "regime": str(rng.choice(["source_like", "overlap", "mask_grow",
                                  "stamp_grow", "aspect", "mixed"])),
        "palette_mode": str(rng.choice(["stable", "varied", "varied", "exchange"])),
        "palette": [int(c) for c in rng.permutation(10)],
        "complexity": int(rng.integers(0, 3)),
        "islands": bool(rng.random() < .45),
        "symmetric": bool(rng.random() < .16),
        "edge_holes": bool(rng.random() < .6),
        "axis": int(rng.integers(0, 2)),
    }


def _array_key(array):
    return (array.shape, tuple(int(x) for x in array.ravel()))


def _encode_grid(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _bbox(binary):
    rows, cols = np.nonzero(binary)
    assert len(rows), "an encoding must have visible cells"
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _crop(grid, box):
    r0, r1, c0, c1 = box
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def _boxes_have_blank_separation(a, b):
    return (a[1] + 1 < b[0] or b[1] + 1 < a[0]
            or a[3] + 1 < b[2] or b[3] + 1 < a[2])


def _component_count(binary, diagonal=False):
    remaining = set(map(tuple, np.argwhere(binary)))
    count = 0
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                cell = (r + dr, c + dc)
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
    return count


def _transforms(stamp):
    return [f(np.rot90(stamp, k)) for k in range(4)
            for f in (lambda x: x, np.fliplr)]


def _orientation_distinct(stamp):
    return all(not np.array_equal(stamp, t) for t in _transforms(stamp)[1:])


def _fully_symmetric(stamp):
    return all(np.array_equal(stamp, t) for t in _transforms(stamp))


def _read_input(grid):
    """Identify color roles and tight rectangles without connectivity assumptions.

    Majority background is a readability constraint on our layouts, not a full
    background rim. Try every remaining color as the instruction color and
    require a single geometrically separated monochrome/multicolor reading.
    """
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    assert int(max(counts)) > grid.size / 2, "background must be evident"
    candidates = []
    for signal in colors:
        signal = int(signal)
        if signal == background:
            continue
        payload = set(map(int, colors)) - {background, signal}
        if len(payload) < 2:
            continue
        mb = _bbox(grid == signal)
        sb = _bbox((grid != background) & (grid != signal))
        if not _boxes_have_blank_separation(mb, sb):
            continue
        mask_crop, stamp = _crop(grid, mb), _crop(grid, sb)
        if not set(map(int, np.unique(mask_crop))) <= {background, signal}:
            continue
        if signal in stamp:
            continue
        mask = mask_crop == signal
        if mask.shape[0] * stamp.shape[0] > 30 or mask.shape[1] * stamp.shape[1] > 30:
            continue
        candidates.append({"background": background, "signal": signal,
                           "payload_colors": payload, "mask": mask, "stamp": stamp,
                           "mask_bbox": mb, "stamp_bbox": sb})
    assert len(candidates) == 1, "encodings must have a unique reading"
    return candidates[0]


def _predict_input(grid):
    # Independent indexing expression rather than the generator's block writes.
    info = _read_input(grid)
    mask, stamp, bg = info["mask"], info["stamp"], info["background"]
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    return np.where(mask[rr // sh, cc // sw], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def solve(train_pairs, test_input):
    """Check the demonstrations and predict with no labels or generation state."""
    for pair in train_pairs:
        assert np.array_equal(_predict_input(pair["input"]), pair["output"])
    return _predict_input(test_input)


def _shape_options(parameters, regime, witness=False):
    train = regime == "source_train"
    assert regime in ("source_train", "target_test")
    kind = parameters["regime"]
    if kind == "source_like":
        mrange, srange = ((2, 4), (3, 4)) if train else ((4, 6), (4, 6))
    elif kind == "mask_grow":
        mrange, srange = ((2, 4) if train else (5, 10)), (2, 4)
    elif kind == "stamp_grow":
        mrange, srange = (2, 4), ((3, 4) if train else (5, 10))
    elif kind == "overlap":
        mrange, srange = (2, 6), (2, 6)
    else:
        upper = (6, 8, 10)[parameters["complexity"]]
        mrange, srange = (2, upper), (2, upper)
    masks = [(h, w) for h in range(mrange[0], mrange[1] + 1)
             for w in range(mrange[0], mrange[1] + 1)
             if not witness or min(h, w) >= 3]
    stamps = [(h, w) for h in range(srange[0], srange[1] + 1)
              for w in range(srange[0], srange[1] + 1)
              if (not witness or min(h, w) >= 3)
              and (not parameters["symmetric"] or h == w >= 3)]
    choices = {}
    for mh, mw in masks:
        feasible = [(sh, sw) for sh, sw in stamps if mh * sh <= 30 and mw * sw <= 30]
        if kind == "aspect":
            tall = bool(parameters["axis"]) == train
            feasible = [(sh, sw) for sh, sw in feasible
                        if (mh * sh > mw * sw if tall else mh * sh < mw * sw)]
        if feasible:
            choices[mh, mw] = feasible
    assert choices
    return choices


def _choose_shapes(rng, parameters, regime, witness, previous=None, islands=False):
    options = _shape_options(parameters, regime, witness)
    if islands:
        options = {m: ss for m, ss in options.items() if min(m) >= 3}
    if previous:
        different = {m: [s for s in ss if s != previous[1]]
                     for m, ss in options.items() if m != previous[0]}
        different = {m: ss for m, ss in different.items() if ss}
        if different:
            options = different
    masks = list(options)
    m = masks[int(rng.integers(len(masks)))]
    stamps = options[m]
    return m, stamps[int(rng.integers(len(stamps)))]


def _path(rng, shape, diagonal):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    r = c = 0
    mask[0, 0] = True
    if diagonal:
        r = c = 1
        mask[r, c] = True
    while r < h - 1 or c < w - 1:
        choices = []
        if r < h - 1:
            choices.append((1, 0))
        if c < w - 1:
            choices.append((0, 1))
        if diagonal and r < h - 1 and c < w - 1:
            choices.extend([(1, 1)] * 4)
        dr, dc = choices[int(rng.integers(len(choices)))]
        r, c = r + dr, c + dc
        mask[r, c] = True
    return mask


def _make_mask(rng, shape, family):
    h, w = shape
    r, c = np.indices(shape)
    if family in ("path", "diagonal", "branches"):
        mask = _path(rng, shape, family != "path")
        if family == "branches":
            # Grow short eight-neighbor branches, preserving the first diagonal
            # bridge instead of filling it in during random growth.
            for _ in range(int(rng.integers(1, max(2, (h + w) // 2)))):
                fringe = [(i, j) for i in range(h) for j in range(w)
                          if not mask[i, j] and (i, j) not in ((0, 1), (1, 0))
                          and np.any(mask[max(0, i-1):i+2, max(0, j-1):j+2])]
                if fringe:
                    mask[fringe[int(rng.integers(len(fringe)))]] = True
    elif family == "ring":
        mask = (r == 0) | (r == h - 1) | (c == 0) | (c == w - 1)
    elif family == "ladder":
        period = int(rng.integers(2, 4))
        mask = (r % period == 0) | (r == h - 1) | (c == int(rng.integers(w)))
    elif family in ("islands", "sparse"):
        if family == "islands":
            mask = (r % 2 == 0) & (c % 2 == 0)
            mask[-1, ::2] = True
            mask[::2, -1] = True
            # A blank row/column is explicitly inside the encoding's bounds.
            if h >= 3:
                mask[h // 2, :] = False
            else:
                mask[:, w // 2] = False
        else:
            mask = rng.random(shape) < float(rng.uniform(.12, .42))
        mask[0, 0] = mask[-1, -1] = True
    elif family == "full":
        mask = np.ones(shape, dtype=bool)
    elif family == "dense":
        mask = np.ones(shape, dtype=bool)
        count = int(rng.integers(1, max(2, h * w // 3)))
        for index in rng.permutation(h * w):
            i, j = divmod(int(index), w)
            mask[i, j] = False
            if (_component_count(mask) != 1 or not _touches_sides(mask)):
                mask[i, j] = True
            else:
                count -= 1
                if count == 0:
                    break
    else:
        raise ValueError("unknown mask family")
    if rng.random() < .5:
        mask = np.flipud(mask)
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _touches_sides(binary):
    return all(np.any(side) for side in
               (binary[0], binary[-1], binary[:, 0], binary[:, -1]))


def _pair_colors(rng, parameters, index):
    mode = parameters["palette_mode"]
    if mode == "varied":
        palette = [int(c) for c in rng.permutation(10)]
    elif mode == "exchange":
        # Rotate roles across demonstrations as well as tests. This optional
        # regime does not impose the source-specific three-color cycle.
        palette = list(np.roll(parameters["palette"], index))
    else:
        palette = parameters["palette"]
    k = 2 if mode == "stable" else int(rng.integers(2, 5 + parameters["complexity"]))
    return int(palette[0]), int(palette[1]), np.array(palette[2:2 + k], dtype=np.int8)


def _stamp_pattern(rng, shape, payload, family):
    h, w = shape
    r, c = np.indices(shape)
    n = len(payload)
    if family == "bands":
        labels = (r if rng.random() < .5 else c) // int(rng.integers(1, 3))
    elif family == "rings":
        labels = np.minimum.reduce([r, c, h - 1 - r, w - 1 - c])
    elif family == "centered":
        labels = (abs(2*r-h+1) + abs(2*c-w+1)) // 2
    elif family == "repeat":
        tile = rng.integers(n, size=(int(rng.integers(1, min(4, h)+1)),
                                     int(rng.integers(1, min(4, w)+1))))
        labels = tile[r % tile.shape[0], c % tile.shape[1]]
    elif family == "diagonal":
        labels = r * int(rng.integers(1, n + 1)) + c
    elif family == "symmetric":
        a, b = np.minimum(r, h-1-r), np.minimum(c, w-1-c)
        if h == w:
            a, b = np.minimum(a, b), np.maximum(a, b)
        orbits = sorted(set(zip(a.ravel(), b.ravel())))
        values = rng.integers(n, size=len(orbits))
        labels = np.zeros(shape, dtype=int)
        for orbit, value in zip(orbits, values):
            labels[(a == orbit[0]) & (b == orbit[1])] = value
    else:
        labels = rng.integers(n, size=shape)
    return payload[(labels + int(rng.integers(n))) % n].copy()


def _make_stamp(rng, shape, bg, payload, symmetric=False, witness=False, edge=False):
    h, w = shape
    family = "symmetric" if symmetric else str(rng.choice(
        ["bands", "rings", "centered", "repeat", "diagonal", "random", "symmetric"]))
    if witness and not symmetric:
        family = "random"
    hole_kind = "witness" if witness else str(rng.choice(
        ["none", "none", "single", "multiple", "large", "edge"] if edge
        else ["none", "none", "single", "multiple", "large"]))
    for attempt in range(80):
        stamp = _stamp_pattern(rng, shape, payload, family)
        holes = np.zeros(shape, dtype=bool)
        interior = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
        if interior and hole_kind != "none":
            if hole_kind == "large":
                r0, c0 = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
                r1, c1 = int(rng.integers(r0+1, h)), int(rng.integers(c0+1, w))
                holes[r0:r1, c0:c1] = True
            else:
                n = 1 if hole_kind == "single" else int(rng.integers(1, len(interior)+1))
                for k in rng.choice(len(interior), size=n, replace=False):
                    holes[interior[int(k)]] = True
        if edge and (hole_kind == "edge" or witness):
            boundary = [(r, c) for r in range(h) for c in range(w)
                        if r in (0, h-1) or c in (0, w-1)]
            for k in rng.choice(len(boundary), size=int(rng.integers(1, max(2, len(boundary)//3))), replace=False):
                holes[boundary[int(k)]] = True
        if symmetric:
            # Square symmetry is a deliberate subfamily. Edge-hole puzzles use
            # asymmetric witnesses instead, so a tiny symmetric stamp need not
            # lose an entire orbit of boundary evidence.
            holes = np.logical_or.reduce(_transforms(holes))
        stamp[holes] = bg
        foreground = stamp != bg
        if not _touches_sides(foreground) or np.count_nonzero(foreground) < stamp.size * .45:
            continue
        if _component_count(foreground, True) != 1:
            continue
        if len(set(map(int, np.unique(stamp))) - {bg}) < 2:
            continue
        if witness and not np.any(holes):
            continue
        if witness and not symmetric and not _orientation_distinct(stamp):
            continue
        return stamp
    # Constructive fallback: a concentric two-color stamp or a short asymmetric
    # corner motif. This is also well-defined for the smallest witness sizes.
    stamp = np.full(shape, payload[0], dtype=np.int8)
    if symmetric:
        stamp[1:-1, 1:-1] = payload[1]
        if witness:
            stamp[0, 0] = stamp[0, -1] = stamp[-1, 0] = stamp[-1, -1] = bg
    else:
        stamp[0, 0:2] = payload[1]
        if witness:
            stamp[1, 1] = bg
            if edge:
                stamp[-1, -1] = bg
    return stamp


def _compose_input(rng, background, signal, mask, stamp, islands=False):
    mh, mw = mask.shape
    sh, sw = stamp.shape
    horizontal = bool(rng.integers(2))
    reverse = bool(rng.integers(2))
    rim = 0 if rng.random() < .25 else 1
    gap = int(rng.integers(2 if islands else 1, 11))
    base_h = (max(mh, sh) if horizontal else mh + sh + gap) + 2 * rim
    base_w = (mw + sw + gap if horizontal else max(mw, sw)) + 2 * rim
    # Every source rectangle is at most 10; even gap=10 fits a 30-cell axis.
    pad_limit = (2, 7, 30)[int(rng.integers(3))]
    h = base_h + int(rng.integers(min(30-base_h, pad_limit)+1))
    w = base_w + int(rng.integers(min(30-base_w, pad_limit)+1))
    foreground_count = int(np.count_nonzero(mask) + np.count_nonzero(stamp != background))
    while h * w <= 2 * foreground_count:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    if horizontal:
        start = rim + int(rng.integers(w-base_w+1))
        mc, sc = ((start+sw+gap, start) if reverse else (start, start+mw+gap))
        mr = rim + int(rng.integers(h-mh-2*rim+1))
        sr = rim + int(rng.integers(h-sh-2*rim+1))
    else:
        start = rim + int(rng.integers(h-base_h+1))
        mr, sr = ((start+sh+gap, start) if reverse else (start, start+mh+gap))
        mc = rim + int(rng.integers(w-mw-2*rim+1))
        sc = rim + int(rng.integers(w-sw-2*rim+1))
    grid = np.full((h, w), background, dtype=np.int8)
    grid[mr:mr+mh, mc:mc+mw] = np.where(mask, signal, background)
    grid[sr:sr+sh, sc:sc+sw] = stamp
    return grid


def _generate_pair(rng, regime, parameters, index, previous=None):
    train = regime == "source_train"
    witness = train and index == 0
    island_witness = train and index == 1 and parameters["islands"]
    mshape, sshape = _choose_shapes(rng, parameters, regime, witness, previous, island_witness)
    families = ["path", "diagonal", "branches", "ring", "ladder", "dense", "dense", "full"]
    if parameters["islands"]:
        families += ["islands", "sparse"]
    family = "islands" if island_witness else str(rng.choice(families))
    if witness:
        family = str(rng.choice(["path", "diagonal", "branches"]))
    # Internal blank rows require room for two visible groups.
    if family == "islands" and min(mshape) < 3:
        family = "diagonal"
    mask = _make_mask(rng, mshape, family)
    if witness:
        # Mask orientation matters too. Tiny symmetric demonstrations can leave
        # a reflected placement rule unresolved even with asymmetric stamps.
        for _ in range(40):
            if _orientation_distinct(mask):
                break
            mask = _make_mask(rng, mshape, str(rng.choice(["path", "diagonal", "branches"])))
        else:
            mask = np.zeros(mshape, dtype=bool)
            mask[0, 0] = True
            mask[1, 1:] = True
            mask[1:, -1] = True
        assert _orientation_distinct(mask)
    if not parameters["islands"] and _component_count(mask, True) > 1:
        mask = _make_mask(rng, mshape, "path")
    bg, signal, payload = _pair_colors(rng, parameters, index + (0 if train else 5))
    edge = parameters["edge_holes"] and not parameters["symmetric"]
    stamp = _make_stamp(rng, sshape, bg, payload, parameters["symmetric"], witness, edge)
    grid = _compose_input(rng, bg, signal, mask, stamp, _component_count(mask, True) > 1)
    return {"input": grid, "output": _build_output(mask, stamp, bg)}


def _edge_holes(info):
    holes = info["stamp"] == info["background"]
    return bool(np.any(holes[0]) or np.any(holes[-1]) or np.any(holes[:, 0]) or np.any(holes[:, -1]))


def _analyze_pair(input_grid, output_grid, regime=None):
    info = _read_input(input_grid)
    assert np.array_equal(output_grid, _predict_input(input_grid)), "incorrect block substitution"
    assert not np.array_equal(input_grid, output_grid)
    assert info["signal"] not in output_grid
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _array_key(pair["input"])
            assert key not in seen
            seen.add(key)
            # Only generated samples exclude the source. The semantic parser
            # and source regressions deliberately accept the original inputs.
            assert _encode_grid(pair["input"]) not in _SOURCE_INPUT_ENCODINGS
            information[split].append(_analyze_pair(pair["input"], pair["output"]))
    train, test = information["train"], information["test"]
    assert len({i["mask"].shape for i in train}) >= 2
    assert len({i["stamp"].shape for i in train}) >= 2
    assert any(np.any(~i["mask"]) for i in train), "demonstrate inactive blocks"
    for transform in range(1, 8):
        if all(np.array_equal(i["mask"], _transforms(i["mask"])[transform]) for i in train):
            assert all(np.array_equal(i["mask"], _transforms(i["mask"])[transform]) for i in test)
    if any(np.any(i["stamp"] == i["background"]) for i in test):
        assert any(np.any(i["stamp"] == i["background"]) for i in train)
    if any(_edge_holes(i) for i in test):
        assert any(_edge_holes(i) for i in train)
    if any(_component_count(i["mask"], True) > 1 for i in test):
        assert any(_component_count(i["mask"], True) > 1 for i in train)
    if not all(_fully_symmetric(i["stamp"]) for i in train + test):
        assert any(_orientation_distinct(i["stamp"]) for i in train)
    for pair in puzzle["test"]:
        predicted = solve(puzzle["train"], pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _select_counts(rng, seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, count in (("train", "source_train", ntrain), ("test", "target_test", ntest)):
        previous = None
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            for _ in range(64):
                pair = _generate_pair(pair_rng, regime, parameters, index, previous)
                try:
                    info = _read_input(pair["input"])
                except AssertionError:
                    continue
                key = _array_key(pair["input"])
                if key in seen or _encode_grid(pair["input"]) in _SOURCE_INPUT_ENCODINGS:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                previous = info["mask"].shape, info["stamp"].shape
                break
            else:
                # Distinct payload color positions and separated rectangles make
                # ambiguous candidates exceptional; never emit an unchecked pair.
                raise RuntimeError("could not construct a unique pair")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _canonical_pattern(array, background):
    mapping = {int(background): 0}
    values = []
    for value in array.ravel():
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        values.append(mapping[value])
    return array.shape, tuple(values)


def _features(pair):
    """Observable review measurements; no sampled labels or hidden parameters."""
    info = _read_input(pair["input"])
    mask, stamp, bg = info["mask"], info["stamp"], info["background"]
    mb, sb = info["mask_bbox"], info["stamp_bbox"]
    gap_x = max(sb[2] - mb[3] - 1, mb[2] - sb[3] - 1)
    gap_y = max(sb[0] - mb[1] - 1, mb[0] - sb[1] - 1)
    layout = ("left" if mb[3] < sb[2] else "right") if gap_x >= 1 else ("above" if mb[1] < sb[0] else "below")
    holes = stamp == bg
    return {
        "background": bg, "signal": info["signal"],
        "mask_shape": mask.shape, "stamp_shape": stamp.shape,
        "output_shape": pair["output"].shape, "input_shape": pair["input"].shape,
        "mask_pattern": _array_key(mask), "stamp_pattern": _canonical_pattern(stamp, bg),
        "active": int(mask.sum()), "density": float(mask.mean()),
        "components4": _component_count(mask), "components8": _component_count(mask, True),
        "blank_internal_line": bool(np.any(~mask.any(axis=0)) or np.any(~mask.any(axis=1))),
        "payload_count": len(info["payload_colors"]),
        "palette_missing_top": bool(info["payload_colors"] - set(map(int, stamp[0]))),
        "holes": int(holes.sum()), "edge_holes": _edge_holes(info),
        "hole_square": bool(any(np.all(holes[r:r+2, c:c+2])
                                 for r in range(stamp.shape[0]-1) for c in range(stamp.shape[1]-1))),
        "symmetric": _fully_symmetric(stamp), "asymmetric": _orientation_distinct(stamp),
        "layout": layout, "gap": max(gap_x, gap_y),
        "edge_aligned": any(b[0] == 0 or b[2] == 0 or b[1] == pair["input"].shape[0]-1
                            or b[3] == pair["input"].shape[1]-1 for b in (mb, sb)),
        "padding_fraction": 1 - (mask.size + stamp.size) / pair["input"].size,
    }


def self_test() -> None:
    automatic = generate_puzzle(17)
    assert _puzzles_equal(automatic, generate_puzzle(17))
    explicit = generate_puzzle(17, 6, 2)
    assert _puzzles_equal(explicit, generate_puzzle(17, 6, 2))
    assert not _puzzles_equal(explicit, generate_puzzle(18, 6, 2))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                    (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, *counts)
        for split, count in zip(("train", "test"), counts):
            assert count is None or len(puzzle[split]) == count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2),
                   (3, 4), (4, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(400, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal count request accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    _source_regression()
    records, puzzles = [], []
    train_counts, test_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        features = {s: [_features(p) for p in puzzle[s]] for s in puzzle}
        puzzles.append(features)
        records.extend(features["train"] + features["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    for feature, minimum in (("mask_shape", 30), ("stamp_shape", 25), ("output_shape", 60),
                             ("input_shape", 80), ("mask_pattern", 160), ("stamp_pattern", 170)):
        assert len({r[feature] for r in records}) >= minimum, feature
    for split in ("train", "test"):
        rows = [r for p in puzzles for r in p[split]]
        assert sum(r["components4"] > 1 and r["components8"] == 1 for r in rows) >= 5
        assert sum(r["components8"] > 1 for r in rows) >= 5
        assert sum(r["symmetric"] for r in rows) >= 5
        assert sum(r["stamp_shape"][0] != r["stamp_shape"][1] for r in rows) >= 10
    assert max(r["active"] for r in records) > 25
    assert sum(r["density"] == 1 for r in records) >= 5
    for low, high in ((0, .35), (.35, .65), (.65, 1)):
        assert sum(low < r["density"] <= high for r in records) >= 20
    for flag in ("blank_internal_line", "palette_missing_top", "edge_holes", "hole_square", "edge_aligned"):
        assert sum(r[flag] for r in records) >= 10, flag
    assert sum(r["payload_count"] > 3 for r in records) >= 10
    assert sum(r["holes"] > 1 for r in records) >= 20
    assert sum(r["gap"] > 3 for r in records) >= 40
    assert {r["layout"] for r in records} == {"left", "right", "above", "below"}
    assert max(max(r["input_shape"]) for r in records) == 30
    assert max(max(r["output_shape"]) for r in records) == 30
    assert sum(len({r["background"] for r in p["train"]}) > 1 for p in puzzles) >= 20
    assert sum(len({(r["background"], r["signal"]) for r in p["train"]+p["test"]}) == 1 for p in puzzles) >= 3
    assert sum(len({r["output_shape"] for r in p["test"]}) > 1 for p in puzzles) >= 15
    assert sum(all(r["symmetric"] for r in p["train"] + p["test"]) for p in puzzles) >= 3
    assert all(len({r["mask_pattern"] for r in p["train"]}) >= 2 for p in puzzles)
    assert all(len({r["stamp_pattern"] for r in p["train"]}) >= 2 for p in puzzles)
    shape_partners = {}
    growth = set()
    for r in records:
        shape_partners.setdefault(r["mask_shape"], set()).add(r["stamp_shape"])
    assert sum(len(shapes) >= 3 for shapes in shape_partners.values()) >= 10
    for p in puzzles:
        for test in p["test"]:
            growth.add((np.prod(test["mask_shape"]) > max(np.prod(t["mask_shape"]) for t in p["train"]),
                        np.prod(test["stamp_shape"]) > max(np.prod(t["stamp_shape"]) for t in p["train"])))
    assert growth == {(False, False), (True, False), (False, True), (True, True)}
    # Validate rejection of a changed output, including a label-blind prediction.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in automatic[s]] for s in automatic}
    output = damaged["test"][0]["output"]
    output[0, 0] = (int(output[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("incorrect output accepted")


# Literal source fixtures are used only for regression, never generation.
_SOURCE_REGRESSIONS = (('444444444444/444444224244/444444242244/444444424244/444444444444/483844444444/434344444444/483844444444/444444444444/444444444444/444444444444/444444444444', '838838444838/343343444343/838838444838/838444838838/343444343343/838444838838/444838444838/444343444343/444838444838'), ('33333333333333333/33333333333333333/33133333333333333/33313333333333333/33313333333333333/33111333888333333/33333333828333333/33333333888333333/33333333333333333/33333333333333333/33333333333333333/33333333333333333', '888333333/828333333/888333333/333888333/333828333/333888333/333888333/333828333/333888333/888888888/828828828/888888888'), ('111111111111/111111111111/166611111111/161611111111/116111111111/111111111111/111118998111/111119449111/111119449111/111118998111/111111111111/111111111111', '899889988998/944994499449/944994499449/899889988998/899811118998/944911119449/944911119449/899811118998/111189981111/111194491111/111194491111/111189981111'), ('8888888888888/8663668888888/8643468888888/8333338888888/8643468888888/8663668888888/8888888888888/8888888111888/8888881888188/8888881111188/8888881818188/8888881111188/8888888888888/8888888888888', '8888866366663666636688888/8888864346643466434688888/8888833333333333333388888/8888864346643466434688888/8888866366663666636688888/6636688888888888888866366/6434688888888888888864346/3333388888888888888833333/6434688888888888888864346/6636688888888888888866366/6636666366663666636666366/6434664346643466434664346/3333333333333333333333333/6434664346643466434664346/6636666366663666636666366/6636688888663668888866366/6434688888643468888864346/3333388888333338888833333/6434688888643468888864346/6636688888663668888866366/6636666366663666636666366/6434664346643466434664346/3333333333333333333333333/6434664346643466434664346/6636666366663666636666366'))

def _source_regression():
    components = []
    for encoded_input, encoded_output in _SOURCE_REGRESSIONS:
        inp = np.array([[int(c) for c in row] for row in encoded_input.split("/")], dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in encoded_output.split("/")], dtype=np.int8)
        info = _analyze_pair(inp, out)
        assert _component_count(info["mask"], True) == 1
        components.append(_component_count(info["mask"]))
    assert components == [3, 2, 2, 2]
    assert int(info["mask"].sum()) == 18
