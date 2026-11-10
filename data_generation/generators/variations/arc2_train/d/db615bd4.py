"""Two-step lattice samples encoding a frame and ordered solid rectangles."""

import hashlib
import numpy as np


SOURCE_HASHES = {
    "84a6d4896de7236a567a393885587eeb4860a38cfed0323c7f63345cd9a4f5fd",
    "17e06a0fa6b979c3badad33fd562527d0bcb801875cc8ca7c9c9625be51dbd48",
    "d344ca659669d03e017aa00bd6ab8cc46229e971b2ed07c491cfb7a51bd7e67e",
    "32a03e9dae967aa36295cc5b3453d9bb01babfedef3046077d95cba25682cac0",
}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _counts(rng, num_train, num_test):
    if num_train is not None:
        num_train = _count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _size(rng, choices):
    values = list(choices)
    if 25 in values:
        values += [25] * 5
    return int(values[int(rng.integers(len(values)))])


def _signature(rng, count, orientation, varying_widths, both_sides):
    for _ in range(500):
        lengths = tuple(int(x) for x in rng.choice((1, 3, 5, 7), size=count))
        if max(lengths) == 1:
            continue
        widths = tuple(int(x) for x in rng.choice((1, 3, 5), size=count))
        if varying_widths and len(set(widths)) < 2:
            continue
        if not varying_widths:
            widths = (widths[0],) * count
        pad = int(rng.integers(1, 4))
        long = sum(lengths) + count + 3
        short = max(widths) + 2 * pad + 2
        minimum_cross = short + max(widths) * (2 if both_sides else 1) + (4 if both_sides else 3)
        if long <= 28 and long > short and minimum_cross <= 30:
            return (orientation, lengths, widths, pad, both_sides)
    raise ValueError("could not sample feasible geometry")


def _base(shape, field, secondary, phase):
    grid = np.full(shape, field, dtype=np.int8)
    grid[phase[0]::2, phase[1]::2] = secondary
    return grid


def _axis_starts(rng, size, lengths, parity):
    span = sum(lengths) + len(lengths) - 1
    units = (size - parity - span) // 2
    assert units >= 0
    allocation = rng.multinomial(units, np.ones(len(lengths) + 1) / (len(lengths) + 1))
    cursor = parity + 2 * int(allocation[0])
    starts = []
    for i, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length + 1
        if i + 1 < len(lengths):
            cursor += 2 * int(allocation[i + 1])
    assert starts[-1] + lengths[-1] <= size
    return starts


def _cross_options(size, short, phase, payload_phase, width, both_sides):
    options = []
    for start in range(phase, size - short + 1, 2):
        if start < 1 or start + short > size - 1:
            continue
        before = [x for x in range(payload_phase, size - width + 1, 2) if x + width <= start]
        after = [x for x in range(payload_phase, size - width + 1, 2) if x > start + short - 1]
        if (before and after) if both_sides else (before or after):
            options.append((start, before, after))
    return options


def _roles(palette, rng, split, index, count):
    if split == "train" and index == 0:
        indices = (0, 1, 2, 3, 4, 5, 6)
    elif split == "train" and index == 1:
        indices = (2, 3, 0, 1, 4, 5, 6)
    elif split == "test":
        frame = 1 if index % 2 == 0 else 3
        rest = tuple(int(x) for x in rng.permutation([x for x in range(10) if x != frame]))
        indices = (rest[0], rest[1], frame, *rest[2:])
    else:
        indices = tuple(int(x) for x in rng.permutation(10))
    colors = tuple(palette[x] for x in indices[:count + 3])
    return colors[:3], colors[3:]


def _pair(rng, signature, roles, phase):
    orientation, lengths, widths, pad, both_sides = signature
    count = len(lengths)
    long = sum(lengths) + count + 3
    short = max(widths) + 2 * pad + 2
    field, secondary, frame = roles[0]
    colors = roles[1]
    axis_phase = phase[1] if orientation == "H" else phase[0]
    cross_phase = phase[0] if orientation == "H" else phase[1]
    feasible_long_sizes = [size for size in range(long + 2, 31)
                           if any(x >= 1 and x + long <= size - 1
                                  for x in range(axis_phase, size - long + 1, 2))]
    assert feasible_long_sizes
    long_size = _size(rng, feasible_long_sizes)
    axis_choices = [x for x in range(axis_phase, long_size - long + 1, 2)
                    if x >= 1 and x + long <= long_size - 1]
    assert axis_choices
    axis_frame = int(rng.choice(axis_choices))
    cross_options = [(size, start, before, after)
                     for size in range(short + 2, 31)
                     for start, before, after in _cross_options(
                         size, short, cross_phase, 1 - cross_phase, max(widths), both_sides)]
    assert cross_options
    cross_size = _size(rng, sorted({x[0] for x in cross_options}))
    choices = [x for x in cross_options if x[0] == cross_size]
    _, cross_frame, before, after = choices[int(rng.integers(len(choices)))]
    if both_sides:
        sides = [int(x) for x in rng.choice((-1, 1), size=count)]
        if len(set(sides)) == 1:
            sides[0], sides[-1] = -1, 1
    else:
        side = -1 if before and (not after or int(rng.integers(2)) == 0) else 1
        sides = [side] * count
    axis_starts = _axis_starts(rng, long_size, lengths, 1 - axis_phase)
    if orientation == "H":
        shape = (cross_size, long_size)
        top, left = cross_frame, axis_frame
        bottom, right = top + short - 1, left + long - 1
    else:
        shape = (long_size, cross_size)
        top, left = axis_frame, cross_frame
        bottom, right = top + long - 1, left + short - 1
    base = _base(shape, field, secondary, phase)
    inp = base.copy()
    inp[top, left:right + 1:2] = frame
    inp[bottom, left:right + 1:2] = frame
    inp[top:bottom + 1:2, left] = frame
    inp[top:bottom + 1:2, right] = frame
    for length, width, color, axis, side in zip(lengths, widths, colors, axis_starts, sides):
        positions = [x for x in range(1 - cross_phase, cross_size - width + 1, 2)
                     if (x + width <= cross_frame if side == -1 else x > cross_frame + short - 1)]
        assert positions
        cross = int(rng.choice(positions))
        if orientation == "H":
            inp[cross:cross + width:2, axis:axis + length:2] = color
        else:
            inp[axis:axis + length:2, cross:cross + width:2] = color
    out = base.copy()
    out[top + 1:bottom, left + 1:right] = field
    out[top, left:right + 1] = frame
    out[bottom, left:right + 1] = frame
    out[top:bottom + 1, left] = frame
    out[top:bottom + 1, right] = frame
    cursor = (left if orientation == "H" else top) + 2
    for length, width, color in zip(lengths, widths, colors):
        cross = (top if orientation == "H" else left) + (short - width) // 2
        if orientation == "H":
            out[cross:cross + width, cursor:cursor + length] = color
        else:
            out[cursor:cursor + length, cross:cross + width] = color
        cursor += length + 1
    assert cursor == (right if orientation == "H" else bottom)
    return {"input": inp, "output": out}


def _key(grid):
    return grid.shape, grid.tobytes()


def _analyze_for_lattice(inp, out, field, secondary, phase):
    base = _base(inp.shape, field, secondary, phase)
    rr, cc = np.indices(inp.shape)
    frame_sites = (rr % 2 == phase[0]) & (cc % 2 == phase[1])
    object_sites = (rr % 2 != phase[0]) & (cc % 2 != phase[1])
    changed = inp != base
    assert np.all(~changed | frame_sites | object_sites)
    fm, pm = changed & frame_sites, changed & object_sites
    assert np.any(fm) and np.any(pm)
    frame_colors = np.unique(inp[fm])
    assert len(frame_colors) == 1
    frame = int(frame_colors[0])
    assert frame not in (field, secondary)
    coords = np.argwhere(fm)
    top, left = (int(x) for x in coords.min(axis=0))
    bottom, right = (int(x) for x in coords.max(axis=0))
    expected = np.zeros(inp.shape, dtype=bool)
    expected[top, left:right + 1:2] = True
    expected[bottom, left:right + 1:2] = True
    expected[top:bottom + 1:2, left] = True
    expected[top:bottom + 1:2, right] = True
    assert np.array_equal(fm, expected)
    assert top >= 1 and left >= 1 and bottom < inp.shape[0] - 1 and right < inp.shape[1] - 1
    assert top % 2 == bottom % 2 == phase[0]
    assert left % 2 == right % 2 == phase[1]
    fh, fw = bottom - top + 1, right - left + 1
    assert fh != fw
    orientation = "H" if fw > fh else "V"
    objects = []
    for color in np.unique(inp[pm]):
        color = int(color)
        assert color not in (field, secondary, frame)
        mask = pm & (inp == color)
        coords = np.argwhere(mask)
        r0, c0 = (int(x) for x in coords.min(axis=0))
        r1, c1 = (int(x) for x in coords.max(axis=0))
        expected = np.zeros(inp.shape, dtype=bool)
        expected[r0:r1 + 1:2, c0:c1 + 1:2] = True
        assert np.array_equal(mask, expected)
        assert r1 < top or r0 > bottom or c1 < left or c0 > right
        length = c1 - c0 + 1 if orientation == "H" else r1 - r0 + 1
        width = r1 - r0 + 1 if orientation == "H" else c1 - c0 + 1
        axis_start = c0 if orientation == "H" else r0
        axis_end = c1 if orientation == "H" else r1
        side = -1 if (r1 < top if orientation == "H" else c1 < left) else 1
        objects.append((axis_start, axis_end, length, width, color, side, (r0, c0, r1, c1)))
    objects.sort()
    assert 2 <= len(objects) <= 4
    assert all(a[1] < b[0] for a, b in zip(objects, objects[1:]))
    lengths = tuple(o[2] for o in objects)
    widths = tuple(o[3] for o in objects)
    assert all(x % 2 == 1 for x in lengths + widths)
    long = fw if orientation == "H" else fh
    short = fh if orientation == "H" else fw
    assert long == sum(lengths) + len(objects) + 3
    assert short > max(widths) + 2
    assert (short - max(widths) - 2) % 2 == 0
    expected_out = base.copy()
    expected_out[top + 1:bottom, left + 1:right] = field
    expected_out[top, left:right + 1] = frame
    expected_out[bottom, left:right + 1] = frame
    expected_out[top:bottom + 1, left] = frame
    expected_out[top:bottom + 1, right] = frame
    cursor = (left if orientation == "H" else top) + 2
    for _, _, length, width, color, _, _ in objects:
        cross = (top if orientation == "H" else left) + (short - width) // 2
        if orientation == "H":
            expected_out[cross:cross + width, cursor:cursor + length] = color
        else:
            expected_out[cursor:cursor + length, cross:cross + width] = color
        cursor += length + 1
    assert cursor == (right if orientation == "H" else bottom)
    assert np.array_equal(out, expected_out)
    assert not np.array_equal(inp, out)
    inner = (slice(top + 1, bottom), slice(left + 1, right))
    exposed = int(np.count_nonzero((base[inner] == secondary) & (out[inner] == field)))
    return {
        "phase": phase, "field": field, "secondary": secondary, "frame": frame,
        "orientation": orientation, "count": len(objects), "lengths": lengths,
        "widths": widths, "pad": (short - max(widths) - 2) // 2,
        "mixed_sides": len({o[5] for o in objects}) > 1,
        "mixed_widths": len(set(widths)) > 1, "exposed": exposed,
        "shape": inp.shape, "frame_bbox": (top, left, bottom, right),
        "source_bboxes": tuple(o[6] for o in objects),
        "payload_colors": tuple(o[4] for o in objects),
        "geometry": (orientation, lengths, widths, (short - max(widths) - 2) // 2),
    }


def _analyze(inp, out):
    assert inp.shape == out.shape
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    field = int(np.argmax(counts))
    found = []
    for secondary in np.argsort(-counts):
        secondary = int(secondary)
        if secondary == field or counts[secondary] == 0:
            continue
        phase_counts = np.array([[np.count_nonzero(inp[r::2, c::2] == secondary)
                                  for c in range(2)] for r in range(2)])
        phase = tuple(int(x) for x in np.unravel_index(np.argmax(phase_counts), (2, 2)))
        try:
            found.append(_analyze_for_lattice(inp, out, field, secondary, phase))
        except AssertionError:
            pass
    assert len(found) == 1
    return found[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            input_key = _key(inp)
            pair_key = (input_key, _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert inp.shape != (25, 25) or hashlib.sha256(inp.astype(np.int8).tobytes()).hexdigest() not in SOURCE_HASHES
            info[section].append(_analyze(inp, out))
    train, test = info["train"], info["test"]
    assert {x["orientation"] for x in train} == {"H", "V"}
    assert {2, 4}.issubset({x["count"] for x in train})
    assert any(x["mixed_sides"] for x in train)
    assert any(not x["mixed_sides"] for x in train)
    assert any(x["mixed_widths"] for x in train)
    assert any(not x["mixed_widths"] for x in train)
    assert any(x["exposed"] > 0 for x in train)
    non_frame = {color for x in train for color in
                 (x["field"], x["secondary"], *x["payload_colors"])}
    assert all(x["frame"] in non_frame and x["count"] in (2, 3, 4) for x in test)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    palette = tuple(int(x) for x in rng.permutation(10))
    puzzle = {"train": [], "test": []}
    seen = set()
    anchor = None
    for section, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
            for _ in range(80):
                if section == "train" and index == 0:
                    count, orientation, varying_widths, both_sides = 2, "H", False, False
                elif section == "train" and index == 1:
                    count, orientation, varying_widths, both_sides = 4, "V", True, True
                else:
                    count = int(pair_rng.integers(2, 5))
                    orientation = "H" if int(pair_rng.integers(2)) == 0 else "V"
                    varying_widths = bool(pair_rng.integers(2))
                    both_sides = bool(pair_rng.integers(2))
                if section == "test" and int(pair_rng.integers(5)) == 0:
                    signature = anchor
                else:
                    signature = _signature(pair_rng, count, orientation, varying_widths, both_sides)
                if section == "train" and index == 1:
                    anchor = signature
                roles = _roles(palette, pair_rng, section, index, len(signature[1]))
                phase = (int(pair_rng.integers(2)), int(pair_rng.integers(2)))
                pair = _pair(pair_rng, signature, roles, phase)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise ValueError("could not create distinct pair")
            puzzle[section].append(pair)
            seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for section in ("train", "test") for p in puzzle[section])


def self_test() -> None:
    features = []
    train_counts, test_counts, keys = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        keys.add(_puzzle_key(puzzle))
        features.extend((section, _analyze(p["input"], p["output"]))
                        for section in ("train", "test") for p in puzzle[section])
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(keys) == 42
    all_info = [x for _, x in features]
    tests = [x for split, x in features if split == "test"]
    assert {x["phase"] for x in all_info} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert {x["orientation"] for x in tests} == {"H", "V"}
    assert {x["count"] for x in tests} == {2, 3, 4}
    assert any(x["mixed_sides"] for x in tests) and any(not x["mixed_sides"] for x in tests)
    assert any(x["mixed_widths"] for x in tests) and any(not x["mixed_widths"] for x in tests)
    assert any(25 in x["shape"] for x in all_info)
    assert any(max(x["shape"]) <= 20 for x in all_info)
    assert any(x["shape"][0] != x["shape"][1] for x in all_info)
    assert len({x["geometry"] for x in all_info}) >= 35
    assert len({(x["shape"], x["frame_bbox"], x["source_bboxes"]) for x in all_info}) >= 150
    explicit = generate_puzzle(7001, 6, 3)
    assert (len(explicit["train"]), len(explicit["test"])) == (6, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(7001, 6, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(7002, 6, 3))
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8000 + 10 * nt + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for seed, nt, ns in ((8101, None, 1), (8102, None, 4),
                         (8103, 2, None), (8104, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
