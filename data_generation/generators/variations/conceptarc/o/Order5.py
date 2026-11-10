"""Generate panels sorted by their marked-cell counts."""

import itertools
import numpy as np


_SOURCE_INPUTS = {
    ("304003000030000300003", "300403000030040300003", "300003040030000300403",
     "340403000030400300003", "304003004030004300003", "300003000030000300003"),
    ("300003000030000300003", "340403400034004300003", "300003000030000340003",
     "300003000030004300003", "304003040030400300003", "300003000030000300003"),
    ("300003040030000304003", "300003000030000300403", "304003040030400340003",
     "300003000030000300043", "340403000030000304003", "300003000030000300003"),
    ("300003000030000304003", "300043000030000300003", "300403000030000304003",
     "304003040030000300003", "340003000030000300003", "300003000030000300003"),
    ("300003000030000300043", "300003000030000300043", "304003000030000300043",
     "304003000030000300003", "304003040030000300003", "304003040034000300003"),
}


def _pair_counts(rng, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                    (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _geometry(rng, panel_count):
    height = int(rng.integers(5, 14))
    max_width = min(6, (29 - panel_count) // panel_count)
    width = int(rng.integers(2, max_width + 1))
    return height, width


def _counts(rng, panel_count, height, width, contrast, sparse, train):
    area = height * width
    with_zero = (bool(rng.random() < (.18 if train else .38)) and not contrast and
                 not (sparse and train))
    minimum = 2 if contrast else 1
    upper = min(area - 1, max(panel_count + 2, int(.82 * area)))
    if contrast:
        upper = min(upper, (height - 1) * (width - 1))
    elif sparse or rng.random() < (.30 if train else .12):
        upper = min(upper, 7)
    needed = panel_count - int(with_zero)
    for _ in range(48):
        values = rng.choice(np.arange(minimum, upper + 1), size=needed,
                            replace=False)
        if not contrast and upper >= max(9, height) and rng.random() < .48:
            values[0] = int(rng.integers(max(8, height), upper + 1))
        result = tuple(sorted(set(int(v) for v in values) | ({0} if with_zero else set())))
        if len(result) == panel_count and sum(result) * 2 < panel_count * area:
            return result
    result = tuple(range(minimum, minimum + needed))
    if with_zero:
        result = (0,) + result
    return result


def _pattern(rng, height, width, count, mode, background, mark):
    panel = np.full((height, width), background, dtype=np.int8)
    if count == 0:
        return panel
    if mode == "corners":
        places = [0, height * width - 1]
        rest = [j for j in range(height * width) if j not in places]
        if count > 2:
            places.extend(int(v) for v in rng.choice(rest, size=count - 2, replace=False))
    elif mode == "compact":
        h, w = height - 1, width - 1
        top = int(rng.integers(0, 2))
        left = int(rng.integers(0, 2))
        selected = rng.choice(h * w, size=count, replace=False)
        places = [(top + int(j) // w) * width + left + int(j) % w for j in selected]
    elif mode == "column" and count >= height:
        column = int(rng.integers(width))
        places = [r * width + column for r in range(height)]
        remainder = [j for j in range(height * width) if j not in places]
        if count > height:
            places.extend(int(v) for v in rng.choice(remainder, size=count - height,
                                                     replace=False))
    elif mode == "run":
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        order = [(row + r) % height * width + (col + c) % width
                 for r in range(height) for c in range(width)]
        places = order[:count]
    elif mode == "cluster":
        center_r = int(rng.integers(height))
        center_c = int(rng.integers(width))
        noise = rng.random(height * width)
        places = sorted(range(height * width),
                        key=lambda j: (abs(j // width - center_r) +
                                       abs(j % width - center_c), noise[j]))[:count]
    else:
        places = rng.choice(height * width, size=count, replace=False)
    panel.flat[places] = mark
    return panel


def _order(rng, panel_count, used, source_like, train, allow_identity):
    if allow_identity:
        return tuple(range(panel_count))
    candidates = list(itertools.permutations(range(panel_count)))
    if source_like and train:
        candidates = [o for o in candidates if o[-1] == 0]
    if not allow_identity:
        candidates = [o for o in candidates if o != tuple(range(panel_count))]
    fresh = [o for o in candidates if o not in used]
    if fresh:
        candidates = fresh
    return candidates[int(rng.integers(len(candidates)))]


def _make_pair(rng, panel_count, geometry, palette, counts, order, contrast):
    height, width = geometry
    background, divider, mark = palette
    panels = []
    for rank in order:
        mode = ("corners" if contrast and rank == 0 else
                "compact" if contrast and rank == panel_count - 1 else
                ("scatter", "cluster", "run", "column")[int(rng.integers(4))])
        panels.append(_pattern(rng, height, width, counts[rank], mode,
                               background, mark))
    stride = width + 1
    inp = np.full((height, panel_count * stride + 1), divider, dtype=np.int8)
    out = inp.copy()
    for position, panel in enumerate(panels):
        inp[:, position * stride + 1:(position + 1) * stride] = panel
    for destination, source in enumerate(sorted(range(panel_count),
                                                key=lambda j: counts[order[j]])):
        out[:, destination * stride + 1:(destination + 1) * stride] = panels[source]
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _pair_counts(rng, num_train, num_test)
    source_like = bool(rng.random() < .16)
    local_palette = not source_like and bool(rng.random() < .55)
    panel_mode = int(rng.integers(3)) if not source_like else 0
    if panel_mode == 0:
        panel_menu = (4,)
    elif panel_mode == 1:
        panel_menu = (int(rng.choice((3, 5, 6))),)
    else:
        panel_menu = tuple(int(v) for v in rng.choice((3, 4, 5, 6), size=2,
                                                       replace=False))
    fixed_geometry = _geometry(rng, 4) if source_like else None
    fixed_palette = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    reuse_counts = (len(panel_menu) == 1 and n_train >= 2 and
                    bool(rng.random() < .48) and not source_like)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_orders = set()
    first_counts = None
    first_geometry = None
    first_palette = None
    for split, length in (("train", n_train), ("test", n_test)):
        for index in range(length):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            if split == "train":
                panel_count = panel_menu[index % len(panel_menu)]
            else:
                panel_count = panel_menu[int(pair_rng.integers(len(panel_menu)))]
            for attempt in range(24):
                geometry = (fixed_geometry if fixed_geometry is not None else
                            first_geometry if split == "train" and index == 1 and
                            reuse_counts else _geometry(pair_rng, panel_count))
                palette = (fixed_palette if not local_palette else
                           tuple(int(v) for v in pair_rng.choice(10, size=3,
                                                                 replace=False)))
                if local_palette and split == "train" and index == 1 and palette == first_palette:
                    palette = (palette[1], palette[2], palette[0])
                contrast = (not source_like and geometry[1] >= 3 and
                            bool(pair_rng.random() < .28))
                counts = (first_counts if split == "train" and index == 1 and
                          reuse_counts else _counts(pair_rng, panel_count,
                                                   *geometry, contrast,
                                                   source_like, split == "train"))
                if reuse_counts and split == "train" and index == 1:
                    contrast = False
                identity = ((split == "train" and index >= 2 or split == "test")
                            and bool(pair_rng.random() < .14))
                order = _order(pair_rng, panel_count, used_orders if split == "train"
                               else set(), source_like, split == "train", identity)
                pair = _make_pair(pair_rng, panel_count, geometry, palette, counts,
                                  order, contrast)
                inp = pair["input"]
                key = (inp.shape, inp.tobytes())
                rows = tuple("".join(str(int(v)) for v in row) for row in inp)
                if key not in used_inputs and rows not in _SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError("Could not construct a unique pair")
            puzzle[split].append(pair)
            used_inputs.add(key)
            if split == "train":
                used_orders.add(order)
                if index == 0:
                    first_counts, first_geometry, first_palette = counts, geometry, palette
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= dimension <= 30 for dimension in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert bool(np.all((grid >= 0) & (grid <= 9)))
    assert inp.shape == out.shape
    height, grid_width = inp.shape
    divider = int(inp[0, 0])
    divider_columns = np.flatnonzero(np.all(inp == divider, axis=0))
    panel_count = len(divider_columns) - 1
    assert 3 <= panel_count <= 6
    assert divider_columns[0] == 0 and divider_columns[-1] == grid_width - 1
    widths = np.diff(divider_columns) - 1
    assert np.all(widths == widths[0]) and 2 <= widths[0] <= 6
    assert np.array_equal(inp[:, divider_columns], out[:, divider_columns])
    assert np.all(out[:, divider_columns] == divider)
    panels = [inp[:, divider_columns[j] + 1:divider_columns[j + 1]].copy()
              for j in range(panel_count)]
    interior = np.concatenate(panels, axis=1)
    colors, frequencies = np.unique(interior, return_counts=True)
    assert len(colors) == 2 and divider not in colors
    assert frequencies[0] != frequencies[1]
    background = int(colors[np.argmax(frequencies)])
    mark = int(colors[np.argmin(frequencies)])
    counts = tuple(int(np.count_nonzero(p == mark)) for p in panels)
    assert len(set(counts)) == panel_count
    expected = inp.copy()
    order = tuple(sorted(range(panel_count), key=lambda j: counts[j]))
    for j, source in enumerate(order):
        expected[:, divider_columns[j] + 1:divider_columns[j + 1]] = panels[source]
    assert np.array_equal(out, expected)
    rows = tuple("".join(str(int(v)) for v in row) for row in inp)
    assert rows not in _SOURCE_INPUTS
    return (panel_count, height, int(widths[0]), background, divider, mark,
            counts, order)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    train_panel_counts, train_orders, test_panel_counts = set(), set(), set()
    train_moves = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            inp, out = pair["input"], pair["output"]
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.shape, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            if split == "train":
                train_panel_counts.add(info[0])
                train_orders.add((info[0], info[-1]))
                train_moves += not np.array_equal(inp, out)
            else:
                test_panel_counts.add(info[0])
    assert test_panel_counts <= train_panel_counts
    assert train_moves >= 2 and len(train_orders) >= 2


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_lengths, test_lengths = set(), set()
    shapes, palettes, panel_counts, structures = set(), set(), set(), set()
    varied_geometry = varied_palette = zeros_train = zeros_test = 0
    identity = high_counts = full_columns = contrast_pairs = reused_counts = 0
    runs = compact_blocks = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        train_info = [_inspect_pair(p) for p in puzzle["train"]]
        all_info = train_info + [_inspect_pair(p) for p in puzzle["test"]]
        varied_geometry += len({x[:3] for x in all_info}) > 1
        varied_palette += len({x[3:6] for x in all_info}) > 1
        reused_counts += len({(x[0], tuple(sorted(x[6]))) for x in train_info}) < len(train_info)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                n, height, width, bg, divider, mark, counts, order = _inspect_pair(pair)
                inp, out = pair["input"], pair["output"]
                shapes.add(inp.shape)
                palettes.add((bg, divider, mark))
                panel_counts.add(n)
                structures.add((n, height, width, tuple(sorted(counts)), order))
                identity += np.array_equal(inp, out)
                high_counts += max(counts) > 7
                zeros_train += split == "train" and 0 in counts
                zeros_test += split == "test" and 0 in counts
                stride = width + 1
                panels = [inp[:, j * stride + 1:(j + 1) * stride] for j in range(n)]
                full_columns += any(np.any(np.all(p == mark, axis=0)) for p in panels)
                for panel in panels:
                    mask = panel == mark
                    runs += bool(np.any(mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]) or
                                 np.any(mask[:-2, :] & mask[1:-1, :] & mask[2:, :]))
                    compact_blocks += bool(np.any(mask[:-1, :-1] & mask[1:, :-1] &
                                                  mask[:-1, 1:] & mask[1:, 1:]))
                if width >= 3:
                    boxes = []
                    for p in panels:
                        rows, cols = np.nonzero(p == mark)
                        boxes.append((0, 0) if len(rows) == 0 else
                                     (int(rows.max() - rows.min() + 1),
                                      int(cols.max() - cols.min() + 1)))
                    contrast_pairs += any(counts[a] > counts[b] >= 2 and
                                          boxes[a][0] < boxes[b][0] and
                                          boxes[a][1] < boxes[b][1]
                                          for a in range(n) for b in range(n))
                assert np.count_nonzero(inp == mark) == np.count_nonzero(out == mark)
                assert sorted(p.tobytes() for p in panels) == sorted(
                    out[:, j * stride + 1:(j + 1) * stride].tobytes()
                    for j in range(n))
                assert tuple(sorted(counts)) == tuple(int(np.count_nonzero(
                    out[:, j * stride + 1:(j + 1) * stride] == mark))
                    for j in range(n))
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert len(shapes) >= 15 and len(palettes) >= 20
    assert panel_counts == {3, 4, 5, 6} and len(structures) >= 150
    assert varied_geometry >= 20 and varied_palette >= 10
    assert zeros_train >= 5 and zeros_test >= 5
    assert identity >= 2 and high_counts >= 30 and full_columns >= 3
    assert contrast_pairs >= 5 and reused_counts >= 2
    assert runs >= 5 and compact_blocks >= 5
    for nt, ns in ((2, 1), (10, 1), (10, 4), (5, 2)):
        puzzle = generate_puzzle(100 + nt * 10 + ns, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
        assert _same_puzzle(puzzle, generate_puzzle(100 + nt * 10 + ns, nt, ns))
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(200 + (nt or 0) * 10 + (ns or 0), nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {(nt, ns)}")
