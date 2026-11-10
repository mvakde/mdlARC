"""Compositional tapered-outline completion puzzles."""

import numpy as np


def _seed_as_int(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _rows(s):
    return tuple(s.split("/"))


_SOURCE = {
    (_rows("0000000000/0000000000/1000010000/0100100000/0011000000/0011000000/0000400004/0000040040/0000004400/0000004400"),
     _rows("0000000000/0000000000/1111110000/0111100000/0011000000/0011000000/0000444444/0000044440/0000004400/0000004400")),
    (_rows("0000000000/0300000030/0030000300/0003003000/0000330000/0000330000/0000000000/0000000000/0000000000/0000000000"),
     _rows("0000000000/0333333330/0033333300/0003333000/0000330000/0000330000/0000000000/0000000000/0000000000/0000000000")),
    (_rows("6000060000/0600600000/0066000000/0066000000/0000000000/0800000080/0080000800/0008008000/0000880000/0000880000"),
     _rows("6666660000/0666600000/0066000000/0066000000/0000000000/0888888880/0088888800/0008888000/0000880000/0000880000")),
    (_rows("0000000000/4004000000/0440700007/0440070070/0000007700/0000007700/0300003000/0030030000/0003300000/0003300000"),
     _rows("0000000000/4444000000/0440777777/0440077770/0000007700/0000007700/0333333000/0033330000/0003300000/0003300000")),
}


def _source_key(inp, out):
    return (tuple("".join(str(int(v)) for v in row) for row in inp),
            tuple("".join(str(int(v)) for v in row) for row in out))


def _direction_set(rng, seed):
    if seed % 4 == 0:
        return ("down",), True
    choices = ("down", "up", "right", "left")
    if seed % 4 == 1:
        return (choices[int(rng.integers(4))],), False
    ids = rng.choice(4, size=2, replace=False)
    return (choices[int(ids[0])], choices[int(ids[1])]), False


def _sample_shape(rng, max_lateral, max_depth, continuous, tall=False):
    maximum = min(max_lateral, 19 if tall else 13)
    if maximum < 4 or max_depth < 3:
        raise ValueError("canvas too small")
    if not tall and rng.random() < 0.24:
        symmetric_widths = [w for w in range(4, maximum + 1, 2)
                            if (w - 2) // 2 + 2 <= max_depth]
        if symmetric_widths:
            width = int(symmetric_widths[int(rng.integers(len(symmetric_widths)))])
            depth = (width - 2) // 2
            terminal = int(rng.integers(2, min(4, max_depth - depth + 1)))
            return tuple([(i, width - 1 - i) for i in range(depth)]
                         + [(depth, depth + 1)] * terminal)
    minimum = min(maximum, 11 if tall else 4)
    for _ in range(30):
        width = int(rng.integers(minimum, maximum + 1))
        tip = int(rng.integers(1, 3))
        left, right = 0, width - 1
        spans = [(left, right)]
        repeat_wide = rng.random() < 0.24
        while right - left + 1 > tip and len(spans) < max_depth - 1:
            if repeat_wide and len(spans) > 1 and rng.random() < 0.14:
                spans.append((left, right))
                repeat_wide = False
                continue
            step = 1 if continuous else 2
            dl = int(rng.integers(step + 1))
            dr = int(rng.integers(step + 1))
            if dl + dr == 0:
                if rng.random() < 0.5:
                    dl = 1
                else:
                    dr = 1
            remaining = right - left + 1 - tip
            if dl + dr > remaining:
                if dl > remaining:
                    dl, dr = remaining, 0
                else:
                    dr = remaining - dl
            left += dl
            right -= dr
            spans.append((left, right))
        if right - left + 1 != tip or len(spans) < 3:
            continue
        terminal = int(rng.integers(1, min(4, max_depth - len(spans) + 1))) if len(spans) < max_depth else 0
        spans.extend([(left, right)] * terminal)
        if len(spans) <= max_depth and spans[0][1] - spans[0][0] + 1 >= 4:
            return tuple(spans)
    raise ValueError("could not sample a fitting taper")


def _dimensions(rng, count, compact=False, tall=False):
    if compact:
        return int(rng.integers(9, 14)), int(rng.integers(9, 15))
    if tall:
        return int(rng.integers(22, 31)), int(rng.integers(12, 25))
    low = 12 if count <= 2 else 16
    return int(rng.integers(low, 31)), int(rng.integers(low, 31))


def _size(spans, direction):
    depth, lateral = len(spans), spans[0][1] + 1
    return (depth, lateral) if direction in ("down", "up") else (lateral, depth)


def _coordinates(spans, direction, row, col):
    depth = len(spans)
    for t, (left, right) in enumerate(spans):
        for u in range(left, right + 1):
            if direction == "down":
                yield (row + t, col + u, u == left or u == right)
            elif direction == "up":
                yield (row + depth - 1 - t, col + u, u == left or u == right)
            elif direction == "right":
                yield (row + u, col + t, u == left or u == right)
            else:
                yield (row + u, col + depth - 1 - t, u == left or u == right)


def _dilate(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]]
                                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)])


def _place_objects(rng, height, width, specs, layout):
    placed = []
    occupied = np.zeros((height, width), dtype=bool)
    for spans, direction in specs:
        h, w = _size(spans, direction)
        if h > height or w > width:
            return None
        options = []
        blocked = _dilate(occupied)
        for row in range(height - h + 1):
            for col in range(width - w + 1):
                mask = np.zeros((height, width), dtype=bool)
                for r, c, _ in _coordinates(spans, direction, row, col):
                    mask[r, c] = True
                if np.any(mask & blocked):
                    continue
                shared = sum(max(0, min(row + h, old[0] + old[2]) - max(row, old[0]))
                             for old in placed)
                if layout == "separated" and shared:
                    continue
                score = shared if layout in ("overlap", "simultaneous") else 0
                if layout == "simultaneous" and len(placed) >= 2:
                    intersection = max(0, min([row + h] + [old[0] + old[2] for old in placed])
                                       - max([row] + [old[0] for old in placed]))
                    score += intersection * 6
                options.append((score, row, col, mask))
        if not options:
            return None
        if layout in ("overlap", "simultaneous") and placed:
            best = max(x[0] for x in options)
            options = [x for x in options if x[0] >= max(1, best - 2)] if best else options
        choice = options[int(rng.integers(len(options)))]
        _, row, col, mask = choice
        occupied |= mask
        placed.append((row, col, h, w, spans, direction))
    if layout in ("overlap", "simultaneous") and len(placed) >= 2:
        if not any(max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
                   for i, a in enumerate(placed) for b in placed[i + 1:]):
            return None
    if layout == "simultaneous" and len(placed) >= 3:
        if max(x[0] for x in placed) >= min(x[0] + x[2] for x in placed):
            return None
    return placed


def _render(height, width, background, placed, colors):
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for item, color in zip(placed, colors):
        row, col, _, _, spans, direction = item
        for r, c, endpoint in _coordinates(spans, direction, row, col):
            if out[r, c] != background:
                raise ValueError("filled objects collide")
            out[r, c] = color
            if endpoint:
                inp[r, c] = color
    if np.array_equal(inp, out):
        raise ValueError("no visible completion")
    values, counts = np.unique(inp, return_counts=True)
    if int(values[int(np.argmax(counts))]) != background or (len(counts) > 1 and sorted(counts)[-1] == sorted(counts)[-2]):
        raise ValueError("background is not identifiable")
    return inp, out


def _make_pair(rng, directions, count, layout, background, colors, compact=False, tall=False, connected=False):
    for _ in range(70):
        height, width = _dimensions(rng, count, compact, tall)
        specs = []
        try:
            for i in range(count):
                direction = directions[i % len(directions)] if i < len(directions) else directions[int(rng.integers(len(directions)))]
                depth_limit = height if direction in ("down", "up") else width
                lateral_limit = width if direction in ("down", "up") else height
                # Shared-color objects must retain connected boundaries for unambiguous grouping.
                continuous = connected or rng.random() < 0.65
                shape = _sample_shape(rng, min(lateral_limit - 2, 19), min(depth_limit - 1, 22), continuous,
                                      tall and i == 0)
                specs.append((shape, direction))
            placed = _place_objects(rng, height, width, specs, layout)
            if placed is None:
                continue
            inp, out = _render(height, width, background, placed, colors)
            try:
                _, reconstructed, _ = _analyze_pair(inp)
            except AssertionError:
                continue
            if np.array_equal(out, reconstructed):
                return inp, out
        except ValueError:
            continue
    raise ValueError("unable to construct a noncolliding pair")


def _components8(cells):
    remaining = set(cells)
    result = []
    while remaining:
        stack = [remaining.pop()]
        part = set(stack)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        part.add(q)
                        stack.append(q)
        result.append(part)
    return result


def _infer_one(cells):
    candidates = []
    for direction in ("down", "up", "right", "left"):
        groups = {}
        for r, c in cells:
            t, u = ((r, c) if direction == "down" else
                    (-r, c) if direction == "up" else
                    (c, r) if direction == "right" else (-c, r))
            groups.setdefault(t, []).append(u)
        ts = sorted(groups)
        if ts != list(range(ts[0], ts[-1] + 1)) or len(ts) < 3:
            continue
        spans = []
        valid = True
        for t in ts:
            us = sorted(groups[t])
            if len(us) not in (1, 2):
                valid = False
                break
            spans.append((us[0], us[-1]))
        if not valid:
            continue
        widths = [right - left + 1 for left, right in spans]
        if widths[0] < 4 or widths[-1] > 2 or not any(x > y for x, y in zip(widths, widths[1:])):
            continue
        if any(a > b or c < d for (a, c), (b, d) in zip(spans, spans[1:])):
            continue
        expected = set()
        for t, (left, right) in zip(ts, spans):
            for u in range(left, right + 1):
                if direction == "down":
                    expected.add((t, u))
                elif direction == "up":
                    expected.add((-t, u))
                elif direction == "right":
                    expected.add((u, t))
                else:
                    expected.add((u, -t))
        candidates.append((direction, expected, tuple(widths), spans))
    if len(candidates) != 1:
        raise AssertionError("taper axis or endpoints ambiguous")
    return candidates[0]


def _analyze_pair(inp):
    values, counts = np.unique(inp, return_counts=True)
    if len(values) < 2 or sum(counts == max(counts)) != 1:
        raise AssertionError("background is ambiguous")
    background = int(values[int(np.argmax(counts))])
    objects = []
    for color in values:
        color = int(color)
        if color == background:
            continue
        cells = set(zip(*np.nonzero(inp == color)))
        try:
            inferred = [_infer_one(cells)]
        except AssertionError:
            parts = _components8(cells)
            if len(parts) < 2:
                raise
            inferred = [_infer_one(part) for part in parts]
            if any(len(_components8(part)) != 1 for part in parts):
                raise AssertionError("same-color object disconnected")
        for direction, expected, widths, spans in inferred:
            objects.append((color, direction, expected, widths, spans))
    out = inp.copy()
    occupied = set()
    for color, direction, cells, _, _ in objects:
        if occupied & cells:
            raise AssertionError("completed spans overlap")
        occupied |= cells
        for r, c in cells:
            if int(inp[r, c]) not in (background, color):
                raise AssertionError("another color interrupts a span")
            out[r, c] = color
    if np.array_equal(out, inp):
        raise AssertionError("completion is invisible")
    return background, out, objects


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    directions = {"train": set(), "test": set()}
    repeats = {"train": False, "test": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert _source_key(inp, out) not in _SOURCE
            _, expected, objects = _analyze_pair(inp)
            assert np.array_equal(expected, out), "wrong object-local completion"
            directions[split].update(x[1] for x in objects)
            colors = [x[0] for x in objects]
            repeats[split] |= len(colors) != len(set(colors))
    assert directions["test"] <= directions["train"], "test orientation missing from training"
    assert not repeats["test"] or repeats["train"], "same-color grouping missing from training"


def _choose_colors(rng, count, pool, background, repeat=False):
    available = [x for x in pool if x != background]
    if not available:
        raise ValueError("no usable object colors")
    if repeat:
        common = int(available[int(rng.integers(len(available)))])
        others = [int(available[int(rng.integers(len(available)))]) for _ in range(count - 2)]
        return [common, common] + others
    if count > len(available):
        available += [x for x in range(10) if x != background and x not in available]
    return [int(x) for x in rng.choice(available, size=count, replace=False)]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed_as_int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    directions, classic = _direction_set(rng, seed)
    repeat_convention = not classic and seed % 3 == 1
    whole_connected = seed % 6 == 0
    bank = [int(x) for x in rng.choice(10, size=4, replace=False)]
    backgrounds = [int(x) for x in rng.permutation(10)]
    puzzle = {"train": [], "test": []}
    used = set()
    train_colors = set()
    for split, count_pairs in (("train", num_train), ("test", num_test)):
        for index in range(count_pairs):
            global_index = index if split == "train" else num_train + index
            background = backgrounds[global_index % 10]
            if split == "train":
                count = (1 + (index + seed) % 2) if classic else (max(2, len(directions)) if index == 0 else 1 + (index + seed) % 4)
                repeat = repeat_convention and index == 0
                layout = "separated" if classic and count > 1 else ("simultaneous" if count >= 3 and index % 3 == 1 else "overlap" if count > 1 and index % 2 == 0 else "free")
                pool = bank
            else:
                count = 3 if classic else 1 + (index + seed // 3) % 5
                count = max(count, len(directions)) if index == 0 else count
                repeat = repeat_convention and count >= 2 and index % 2 == 0
                layout = "overlap" if classic else ("simultaneous" if count >= 3 and index % 3 == 0 else "overlap" if count > 1 and index % 2 == 0 else "free")
                mode = (seed + index) % 3
                unseen = [x for x in range(10) if x not in train_colors and x != background]
                familiar = [x for x in train_colors if x != background]
                if mode == 0 or not unseen:
                    pool = familiar or bank
                elif mode == 1:
                    pool = (familiar[:2] or bank[:2]) + unseen[:2]
                else:
                    pool = unseen
            colors = _choose_colors(rng, count, pool, background, repeat)
            compact = (seed + index) % 7 == 2 and count <= 3
            tall = (seed + index) % 9 == 5 and count <= 2 and not compact
            pair = None
            for attempt in range(5):
                try:
                    inp, out = _make_pair(rng, directions, count, layout if attempt < 2 else "free",
                                          background, colors, compact and attempt < 2, tall and attempt < 2,
                                          repeat or whole_connected)
                except ValueError:
                    continue
                key = _grid_key(inp)
                if key in used or _source_key(inp, out) in _SOURCE:
                    continue
                pair = {"input": inp, "output": out}
                used.add(key)
                break
            if pair is None:
                raise ValueError("unable to generate unique pair")
            puzzle[split].append(pair)
            if split == "train":
                train_colors.update(colors)
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    observed = {}
    def add(name):
        observed[name] = observed.get(name, 0) + 1
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        backgrounds = []
        pair_counts = []
        all_connected = True
        train_palette = set()
        puzzle_directions = set()
        regimes = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                background, expected, objects = _analyze_pair(inp)
                assert np.array_equal(expected, pair["output"])
                backgrounds.append(background)
                pair_counts.append(len(objects))
                colors = {item[0] for item in objects}
                if split == "train":
                    train_palette.update(colors)
                else:
                    new = colors - train_palette
                    add("new_palette" if new == colors else "partial_palette" if new else "familiar_palette")
                    if len(objects) >= 3 and max(inp.shape) <= 15:
                        add("compact_multi_test")
                if len(objects) >= 3:
                    ranges = [set(r for r, _ in item[2]) for item in objects]
                    if set.intersection(*ranges):
                        add("simultaneous_" + split)
                pair_overlap = False
                for i, first in enumerate(objects):
                    for second in objects[i + 1:]:
                        first_rows = {r for r, _ in first[2]}
                        second_rows = {r for r, _ in second[2]}
                        if first_rows & second_rows:
                            add("overlap_" + split)
                            pair_overlap = True
                            if first[0] == second[0]:
                                add("same_color_overlap_" + split)
                        elif max(first_rows) + 1 == min(second_rows) or max(second_rows) + 1 == min(first_rows):
                            add("consecutive_bands_" + split)
                regimes[split].append((len(objects), pair_overlap))
                for color, direction, cells, widths, spans in objects:
                    add("direction_" + direction)
                    puzzle_directions.add(direction)
                    endpoints = {point for point in cells if inp[point] == color}
                    if len(_components8(endpoints)) == 1:
                        add("connected")
                    else:
                        add("gapped")
                        all_connected = False
                    if len(widths) >= 10:
                        add("tall")
                    if widths[-1] == 1:
                        add("one_cell_tip")
                    if widths[-1] == 2:
                        first_tip = widths.index(2)
                        if (first_tip >= 1
                            and all(spans[i + 1][0] - spans[i][0] == 1
                                    and spans[i][1] - spans[i + 1][1] == 1
                                    for i in range(first_tip))
                            and all(width == 2 for width in widths[first_tip:])):
                            add("symmetric_two_cell")
                    if any(a == b and a > 2 for a, b in zip(widths, widths[1:])):
                        add("wide_repeat")
                    if any(b[0] - a[0] != a[1] - b[1] for a, b in zip(spans, spans[1:])):
                        add("asymmetric")
        if len(set(backgrounds)) > 1:
            add("varying_background_puzzles")
        if len(set(pair_counts)) > 1:
            add("varying_count_puzzles")
        if all_connected:
            add("all_connected_puzzles")
        if len(puzzle_directions) > 1:
            add("mixed_direction_puzzles")
        if ({number for number, _ in regimes["train"]} == {1, 2}
            and all(not overlap for _, overlap in regimes["train"])
            and all(number == 3 and overlap for number, overlap in regimes["test"])):
            add("classic_shift_puzzles")
    for name, minimum in {
        "varying_background_puzzles": 42, "varying_count_puzzles": 35,
        "connected": 250, "gapped": 100, "all_connected_puzzles": 4,
        "tall": 25, "one_cell_tip": 100, "wide_repeat": 15,
        "symmetric_two_cell": 25,
        "asymmetric": 200, "same_color_overlap_train": 5,
        "same_color_overlap_test": 5, "simultaneous_train": 10,
        "simultaneous_test": 5, "consecutive_bands_train": 2,
        "overlap_train": 15, "overlap_test": 15,
        "new_palette": 10, "partial_palette": 10, "familiar_palette": 10,
        "compact_multi_test": 1,
        "direction_down": 15, "direction_up": 15,
        "direction_left": 15, "direction_right": 15,
        "mixed_direction_puzzles": 10, "classic_shift_puzzles": 5,
    }.items():
        assert observed.get(name, 0) >= minimum, (name, observed.get(name, 0))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_grid_key(pair["input"]) for p in puzzles for s in p.values() for pair in s}) > 150
    assert len({pair["input"].shape for p in puzzles for s in p.values() for pair in s}) > 25
    assert len({int(np.unique(pair["input"], return_counts=True)[0][np.argmax(np.unique(pair["input"], return_counts=True)[1])])
                for p in puzzles for s in p.values() for pair in s}) == 10
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 2), (8, 3), (9, 4)):
        validate_generated_puzzle(generate_puzzle(2000 + train * 10 + test, train, test))
    for kwargs in ({"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4},
                   {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None}):
        validate_generated_puzzle(generate_puzzle(3001, **kwargs))
    assert _equal(generate_puzzle(456789, 8, 3), generate_puzzle(456789, 8, 3))
    assert not _equal(generate_puzzle(456789, 8, 3), generate_puzzle(456790, 8, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(9, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
