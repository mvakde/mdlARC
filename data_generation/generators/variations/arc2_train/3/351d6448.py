from __future__ import annotations

import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "0000000000000",
            "1100000000000",
            "0000000000000",
            "5555555555555",
            "0000000000000",
            "1111000000000",
            "0000000000000",
            "5555555555555",
            "0000000000000",
            "1111110000000",
            "0000000000000",
            "5555555555555",
            "0000000000000",
            "1111111100000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "1111111111000",
            "0000000000000",
        ),
    ),
    (
        (
            "0000000000000",
            "0223000000000",
            "0000000000000",
            "5555555555555",
            "0000000000000",
            "0022300000000",
            "0000000000000",
            "5555555555555",
            "0000000000000",
            "0002230000000",
            "0000000000000",
            "5555555555555",
            "0000000000000",
            "0000223000000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0000022300000",
            "0000000000000",
        ),
    ),
    (
        (
            "0020002000200",
            "0322022202220",
            "3322222222222",
            "5555555555555",
            "0030002000200",
            "0332022202220",
            "3332222222222",
            "5555555555555",
            "0030002000200",
            "0333022202220",
            "3333222222222",
            "5555555555555",
            "0030002000200",
            "0333022202220",
            "3333322222222",
        ),
        (
            "0030002000200",
            "0333032202220",
            "3333332222222",
        ),
    ),
)


def _rows_to_grid(rows) -> np.ndarray:
    return np.array(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _same_as_source(inp: np.ndarray, out: np.ndarray) -> bool:
    return any(
        np.array_equal(inp, source_input)
        and np.array_equal(out, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _value_signature(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.uint8, copy=False)
    return tuple(grid.shape), normalized.tobytes()


def _pair_signature(inp: np.ndarray, out: np.ndarray) -> tuple:
    return _value_signature(inp), _value_signature(out)


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    def checked(
        value: int,
        low: int,
        high: int,
        name: str,
    ) -> int:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{name} must be an integer or None")

        result = int(value)
        if not low <= result <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}"
            )
        return result

    if num_train is not None:
        num_train = checked(num_train, 2, 10, "num_train")

    if num_test is not None:
        num_test = checked(num_test, 1, 4, "num_test")

    if num_train is None and num_test is None:
        # These formulas deliberately cover all four legal test counts and
        # many train counts over consecutive seeds.
        num_test = 1 + seed % 4
        base_train = 2 + (seed // 4) % 9
        num_train = max(base_train, num_test + 1)

    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test




def _pack(frames, divider, layout):
    if layout == "stack":
        pieces = []
        for i, frame in enumerate(frames):
            if i:
                pieces.append(np.full((1, frame.shape[1]), divider, np.int8))
            pieces.append(frame)
        return np.vstack(pieces)
    pieces = []
    for i, frame in enumerate(frames):
        if i:
            pieces.append(np.full((frame.shape[0], 1), divider, np.int8))
        pieces.append(frame)
    return np.hstack(pieces)


def _axis_array(frame, axis):
    return frame if axis == 1 else frame.T


def _from_axis(frame, axis):
    return frame if axis == 1 else frame.T


def _scene(rng, layout):
    long_size = int(rng.integers(12, 31))
    short_size = int(rng.integers(2, 7))
    if rng.random() < 0.18:
        short_size = 6
        axis = 0 if layout == "stack" else 1
    else:
        axis = 1 if layout == "stack" else 0
    shape = (short_size, long_size) if layout == "stack" else (long_size, short_size)
    return shape, axis


def _translation(rng, shape, axis, colors):
    background = colors[0]
    cross, length = _axis_array(np.empty(shape), axis).shape
    step = int(rng.integers(1, min(3, (length - 2) // 4) + 1))
    motif_len = int(rng.integers(1, min(10, length - 4 * step) + 1))
    depth = int(rng.integers(1, min(4, cross) + 1))
    motif = np.full((depth, motif_len), background, np.int8)
    count_colors = int(rng.choice([1, 1, 2, 2, 3]))
    for col in range(motif_len):
        rows = np.flatnonzero(rng.random(depth) < float(rng.uniform(0.4, 0.9)))
        if len(rows) == 0 and (col == 0 or col == motif_len - 1 or rng.random() < 0.7):
            rows = np.array([int(rng.integers(depth))])
        for row in rows:
            motif[row, col] = colors[2 + int(rng.integers(count_colors))]
    motif[int(rng.integers(depth)), 0] = colors[2]
    motif[int(rng.integers(depth)), -1] = colors[2 + count_colors - 1]
    direction = 1 if rng.random() < 0.5 else -1
    first = int(rng.integers(0, length - motif_len - 4 * step + 1))
    if direction < 0:
        first += 4 * step
    cross_start = int(rng.integers(0, cross - depth + 1))
    frames = []
    for time in range(5):
        frame = np.full((cross, length), background, np.int8)
        x = first + direction * time * step
        frame[cross_start:cross_start + depth, x:x + motif_len] = motif
        frames.append(_from_axis(frame, axis))
    return frames


def _growth(rng, shape, axis, colors, shrink):
    background = colors[0]
    cross, length = _axis_array(np.empty(shape), axis).shape
    step = int(rng.integers(1, min(3, (length - 2) // 4) + 1))
    period = int(rng.integers(1, min(3, length - 4 * step) + 1))
    first_len = int(rng.integers(max(2, period), length - 4 * step + 1))
    depth = int(rng.integers(1, min(4, cross) + 1))
    count_colors = int(rng.choice([1, 1, 2, 3]))
    tile = np.full((depth, period), background, np.int8)
    for j in range(period):
        present = np.flatnonzero(rng.random(depth) < 0.78)
        if len(present) == 0:
            present = np.array([int(rng.integers(depth))])
        for row in present:
            tile[row, j] = colors[2 + int(rng.integers(count_colors))]
    direction = 1 if rng.random() < 0.5 else -1
    max_len = first_len + 4 * step
    low = int(rng.integers(0, length - max_len + 1))
    cross_start = int(rng.integers(0, cross - depth + 1))
    lengths = [first_len + t * step for t in range(5)]
    if shrink:
        lengths.reverse()
    frames = []
    for current_len in lengths:
        start = low if direction > 0 else low + max_len - current_len
        frame = np.full((cross, length), background, np.int8)
        for j in range(current_len):
            frame[cross_start:cross_start + depth, start + j] = tile[:, (start + j - low) % period]
        frames.append(_from_axis(frame, axis))
    return frames


def _mask(rng, cross, length, kind):
    mask = np.zeros((cross, length), bool)
    if kind == "periodic":
        period = int(rng.integers(2, min(5, length) + 1))
        tile = rng.random((cross, period)) < float(rng.uniform(0.55, 0.85))
        mask[:] = tile[:, np.arange(length) % period]
    elif kind == "solid":
        depth = int(rng.integers(1, cross + 1))
        start = int(rng.integers(0, cross - depth + 1))
        mask[start:start + depth, :] = True
    elif kind == "sparse":
        mask[:] = rng.random((cross, length)) < float(rng.uniform(0.25, 0.52))
    elif kind == "irregular":
        mask[:] = rng.random((cross, length)) < float(rng.uniform(0.45, 0.82))
    else:
        for _ in range(int(rng.integers(2, 6))):
            a = int(rng.integers(cross))
            b = int(rng.integers(a + 1, cross + 1))
            x = int(rng.integers(length))
            y = int(rng.integers(x + 1, length + 1))
            mask[a:b, x:y] = True
    if length >= 12 and rng.random() < 0.37:
        for _ in range(int(rng.integers(1, 4))):
            mask[:, int(rng.integers(length))] = False
    return mask


def _frontier(rng, shape, axis, colors):
    background, old, new = colors[0], colors[2], colors[3]
    cross, length = _axis_array(np.empty(shape), axis).shape
    step = int(rng.integers(1, min(3, (length - 2) // 4) + 1))
    direction = 1 if rng.random() < 0.5 else -1
    for _ in range(180):
        kind = str(rng.choice(["periodic", "solid", "sparse", "irregular", "silhouette"]))
        mask = _mask(rng, cross, length, kind)
        if not np.any(mask):
            continue
        if direction > 0:
            boundary = int(rng.integers(0, length - 4 * step))
        else:
            boundary = int(rng.integers(4 * step, length))
        positions = np.arange(length)
        zones = []
        for time in range(5):
            current = boundary + direction * time * step
            zones.append(positions <= current if direction > 0 else positions >= current)
        if not np.any(mask & zones[0][None, :]) or not np.any(mask & ~zones[3][None, :]):
            continue
        good = True
        for time in range(1, 5):
            changed_zone = zones[time] & ~zones[time - 1]
            changed = np.flatnonzero(np.any(mask & changed_zone[None, :], axis=0))
            if len(changed) == 0:
                good = False
                break
            expected = np.flatnonzero(changed_zone)
            if changed[0] != expected[0] or changed[-1] != expected[-1]:
                good = False
                break
        if not good:
            continue
        frames = []
        for zone in zones:
            frame = np.full((cross, length), background, np.int8)
            frame[mask] = old
            frame[mask & zone[None, :]] = new
            frames.append(_from_axis(frame, axis))
        if (any(int(frame[0, 0]) != background for frame in frames) or
            any(not (np.any(frame == old) and np.any(frame == new)) for frame in frames)):
            continue
        return frames
    raise RuntimeError("could not construct a visible frontier")


def _split(inp, out):
    candidates = []
    if (inp.shape[0] - 3) % 4 == 0:
        h = (inp.shape[0] - 3) // 4
        if h >= 1 and out.shape == (h, inp.shape[1]):
            rows = [h, 2*h + 1, 3*h + 2]
            divider = int(inp[rows[0], 0])
            if np.all(inp[rows, :] == divider):
                frames = [inp[i*(h+1):i*(h+1)+h, :] for i in range(4)]
                if all(not np.any(f == divider) for f in frames) and not np.any(out == divider):
                    candidates.append((frames, divider, "stack"))
    if (inp.shape[1] - 3) % 4 == 0:
        w = (inp.shape[1] - 3) // 4
        if w >= 1 and out.shape == (inp.shape[0], w):
            cols = [w, 2*w + 1, 3*w + 2]
            divider = int(inp[0, cols[0]])
            if np.all(inp[:, cols] == divider):
                frames = [inp[:, i*(w+1):i*(w+1)+w] for i in range(4)]
                if all(not np.any(f == divider) for f in frames) and not np.any(out == divider):
                    candidates.append((frames, divider, "side"))
    assert len(candidates) == 1, "four panels and three distinct separators required"
    return candidates[0]


def _shift(frame, dr, dc, background):
    result = np.full_like(frame, background)
    h, w = frame.shape
    r0, r1 = max(0, -dr), min(h, h-dr)
    c0, c1 = max(0, -dc), min(w, w-dc)
    if r0 < r1 and c0 < c1:
        result[r0+dr:r1+dr, c0+dc:c1+dc] = frame[r0:r1, c0:c1]
    return result


def _predictions(frames, background):
    predictions = []
    first = frames[0]
    for axis in (0, 1):
        for displacement in (-3, -2, -1, 1, 2, 3):
            dr, dc = (displacement, 0) if axis == 0 else (0, displacement)
            states = [_shift(first, t*dr, t*dc, background) for t in range(5)]
            if (all(np.array_equal(a, b) for a, b in zip(states[:4], frames)) and
                all(np.count_nonzero(s != background) == np.count_nonzero(first != background) for s in states)):
                predictions.append(("translation", states[4]))
    for axis in (0, 1):
        arrays = [_axis_array(f, axis) for f in frames]
        present = [np.flatnonzero(np.any(a != background, axis=0)) for a in arrays]
        if any(len(p) == 0 or p[-1] - p[0] + 1 != len(p) for p in present):
            continue
        starts = [int(p[0]) for p in present]
        ends = [int(p[-1])+1 for p in present]
        lengths = [e-s for s, e in zip(starts, ends)]
        delta = lengths[1] - lengths[0]
        if delta == 0 or abs(delta) > 3 or any(lengths[i+1]-lengths[i] != delta for i in range(3)):
            continue
        if not (len(set(starts)) == 1 or len(set(ends)) == 1):
            continue
        longest = arrays[3] if delta > 0 else arrays[0]
        base_start, base_end = (starts[3], ends[3]) if delta > 0 else (starts[0], ends[0])
        if not all(np.array_equal(a[:, s:e], longest[:, s:e]) and
                   np.all(a[:, :s] == background) and np.all(a[:, e:] == background)
                   for a, s, e in zip(arrays, starts, ends)):
            continue
        if delta > 0:
            next_start = starts[3] - abs(delta) if len(set(ends)) == 1 else starts[3]
            next_end = ends[3] + abs(delta) if len(set(starts)) == 1 else ends[3]
        else:
            next_start = starts[3] + abs(delta) if len(set(ends)) == 1 else starts[3]
            next_end = ends[3] - abs(delta) if len(set(starts)) == 1 else ends[3]
        if next_start < 0 or next_end > longest.shape[1] or next_start >= next_end:
            continue
        segment = longest[:, base_start:base_end]
        for period in range(1, min(3, base_end-base_start)+1):
            if not all(np.array_equal(segment[:, j], segment[:, j % period]) for j in range(segment.shape[1])):
                continue
            predicted = np.full_like(longest, background)
            for x in range(next_start, next_end):
                predicted[:, x] = segment[:, (x-base_start) % period]
            predictions.append(("growth" if delta > 0 else "shrink", _from_axis(predicted, axis)))
    for axis in (0, 1):
        arrays = [_axis_array(f, axis) for f in frames]
        mask = arrays[0] != background
        if not np.any(mask) or not all(np.array_equal(a != background, mask) for a in arrays[1:]):
            continue
        changes = [(a != b) for a, b in zip(arrays[:-1], arrays[1:])]
        if any(not np.any(c) for c in changes):
            continue
        old_values = set(int(x) for a, c in zip(arrays[:-1], changes) for x in a[c])
        new_values = set(int(x) for a, c in zip(arrays[1:], changes) for x in a[c])
        if len(old_values) != 1 or len(new_values) != 1:
            continue
        old, new = next(iter(old_values)), next(iter(new_values))
        if old == new or background in (old, new):
            continue
        size = arrays[0].shape[1]
        coords = np.arange(size)
        for direction in (-1, 1):
            for step in (1, 2, 3):
                for boundary in range(-1, size+1):
                    result = []
                    for time in range(5):
                        b = boundary + direction*time*step
                        zone = coords <= b if direction > 0 else coords >= b
                        scene = np.full_like(arrays[0], background)
                        scene[mask] = old
                        scene[mask & zone[None, :]] = new
                        result.append(scene)
                    if all(np.array_equal(a, b) for a, b in zip(result[:4], arrays)):
                        predictions.append(("frontier", _from_axis(result[4], axis)))
    return predictions


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert not np.array_equal(inp, out)
    assert not _same_as_source(inp, out)
    frames, divider, layout = _split(inp, out)
    background = int(frames[0][0, 0])
    assert background != divider
    assert all(int(f[0, 0]) == background for f in frames) and int(out[0, 0]) == background
    possibilities = _predictions(frames, background)
    assert possibilities, "no observable constant-rate temporal rule"
    answers = {_value_signature(prediction) for _, prediction in possibilities}
    assert len(answers) == 1, "observed sequence has divergent next frames"
    assert _value_signature(out) in answers, "output is not the inferred next frame"
    return {
        "operations": sorted({name for name, _ in possibilities}),
        "layout": layout,
        "background": background,
        "divider": divider,
        "frame_shape": out.shape,
        "colors": tuple(int(x) for x in np.unique(inp)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    layouts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            layouts.add(info["layout"])
            key = (_value_signature(pair["input"]), _value_signature(pair["output"]))
            assert key[0] not in inputs and key not in pairs
            inputs.add(key[0])
            pairs.add(key)
    assert len(layouts) == 1, "chronological packing must agree throughout the puzzle"


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    layout = "stack" if rng.random() < 0.7 else "side"
    if seed % 7 == 0:
        operations = ["frontier"]
    elif seed % 7 == 1:
        operations = ["growth", "translation"]
    else:
        operations = ["growth", "translation", "frontier", "shrink"]
    selected_train = []
    selected_test = []
    for i in range(num_train):
        selected_train.append(operations[i % len(operations)] if i < len(operations) else str(rng.choice(operations)))
    for i in range(num_test):
        if seed % 7 == 1 and i == 0:
            selected_test.append("frontier")
        else:
            selected_test.append(str(rng.choice(operations)))
    rng.shuffle(selected_train)
    rng.shuffle(selected_test)
    seen = set()
    def make_pair(operation):
        for _ in range(500):
            child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            shape, axis = _scene(child_rng, layout)
            colors = [int(x) for x in child_rng.choice(10, size=5, replace=False)]
            try:
                if operation == "translation":
                    frames = _translation(child_rng, shape, axis, colors)
                elif operation in ("growth", "shrink"):
                    frames = _growth(child_rng, shape, axis, colors, operation == "shrink")
                else:
                    frames = _frontier(child_rng, shape, axis, colors)
            except (ValueError, RuntimeError):
                continue
            inp = _pack(frames[:4], colors[1], layout)
            pair = {"input": inp, "output": frames[4]}
            if any(int(frame[0, 0]) != colors[0] for frame in frames):
                continue
            key = _value_signature(inp)
            if key in seen:
                continue
            try:
                _inspect_pair(pair)
            except AssertionError:
                continue
            seen.add(key)
            return pair
        raise RuntimeError("could not construct a unique, determinate pair")
    puzzle = {"train": [make_pair(op) for op in selected_train],
              "test": [make_pair(op) for op in selected_test]}
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts, layouts, operations = set(), set(), set(), set()
    backgrounds, shapes, signatures = set(), set(), set()
    split_operations = {"train": set(), "test": set()}
    mixed_backgrounds = 0
    mixed_directions = 0
    frontier_gaps = 0
    uneven_frontiers = 0
    monochrome_moves = 0
    multicolor_moves = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_backgrounds = set()
        local_directions = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                layouts.add(info["layout"])
                operations.update(info["operations"])
                split_operations[split].update(info["operations"])
                backgrounds.add(info["background"])
                local_backgrounds.add(info["background"])
                shapes.add(info["frame_shape"])
                signatures.add(_value_signature(pair["input"]))
                frames, _, _ = _split(pair["input"], pair["output"])
                changes = [frames[i] != frames[i+1] for i in range(3)]
                center0 = np.argwhere(changes[0]).mean(axis=0)
                center2 = np.argwhere(changes[2]).mean(axis=0)
                displacement = center2 - center0
                axis = int(np.argmax(np.abs(displacement)))
                local_directions.add((axis, 1 if displacement[axis] > 0 else -1))
                if "translation" in info["operations"]:
                    foreground = frames[0][frames[0] != info["background"]]
                    if len(np.unique(foreground)) == 1:
                        monochrome_moves += 1
                    else:
                        multicolor_moves += 1
                if "frontier" in info["operations"]:
                    mask = frames[0] != info["background"]
                    if np.any(~np.any(mask, axis=1-axis)):
                        frontier_gaps += 1
                    if len({int(change.sum()) for change in changes}) > 1:
                        uneven_frontiers += 1
        mixed_backgrounds += len(local_backgrounds) > 1
        mixed_directions += len(local_directions) > 1
    assert len(train_counts) >= 4 and len(test_counts) == 4
    assert layouts == {"stack", "side"}
    assert {"growth", "shrink", "translation", "frontier"} <= operations
    assert {"frontier", "growth", "translation"} <= split_operations["train"]
    assert {"frontier", "growth", "translation"} <= split_operations["test"]
    assert len(backgrounds) >= 8 and len(shapes) >= 25 and len(signatures) >= 150
    assert mixed_backgrounds >= 30 and mixed_directions >= 25
    assert frontier_gaps >= 10 and uneven_frontiers >= 10
    assert monochrome_moves >= 10 and multicolor_moves >= 10
    for source_input, source_output in _SOURCE_PAIRS:
        frames, _, _ = _split(source_input, source_output)
        answers = {_value_signature(prediction) for _, prediction in
                   _predictions(frames, int(frames[0][0, 0]))}
        assert answers == {_value_signature(source_output)}
    for seed, nt, ne in ((200, 2, 1), (201, 10, 1), (202, 10, 4)):
        p = generate_puzzle(seed, nt, ne)
        assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for args in ({"num_train": 1}, {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(99, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3)
    assert all(_value_signature(x["input"]) == _value_signature(y["input"]) and
               _value_signature(x["output"]) == _value_signature(y["output"])
               for split in ("train", "test") for x, y in zip(a[split], b[split]))
