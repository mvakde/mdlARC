"""Rectangles translated by their one-cell-wide extensions."""

import hashlib
import numpy as np


VECTORS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
SOURCE_PAIR_DIGESTS = {
    "f94c68d4d5733b56eaa7e44280f8f4be6eadd1dbbd85153a7edb0922f84aae65",
    "d06651f5d85e079c071d1ce1b2d80c33e9a2ace0207d0353b7611ea0f2674685",
    "777c0e0322113f8bef06501fb7320d379e9ba3b78549bc2d1200cbd5bd8fa368",
    "5932427150b3e02d044d09fe8fbf1f926fa67372e408c995c3e4e89a5efd9505",
}


def _cells(rect):
    r0, r1, c0, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _shift(rect, direction, distance):
    dr, dc = VECTORS[direction]
    r0, r1, c0, c1 = rect
    return (r0 + dr * distance, r1 + dr * distance,
            c0 + dc * distance, c1 + dc * distance)


def _tail(rect, direction, distance, attach):
    r0, r1, c0, c1 = rect
    if direction == "U":
        return {(r0 - k, attach) for k in range(1, distance + 1)}
    if direction == "D":
        return {(r1 + k, attach) for k in range(1, distance + 1)}
    if direction == "L":
        return {(attach, c0 - k) for k in range(1, distance + 1)}
    return {(attach, c1 + k) for k in range(1, distance + 1)}


def _mover_candidates(grid, color):
    observed = set(map(tuple, np.argwhere(grid == color)))
    if not observed:
        return []
    rows = [p[0] for p in observed]
    cols = [p[1] for p in observed]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    h, w, area = r1 - r0 + 1, c1 - c0 + 1, len(observed)
    candidates = []
    # Area fixes the extension length. This leaves only the attachment
    # coordinate to enumerate, including the two ends of each side.
    for direction in VECTORS:
        cross, axis = (h, w) if direction in "LR" else (w, h)
        if cross < 3 or (h * w - area) % (cross - 1):
            continue
        distance = (h * w - area) // (cross - 1)
        if not 1 <= distance <= axis - 3:
            continue
        if direction == "L":
            rect = (r0, r1, c0 + distance, c1)
            attachments = range(r0, r1 + 1)
        elif direction == "R":
            rect = (r0, r1, c0, c1 - distance)
            attachments = range(r0, r1 + 1)
        elif direction == "U":
            rect = (r0 + distance, r1, c0, c1)
            attachments = range(c0, c1 + 1)
        else:
            rect = (r0, r1 - distance, c0, c1)
            attachments = range(c0, c1 + 1)
        for attach in attachments:
            if _cells(rect) | _tail(rect, direction, distance, attach) == observed:
                candidates.append({"color": color, "rect": rect,
                                   "direction": direction, "distance": distance,
                                   "attach": attach,
                                   "dest": _shift(rect, direction, distance)})
    return candidates


def _paint(grid, rect, color):
    r0, r1, c0, c1 = rect
    grid[r0:r1 + 1, c0:c1 + 1] = color


def _render(shape, background, fixed, movers):
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for obj in fixed:
        _paint(inp, obj["rect"], obj["color"])
        _paint(out, obj["rect"], obj["color"])
    for obj in movers:
        _paint(inp, obj["rect"], obj["color"])
        for r, c in _tail(obj["rect"], obj["direction"],
                          obj["distance"], obj["attach"]):
            inp[r, c] = obj["color"]
        _paint(out, obj["dest"], obj["color"])
    return inp, out


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    background = int(inp[0, 0])
    for grid in (inp, out):
        assert np.all(grid[0] == background)
        assert np.all(grid[-1] == background)
        assert np.all(grid[:, 0] == background)
        assert np.all(grid[:, -1] == background)
    colors = {int(v) for v in np.unique(inp)} - {background}
    assert colors and {int(v) for v in np.unique(out)} - {background} == colors
    movers, fixed = [], []
    for color in sorted(colors):
        candidates = _mover_candidates(inp, color)
        assert len(candidates) <= 1
        if candidates:
            movers.append(candidates[0])
        else:
            points = np.argwhere(inp == color)
            r0, c0 = points.min(axis=0)
            r1, c1 = points.max(axis=0)
            rect = (int(r0), int(r1), int(c0), int(c1))
            assert r1 - r0 >= 2 and c1 - c0 >= 2
            fixed.append({"color": color, "rect": rect})
    assert movers and fixed
    fixed_sets = [_cells(obj["rect"]) for obj in fixed]
    source_sets = [_cells(obj["rect"]) |
                   _tail(obj["rect"], obj["direction"], obj["distance"],
                         obj["attach"]) for obj in movers]
    dest_sets = [_cells(obj["dest"]) for obj in movers]
    for collection in (fixed_sets, source_sets, dest_sets):
        for i, current in enumerate(collection):
            assert all(current.isdisjoint(prior) for prior in collection[:i])
    for obj in movers:
        r0, r1, c0, c1 = obj["dest"]
        assert 1 <= r0 <= r1 < inp.shape[0] - 1
        assert 1 <= c0 <= c1 < inp.shape[1] - 1
    expected_in, expected_out = _render(inp.shape, background, fixed, movers)
    # Both reconstructions matter: checking only the output admits holes
    # or stray cells in a supposedly solid fixed body.
    assert np.array_equal(inp, expected_in)
    assert np.array_equal(out, expected_out)
    fixed_union = set().union(*fixed_sets)
    body_union = set().union(*(_cells(o["rect"]) for o in movers))
    tail_union = set().union(*(_tail(o["rect"], o["direction"],
                                      o["distance"], o["attach"]) for o in movers))
    source_cover = fixed_union & set().union(*source_sets)
    dest_cover = fixed_union & set().union(*dest_sets)
    hidden_corners = 0
    for obj in fixed:
        r0, r1, c0, c1 = obj["rect"]
        hidden_corners += sum(inp[r, c] != obj["color"]
                              for r, c in ((r0, c0), (r0, c1),
                                           (r1, c0), (r1, c1)))
    return {"movers": movers, "fixed": fixed, "background": background,
            "source_cover": bool(source_cover),
            "dest_cover": bool(dest_cover),
            "restoration": bool(source_cover - dest_cover),
            "new_occlusion": bool(dest_cover - source_cover),
            "tail_cover": bool(fixed_union & tail_union),
            "body_cover": bool(fixed_union & body_union),
            "hidden_corners": hidden_corners}


def _sample_mover(rng, shape, direction, color, sources, destinations):
    height, width = shape
    for _ in range(110):
        body_h = int(rng.integers(3, min(9, height - 4) + 1))
        body_w = int(rng.integers(3, min(9, width - 4) + 1))
        axis = height if direction in "UD" else width
        body_axis = body_h if direction in "UD" else body_w
        maximum = min(11, axis - body_axis - 2)
        if maximum < 1:
            continue
        distance = int(rng.integers(1, maximum + 1))
        if direction == "U":
            row_limits = (distance + 1, height - body_h)
            col_limits = (1, width - body_w)
        elif direction == "D":
            row_limits = (1, height - body_h - distance)
            col_limits = (1, width - body_w)
        elif direction == "L":
            row_limits = (1, height - body_h)
            col_limits = (distance + 1, width - body_w)
        else:
            row_limits = (1, height - body_h)
            col_limits = (1, width - body_w - distance)
        if row_limits[1] < row_limits[0] or col_limits[1] < col_limits[0]:
            continue
        r0 = int(rng.integers(row_limits[0], row_limits[1] + 1))
        c0 = int(rng.integers(col_limits[0], col_limits[1] + 1))
        rect = (r0, r0 + body_h - 1, c0, c0 + body_w - 1)
        attach = int(rng.integers(c0, c0 + body_w)) if direction in "UD" \
            else int(rng.integers(r0, r0 + body_h))
        tail = _tail(rect, direction, distance, attach)
        source = _cells(rect) | tail
        dest = _shift(rect, direction, distance)
        dest_cells = _cells(dest)
        if any(not source.isdisjoint(other) for other in sources):
            continue
        if any(not dest_cells.isdisjoint(other) for other in destinations):
            continue
        return {"color": color, "rect": rect, "direction": direction,
                "distance": distance, "attach": attach, "tail": tail,
                "source": source, "dest": dest, "dest_cells": dest_cells}
    return None


def _integral(shape, cells):
    mask = np.zeros(shape, dtype=np.int16)
    for r, c in cells:
        mask[r, c] = 1
    return np.pad(mask.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _rect_sum(prefix, r0, r1, c0, c1):
    return int(prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] -
               prefix[r1 + 1, c0] + prefix[r0, c0])


def _sample_fixed(rng, shape, count, movers, mode, colors):
    height, width = shape
    source = set().union(*(m["source"] for m in movers))
    bodies = set().union(*(_cells(m["rect"]) for m in movers))
    tails = set().union(*(m["tail"] for m in movers))
    destination = set().union(*(m["dest_cells"] for m in movers))
    source_p = _integral(shape, source)
    body_p = _integral(shape, bodies)
    tail_p = _integral(shape, tails)
    dest_p = _integral(shape, destination)
    source_only_p = _integral(shape, source - destination)
    dest_only_p = _integral(shape, destination - source)
    fixed = []
    occupied = np.zeros(shape, dtype=np.int16)
    for index in range(count):
        occupied_p = np.pad(occupied.cumsum(axis=0).cumsum(axis=1),
                            ((1, 0), (1, 0)))
        candidates = []
        # The first rectangle establishes the requested overlap class.
        # The rest are sampled independently, subject to the same visible
        # side-extents condition.
        for _ in range(5):
            rect_h = int(rng.integers(3, min(10, height - 4) + 1))
            rect_w = int(rng.integers(3, min(10, width - 4) + 1))
            for r0 in range(1, height - rect_h):
                for c0 in range(1, width - rect_w):
                    rect = (r0, r0 + rect_h - 1, c0, c0 + rect_w - 1)
                    r1, c1 = rect[1], rect[3]
                    if _rect_sum(occupied_p, r0, r1, c0, c1):
                        continue
                    s = _rect_sum(source_p, r0, r1, c0, c1)
                    d = _rect_sum(dest_p, r0, r1, c0, c1)
                    if rect_h * rect_w - s < max(6, rect_h * rect_w // 4):
                        continue
                    if (_rect_sum(source_p, r0, r0, c0, c1) == rect_w or
                            _rect_sum(source_p, r1, r1, c0, c1) == rect_w or
                            _rect_sum(source_p, r0, r1, c0, c0) == rect_h or
                            _rect_sum(source_p, r0, r1, c1, c1) == rect_h):
                        continue
                    if mode == "clean" and (s or d):
                        continue
                    if index == 0:
                        if mode == "tail" and not (
                                _rect_sum(tail_p, r0, r1, c0, c1) and
                                not _rect_sum(body_p, r0, r1, c0, c1) and not d):
                            continue
                        if mode == "body" and not (
                                _rect_sum(body_p, r0, r1, c0, c1) and
                                _rect_sum(source_only_p, r0, r1, c0, c1)):
                            continue
                        if mode == "dest" and not (d and not s):
                            continue
                        if mode == "combined" and not (
                                _rect_sum(source_only_p, r0, r1, c0, c1) and
                                _rect_sum(dest_only_p, r0, r1, c0, c1)):
                            continue
                    candidates.append(rect)
            if candidates:
                break
        if not candidates:
            return None
        rect = candidates[int(rng.integers(len(candidates)))]
        fixed.append({"color": colors[index], "rect": rect})
        r0, r1, c0, c1 = rect
        occupied[r0:r1 + 1, c0:c1 + 1] = 1
    return fixed


def _sample_shape(rng, object_count):
    if rng.random() < 0.09:
        return (20, 20)
    if object_count <= 4 and rng.random() < 0.27:
        return (int(rng.integers(16, 21)), int(rng.integers(16, 22)))
    if rng.random() < 0.57:
        return (int(rng.integers(19, 26)), int(rng.integers(19, 26)))
    return (int(rng.integers(24, 31)), int(rng.integers(22, 31)))


def _generate_pair(rng, palette, mode, regime):
    for _ in range(160):
        n_movers = int(rng.choice([1, 2, 3, 4],
                                  p=[0.19, 0.42, 0.33, 0.06]))
        n_fixed = int(rng.choice([1, 2, 3], p=[0.27, 0.48, 0.25]))
        shape = _sample_shape(rng, n_movers + n_fixed)
        background = int(palette[int(rng.integers(10))])
        colors = [int(v) for v in palette if int(v) != background]
        rng.shuffle(colors)
        weights = [0.33, 0.24, 0.25, 0.18] if regime == "source_train" \
            else [0.18, 0.25, 0.20, 0.37]
        directions = list(rng.choice(["R", "L", "D", "U"],
                                     size=n_movers, p=weights))
        movers, sources, destinations = [], [], []
        for color, direction in zip(colors[:n_movers], directions):
            obj = _sample_mover(rng, shape, direction, color,
                                sources, destinations)
            if obj is None:
                break
            movers.append(obj)
            sources.append(obj["source"])
            destinations.append(obj["dest_cells"])
        if len(movers) != n_movers:
            continue
        fixed = _sample_fixed(rng, shape, n_fixed, movers, mode,
                              colors[n_movers:n_movers + n_fixed])
        if fixed is None:
            continue
        inp, out = _render(shape, background, fixed, movers)
        if np.array_equal(inp, out):
            continue
        try:
            analysis = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if len(analysis["movers"]) != n_movers or len(analysis["fixed"]) != n_fixed:
            continue
        recovered = {o["color"]: o["rect"] for o in analysis["fixed"]}
        if any(recovered.get(o["color"]) != o["rect"] for o in fixed):
            continue
        return {"input": inp, "output": out}
    raise ValueError("could not construct a recoverable scene")


def _coerce_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _coerce_count(num_train, "num_train")
    num_test = _coerce_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    palette = rng.permutation(10)
    train_modes = ["tail" if rng.random() < 0.36 else "body",
                   "combined" if rng.random() < 0.34 else "dest"]
    rng.shuffle(train_modes)
    other_modes = ["clean", "tail", "body", "dest", "combined"]
    puzzle = {"train": [], "test": []}
    signatures = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                mode = train_modes[index]
            else:
                weights = [0.27, 0.18, 0.17, 0.20, 0.18] if split == "train" \
                    else [0.12, 0.20, 0.18, 0.22, 0.28]
                mode = str(rng.choice(other_modes, p=weights))
            for _ in range(25):
                pair = _generate_pair(rng, palette, mode,
                                      "source_train" if split == "train" else "target_test")
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in signatures:
                    signatures.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not construct distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            if inp.shape == (20, 20):
                digest = hashlib.sha256(
                    np.asarray(inp, dtype=np.int8).tobytes() +
                    np.asarray(out, dtype=np.int8).tobytes()
                ).hexdigest()
                assert digest not in SOURCE_PAIR_DIGESTS
            analysis = _analyze_pair(inp, out)
            if split == "train":
                train_analyses.append(analysis)
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.tobytes())
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
    # Training demonstrates both restoration and destination compositing.
    assert any(a["restoration"] for a in train_analyses)
    assert any(a["new_occlusion"] for a in train_analyses)


def self_test() -> None:
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
    first = generate_puzzle(314159, 7, 3)
    second = generate_puzzle(314159, 7, 3)
    for split in ("train", "test"):
        for a, b in zip(first[split], second[split]):
            assert np.array_equal(a["input"], b["input"])
            assert np.array_equal(a["output"], b["output"])
    for counts in ((1, None), (11, None), (None, 0),
                   (None, 5), (2, 2), (2, 4)):
        try:
            generate_puzzle(7, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # An interior hole in a fixed body must fail input reconstruction.
    damaged = generate_puzzle(26, 2, 1)
    pair = damaged["train"][0]
    analysis = _analyze_pair(pair["input"], pair["output"])
    for obj in analysis["fixed"]:
        r0, r1, c0, c1 = obj["rect"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if pair["input"][r, c] == obj["color"]:
                    corrupted = {"train": [
                        {"input": p["input"].copy(), "output": p["output"].copy()}
                        for p in damaged["train"]],
                        "test": damaged["test"]}
                    corrupted["train"][0]["input"][r, c] = analysis["background"]
                    try:
                        validate_generated_puzzle(corrupted)
                    except AssertionError:
                        return
                    raise AssertionError("input hole accepted")
