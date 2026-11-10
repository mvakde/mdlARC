"""Compositional missing-panel ARC puzzle generator."""

import hashlib
import json

import numpy as np


_SOURCE_HASHES = {
    "106b91fd9596881c7b2a490c023398acc64e4d5b547495b45183a8e2d8b0a8cc",
    "88fb6781196cc482c405e479bf7dec7fc51013c8ad6f9ec5569404ac734ab8ea",
    "76cfa81df6e309e2c315b5bf557e9d7c2cb3a7cdf3141587df54b6c7ed433eb4",
    "8e6e702788aea6564f4116a49cc5fa5a493ff1fab3cc3a703b1ad5201ea82cc1",
}


def _counts(seed, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be between {low} and {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4 + seed) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, int(num_test) + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _frame(rng, height, width, background, structure, forbidden):
    grid = np.full((height, width), background, dtype=np.int8)
    style = int(rng.integers(6))
    if style == 0:
        grid[0, :] = grid[-1, :] = structure
        grid[:, 0] = grid[:, -1] = structure
    elif style == 1:
        grid[0, ::2] = grid[-1, ::2] = structure
        grid[::2, 0] = grid[::2, -1] = structure
    elif style == 2:
        grid[:, 0] = grid[:, -1] = structure
    elif style == 3:
        grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = structure
        grid[0, 2::3] = grid[-1, 2::3] = structure
    elif style == 4:
        grid[0, ::3] = grid[-1, ::3] = structure
        grid[1::3, 0] = grid[1::3, -1] = structure
    else:
        grid[0, :] = grid[-1, :] = structure
        grid[1::2, 0] = grid[1::2, -1] = structure
    candidates = [(r, c) for r in range(1, height - 1)
                  for c in range(1, width - 1) if (r, c) not in forbidden]
    rng.shuffle(candidates)
    for r, c in candidates[:int(rng.integers(0, min(5, len(candidates) + 1)))]:
        grid[r, c] = structure
    return grid


def _shape(rng):
    width, height = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    style = int(rng.integers(5))
    if style == 0:
        return {(t, 0) for t in range(height)} | {(height - 1, u) for u in range(width)}
    if style == 1:
        return {(0, u) for u in range(width)} | {(t, width // 2) for t in range(height)}
    if style == 2:
        return {(t, u) for t in range(height) for u in range(width)
                if u in (min(width - 1, t * width // height),
                         min(width - 1, t * width // height + 1))}
    if style == 3:
        return {(t, u) for t in range(height) for u in range(width)
                if t in (0, height - 1) or u == 0}
    return {(t, u) for t in range(height) for u in range(width)
            if t == height // 2 or u == width // 2}


def _local_states(rng, kind, steps, axis):
    sign = _choice(rng, (-1, 1))
    rate = int(rng.integers(1, 3))
    if kind in ("growth", "pattern_growth"):
        span = int(rng.integers(1 if kind == "growth" else 3, 6))
        base = int(rng.integers(1, 3))
        pattern = _choice(rng, ("solid", "rails", "stripe", "spine", "spine", "marker", "marker", "marker", "holes")) if kind == "pattern_growth" else _choice(rng, ("solid", "rails", "stripe"))
        if span < 3 and pattern in ("rails", "spine", "marker", "holes"):
            pattern = "solid"
        cap = pattern == "spine" or bool(rng.integers(2))
        cap_style = "full" if pattern == "spine" else _choice(rng, ("full", "ends", "middle"))
        states = []
        for s in range(steps):
            length = base + rate * s
            cells = {}
            for t in range(length):
                for u in range(span):
                    if pattern == "rails" and 0 < u < span - 1:
                        continue
                    if pattern == "holes" and u == span // 2 and t % 2:
                        continue
                    role = "accent" if (pattern == "stripe" and u % 2) or (pattern == "spine" and u == span // 2) or (pattern == "marker" and u == span // 2 and t % 2 == 0) else "main"
                    cells[(sign * t, u)] = role
            if cap:
                for u in range(span):
                    if cap_style == "ends" and u not in (0, span - 1):
                        continue
                    if cap_style == "middle" and u != span // 2:
                        continue
                    cells[(sign * length, u)] = "cap"
            states.append(cells)
    elif kind == "translation":
        shape = _shape(rng)
        states = [{(sign * (t + rate * s), u): "main" for t, u in shape}
                  for s in range(steps)]
    else:
        span = int(rng.integers(3, 6))
        center = int(rng.integers(1, span - 1))
        thickness = int(rng.integers(1, min(3, span - center + 1)))
        base, extension = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        states = []
        for s in range(steps):
            head = rate * s
            cells = {(sign * head, u): "cap" for u in range(span)}
            for t in range(head + 1, head + base + extension * s + 1):
                for u in range(center, center + thickness):
                    cells[(sign * t, u)] = "main"
            states.append(cells)
    if axis == "horizontal":
        states = [{(u, t): role for (t, u), role in cells.items()} for cells in states]
    return states


def _place(rng, states, height, width, occupied):
    points = set().union(*(set(state) for state in states))
    rmin, rmax = min(r for r, _ in points), max(r for r, _ in points)
    cmin, cmax = min(c for _, c in points), max(c for _, c in points)
    if rmax - rmin > height - 3 or cmax - cmin > width - 3:
        return None
    positions = [(r, c) for r in range(1 - rmin, height - 1 - rmax)
                 for c in range(1 - cmin, width - 1 - cmax)]
    rng.shuffle(positions)
    for dr, dc in positions:
        moved = [{(r + dr, c + dc): role for (r, c), role in state.items()}
                 for state in states]
        if all(not (set(frame) & occupied[s]) for s, frame in enumerate(moved)):
            return moved
    return None


def _interleaved_pair(rng, layout):
    """A moving, lengthening T runs through the gap in two growing strips."""
    n = int(rng.integers(3, 5))
    long_size = int(rng.integers(10 if n == 3 else 12, 15))
    span = int(rng.integers(5, 8))
    short_size = span + 2
    # Four panels of height 8 would exceed the canvas when stacked.
    if layout == "vertical" and n * short_size > 30:
        n = 3
    if layout == "horizontal" and n * short_size > 30:
        n = 3
    height, width = (long_size, short_size) if layout == "horizontal" else (short_size, long_size)
    reverse = bool(rng.integers(2))
    stem_u = 1 + span // 2
    left = (1, 2)
    right = (span - 1, span)
    end = long_size - 2
    logical = []
    for s in range(n):
        cells = {}
        head = 2 + s
        for u in range(1, span + 1):
            cells[(head, u)] = "t"
        for t in range(head + 1, head + 2 + s):
            cells[(t, stem_u)] = "t"
        for t in range(end - s, end + 1):
            for u in left + right:
                cells[(t, u)] = "strips"
        if reverse:
            cells = {(long_size - 1 - t, u): role for (t, u), role in cells.items()}
        if layout == "vertical":
            cells = {(u, t): role for (t, u), role in cells.items()}
        logical.append(cells)
    # Reject geometry that would turn the two behaviors into an occlusion.
    assert all(len(frame) == span + (1 + s) + 4 * (1 + s)
               for s, frame in enumerate(logical))
    palette = [int(x) for x in rng.permutation(10)]
    background, structure, t_color, strips_color = palette[:4]
    points = set().union(*(set(frame) for frame in logical))
    scaffold = _frame(rng, height, width, background, structure, points)
    frames = []
    for cells in logical:
        grid = scaffold.copy()
        for point, role in cells.items():
            grid[point] = t_color if role == "t" else strips_color
        frames.append(grid)
    inp = np.concatenate(frames[:-1] + [scaffold], axis=0 if layout == "vertical" else 1)
    return {"input": inp, "output": frames[-1]}


def _pair(rng, kind, layout, challenge):
    if kind == "combined" and rng.integers(4 if challenge else 7) == 0:
        return _interleaved_pair(rng, layout)
    for _ in range(140):
        n = int(rng.integers(3, 6)) if challenge else int(rng.integers(3, 5))
        if kind == "pattern_growth":
            n = max(4, n)
        max_short = 30 // n
        if layout == "vertical":
            height, width = int(rng.integers(5, max_short + 1)), int(rng.integers(7, 15))
        else:
            height, width = int(rng.integers(7, 15)), int(rng.integers(5, max_short + 1))
        motif_count = int(rng.choice((1, 1, 1, 2))) if n == 5 else (int(rng.choice((1, 1, 2, 2, 3))) if challenge else int(rng.choice((1, 1, 1, 2, 2, 3))))
        kinds = [kind] + [_choice(rng, ("growth", "pattern_growth", "translation", "combined"))
                          for _ in range(motif_count - 1)]
        occupied = [set() for _ in range(n)]
        motifs = []
        for motif_kind in kinds:
            axes = ["vertical", "horizontal"]
            if n == 5:
                axes = ["vertical", "horizontal"] if height >= width else ["horizontal", "vertical"]
            else:
                rng.shuffle(axes)
            candidate = None
            for axis in axes:
                candidate = _place(rng, _local_states(rng, motif_kind, n, axis),
                                   height, width, occupied)
                if candidate is not None:
                    break
            if candidate is None:
                break
            for s, frame in enumerate(candidate):
                occupied[s].update(frame)
            motifs.append(candidate)
        if len(motifs) != motif_count:
            continue
        all_points = set().union(*occupied)
        palette = [int(x) for x in rng.permutation(10)]
        background, structure = palette[:2]
        scaffold = _frame(rng, height, width, background, structure, all_points)
        available = palette[2:]
        role_sets = [set().union(*(set(frame.values()) for frame in frames))
                     for frames in motifs]
        next_color = len(motifs)
        mains = available[:len(motifs)]
        accents = []
        for main, roles in zip(mains, role_sets):
            if "accent" in roles and next_color < len(available):
                accents.append(available[next_color])
                next_color += 1
            else:
                accents.append(main)
        alternates = []
        for main, motif_kind in zip(mains, kinds):
            cycle = n >= 4 and next_color < len(available) and bool(rng.integers(2 if motif_kind == "pattern_growth" else 4) == 0)
            alternates.append(available[next_color] if cycle else main)
            next_color += int(cycle)
        maps = []
        for main, alternate, accent, roles in zip(mains, alternates, accents, role_sets):
            cap_mode = int(rng.integers(3)) if "cap" in roles else 0
            if cap_mode == 2 and next_color < len(available):
                cap = available[next_color]
                next_color += 1
            else:
                cap = accent if cap_mode == 1 else main
            maps.append((main, alternate, accent, cap))
        frames = []
        for s in range(n):
            grid = scaffold.copy()
            for motif, colors in zip(motifs, maps):
                main, alternate, accent, cap = colors
                for point, role in motif[s].items():
                    grid[point] = (main if s % 2 == 0 else alternate) if role == "main" else accent if role == "accent" else cap
            frames.append(grid)
        if any(np.array_equal(frame, scaffold) for frame in frames):
            continue
        inp = np.concatenate(frames[:-1] + [scaffold], axis=0 if layout == "vertical" else 1)
        return {"input": inp, "output": frames[-1]}
    raise ValueError("bounded motif construction exhausted")


def _key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    preferred = _choice(rng, ("vertical", "horizontal"))
    opposite = "horizontal" if preferred == "vertical" else "vertical"
    modes = ("growth", "combined", "pattern_growth", "translation")
    offset = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", train_count), ("test", test_count)):
        for i in range(count):
            kind = modes[(i + offset + (1 if split == "test" else 0)) % 4]
            layout = (opposite if i % 3 == 1 else preferred) if split == "train" else (opposite if i % 2 == 0 else preferred)
            for _ in range(20):
                pair = _pair(rng, kind, layout, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _panels(pair):
    inp, out = pair["input"], pair["output"]
    h, w = out.shape
    if inp.shape[1] == w and inp.shape[0] % h == 0 and inp.shape[0] > h:
        n = inp.shape[0] // h
        panels, layout = [inp[i*h:(i+1)*h, :] for i in range(n)], "vertical"
    elif inp.shape[0] == h and inp.shape[1] % w == 0 and inp.shape[1] > w:
        n = inp.shape[1] // w
        panels, layout = [inp[:, i*w:(i+1)*w] for i in range(n)], "horizontal"
    else:
        raise AssertionError("input must be a one-axis panel sequence")
    assert 3 <= n <= 5
    return panels, layout


def _source_digest(pair):
    value = {name: pair[name].tolist() for name in ("input", "output")}
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _visible_interleaving(pair):
    panels, _ = _panels(pair)
    base = panels[-1]
    frames = panels[:-1] + [pair["output"]]
    out = frames[-1]
    colors = set(int(v) for v in np.unique(out[out != base]))
    for t_color in colors:
        points = np.argwhere((out == t_color) & (out != base))
        for axis in (0, 1):
            t, u = points[:, axis], points[:, 1 - axis]
            for head in (int(t.min()), int(t.max())):
                cap_u = sorted(int(x) for x in u[t == head])
                stem_u = sorted(set(int(x) for x in u[t != head]))
                if (len(cap_u) < 5 or cap_u != list(range(cap_u[0], cap_u[-1] + 1))
                        or len(stem_u) != 1 or not cap_u[0] < stem_u[0] < cap_u[-1]):
                    continue
                stem_t = set(int(x) for x in t[t != head])
                if len(stem_t) < 2:
                    continue
                for strip_color in colors - {t_color}:
                    q = np.argwhere((out == strip_color) & (out != base))
                    qt, qu = q[:, axis], q[:, 1 - axis]
                    if (np.any(qu < stem_u[0]) and np.any(qu > stem_u[0])
                            and bool(stem_t & set(int(x) for x in qt))):
                        return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            key = (_key(pair["input"]), _key(pair["output"]))
            assert key[0] not in seen_inputs and key not in seen_pairs
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            assert _source_digest(pair) not in _SOURCE_HASHES
            panels, _ = _panels(pair)
            base = panels[-1]
            colors, counts = np.unique(base[1:-1, 1:-1], return_counts=True)
            background = int(colors[np.argmax(counts)])
            protected = base != background
            assert np.any(protected)
            assert all(np.array_equal(frame[protected], base[protected])
                       for frame in panels[:-1] + [pair["output"]])
            supports = [frame != base for frame in panels[:-1] + [pair["output"]]]
            assert all(np.any(support) for support in supports)
            assert all(not np.array_equal(panels[i], panels[i + 1])
                       for i in range(len(panels) - 2))
            assert not np.array_equal(pair["output"], base)
            observed = set(int(v) for frame in panels[:-1] for v in np.unique(frame))
            assert set(int(v) for v in np.unique(pair["output"])) <= observed


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_key(pair["input"]) for split in ("train", "test")
                      for pair in p[split]) for p in puzzles}) >= 40
    layouts, sizes, colors = set(), set(), set()
    measured = {name: 0 for name in ("five_panels", "translation_like",
                                   "motion_growth_like", "color_recurrence",
                                   "stable_and_changing", "zero_landmarks",
                                   "several_landmarks", "five_colors",
                                   "interleaving")}
    within_palette_changes = 0
    within_landmark_changes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        palettes = set()
        marks_seen = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                panels, layout = _panels(pair)
                layouts.add((split, layout, len(panels)))
                sizes.add(pair["output"].shape)
                palette = tuple(int(v) for v in np.unique(pair["input"]))
                colors.add(palette)
                palettes.add(palette)
                measured["five_panels"] += len(panels) == 5
                measured["five_colors"] += len(palette) >= 5
                measured["interleaving"] += _visible_interleaving(pair)
                base = panels[-1]
                values, frequencies = np.unique(base[1:-1, 1:-1], return_counts=True)
                background = int(values[np.argmax(frequencies)])
                marks = int(np.count_nonzero(base[1:-1, 1:-1] != background))
                marks_seen.add(marks)
                measured["zero_landmarks"] += marks == 0
                measured["several_landmarks"] += marks >= 2
                frames = panels[:-1] + [pair["output"]]
                masks = [frame != base for frame in frames]
                areas = [int(mask.sum()) for mask in masks]
                boxes = []
                dynamic_palettes = []
                for frame, mask in zip(frames, masks):
                    rows, cols = np.where(mask)
                    boxes.append((int(rows.min()), int(rows.max()),
                                  int(cols.min()), int(cols.max())))
                    dynamic_palettes.append(set(int(v) for v in np.unique(frame[mask])))
                measured["translation_like"] += len(set(areas)) == 1 and boxes[0] != boxes[-1]
                measured["motion_growth_like"] += areas[-1] > areas[0] and (
                    (boxes[0][0] != boxes[-1][0] and boxes[0][1] != boxes[-1][1]) or
                    (boxes[0][2] != boxes[-1][2] and boxes[0][3] != boxes[-1][3]))
                measured["color_recurrence"] += (len(frames) >= 4 and
                    dynamic_palettes[0] == dynamic_palettes[2] and
                    dynamic_palettes[0] != dynamic_palettes[1])
                stable = set.intersection(*dynamic_palettes)
                measured["stable_and_changing"] += (bool(stable) and
                    len(set.union(*dynamic_palettes)) > len(stable) and
                    areas[-1] > areas[0])
        within_palette_changes += len(palettes) > 1
        within_landmark_changes += len(marks_seen) > 1
    assert {layout for _, layout, _ in layouts} == {"vertical", "horizontal"}
    assert len(sizes) >= 15 and len(colors) >= 30
    assert within_palette_changes >= 35
    assert within_landmark_changes >= 30
    assert measured["five_panels"] >= 5
    assert measured["five_colors"] >= 30
    assert measured["translation_like"] >= 20
    assert measured["motion_growth_like"] >= 20
    assert measured["color_recurrence"] >= 20
    assert measured["stable_and_changing"] >= 10
    assert measured["zero_landmarks"] >= 10
    assert measured["several_landmarks"] >= 20
    assert measured["interleaving"] >= 2
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None), (7, 3)):
        a, b = generate_puzzle(12345, *args), generate_puzzle(12345, *args)
        assert all(np.array_equal(x[role], y[role])
                   for split in ("train", "test") for x, y in zip(a[split], b[split])
                   for role in ("input", "output"))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1, *args)
        except ValueError:
            continue
        raise AssertionError("invalid pair counts accepted")
