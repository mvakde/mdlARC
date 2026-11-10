"""Wavefront-triggered jumps: input-only geometry and bounded random construction."""

from __future__ import annotations

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    offsets = N8 if diagonal else N4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                other = r + dr, c + dc
                if other in remaining:
                    remaining.remove(other)
                    part.add(other)
                    stack.append(other)
        result.append(part)
    return result


def _cells(grid, color):
    return set(map(tuple, np.argwhere(grid == color).tolist()))


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _width(cells):
    _, _, left, right = _bbox(cells)
    return right - left + 1


def _shift(cells, distance):
    return {(r, c + distance) for r, c in cells}


def _near(a, b):
    return any((r + dr, c + dc) in b for r, c in a
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _diamond(center, radius, shape):
    r, c = center
    h, w = shape
    return {(rr, cc) for rr in range(max(0, r - radius), min(h, r + radius + 1))
            for cc in range(max(0, c - radius), min(w, c + radius + 1))
            if abs(rr - r) + abs(cc - c) == radius}


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer rectangles of size 1..30 with colors 0..9")
    return a.astype(np.int8, copy=False)


def _parse(input_grid, diagonal=True):
    a = _array(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    background = int(colors[np.argmax(counts)])
    if counts.max() * 2 <= a.size or background in (1, 2):
        raise ValueError("background must occupy a strict majority and cannot be a pulse color")
    waves = []
    for color, delta in ((1, 1), (2, -1)):
        for cells in _components(_cells(a, color)):
            top, bottom, left, right = _bbox(cells)
            if bottom - top != right - left or (right - left) % 2:
                raise ValueError("input pulse must be a complete diamond")
            radius = (right - left) // 2
            center = (top + radius, left + radius)
            if not 2 <= radius <= 5 or cells != _diamond(center, radius, a.shape):
                raise ValueError("input pulse must be a full radius-2..5 diamond")
            waves.append({"color": color, "center": center, "radius": radius,
                          "source": cells, "next": _diamond(center, radius + delta, a.shape)})
    if not 1 <= len(waves) <= 3:
        raise ValueError("expected one to three pulses")
    payloads = []
    for color in sorted(set(map(int, colors)) - {background, 1, 2}):
        for cells in _components(_cells(a, color), diagonal):
            hits = [i for i, wave in enumerate(waves) if cells & wave["next"]]
            payloads.append({"color": color, "source": cells, "width": _width(cells),
                             "hits": hits})
    if not payloads:
        raise ValueError("expected cargo")
    return a, background, waves, payloads


def _paint(shape, background, layers):
    out = np.full(shape, background, dtype=np.int8)
    occupied = set()
    for color, cells in layers:
        if cells & occupied or any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in cells):
            raise ValueError("output collision or out-of-bounds cargo")
        occupied.update(cells)
        for r, c in cells:
            out[r, c] = color
    return out


def _prediction(input_grid, alternative=None):
    a, background, waves, payloads = _parse(input_grid, diagonal=alternative != "four")
    fronts = set().union(*(wave["next"] for wave in waves))
    if alternative == "old":
        fronts = set().union(*(wave["source"] for wave in waves))
    elif alternative == "inside":
        fronts = {(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])
                  if any(abs(r - wave["center"][0]) + abs(c - wave["center"][1])
                         <= wave["radius"] + (1 if wave["color"] == 1 else -1) for wave in waves)}
    elif alternative == "reverse":
        fronts = set().union(*(_diamond(w["center"], w["radius"] + (-1 if w["color"] == 1 else 1), a.shape)
                              for w in waves))
    layers = [(w["color"], w["next"]) for w in waves]
    for cargo in payloads:
        source = cargo["source"]
        hit = bool(source & fronts)
        if alternative == "all":
            hit = True
        elif alternative == "none":
            hit = False
        distance = 1 if alternative == "unit" else cargo["width"]
        if alternative == "area":
            distance = len(source)
        elif alternative == "height":
            top, bottom, _, _ = _bbox(source)
            distance = bottom - top + 1
        layers.append((cargo["color"], _shift(source, distance) if hit else source))
    return _paint(a.shape, background, layers)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute synchronous contact and motion from this input alone."""
    return _prediction(input_grid)


def _normalize(cells):
    top, _, left, _ = _bbox(cells)
    return {(r - top, c - left) for r, c in cells}


def _make_shape(rng, width, kind):
    if kind == "control":
        return {(0, 0)}
    if kind == "column":
        return {(r, 0) for r in range(int(rng.integers(3, 5)))}
    if width == 1:
        return {(r, 0) for r in range(int(rng.integers(1, 5)))}
    if kind == "diagonal":
        # A zigzag can be much wider than it is tall; four-connectivity splits it.
        rows = [0]
        for _ in range(width - 1):
            rows.append(rows[-1] + (1 if rows[-1] == 0 else int(rng.choice((-1, 1)))))
        return _normalize({(r, c) for c, r in enumerate(rows)})
    height = int(rng.integers(2, min(7, width + 3)))
    if kind == "hollow" and width >= 3:
        height = max(3, height)
        return {(r, c) for r in range(height) for c in range(width)
                if r in (0, height - 1) or c in (0, width - 1)}
    if kind == "solid":
        return {(r, c) for r in range(height) for c in range(width)}
    row = int(rng.integers(height))
    cells = {(row, 0)}
    for col in range(1, width):
        row = int(rng.choice([v for v in (row - 1, row, row + 1) if 0 <= v < height]))
        cells.add((row, col))
    for _ in range(int(rng.integers(1, width + height))):
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in N8
                           if 0 <= r + dr < height and 0 <= c + dc < width} - cells)
        if frontier:
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize(cells)


def _place_waves(rng, shape, specifications, clipped_index=None):
    waves = []
    for index, (color, radius) in enumerate(specifications):
        positions = [(r, c) for r in range(radius, shape[0] - radius)
                     for c in range(radius, shape[1] - radius)
                     if index != clipped_index or min(r, c, shape[0] - 1 - r, shape[1] - 1 - c) == radius]
        for j in rng.permutation(len(positions)):
            center = positions[int(j)]
            source = _diamond(center, radius, shape)
            target = _diamond(center, radius + (1 if color == 1 else -1), shape)
            if any(source & w["source"] or target & w["next"] or
                   (color == w["color"] and _near(source, w["source"])) for w in waves):
                continue
            waves.append({"color": color, "center": center, "radius": radius,
                          "source": source, "next": target})
            break
        else:
            raise RuntimeError("wave placement failed")
    return waves


def _place_cargo(rng, shape, waves, cargo, local_shape, color, wanted, wave_index=None, interior=False):
    old_front = set().union(*(w["source"] for w in waves))
    new_front = set().union(*(w["next"] for w in waves))
    h = max(r for r, _ in local_shape) + 1
    width = _width(local_shape)
    candidates = [(r, c) for r in range(shape[0] - h + 1)
                  for c in range(shape[1] - width + 1)]
    for j in rng.permutation(len(candidates)):
        top, left = candidates[int(j)]
        source = {(top + r, left + c) for r, c in local_shape}
        if source & old_front or any(_near(source, old["source"]) for old in cargo):
            continue
        if interior and not any(all(abs(r - w["center"][0]) + abs(c - w["center"][1])
                                    < w["radius"] + (1 if w["color"] == 1 else -1)
                                    for r, c in source) for w in waves):
            continue
        hits = [i for i, wave in enumerate(waves) if source & wave["next"]]
        # Generated scenes use a unique contacting pulse per cargo. The rule
        # still has a well-defined Boolean trigger for multiple contacts.
        if (bool(hits) != wanted or len(hits) > 1
                or (wanted and wave_index is not None and hits != [wave_index])):
            continue
        target = _shift(source, width) if hits else source
        if (any(c >= shape[1] for _, c in target) or target & new_front
                or any(target & old["target"] for old in cargo)):
            continue
        return {"color": color, "source": source, "target": target, "hits": hits}
    return None


def _build_pair(rng, palette, split, index, shift):
    background, cargo_colors = palette
    for _ in range(100):
        if split == "train" and index == 0:
            specifications = [(1, int(rng.integers(2, 5)))]
            requests = [(True, 0, int(rng.integers(3, 5)), "diagonal", cargo_colors[0]),
                        (True, 0, 1, "column", cargo_colors[0]),
                        (False, None, 1, "control", cargo_colors[0])]
        elif split == "train" and index == 1:
            specifications = [(2, int(rng.integers(3, 5))), (1, int(rng.integers(2, 4)))]
            requests = [(True, 0, int(rng.integers(2, 4)), "branch", cargo_colors[0]),
                        (True, 1, int(rng.integers(2, 4)), "branch", cargo_colors[-1]),
                        (False, None, int(rng.integers(1, 4)), "hollow", cargo_colors[min(1, len(cargo_colors) - 1)])]
        else:
            count = (int(rng.integers(1, 3 if shift == "more_pulses" else 4)) if split == "train"
                     else (3 if shift == "more_pulses" else int(rng.integers(1, 4))))
            specifications = [(int(rng.choice((1, 2))), int(rng.integers(2, 6))) for _ in range(count)]
            requests = []
            for i in range(count):
                width = int(rng.integers(2, 5))
                kind = str(rng.choice(("diagonal", "branch", "hollow", "solid")))
                if split == "test" and shift == "wider_cargo" and i == 0:
                    width = int(rng.integers(5, 7))
                    # Wide cargo can contact the contracting interior by
                    # diagonal bridges, while dense shapes often cannot.
                    kind = str(rng.choice(("diagonal", "branch")))
                requests.append((True, i, width, kind, int(rng.choice(cargo_colors))))
            for _ in range(int(rng.integers(1, 4))):
                requests.append((False, None, int(rng.integers(1, 6)),
                                 str(rng.choice(("branch", "hollow", "solid", "diagonal"))),
                                 int(rng.choice(cargo_colors))))
        # More pulses need space, but positions are never arranged into panels.
        low = 15 if len(specifications) == 1 else 21
        shape = (int(rng.integers(low, 31)), int(rng.integers(max(low, 20), 31)))
        try:
            waves = _place_waves(rng, shape, specifications,
                                 clipped_index=1 if split == "train" and index == 1 else None)
        except RuntimeError:
            continue
        cargo = []
        for wanted, wave_index, width, kind, color in requests:
            placed = None
            for _ in range(8):
                local = _make_shape(rng, width, kind)
                placed = _place_cargo(rng, shape, waves, cargo, local, color, wanted, wave_index,
                                      interior=kind == "control")
                if placed is not None:
                    break
            if placed is None:
                break
            cargo.append(placed)
        if len(cargo) != len(requests):
            continue
        inp = _paint(shape, background, [(w["color"], w["source"]) for w in waves]
                     + [(c["color"], c["source"]) for c in cargo])
        out = _paint(shape, background, [(w["color"], w["next"]) for w in waves]
                     + [(c["color"], c["target"]) for c in cargo])
        if min(np.count_nonzero(inp == background), np.count_nonzero(out == background)) * 2 <= inp.size:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a collision-free scene")


def _check_counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    available = [0, 3, 4, 5, 6, 7, 8, 9]
    background = int(rng.choice(available))
    cargo_colors = tuple(map(int, rng.choice([c for c in available if c != background],
                                            size=int(rng.integers(1, 4)), replace=False)))
    shift = str(rng.choice(("wider_cargo", "more_pulses")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(int(count)):
            for _ in range(12):
                pair = _build_pair(rng, (background, cargo_colors), split, index, shift)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate-input construction exhausted")
    return puzzle


def _alternative_differs(pair, name):
    try:
        return not np.array_equal(_prediction(pair["input"], name), pair["output"])
    except ValueError:
        # A shortcut that causes a collision or exits the canvas also fails.
        return True


def _analyze_pair(pair):
    inp, out = _array(pair["input"]), _array(pair["output"])
    assert inp.shape == out.shape
    a, background, waves, cargo = _parse(inp)
    assert np.array_equal(solve(inp), out)
    assert np.count_nonzero(out == background) * 2 > out.size
    assert 2 <= len(cargo) <= 6
    assert any(c["hits"] for c in cargo) and any(not c["hits"] for c in cargo)
    assert all(len(c["hits"]) <= 1 for c in cargo)
    for i, c in enumerate(cargo):
        assert 1 <= c["width"] <= 6
        assert all(not _near(c["source"], other["source"]) for other in cargo[i + 1:])
    # Validate each semantic output directly by color and the input components.
    # Wave colors and cargo colors have disjoint roles.
    for color in (1, 2):
        expected = set().union(*(w["next"] for w in waves if w["color"] == color))
        assert _cells(out, color) == expected
    for color in {c["color"] for c in cargo}:
        expected = set().union(*(_shift(c["source"], c["width"]) if c["hits"] else c["source"]
                                for c in cargo if c["color"] == color))
        assert _cells(out, color) == expected
        assert len(_cells(inp, color)) == len(_cells(out, color))
    return {"background": background, "waves": waves, "cargo": cargo, "shape": a.shape,
            "clipped": any(len(w["next"]) < 4 * (w["radius"] + 1) for w in waves if w["color"] == 1)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, info = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        key = _key(pair["input"])
        assert key not in seen
        seen.add(key)
        info.append(_analyze_pair(pair))
    assert len({a["background"] for a in info}) == 1
    training = info[:len(puzzle["train"])]
    demonstrated_colors = {c["color"] for a in training for c in a["cargo"]}
    assert all(c["color"] in demonstrated_colors for a in info for c in a["cargo"])
    assert {w["color"] for a in training for w in a["waves"]} == {1, 2}
    assert any(a["clipped"] for a in training)
    assert all(any(c["hits"] == [i] for c in a["cargo"]) for a in training for i in range(len(a["waves"])))
    first = training[0]
    assert len({c["width"] for c in first["cargo"] if c["hits"]}) >= 2
    assert any(c["hits"] and len(_components(c["source"], False)) > 1 for c in first["cargo"])
    assert len({c["color"] for c in first["cargo"]}) == 1
    for alternative in ("four", "unit", "area", "height", "all", "none", "old", "inside", "reverse"):
        assert any(_alternative_differs(pair, alternative) for pair in puzzle["train"]), alternative
    assert all(_alternative_differs(pair, "none") and _alternative_differs(pair, "unit") for pair in puzzle["test"])
    max_training_width = max(c["width"] for a in training for c in a["cargo"] if c["hits"])
    max_training_waves = max(len(a["waves"]) for a in training)
    for a in info[len(training):]:
        assert (max(c["width"] for c in a["cargo"] if c["hits"]) > max_training_width
                or len(a["waves"]) > max_training_waves)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded structural/semantic audit and diversity checks on 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    shapes, backgrounds, radii, wave_counts, widths, geometries = set(), set(), set(), set(), set(), set()
    clipped = diagonal = hollow = multi_color = 0
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            a = _analyze_pair(pair)
            shapes.add(a["shape"])
            backgrounds.add(a["background"])
            wave_counts.add(len(a["waves"]))
            radii.update(w["radius"] for w in a["waves"])
            clipped += a["clipped"]
            multi_color += len({c["color"] for c in a["cargo"]}) > 1
            for c in a["cargo"]:
                widths.add(c["width"])
                geometries.add(tuple(sorted(_normalize(c["source"]))))
                diagonal += len(_components(c["source"], False)) > 1
                t, b, l, r = _bbox(c["source"])
                empty = {(rr, cc) for rr in range(t, b + 1) for cc in range(l, r + 1)} - c["source"]
                hollow += any(all(t < rr < b and l < cc < r for rr, cc in part)
                              for part in _components(empty, False))
    assert len(shapes) >= 35 and len(backgrounds) >= 6
    assert radii == {2, 3, 4, 5} and wave_counts == {1, 2, 3}
    assert widths == {1, 2, 3, 4, 5, 6} and len(geometries) >= 50
    assert clipped >= 5 and diagonal >= 42 and hollow >= 5 and multi_color >= 20
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(123, train, test)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "canvas_shapes": len(shapes), "cargo_geometries": len(geometries),
            "backgrounds": len(backgrounds), "clipped_pairs": clipped,
            "diagonal_cargo": diagonal, "hollow_cargo": hollow}
