"""Synthetic instances of the same-color, center-aligned rectangle bridge task."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = {
    "7c56b6df9fe72704a57e2fb235c3ae86ecd56ea7ed14dc0438b50792c9afb3ee",
    "a075988b35e14c952c7c5bede36a5244d2804a2da0a19882be607ec79af39c74",
    "a3e63b6e6d1771547f61cde48368594c3443fbbf0ef9f21997373d4a8d8f0037",
    "486e79bc8e5ddf6be27087618fadf1318ace1666caa075051fa052a0c8f23aa9",
}


def _validate_optional_count(value, name, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(seed, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif num_train is None:
        test = int(num_test)
        train = test + 1 + seed % (10 - test)
    elif num_test is None:
        train = int(num_train)
        test = 1 + seed % min(4, train - 1)
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_rectangles(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < height and 0 <= xx < width and not seen[yy, xx] and int(grid[yy, xx]) != background:
                        assert int(grid[yy, xx]) == color
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            r0 = min(y for y, _ in cells)
            r1 = max(y for y, _ in cells)
            c0 = min(x for _, x in cells)
            c1 = max(x for _, x in cells)
            assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
            assert np.all(grid[r0:r1 + 1, c0:c1 + 1] == color)
            assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
            result.append((r0, r1, c0, c1, color))
    result.sort()
    return result


def _link_specs(grid, rectangles, background):
    rows, cols = {}, {}
    for i, (r0, r1, c0, c1, _) in enumerate(rectangles):
        rows.setdefault(r0 + r1, []).append(i)
        cols.setdefault(c0 + c1, []).append(i)
    specs = []
    for members in rows.values():
        ordered = sorted(members, key=lambda i: rectangles[i][2])
        for a, b in zip(ordered, ordered[1:]):
            left, right = rectangles[a], rectangles[b]
            if left[4] != right[4]:
                continue
            r0, r1 = max(left[0], right[0]) + 1, min(left[1], right[1]) - 1
            c0, c1 = left[3] + 1, right[2] - 1
            if r0 <= r1 and c0 <= c1 and np.all(grid[r0:r1 + 1, c0:c1 + 1] == background):
                specs.append(("H", a, b, r0, r1, c0, c1))
    for members in cols.values():
        ordered = sorted(members, key=lambda i: rectangles[i][0])
        for a, b in zip(ordered, ordered[1:]):
            top, bottom = rectangles[a], rectangles[b]
            if top[4] != bottom[4]:
                continue
            r0, r1 = top[1] + 1, bottom[0] - 1
            c0, c1 = max(top[2], bottom[2]) + 1, min(top[3], bottom[3]) - 1
            if r0 <= r1 and c0 <= c1 and np.all(grid[r0:r1 + 1, c0:c1 + 1] == background):
                specs.append(("V", a, b, r0, r1, c0, c1))
    return specs


def _transform(grid, background, connector):
    rectangles = _extract_rectangles(grid, background)
    specs = _link_specs(grid, rectangles, background)
    output = grid.copy()
    for _, _, _, r0, r1, c0, c1 in specs:
        output[r0:r1 + 1, c0:c1 + 1] = connector
    return output, rectangles, specs


def _tracks(rng, length, count, minimum_gap):
    slack = length - (5 + minimum_gap * (count - 1))
    assert slack >= 0
    parts = rng.multinomial(slack, np.full(count + 1, 1 / (count + 1)))
    integer_centers = [2 + int(parts[0])]
    for i in range(1, count):
        integer_centers.append(integer_centers[-1] + minimum_gap + int(parts[i]))
    return [2 * x + int(rng.integers(0, 2)) for x in integer_centers]


def _side_choices(centers, index):
    parity = centers[index] % 2
    distances = [abs(centers[index] - other) / 2 for j, other in enumerate(centers) if j != index]
    cap = min(8, int(min(distances)) - 1) if distances else 8
    return [s for s in range(3, cap + 1) if s % 2 != parity]


def _bounds(center2, side):
    first = (center2 - side + 1) // 2
    return first, first + side - 1


def _draw_layout(rng, height, width, row_centers, col_centers, occupied, palette, background, mode, orientation, near_miss):
    grid = np.full((height, width), background, dtype=np.int8)
    row_choices = [_side_choices(row_centers, i) for i in range(len(row_centers))]
    col_choices = [_side_choices(col_centers, i) for i in range(len(col_centers))]
    if any(not choices for choices in row_choices + col_choices):
        return None
    row_sizes = {slot: int(rng.choice(row_choices[slot[0]])) for slot in occupied}
    col_sizes = {slot: int(rng.choice(col_choices[slot[1]])) for slot in occupied}
    colors = {slot: int(rng.choice(palette)) for slot in occupied}
    primary = [(0, 0), (0, 1), (0, 2)] if orientation == "H" else [(0, 0), (1, 0), (2, 0)]
    if mode != "free":
        colors[primary[0]] = colors[primary[1]] = palette[0]
        colors[primary[2]] = palette[1]
        perpendicular = row_sizes if orientation == "H" else col_sizes
        choices = row_choices if orientation == "H" else col_choices
        track = 0
        desired = min(choices[track]) if mode == "intro" else max(choices[track])
        if mode == "evidence" and desired < 5:
            return None
        perpendicular[primary[0]] = perpendicular[primary[1]] = desired
    else:
        colors[primary[0]] = colors[primary[1]] = palette[0]
        if primary[2] in colors and len(palette) > 1 and rng.random() < 0.55:
            colors[primary[2]] = palette[1]
    near = (1, 0) if orientation == "H" else (0, 1)
    if near_miss:
        colors[near] = palette[0]
    for slot in sorted(occupied):
        ri, ci = slot
        row2, col2 = row_centers[ri], col_centers[ci]
        if near_miss and slot == near:
            if orientation == "H":
                col2 += 2
            else:
                row2 += 2
        r0, r1 = _bounds(row2, row_sizes[slot])
        c0, c1 = _bounds(col2, col_sizes[slot])
        if not (0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width):
            return None
        # A one-cell border keeps equal-colored rectangles separately segmentable.
        rr0, rr1 = max(0, r0 - 1), min(height, r1 + 2)
        cc0, cc1 = max(0, c0 - 1), min(width, c1 + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != background):
            return None
        grid[r0:r1 + 1, c0:c1 + 1] = colors[slot]
    return grid


def _make_pair(rng, connector, background, mode, orientation, regime):
    for _ in range(80):
        if mode == "free":
            nrows = int(rng.integers(2, 5))
            ncols = int(rng.integers(2, 5))
            # Two occupied positions are enough for a simple one-link case.
            orientation = "H" if ncols >= 3 and (nrows < 3 or rng.random() < 0.5) else "V"
            if orientation == "H" and ncols < 3:
                ncols = 3
            if orientation == "V" and nrows < 3:
                nrows = 3
            gap = int(rng.integers(5, 8))
        else:
            gap = 5 if mode == "intro" else 7
            nrows = int(rng.integers(2, 4)) if orientation == "H" else int(rng.integers(3, 5))
            ncols = int(rng.integers(3, 5)) if orientation == "H" else int(rng.integers(2, 4))
        min_h = 5 + gap * (nrows - 1)
        min_w = 5 + gap * (ncols - 1)
        if min_h > 30 or min_w > 30:
            continue
        height = int(rng.integers(min_h, 31))
        width = int(rng.integers(min_w, 31))
        row_centers = _tracks(rng, height, nrows, gap)
        col_centers = _tracks(rng, width, ncols, gap)
        primary = [(0, 0), (0, 1), (0, 2)] if orientation == "H" else [(0, 0), (1, 0), (2, 0)]
        occupied = set(primary[:2] if mode == "free" else primary)
        near_miss = mode == "evidence" or (mode == "free" and rng.random() < (0.27 if regime == "source_train" else 0.48))
        if near_miss:
            occupied.add((1, 0) if orientation == "H" else (0, 1))
        probability = float(rng.uniform(0.20, 0.77)) if regime == "source_train" else float(rng.uniform(0.36, 0.91))
        for slot in ((r, c) for r in range(nrows) for c in range(ncols)):
            if slot not in occupied and rng.random() < probability:
                occupied.add(slot)
        if len(occupied) > 14:
            continue
        number_colors = int(rng.integers(2, 5)) if mode != "free" else int(rng.integers(1, 5 if regime == "source_train" else 7))
        available = [c for c in range(10) if c not in (background, connector)]
        palette = [int(x) for x in rng.choice(available, size=number_colors, replace=False)]
        grid = _draw_layout(rng, height, width, row_centers, col_centers, occupied, palette, background, mode, orientation, near_miss)
        if grid is None or _background(grid) != background:
            continue
        output, rectangles, specs = _transform(grid, background, connector)
        if not specs:
            continue
        cover = np.zeros(grid.shape, dtype=np.int8)
        for _, _, _, r0, r1, c0, c1 in specs:
            cover[r0:r1 + 1, c0:c1 + 1] += 1
        if np.max(cover) > 1:
            continue
        if near_miss and not _has_near_miss(rectangles):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("unable to construct a valid rectangle layout")


def _has_near_miss(rectangles):
    for i, a in enumerate(rectangles):
        for b in rectangles[i + 1:]:
            if a[4] != b[4]:
                continue
            row_difference = abs((a[0] + a[1]) - (b[0] + b[1]))
            col_difference = abs((a[2] + a[3]) - (b[2] + b[3]))
            if row_difference == 2 and (a[3] < b[2] or b[3] < a[2]):
                return True
            if col_difference == 2 and (a[1] < b[0] or b[1] < a[0]):
                return True
    return False


def _input_key(grid):
    return (grid.shape, grid.tobytes())


def _source_digest(grid):
    return hashlib.sha256(bytes(grid.shape) + np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    connector = int(rng.integers(1, 10))
    backgrounds = [c for c in range(10) if c != connector]
    rng.shuffle(backgrounds)
    first_orientation = "H" if rng.random() < 0.5 else "V"
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train" and index == 0:
                mode, orientation = "intro", first_orientation
            elif split == "train" and index == 1:
                mode, orientation = "evidence", "V" if first_orientation == "H" else "H"
            else:
                mode, orientation = "free", "H"
            background = backgrounds[(index + (0 if split == "train" else train_count)) % len(backgrounds)]
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(12):
                pair = _make_pair(pair_rng, connector, background, mode, orientation, "source_train" if split == "train" else "target_test")
                key = _input_key(pair["input"])
                if key not in used and _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs = set(), set()
    connector = None
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert inp.shape == out.shape
            key = _input_key(inp)
            pair_key = (key, _input_key(out))
            assert key not in used_inputs and pair_key not in used_pairs
            used_inputs.add(key)
            used_pairs.add(pair_key)
            assert _source_digest(inp) not in _SOURCE_INPUT_DIGESTS
            background = _background(inp)
            changed = inp != out
            assert np.any(changed)
            assert np.all(inp[changed] == background)
            colors = set(int(x) for x in np.unique(out[changed]))
            assert len(colors) == 1
            found_connector = colors.pop()
            assert found_connector not in set(int(x) for x in np.unique(inp))
            if connector is None:
                connector = found_connector
            assert found_connector == connector
            expected, rectangles, specs = _transform(inp, background, connector)
            assert np.array_equal(out, expected) and specs
            assert len(rectangles) >= 2
            for orientation, a, b, r0, r1, c0, c1 in specs:
                assert rectangles[a][4] == rectangles[b][4]
                if orientation == "H":
                    assert r1 - r0 + 1 == min(rectangles[a][1] - rectangles[a][0], rectangles[b][1] - rectangles[b][0]) - 1
                else:
                    assert c1 - c0 + 1 == min(rectangles[a][3] - rectangles[a][2], rectangles[b][3] - rectangles[b][2]) - 1
            if split == "train":
                training.append((rectangles, specs))
    # Both orientations, an aligned mixed-color negative, a geometric near miss,
    # and two connector thicknesses are observable before any test input.
    assert {s[0] for _, specs in training for s in specs} == {"H", "V"}
    assert any(_has_near_miss(rects) for rects, _ in training)
    assert any(_has_mixed_alignment(rects) for rects, _ in training)
    assert len({_thickness(s) for _, specs in training for s in specs}) >= 2


def _thickness(spec):
    return spec[4] - spec[3] + 1 if spec[0] == "H" else spec[6] - spec[5] + 1


def _has_mixed_alignment(rectangles):
    for i, a in enumerate(rectangles):
        for b in rectangles[i + 1:]:
            if a[4] == b[4]:
                continue
            if a[0] + a[1] == b[0] + b[1] and (a[3] < b[2] or b[3] < a[2]):
                return True
            if a[2] + a[3] == b[2] + b[3] and (a[1] < b[0] or b[1] < a[0]):
                return True
    return False


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    first_inputs = [p["train"][0]["input"] for p in puzzles]
    assert len({_input_key(grid) for grid in first_inputs}) >= 40
    assert len({grid.shape for grid in first_inputs}) >= 12
    assert len({_background(grid) for grid in first_inputs}) >= 5
    connectors = {int(p["train"][0]["output"][p["train"][0]["input"] != p["train"][0]["output"]][0]) for p in puzzles}
    assert len(connectors) >= 5
    all_pairs = [pair for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    rectangles = [_extract_rectangles(pair["input"], _background(pair["input"])) for pair in all_pairs]
    assert any(any((r[1] - r[0] + 1) % 2 == 0 or (r[3] - r[2] + 1) % 2 == 0 for r in rs) for rs in rectangles)
    assert len({len(rs) for rs in rectangles}) >= 7
    assert any(pair["input"].shape == (30, 30) for pair in all_pairs)
    assert any(min(pair["input"].shape) <= 17 for pair in all_pairs)
    assert any(_has_near_miss(rs) for rs in rectangles)
    assert any(_has_mixed_alignment(rs) for rs in rectangles)
    counts = {name: 0 for name in (
        "train_one_link", "train_straight", "train_junction", "train_long_gap",
        "train_one_color", "train_four_colors", "test_near", "test_mixed",
        "test_no_isolate", "test_both_axes", "test_long_gap", "test_more_links",
        "test_more_colors", "test_one_link", "unequal_network_gaps",
        "reused_color_networks", "thin_long", "thick_short", "combined_variations",
    )}
    thicknesses, gaps_seen, component_patterns = set(), set(), set()
    for puzzle in puzzles:
        train_links, train_colors = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg = _background(inp)
                rects = _extract_rectangles(inp, bg)
                specs = _link_specs(inp, rects, bg)
                degree = [0] * len(rects)
                neighbors = [set() for _ in rects]
                for spec in specs:
                    a, b = spec[1:3]
                    degree[a] += 1
                    degree[b] += 1
                    neighbors[a].add(b)
                    neighbors[b].add(a)
                seen, sizes, components = set(), [], []
                for start in range(len(rects)):
                    if start in seen:
                        continue
                    stack = [start]
                    seen.add(start)
                    size = 0
                    component = []
                    while stack:
                        node = stack.pop()
                        size += 1
                        component.append(node)
                        for next_node in neighbors[node]:
                            if next_node not in seen:
                                seen.add(next_node)
                                stack.append(next_node)
                    sizes.append(size)
                    components.append(component)
                component_patterns.add(tuple(sorted(sizes)))
                gap_lengths = [s[6] - s[5] + 1 if s[0] == "H" else s[4] - s[3] + 1 for s in specs]
                gaps_seen.update(gap_lengths)
                thicknesses.update(_thickness(s) for s in specs)
                counts["unequal_network_gaps"] += any(
                    len({gap for spec, gap in zip(specs, gap_lengths) if spec[1] in component and spec[2] in component}) > 1
                    for component in components
                )
                counts["reused_color_networks"] += any(
                    {rects[node][4] for node in first} & {rects[node][4] for node in second}
                    for i, first in enumerate(components) for second in components[i + 1:]
                )
                counts["thin_long"] += 1 in (_thickness(s) for s in specs) and max(gap_lengths) >= 4
                counts["thick_short"] += max(_thickness(s) for s in specs) >= 3 and min(gap_lengths) <= 2
                color_count = len({r[4] for r in rects})
                mixed = _has_mixed_alignment(rects)
                near = _has_near_miss(rects)
                even = any((r[1] - r[0] + 1) % 2 == 0 or (r[3] - r[2] + 1) % 2 == 0 for r in rects)
                long = max(gap_lengths) >= 4
                counts["combined_variations"] += mixed and near and even and long
                if split == "train":
                    train_links.append(len(specs))
                    train_colors.append(color_count)
                    counts["train_one_link"] += len(specs) == 1
                    counts["train_straight"] += len(specs) >= 2 and max(degree) <= 2 and len({s[0] for s in specs}) == 1
                    counts["train_junction"] += max(degree) >= 3
                    counts["train_long_gap"] += long
                    counts["train_one_color"] += color_count == 1
                    counts["train_four_colors"] += color_count >= 4
                else:
                    counts["test_near"] += near
                    counts["test_mixed"] += mixed
                    counts["test_one_link"] += len(specs) == 1
                    counts["test_no_isolate"] += 0 not in degree
                    counts["test_both_axes"] += len({s[0] for s in specs}) == 2
                    counts["test_long_gap"] += long
                    counts["test_more_links"] += len(specs) > max(train_links)
                    counts["test_more_colors"] += color_count > max(train_colors)
    assert all(counts[name] >= 5 for name in counts)
    assert len(component_patterns) >= 30
    assert len(gaps_seen) >= 12 and min(gaps_seen) == 1 and max(gaps_seen) >= 10
    assert thicknesses == {1, 2, 3, 4, 5, 6}
    a, b = generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(123457, 7, 3))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, train, test)
        assert (train is None or len(p["train"]) == train) and (test is None or len(p["test"]) == test)
        validate_generated_puzzle(p)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
