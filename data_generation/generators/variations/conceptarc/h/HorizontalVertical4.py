"""ARC generator: recolor horizontal marker runs and retain vertical ones."""

import hashlib
import numpy as np

_SOURCE_HASHES = {
    "aa5d49c362fcfb1c42d3ea676fefb32d6362724a0a52cd8a25278728c5d89c6c",
    "f4a89f71d86af18b4f9baeffa19d6515a1e11eb7c164e7ff5cfee6afeadf0d78",
    "e4d194e3a095c4911746bf92235006b8cc17fb868d22764c69e54548ed10ded6",
    "f850a626c4fabc064b5ec50ea81643afa75c95531fb0802c793ebdcf7ed2e90c",
    "c82fa535b8c11e9c65cdcaeee3dfb2cc4754e9e3c4efbedf6a4a3271573b600c",
}


def _source_hash(a, b):
    return hashlib.sha256(str(a.shape).encode() + np.asarray(a, dtype=np.int8).tobytes()
                          + np.asarray(b, dtype=np.int8).tobytes()).hexdigest()


def _check_count(value, name, low, high):
    if value is not None and (isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer))
                              or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _outline_masks(height, width, kind, rng):
    """Make a row-convex filled shape and its four-neighbor interior."""
    left = np.zeros(height, dtype=int)
    right = np.zeros(height, dtype=int)
    if kind in ("lozenge", "pointed"):
        depth = min((width - 1) // 2, (height - 1) // 2)
        for row in range(height):
            d = max(0, depth - min(row, height - row - 1))
            left[row] = d
            right[row] = d if kind == "lozenge" else max(0, d - (row >= height // 2))
    elif kind == "asymmetric":
        depth = min((width - 4) // 2, (height - 2) // 2, 3)
        side = int(rng.integers(2))
        for row in range(height):
            d = max(0, depth - min(row, height - row - 1))
            if side:
                left[row], right[row] = d, max(0, d - 2)
            else:
                right[row], left[row] = d, max(0, d - 2)
    elif kind == "stepped" and height >= 6 and width >= 6:
        start = int(rng.integers(1, height - 2))
        stop = min(height - 1, start + int(rng.integers(2, 4)))
        side = left if rng.random() < .5 else right
        side[start:stop] = int(rng.integers(1, min(3, width - 3)))
        other = right if side is left else left
        if rng.random() < .65:
            other[1:max(2, height // 3)] = 1
    filled = np.zeros((height, width), dtype=bool)
    for row in range(height):
        lo, hi = int(left[row]), width - int(right[row])
        if hi - lo < 3:
            return None
        filled[row, lo:hi] = True
    interior = filled.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    interior[1:, :] &= filled[:-1, :]
    interior[:-1, :] &= filled[1:, :]
    interior[:, 1:] &= filled[:, :-1]
    interior[:, :-1] &= filled[:, 1:]
    return filled & ~interior, interior


def _run_candidates(interior, direction):
    runs = []
    lines = interior if direction == "H" else interior.T
    for fixed, line in enumerate(lines):
        start = 0
        while start < len(line):
            if not line[start]:
                start += 1
                continue
            end = start
            while end < len(line) and line[end]:
                end += 1
            for length in range(2, end - start + 1):
                for offset in range(start, end - length + 1):
                    runs.append((fixed, offset, length, direction) if direction == "H"
                                else (offset, fixed, length, direction))
            start = end
    return runs


def _separated(a, b):
    ar, ac, al, ad = a
    br, bc, bl, bd = b
    aer, aec = ar + (al - 1 if ad == "V" else 0), ac + (al - 1 if ad == "H" else 0)
    ber, bec = br + (bl - 1 if bd == "V" else 0), bc + (bl - 1 if bd == "H" else 0)
    return aer < br - 1 or ber < ar - 1 or aec < bc - 1 or bec < ac - 1


def _choose_runs(rng, interior, directions):
    candidates = {d: _run_candidates(interior, d) for d in ("H", "V")}
    if any(not candidates[d] for d in directions):
        return None
    for attempt in range(36):
        order = list(directions)
        rng.shuffle(order)
        selected = []
        for direction in order:
            possible = [c for c in candidates[direction]
                        if all(_separated(c, old) for old in selected)]
            if not possible:
                break
            longest = max(c[2] for c in possible)
            mode = int(rng.integers(3)) if attempt < 24 else 1
            if mode == 0:
                preferred = [c for c in possible if c[2] <= 3]
            elif mode == 2:
                preferred = [c for c in possible if c[2] >= max(4, longest - 2)]
            else:
                preferred = [c for c in possible if 3 <= c[2] <= max(5, longest - 2)]
            pool = preferred or possible
            selected.append(pool[int(rng.integers(len(pool)))])
        if len(selected) == len(directions):
            return selected
    return None


def _object_size(rng, count, directions):
    if len(directions) >= 4:
        return int(rng.integers(12, 18)), int(rng.integers(12, 18))
    if len(set(directions)) == 2:
        return int(rng.integers(8, 15)), int(rng.integers(8, 15))
    if count >= 4:
        return int(rng.integers(4, 9)), int(rng.integers(4, 10))
    mode = int(rng.integers(4))
    if mode == 0:
        return int(rng.integers(4, 7)), int(rng.integers(8, 16))
    if mode == 1:
        return int(rng.integers(8, 16)), int(rng.integers(4, 7))
    return int(rng.integers(5, 15)), int(rng.integers(5, 15))


def _directions(rng, regime, index, count, dense):
    if regime == "source_train" and index == 0:
        return [["H", "V"], ["H" if rng.random() < .5 else "V"]]
    if regime == "source_train" and index == 1 and count >= 4:
        result = [["H"], ["H"], ["V"], ["V"]]
        if count == 5:
            result.append(["H" if rng.random() < .5 else "V"])
        rng.shuffle(result)
        return result
    if dense and count == 1:
        return [["H", "H", "V", "V"]]
    choices = [["H"], ["V"], ["H", "H"], ["V", "V"], ["H", "V"]]
    result = [choices[int(rng.integers(len(choices)))].copy() for _ in range(count)]
    if regime == "source_train" and not any("H" in d for d in result):
        result[0] = ["H"]
    return result


def _layout(rng, objects):
    count = len(objects)
    if count == 1:
        rows = [objects]
    elif count == 2:
        rows = [objects] if rng.random() < .5 else [[o] for o in objects]
    elif count == 3:
        rows = [objects[:1], objects[1:]] if rng.random() < .5 else [objects[:2], objects[2:]]
    elif count == 4:
        rows = [objects[:2], objects[2:]] if rng.random() < .8 else [objects[:1], objects[1:]]
    else:
        rows = [objects[:2], objects[2:]] if rng.random() < .5 else [objects[:3], objects[3:]]
    col_gaps = [int(rng.integers(1, 4)) for _ in rows]
    row_gap = int(rng.integers(1, 4))
    widths = [sum(o["width"] for o in row) + col_gaps[i] * (len(row) - 1)
              for i, row in enumerate(rows)]
    heights = [max(o["height"] for o in row) for row in rows]
    top, left = int(rng.integers(4)), int(rng.integers(4))
    bottom, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    h = top + sum(heights) + row_gap * (len(rows) - 1) + bottom
    w = left + max(widths) + right
    if h > 30 or w > 30:
        return None
    positions = []
    y = top
    for i, row in enumerate(rows):
        x = left + int(rng.integers(max(widths) - widths[i] + 1))
        for obj in row:
            dy = int(rng.integers(heights[i] - obj["height"] + 1))
            positions.append((y + dy, x))
            x += obj["width"] + col_gaps[i]
        y += heights[i] + row_gap
    return h, w, positions


def _generate_pair(rng, regime, index, palette, background, dense=False):
    if regime == "source_train" and index == 0:
        count = 2
    elif regime == "source_train" and index == 1:
        count = int(rng.integers(4, 6))
    elif dense:
        count = 1
    else:
        count = int(rng.choice((1, 2, 2, 3, 4, 5)))
    plans = _directions(rng, regime, index, count, dense)
    color_pool = [c for c in palette["outlines"] if c != background]
    if regime == "source_train" and index == 0:
        colors = [color_pool[0]] * count
    elif regime == "source_train" and index == 1:
        colors = [color_pool[i % len(color_pool)] for i in range(count)]
    elif rng.random() < .38:
        colors = [color_pool[int(rng.integers(len(color_pool)))]] * count
    else:
        colors = [color_pool[int(rng.integers(len(color_pool)))] for _ in range(count)]
    objects = []
    for directions, color in zip(plans, colors):
        for _ in range(24):
            height, width = _object_size(rng, count, directions)
            kind = ("rectangle", "lozenge", "pointed", "asymmetric", "stepped")[int(rng.integers(5))]
            masks = _outline_masks(height, width, kind, rng)
            if masks is None:
                continue
            border, interior = masks
            runs = _choose_runs(rng, interior, directions)
            if runs is not None:
                objects.append({"height": height, "width": width, "border": border,
                                "runs": runs, "color": color})
                break
        else:
            raise RuntimeError("Could not fit marked runs")
    layout = _layout(rng, objects)
    if layout is None:
        raise RuntimeError("Could not fit outlines")
    height, width, positions = layout
    a = np.full((height, width), background, dtype=np.int8)
    b = a.copy()
    for obj, (top, left) in zip(objects, positions):
        oh, ow = obj["height"], obj["width"]
        a[top:top+oh, left:left+ow][obj["border"]] = obj["color"]
        b[top:top+oh, left:left+ow][obj["border"]] = obj["color"]
        for row, col, length, direction in obj["runs"]:
            rr = slice(top + row, top + row + (1 if direction == "H" else length))
            cc = slice(left + col, left + col + (length if direction == "H" else 1))
            a[rr, cc] = palette["marker"]
            b[rr, cc] = palette["replacement"] if direction == "H" else palette["marker"]
    return a, b


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [int(x) for x in rng.permutation(np.arange(1, 10))]
    palette = {"marker": colors[0], "replacement": colors[1],
               "outlines": colors[2:6] + colors[7:9]}
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    backgrounds = [0, colors[6]]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, key, count in (("source_train", "train", num_train),
                               ("target_test", "test", num_test)):
        for index in range(count):
            bg = (backgrounds[index] if key == "train" and index < 2
                  else backgrounds[int(rng.integers(2))])
            dense = key == "test" and index == 0 and rng.random() < .40
            for _ in range(120):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                try:
                    a, b = _generate_pair(pair_rng, regime, index, palette, bg, dense)
                except RuntimeError:
                    continue
                sig = (a.shape, a.tobytes())
                if sig not in seen and _source_hash(a, b) not in _SOURCE_HASHES:
                    seen.add(sig)
                    puzzle[key].append({"input": a, "output": b})
                    break
            else:
                raise RuntimeError("Could not generate a distinct valid pair")
    return puzzle


def _components(mask, diagonal=False):
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    h, w = mask.shape
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _inspect_pair(a, b, marker, replacement):
    assert a.shape == b.shape and not np.any(a == replacement)
    expected = a.copy()
    runs = []
    for cells in _components(a == marker):
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        assert len(cells) >= 2
        assert (len(rows) == 1) != (len(cols) == 1)
        direction = "H" if len(rows) == 1 else "V"
        assert (max(cols) - min(cols) + 1 if direction == "H"
                else max(rows) - min(rows) + 1) == len(cells)
        if direction == "H":
            for r, c in cells:
                expected[r, c] = replacement
        runs.append((cells, direction))
    assert runs and np.array_equal(expected, b)
    background = int(a[-1, -1])
    assert background not in (marker, replacement)
    outline_colors = set(int(x) for x in np.unique(a)) - {background, marker}
    assert outline_colors
    walkable = (a == background) | (a == marker)
    exterior = np.zeros_like(walkable)
    stack = [(0, c) for c in range(a.shape[1]) if walkable[0, c]]
    stack += [(a.shape[0]-1, c) for c in range(a.shape[1]) if walkable[-1, c]]
    stack += [(r, 0) for r in range(a.shape[0]) if walkable[r, 0]]
    stack += [(r, a.shape[1]-1) for r in range(a.shape[0]) if walkable[r, -1]]
    while stack:
        r, c = stack.pop()
        if exterior[r, c]:
            continue
        exterior[r, c] = True
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and walkable[rr, cc] and not exterior[rr, cc]:
                stack.append((rr, cc))
    assert not np.any(exterior & (a == marker))
    pockets = []
    for cells in _components(walkable & ~exterior):
        cell_set = set(cells)
        directions = [direction for run, direction in runs if run[0] in cell_set]
        assert directions
        pockets.append(directions)
    return {"runs": runs, "background": background,
            "outline_colors": outline_colors, "pockets": pockets}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    marker = replacement = None
    train_directions = set()
    train_backgrounds = set()
    for key in ("train", "test"):
        for pair in puzzle[key]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape
            sig = (a.shape, a.tobytes())
            assert sig not in seen
            seen.add(sig)
            assert _source_hash(a, b) not in _SOURCE_HASHES
            changed = a != b
            if np.any(changed):
                old, new = np.unique(a[changed]), np.unique(b[changed])
                assert len(old) == len(new) == 1
                if marker is None:
                    marker, replacement = int(old[0]), int(new[0])
                assert int(old[0]) == marker and int(new[0]) == replacement
            else:
                assert marker is not None
            assert marker != replacement
            info = _inspect_pair(a, b, marker, replacement)
            if key == "train":
                train_directions.update(d for _, d in info["runs"])
                train_backgrounds.add(info["background"])
    assert train_directions == {"H", "V"}
    assert len(train_backgrounds) >= 2


def _same_puzzle(a, b):
    return all(len(a[key]) == len(b[key]) and all(
        np.array_equal(pa[field], pb[field])
        for pa, pb in zip(a[key], b[key]) for field in ("input", "output"))
        for key in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, mappings = set(), set(), set(), set()
    run_lengths, backgrounds, color_counts, direction_sets, firsts = set(), set(), set(), set(), set()
    object_counts = set()
    review = {name: 0 for name in (
        "small", "shallow", "narrow", "elongated", "edge", "one_gap",
        "nonrectangle", "mixed_train", "mixed_test", "dense_runs",
        "four_train", "five_train", "four_test", "five_test",
        "vertical_only_test", "familiar_test", "novel_test", "combined")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        a, b = puzzle["train"][0]["input"], puzzle["train"][0]["output"]
        marker = int(np.unique(a[a != b])[0])
        replacement = int(np.unique(b[a != b])[0])
        mappings.add((marker, replacement))
        firsts.add((a.shape, a.tobytes()))
        train_outline_colors = set()
        for pair in puzzle["train"]:
            aa = pair["input"]
            train_outline_colors.update(set(int(c) for c in np.unique(aa))
                                        - {int(aa[-1, -1]), marker})
        pair_properties = []
        for key in ("train", "test"):
            for pair in puzzle[key]:
                a, b = pair["input"], pair["output"]
                info = _inspect_pair(a, b, marker, replacement)
                sizes.add(a.shape)
                backgrounds.add(info["background"])
                color_counts.add(len(info["outline_colors"]))
                direction_sets.add(frozenset(d for _, d in info["runs"]))
                run_lengths.update(len(cells) for cells, _ in info["runs"])
                outlines = []
                for color in info["outline_colors"]:
                    outlines.extend((color, cells) for cells in _components(a == color, True))
                count = len(outlines)
                object_counts.add(count)
                assert count == len(info["pockets"])
                pair_properties.append((a.shape, count, info["background"],
                                        len(info["outline_colors"])))
                bounds = []
                nonrectangle = False
                edge = False
                for _, cells in outlines:
                    rows = [r for r, c in cells]
                    cols = [c for r, c in cells]
                    top, bottom = min(rows), max(rows)
                    left, right = min(cols), max(cols)
                    height, width = bottom - top + 1, right - left + 1
                    bounds.append((top, bottom, left, right))
                    review["small"] += min(height, width) <= 6
                    review["shallow"] += height <= 5 and width >= 8
                    review["narrow"] += width <= 5 and height >= 8
                    review["elongated"] += max(height, width) >= 2 * min(height, width)
                    touches = top == 0 or left == 0 or bottom == a.shape[0]-1 or right == a.shape[1]-1
                    review["edge"] += touches
                    edge |= touches
                    rectangle = ({(r, c) for r in range(top, bottom+1) for c in (left, right)}
                                 | {(r, c) for r in (top, bottom) for c in range(left, right+1)})
                    nonrectangle |= set(cells) != rectangle
                review["nonrectangle"] += nonrectangle
                gap_one = False
                for i, (t1, b1, l1, r1) in enumerate(bounds):
                    for t2, b2, l2, r2 in bounds[i+1:]:
                        hgap = max(l2-r1-1, l1-r2-1, 0)
                        vgap = max(t2-b1-1, t1-b2-1, 0)
                        gap_one |= (hgap == 1 and vgap == 0) or (vgap == 1 and hgap == 0)
                review["one_gap"] += gap_one
                mixed = any({"H", "V"} <= set(d) for d in info["pockets"])
                review["mixed_train" if key == "train" else "mixed_test"] += mixed
                if key == "test":
                    review["dense_runs"] += any(d.count("H") >= 2 and d.count("V") >= 2
                                                for d in info["pockets"])
                    review["vertical_only_test"] += all(d == "V" for _, d in info["runs"])
                    novelty = bool(info["outline_colors"] - train_outline_colors)
                    review["novel_test" if novelty else "familiar_test"] += 1
                review[("four_train" if key == "train" else "four_test")] += count >= 4
                review[("five_train" if key == "train" else "five_test")] += count >= 5
                review["combined"] += (count >= 4 and nonrectangle and
                                       info["background"] != 0 and
                                       len(info["outline_colors"]) >= 2 and edge)
        assert len({p[0] for p in pair_properties}) > 1
        assert len({p[1] for p in pair_properties}) > 1
        assert len({p[2] for p in pair_properties}) > 1
        assert len({p[3] for p in pair_properties}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 35 and len(mappings) >= 20 and len(firsts) >= 40
    assert max(run_lengths) >= 9 and 2 in run_lengths
    assert 0 in backgrounds and len(backgrounds) >= 3
    assert {1, 2} <= color_counts
    assert {frozenset({"H"}), frozenset({"V"}), frozenset({"H", "V"})} <= direction_sets
    assert object_counts == {1, 2, 3, 4, 5}
    assert all(review[name] > 0 for name in review)
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (103, None, 1), (104, None, 4), (105, 2, None),
                         (106, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed, nt, ns))
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {kwargs}")
    assert not _same_puzzle(generate_puzzle(4), generate_puzzle(5))
