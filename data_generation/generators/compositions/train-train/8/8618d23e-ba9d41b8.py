"""Woven windows: a perforated upper sheet reveals a diagonally shifted lower sheet.

Only the background color (the upper-left input cell) has a fixed role.  The
solver reads no generation metadata.  See task_spec.md for the complete domain.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be an integer 1..30 rectangle with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(int(v) for v in a.flat)


def _separate(a, b):
    t, d, l, r, _ = a
    u, e, m, n, _ = b
    # Exclusive bounds: at least one completely blank row or column separates
    # each pair of bounding boxes.  This also rules out diagonal contacts.
    return d < u or e < t or r < m or n < l


def _read_input(value):
    grid = _grid(value)
    rows, width = grid.shape
    if rows % 2 or rows > 28 or width > 29:
        raise ValueError("input must be two equal stacked sheets, with room for expansion")
    height = rows // 2
    upper, lower = grid[:height], grid[height:]
    background = int(upper[0, 0])
    seen = np.zeros(upper.shape, dtype=bool)
    rectangles = []
    for row in range(height):
        for col in range(width):
            color = int(upper[row, col])
            if seen[row, col] or color == background:
                continue
            stack, cells = [(row, col)], []
            seen[row, col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr, nc] and int(upper[nr, nc]) == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells) + 1
            left = min(c for _, c in cells)
            right = max(c for _, c in cells) + 1
            if (bottom-top < 3 or right-left < 3
                    or len(cells) != (bottom-top)*(right-left)):
                raise ValueError("upper-sheet objects must be solid rectangles with interiors")
            rectangles.append((top, bottom, left, right, color))
    if not rectangles:
        raise ValueError("the upper sheet needs at least one rectangle")
    for i, a in enumerate(rectangles):
        if any(not _separate(a, b) for b in rectangles[i+1:]):
            raise ValueError("upper rectangles need a background gap")
    return upper, lower, background, sorted(rectangles)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Resolve every output cell from the two visible sheets, without a seed."""
    upper, lower, background, rectangles = _read_input(input_grid)
    h, w = upper.shape
    result = np.full((h+1, w+1), background, dtype=np.int8)
    # Place the lower sheet first, one cell down and right.
    result[1:, 1:] = lower
    # Upper background is transparent.  Opaque rectangle cells cover the lower
    # sheet; checkerboard windows simply leave its already placed value visible.
    for top, bottom, left, right, color in rectangles:
        for row in range(top, bottom):
            for col in range(left, right):
                perimeter = row in (top, bottom-1) or col in (left, right-1)
                window = (row-top + col-left) % 2 == 0
                if perimeter or not window:
                    result[row, col] = color
    return result


def _render(upper, lower, bg, rectangles, *, offset=1, holes=True, phase=None,
            all_interior=False, background_opaque=False):
    """Vectorized construction, also used for explicit alternative-rule checks."""
    h, w = upper.shape
    out = np.full((h+1, w+1), bg, dtype=np.int8)
    out[offset:offset+h, offset:offset+w] = lower
    if background_opaque:
        out[:h, :w][upper == bg] = bg
    for t, b, l, r, color in rectangles:
        rr, cc = np.indices((b-t, r-l))
        interior = (rr > 0) & (rr < b-t-1) & (cc > 0) & (cc < r-l-1)
        even = (rr+cc) % 2 == 0 if phase is None else (rr+t+cc+l) % 2 == phase
        transparent = interior & (even | all_interior) if holes else np.zeros_like(interior)
        area = out[t:b, l:r]
        area[~transparent] = color
    return out


def _capacity(box, minimum):
    t, b, l, r = box
    return ((b-t+1)//(minimum+1))*((r-l+1)//(minimum+1))


def _partition(rng, box, count, minimum):
    """Random recursive allocations; gaps are real background, not panel dividers."""
    if count == 1:
        return [box]
    t, b, l, r = box
    candidates = []
    for axis in (0, 1):
        length = b-t if axis == 0 else r-l
        for cut in range(minimum, length-minimum):
            if axis == 0:
                a, z = (t, t+cut, l, r), (t+cut+1, b, l, r)
            else:
                a, z = (t, b, l, l+cut), (t, b, l+cut+1, r)
            for n in range(1, count):
                if _capacity(a, minimum) >= n and _capacity(z, minimum) >= count-n:
                    candidates.append((a, z, n))
    if not candidates:
        return None
    for i in rng.permutation(len(candidates)):
        a, z, n = candidates[int(i)]
        aa = _partition(rng, a, n, minimum)
        zz = _partition(rng, z, count-n, minimum)
        if aa is not None and zz is not None:
            return aa + zz
    return None


def _stencil(rng, count, bg, palette, *, mixed=False, thin=False, large=False):
    minimum = 3 if thin else 4
    for _ in range(160):
        h, w = int(rng.integers(7, 15)), int(rng.integers(11, 30))
        regions = _partition(rng, (0, h, 0, w), count, minimum)
        if regions is None:
            continue
        rng.shuffle(regions)
        colors = [int(c) for c in rng.choice(palette, size=count, replace=False)]
        if count > 1 and rng.random() < 0.38:
            colors[int(rng.integers(1, count))] = colors[0]
        rectangles = []
        for i, ((t, b, l, r), color) in enumerate(zip(regions, colors)):
            rh, rw = b-t, r-l
            smallest = 5 if large and i == 0 else minimum
            if rh < smallest or rw < smallest:
                break
            profile = int(rng.integers(4))
            if thin and i == 0:
                if max(rh, rw) < 4:
                    break
                if rw >= 4 and (rh < 4 or rng.random() < 0.5):
                    sh, sw = 3, int(rng.integers(max(4, rw//2), rw+1))
                else:
                    sh, sw = int(rng.integers(max(4, rh//2), rh+1)), 3
            elif profile == 0:  # compact
                sh = int(rng.integers(smallest, min(rh, smallest+3)+1))
                sw = int(rng.integers(smallest, min(rw, smallest+3)+1))
            elif profile == 1:  # almost fills the allocated region
                sh = max(smallest, rh-int(rng.integers(3)))
                sw = max(smallest, rw-int(rng.integers(3)))
            elif profile == 2:  # long strip, in either orientation
                if rng.random() < 0.5:
                    sh, sw = smallest, max(smallest, rw-int(rng.integers(3)))
                else:
                    sh, sw = max(smallest, rh-int(rng.integers(3))), smallest
            else:
                sh, sw = int(rng.integers(smallest, rh+1)), int(rng.integers(smallest, rw+1))
            positions = [(y, x) for y in range(t, b-sh+1) for x in range(l, r-sw+1)
                         if (y != 0 or x != 0)
                         and (not mixed or i >= 2 or (y+x) % 2 == i)]
            if not positions:
                break
            y, x = positions[int(rng.integers(len(positions)))]
            rectangles.append((y, y+sh, x, x+sw, color))
        if len(rectangles) == count:
            upper = np.full((h, w), bg, dtype=np.int8)
            for t, b, l, r, color in rectangles:
                upper[t:b, l:r] = color
            return upper, sorted(rectangles)
    raise RuntimeError("bounded stencil sampler exhausted")


def _underlay(rng, shape, bg, palette, mode):
    h, w = shape
    ink_count = 2 if mode == "anchor" else int(rng.integers(2, min(5, len(palette))+1))
    inks = np.asarray(rng.choice(palette, size=ink_count, replace=False), dtype=np.int8)
    colors = np.concatenate((np.array([bg], dtype=np.int8), inks))
    colors = rng.permutation(colors)
    rr, cc = np.indices(shape)
    if mode in ("anchor", "stripes"):
        axis = rr if rng.random() < 0.5 else cc
        thickness = 1 if mode == "anchor" else int(rng.integers(1, 4))
        return colors[(axis//thickness + int(rng.integers(len(colors)))) % len(colors)]
    lower = np.full(shape, bg, dtype=np.int8)
    if mode == "patches":
        for _ in range(int(rng.integers(7, 18))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            y1, x1 = int(rng.integers(y+1, h+1)), int(rng.integers(x+1, w+1))
            lower[y:y1, x:x1] = colors[int(rng.integers(len(colors)))]
    elif mode == "ribbons":
        for _ in range(int(rng.integers(5, 12))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            color = inks[int(rng.integers(len(inks)))]
            if rng.random() < 0.5:
                lower[y:min(h, y+int(rng.integers(1, 3))), :x+1] = color
            else:
                lower[:y+1, x:min(w, x+int(rng.integers(1, 3)))] = color
    elif mode == "mosaic":
        sy, sx = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        coarse = rng.choice(colors, size=((h+sy-1)//sy, (w+sx-1)//sx))
        lower = np.repeat(np.repeat(coarse, sy, axis=0), sx, axis=1)[:h, :w].copy()
    elif mode == "rings":
        depth = np.minimum(np.minimum(rr, h-1-rr), np.minimum(cc, w-1-cc))
        lower = colors[(depth+int(rng.integers(len(colors)))) % len(colors)]
    else:
        raise ValueError("unknown underlay mode")
    return lower.astype(np.int8)


def _observations(upper, lower, bg, rects):
    exposed, anchors, opaque_witness = [], [], False
    for t, b, l, r, color in rects:
        anchors.append(int(lower[t, l]) not in (bg, color))
        for y in range(t+1, b-1):
            for x in range(l+1, r-1):
                value = int(lower[y-1, x-1])
                if (y-t+x-l) % 2 == 0:
                    exposed.append(value)
                elif value not in (bg, color):
                    opaque_witness = True
    return set(exposed), all(anchors), opaque_witness


def _draw_pair(rng, count, bg, pool, seen, *, mixed=False, thin=False, anchor=False):
    for _ in range(60):
        upper, rects = _stencil(rng, count, bg, pool, mixed=mixed,
                                thin=thin, large=anchor)
        mode = "anchor" if anchor else str(rng.choice(
            ["stripes", "patches", "ribbons", "mosaic", "rings"]))
        for _ in range(100):
            lower = _underlay(rng, upper.shape, bg, pool, mode)
            exposed, good_anchors, opaque_witness = _observations(upper, lower, bg, rects)
            if not good_anchors or (anchor and (bg not in exposed or len(exposed-{bg}) < 2)):
                continue
            if not opaque_witness and not thin:
                continue
            out = _render(upper, lower, bg, rects)
            if np.array_equal(out, _render(upper, lower, bg, rects, offset=0)):
                continue
            if np.array_equal(out, _render(upper, lower, bg, rects, all_interior=True)):
                continue
            if np.array_equal(out, _render(upper, lower, bg, rects, background_opaque=True)):
                continue
            inp = np.vstack((upper, lower))
            signature = _key(inp)
            if signature not in seen:
                seen.add(signature)
                return {"input": inp, "output": out}
    raise RuntimeError("bounded visible-witness sampler exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    bg = int(rng.integers(10))
    palette = np.array([c for c in range(10) if c != bg], dtype=np.int8)
    # Some puzzles reserve one ink for tests; others freely reuse all colors.
    rng.shuffle(palette)
    reserve = bool(rng.integers(2))
    train_pool = palette[:-1] if reserve else palette
    count_shift = bool(rng.integers(2))
    seen = set()
    puzzle = {"train": [], "test": []}
    # Two complementary witnesses survive even the smallest legal train count.
    requests = [(1, False, True), (2, True, False)]
    requests += [(int(rng.integers(1, 3 if count_shift else 4)), False, False)
                 for _ in range(num_train-2)]
    rng.shuffle(requests)
    for count, mixed, anchor in requests:
        puzzle["train"].append(_draw_pair(rng, count, bg, train_pool, seen,
                                          mixed=mixed, anchor=anchor))
    for _ in range(num_test):
        count = int(rng.integers(3, 5)) if count_shift else int(rng.integers(1, 5))
        puzzle["test"].append(_draw_pair(rng, count, bg, palette, seen,
                                        thin=not count_shift))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check sheet geometry, visibility, phase witnesses, ablations and shifts."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal counts")
    backgrounds, seen, counts = set(), set(), {"train": [], "test": []}
    mixed_phase, color_window_demo = False, False
    test_thin = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            upper, lower, bg, rects = _read_input(inp)
            h, w = upper.shape
            backgrounds.add(bg)
            counts[split].append(len(rects))
            _require(1 <= len(rects) <= 4, "object count outside generated family")
            _require(out.shape == (h+1, w+1), "wrong collapsed-sheet dimensions")
            _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
            # Explicit cell invariants, including the unoccluded shifted fringe.
            opaque = np.zeros((h+1, w+1), dtype=bool)
            for t, b, l, r, color in rects:
                _require(np.all(out[t, l:r] == color) and np.all(out[b-1, l:r] == color)
                         and np.all(out[t:b, l] == color) and np.all(out[t:b, r-1] == color),
                         "stencil perimeter changed")
                for y in range(t, b):
                    for x in range(l, r):
                        border = y in (t, b-1) or x in (l, r-1)
                        window = not border and (y-t+x-l) % 2 == 0
                        if window:
                            _require(out[y, x] == lower[y-1, x-1], "window sampled wrong lower cell")
                        else:
                            opaque[y, x] = True
                            _require(out[y, x] == color, "opaque stencil cell changed")
            shifted = np.full(out.shape, bg, dtype=np.int8)
            shifted[1:, 1:] = lower
            _require(np.array_equal(out[~opaque], shifted[~opaque]), "exterior transparency failed")
            exposed, good_anchors, opaque_witness = _observations(upper, lower, bg, rects)
            _require(good_anchors, "each rectangle must visibly reveal contrasting ink")
            for kwargs in ({"offset": 0}, {"holes": False}, {"all_interior": True},
                           {"background_opaque": True}):
                _require(not np.array_equal(out, _render(upper, lower, bg, rects, **kwargs)),
                         f"a simpler alternative fits this example: {kwargs}")
            if split == "train":
                _require(all(b-t >= 4 and r-l >= 4 for t, b, l, r, _ in rects),
                         "training rectangle too thin")
                _require(opaque_witness, "training must show interior occlusion")
                if {(t+l) % 2 for t, _, l, _, _ in rects} == {0, 1}:
                    for phase in (0, 1):
                        _require(not np.array_equal(out, _render(upper, lower, bg, rects, phase=phase)),
                                 "a global checkerboard phase still fits")
                    mixed_phase = True
                color_window_demo |= bg in exposed and len(exposed-{bg}) >= 2
            else:
                test_thin.append(any(b-t == 3 or r-l == 3 for t, b, l, r, _ in rects))
    _require(len(backgrounds) == 1, "generated puzzles keep one background convention")
    _require(mixed_phase, "need opposite stencil-origin parities in one training grid")
    _require(color_window_demo, "need both empty and differently colored windows in training")
    _require(min(counts["test"]) > max(counts["train"]) or all(test_thin),
             "test must shift object count or include a three-cell-thick stencil")


def _fingerprint(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed semantic/diversity audit plus all explicit count choices."""
    shapes, sizes, structures, backgrounds, fingerprints = set(), set(), set(), set(), set()
    test_counts, aspects, edge_sides = set(), set(), set()
    counts_shift = thin_shift = repeated = sharing = disjoint = 0
    nonmodal_background = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic seed")
        train_n, test_n = [], []
        has_thin = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                upper, lower, bg, rects = _read_input(pair["input"])
                h, w = upper.shape
                backgrounds.add(bg)
                shapes.add((h, w))
                structures.add((upper.shape, tuple((t, b, l, r) for t, b, l, r, _ in rects)))
                (train_n if split == "train" else test_n).append(len(rects))
                if split == "test":
                    test_counts.add(len(rects))
                    has_thin |= any(b-t == 3 or r-l == 3 for t, b, l, r, _ in rects)
                object_colors = [c for *_, c in rects]
                repeated += len(set(object_colors)) < len(object_colors)
                overlap = set(object_colors) & (set(int(x) for x in np.unique(lower))-{bg})
                sharing += bool(overlap)
                disjoint += not bool(overlap)
                values, amounts = np.unique(upper, return_counts=True)
                nonmodal_background += int(values[int(np.argmax(amounts))]) != bg
                for t, b, l, r, _ in rects:
                    size = b-t, r-l
                    sizes.add(size)
                    aspects.add("square" if size[0] == size[1] else "tall" if size[0] > size[1] else "wide")
                    if t == 0:
                        edge_sides.add("top")
                    if b == h:
                        edge_sides.add("bottom")
                    if l == 0:
                        edge_sides.add("left")
                    if r == w:
                        edge_sides.add("right")
        counts_shift += min(test_n) > max(train_n)
        thin_shift += has_thin
    _require(len(fingerprints) == 42, "puzzles repeated across seeds")
    _require(len(shapes) >= 65 and len(sizes) >= 45 and len(structures) >= 150,
             "insufficient structural variation")
    _require(len(backgrounds) >= 8, "background roles lack variation")
    _require(test_counts == {1, 2, 3, 4} and aspects == {"square", "wide", "tall"},
             "counts or rectangle proportions lack variation")
    _require(edge_sides == {"top", "bottom", "left", "right"}, "boundary cases absent")
    _require(counts_shift >= 8 and thin_shift >= 8, "train/test shifts lack variation")
    _require(repeated >= 8 and sharing >= 20 and disjoint >= 8, "palette relations lack variation")
    _require(nonmodal_background >= 1, "corner-background cue was never needed")
    # A metamorphic color permutation must commute with solving, including 0.
    p = generate_puzzle(917, 2, 1)
    perm = np.array([8, 5, 0, 9, 2, 7, 3, 1, 6, 4], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]),
                 "solver depends on literal color values")
    # Solve also supports valid, sparse lower sheets outside the sampler's
    # contrasting-witness distribution, and does not require a modal background.
    upper = np.zeros((6, 8), dtype=np.int8)
    upper[1:6, 1:8] = 7
    lower = np.zeros_like(upper)
    lower[1, 1] = 4
    result = solve(np.vstack((upper, lower)))
    _require(result[2, 2] == 4 and result[2, 3] == 7 and result[0, 0] == 0,
             "sparse transparency boundary case failed")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(42000+a*10+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "sheet_shapes": len(shapes),
            "rectangle_sizes": len(sizes), "stencil_layouts": len(structures),
            "count_shift_puzzles": counts_shift, "thin_shift_puzzles": thin_shift,
            "repeated_color_pairs": repeated, "shared_palette_pairs": sharing,
            "disjoint_palette_pairs": disjoint, "nonmodal_background_pairs": nonmodal_background}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
