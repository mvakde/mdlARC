"""Fold inferred diagonal bridge lengths onto a clockwise square rim.

The solver reads pixels only. Generation metadata is used solely to construct
an independent expected answer; validation recovers all objects from the grid.
"""

import numpy as np


_SIGNS = ((-1, -1), (-1, 1), (1, 1), (1, -1))  # UL, UR, LR, LL


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((0 <= a) & (a <= 9)), "grid colors must be integers 0..9")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(v) for v in a.flat)


def _components(a, color):
    unseen = {tuple(map(int, p)) for p in np.argwhere(a == color)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, points = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
                    points.append(p)
        rows, cols = zip(*points)
        top, left, size = min(rows), min(cols), max(rows) - min(rows) + 1
        _require(max(cols) - left + 1 == size and len(points) == size * size,
                 "endpoints must be filled squares")
        result.append((top, left, size))
    return sorted(result)


def _read_input(input_grid):
    """Return background, center and bridge records, inferred without outputs."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1,
             "background must be the unique most frequent color")
    bg = int(colors[np.argmax(counts)])
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1 and c != bg]
    _require(len(singletons) == 1, "exactly one singleton center color is needed")
    center_color = singletons[0]
    cy, cx = map(int, np.argwhere(a == center_color)[0])
    _require(np.all(a[cy, np.arange(a.shape[1]) != cx] == bg) and
             np.all(a[np.arange(a.shape[0]) != cy, cx] == bg),
             "center row and column must separate the four quadrants")
    records, all_bridge_cells = [], set()
    for raw_color in colors:
        color = int(raw_color)
        if color in (bg, center_color):
            continue
        squares = _components(a, color)
        _require(len(squares) == 2, "each thread color needs exactly two squares")
        (r, c, s), (rr, cc, ss) = squares
        d = rr - r
        _require(s == ss and d > s and abs(cc - c) == d,
                 "same-colored congruent squares need a positive 45-degree gap")
        slope = 1 if cc > c else -1
        gap = d - s
        cells = tuple((r + s + j, c + s + j if slope == 1 else c - 1 - j)
                      for j in range(gap))
        _require(all(a[p] == bg and p not in all_bridge_cells for p in cells),
                 "bridges must be empty and cannot share a cell")
        all_bridge_cells.update(cells)
        # Twice the bridge midpoint keeps half-cell radial positions exact.
        my, mx = cells[0][0] + cells[-1][0], cells[0][1] + cells[-1][1]
        dy, dx = my - 2 * cy, mx - 2 * cx
        _require(dy != 0 and abs(dy) == abs(dx),
                 "bridge midpoints must lie on the four center diagonals")
        q = _SIGNS.index((1 if dy > 0 else -1, 1 if dx > 0 else -1))
        sy, sx = _SIGNS[q]
        occupied = [(y, x) for top, left, size in squares
                    for y in range(top, top + size) for x in range(left, left + size)]
        _require(all((y - cy) * sy > 0 and (x - cx) * sx > 0
                     for y, x in occupied + list(cells)),
                 "a whole square pair and its bridge must stay in one open quadrant")
        records.append(dict(color=color, size=s, gap=gap, slope=slope,
                            quadrant=q, radius2=abs(dy), cells=cells,
                            squares=tuple(squares)))
    _require(4 <= len(records) <= 8, "four to eight thread colors are supported")
    words = []
    for q in range(4):
        group = sorted((r for r in records if r["quadrant"] == q),
                       key=lambda r: -r["radius2"])
        _require(group and len({r["radius2"] for r in group}) == len(group),
                 "each quadrant needs bridges with distinct radial ranks")
        words.append([r["color"] for r in group for _ in range(r["gap"])])
    lengths = {len(w) for w in words}
    _require(len(lengths) == 1, "four total bridge lengths must be equal")
    length = len(words[0])
    _require(length >= 2 and length % 2 == 0 and length + 1 <= 30,
             "the common bridge length must be positive and even")
    return bg, center_color, (cy, cx), records, words


def _fold(words, bg, center_color):
    length = len(words[0])
    out = np.full((length + 1, length + 1), bg, dtype=np.int8)
    # Each side owns its initial corner and excludes its final corner.
    for j in range(length):
        out[0, j] = words[0][j]
        out[j, length] = words[1][j]
        out[length, length - j] = words[2][j]
        out[length - j, 0] = words[3][j]
    out[length // 2, length // 2] = center_color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg, center_color, _, _, words = _read_input(input_grid)
    return _fold(words, bg, center_color)


def _nearer_wrong_color(a, records, bg):
    """Count clear shorter diagonal connections to an unlike-colored square."""
    found = 0
    for i, first in enumerate(records):
        for second in records[i + 1:]:
            if first["size"] != second["size"]:
                continue
            size = first["size"]
            for p in first["squares"]:
                for t in second["squares"]:
                    (r, c, _), (rr, cc, _) = sorted((p, t))
                    d = rr - r
                    if d <= size or abs(cc - c) != d:
                        continue
                    gap = d - size
                    if gap >= max(first["gap"], second["gap"]):
                        continue
                    cells = [(r + size + j, c + size + j if cc > c else c - 1 - j)
                             for j in range(gap)]
                    found += all(a[p] == bg for p in cells)
    return found


def _integer(value, name, lo, hi):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)) and lo <= int(value) <= hi,
             f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _partition(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(int(v) for v in rng.choice(np.arange(1, total), count - 1,
                                             replace=False))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _count_vector(rng, length, allow_three):
    maximum = min(length, 3 if allow_three and length <= 6 else 2)
    while True:
        counts = [int(v) for v in rng.integers(1, maximum + 1, size=4)]
        if sum(counts) <= 8:
            return counts


def _make_pair(rng, length, counts, teaching=False):
    """Construct objects in local reflected quadrants, and the answer separately."""
    bg, center_color = (int(v) for v in rng.choice(10, 2, replace=False))
    palette = [int(v) for v in rng.permutation(
        [c for c in range(10) if c not in (bg, center_color)])]
    local_objects, words, reaches = [], [], []
    color_index = 0
    for q, count in enumerate(counts):
        gaps = _partition(rng, length, count)  # inner to outer
        if teaching:
            gaps = ([1, 3], [3, 1], [2, 2], [1, 3])[q]
        sizes = [1] * count
        minimum_span = 2 * count + length + count - 1
        span_cap = int(rng.integers(minimum_span, 15))
        # Size and spacing share the finite quadrant budget, but sizes never
        # change a bridge's output multiplicity.
        size_budget = (span_cap - length - (count - 1)) // 2
        for _ in range(size_budget - count):
            candidates = [i for i, s in enumerate(sizes) if s < 3]
            if not candidates or rng.random() < 0.28:
                break
            sizes[int(rng.choice(candidates))] += 1
        if teaching and q == 3:
            # A nearer wrong-colored diagonal is a deliberate teaching case.
            sizes = [1, 1]
        used = sum(2 * s + g for s, g in zip(sizes, gaps)) + count - 1
        slack = span_cap - used
        # Allocate optional leading/inter-object blank distance. Trailing
        # unused space disappears when the whole scene is cropped.
        spaces = [0] * count
        for _ in range(int(rng.integers(slack + 1))):
            spaces[int(rng.integers(count))] += 1
        if teaching and q == 3:
            spaces[1] = 0
        offset, group = 1 + spaces[0], []
        for i, (size, gap) in enumerate(zip(sizes, gaps)):
            color = palette[color_index]
            color_index += 1
            tangent = bool(rng.integers(2))
            if teaching and q == 0:
                tangent = bool(i)
            if teaching and q == 3:
                tangent = False
            d = size + gap
            starts = (((offset, offset + d), (offset + d, offset)) if tangent
                      else ((offset, offset), (offset + d, offset + d)))
            local_objects.append((q, color, size, starts))
            group.append((color, gap))
            end = offset + 2 * size + gap - 1
            offset = end + 2 + (spaces[i + 1] if i + 1 < count else 0)
        reaches.append(end)
        words.append([color for color, gap in reversed(group) for _ in range(gap)])
    top, right = max(reaches[0:2]), max(reaches[1:3])
    bottom, left = max(reaches[2:4]), max(reaches[0], reaches[3])
    # Independent exterior margins yield translations and rectangular canvases.
    h, w = top + bottom + 1, left + right + 1
    extra_y = int(rng.integers(min(4, 30 - h) + 1))
    extra_x = int(rng.integers(min(4, 30 - w) + 1))
    before_y, before_x = int(rng.integers(extra_y + 1)), int(rng.integers(extra_x + 1))
    cy, cx = top + before_y, left + before_x
    inp = np.full((h + extra_y, w + extra_x), bg, dtype=np.int8)
    inp[cy, cx] = center_color
    for q, color, size, starts in local_objects:
        sy, sx = _SIGNS[q]
        for u, v in starts:
            for du in range(size):
                for dv in range(size):
                    inp[cy + sy * (u + du), cx + sx * (v + dv)] = color
    # This answer uses the sampled lengths and ordering, not solve().
    out = np.full((length + 1, length + 1), bg, dtype=np.int8)
    perimeter = ([(0, j) for j in range(length)] +
                 [(j, length) for j in range(length)] +
                 [(length, j) for j in range(length, 0, -1)] +
                 [(j, 0) for j in range(length, 0, -1)])
    for p, color in zip(perimeter, sum(words, [])):
        out[p] = color
    out[length // 2, length // 2] = center_color
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and
             isinstance(seed, (int, np.integer)) and int(seed) >= 0,
             "seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    cap = int(rng.choice([4, 6]))
    puzzle, seen = {"train": [], "test": []}, set()
    has_three = False
    for index in range(nt + nq):
        training = index < nt
        if index == 0:
            length, counts = 4, [2, 2, 2, 2]
        elif index == 1:
            length = 2 if cap == 4 else 6
            counts = ([1, 2, 1, 2] if cap == 4 else [3, 1, 2, 1])
            counts = list(np.roll(counts, int(rng.integers(4))))
        else:
            length = int(rng.choice(np.arange(2, cap + 1, 2))) if training else cap + 2
            counts = _count_vector(rng, length, allow_three=training or has_three)
        for _ in range(32):
            pair = _make_pair(rng, length, counts, teaching=index == 0)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("failed to construct distinct inputs")
        seen.add(key)
        puzzle["train" if training else "test"].append(pair)
        if training:
            has_three |= 3 in counts
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _integer(nt, "training count", 2, 10)
    _integer(nq, "test count", 1, 4)
    _require(nt > nq, "training count must exceed test count")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs need exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            bg, center, _, records, words = _read_input(inp)
            length = len(words[0])
            _require(np.array_equal(out, _fold(words, bg, center)), "incorrect folded rim")
            # Check conservation separately from the coordinate-transfer code.
            _require(out.shape == (length + 1, length + 1), "wrong rim dimensions")
            _require(np.count_nonzero(out == center) == 1, "copy the center exactly once")
            for r in records:
                _require(np.count_nonzero(out == r["color"]) == r["gap"],
                         "each thread must contribute exactly its missing cells")
            _require(np.count_nonzero(out != bg) == 4 * length + 1,
                     "only the rim and center may be foreground")
            infos[split].append((length, records, words))
    first_length, first_records, first_words = infos["train"][0]
    first_input = np.asarray(puzzle["train"][0]["input"])
    first_bg = _read_input(first_input)[0]
    _require(all(len({r["color"] for r in first_records if r["quadrant"] == q}) >= 2
                 for q in range(4)), "first demonstration must expose radial order in all quadrants")
    _require(len({r["gap"] for r in first_records}) > 1,
             "training must distinguish bridge length from one token per pair")
    _require(any(r["gap"] != r["size"] for r in first_records),
             "training must distinguish missing length from square size")
    _require(_nearer_wrong_color(first_input, first_records, first_bg) > 0,
             "training must distinguish color pairing from nearer geometric pairing")
    _require(len({x[0] for x in infos["train"]}) >= 2,
             "training must expose output size as a measured quantity")
    maximum = max(x[0] for x in infos["train"])
    _require(all(x[0] > maximum for x in infos["test"]),
             "tests must extrapolate total missing length")
    train_max_group = max(sum(r["quadrant"] == q for r in records)
                          for _, records, _ in infos["train"] for q in range(4))
    _require(all(sum(r["quadrant"] == q for r in records) <= train_max_group
                 for _, records, _ in infos["test"] for q in range(4)),
             "test group counts need training evidence")


def self_test():
    """Bounded 42-seed structural audit, count contract and ablation witnesses."""
    shapes, layouts, palettes, sizes, gaps, groups, lengths = (set() for _ in range(7))
    tangent = radial = counterfactual_a = counterfactual_b = tempting = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            bg, center, (cy, cx), records, words = _read_input(inp)
            tempting += _nearer_wrong_color(inp, records, bg) > 0
            _require(np.array_equal(solve(inp.tolist()), out), "input-only solve failed")
            shapes.add(inp.shape)
            palettes.add((bg, center))
            lengths.add(len(words[0]))
            sizes.update(r["size"] for r in records)
            gaps.update(r["gap"] for r in records)
            groups.add(tuple(sum(r["quadrant"] == q for r in records) for q in range(4)))
            layouts.add((inp.shape, cy, cx, tuple(sorted(
                (r["quadrant"], r["radius2"], r["size"], r["gap"], r["slope"])
                for r in records))))
            for r in records:
                sy, sx = _SIGNS[r["quadrant"]]
                radial += r["slope"] == sy * sx
                tangent += r["slope"] != sy * sx
            if all(sum(r["quadrant"] == q for r in records) == 2 for q in range(4)):
                # Ignore A's metric: assign equal multiplicity to all pairs.
                equal_words = []
                for q in range(4):
                    ordered = sorted((r for r in records if r["quadrant"] == q),
                                     key=lambda r: -r["radius2"])
                    equal_words.append([r["color"] for r in ordered
                                        for _ in range(len(words[0]) // 2)])
                counterfactual_a += not np.array_equal(_fold(equal_words, bg, center), out)
            backwards = []
            for q in range(4):
                ordered = sorted((r for r in records if r["quadrant"] == q),
                                 key=lambda r: r["radius2"])
                backwards.append([r["color"] for r in ordered for _ in range(r["gap"])])
            counterfactual_b += not np.array_equal(_fold(backwards, bg, center), out)
    _require(len(shapes) >= 30 and len(layouts) >= 180 and len(palettes) >= 40,
             "insufficient geometry and role variation")
    _require(sizes == {1, 2, 3} and gaps >= {1, 2, 3, 4, 6, 8} and
             lengths == {2, 4, 6, 8} and len(groups) >= 25,
             "insufficient size, length, or group-count variation")
    _require(min(radial, tangent) >= 100 and min(counterfactual_a, counterfactual_b) >= 42,
             "both inherited mechanisms need distinguishing witnesses")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345 + nt * 10 + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "wrong counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4),
                   (2.0, 1), (True, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p1, p2 = generate_puzzle(98765, 10, 4), generate_puzzle(98765, 10, 4)
    _require(all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
                 for a, b in zip(p1[s], p2[s]) for k in ("input", "output")),
             "seed does not reproduce the whole puzzle")
    # Valid larger members outside the production sampler's 2..8 length range.
    for length in (10, 12):
        pair = _make_pair(np.random.default_rng(500 + length), length, [1, 1, 1, 1])
        predicted = solve(pair["input"])
        _require(np.array_equal(predicted, pair["output"]) and
                 predicted.shape == (length + 1, length + 1),
                 "solver overfits sampled rim lengths")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "geometries": len(layouts), "palette_roles": len(palettes),
            "group_count_vectors": len(groups), "bridge_lengths": sorted(gaps),
            "rim_lengths": sorted(lengths), "radial_bridges": radial,
            "tangential_bridges": tangent, "A_ablation_witnesses": counterfactual_a,
            "B_ablation_witnesses": counterfactual_b,
            "nearer_wrong_color_scenes": tempting}
