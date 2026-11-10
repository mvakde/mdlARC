"""Corner-wave consensus on a lattice whose rooms may have unequal sizes.

Only the standard library and NumPy are used.  No parent code or data is loaded.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside the legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a


def _bands(length, lines):
    bands, start = [], 0
    for line in lines:
        if not start < line < length - 1:
            raise ValueError("Dividers must be internal, separated, and one cell wide")
        bands.append((start, line))
        start = line + 1
    bands.append((start, length))
    return tuple(bands)


def _corners(rows, cols):
    return ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1))


def _parse(input_grid):
    """Read all conventions from this input, including the background color."""
    grid = _grid(input_grid)
    candidates = []
    for raw in np.unique(grid):
        color = int(raw)
        lines_r = tuple(int(x) for x in np.flatnonzero(np.all(grid == color, axis=1)))
        lines_c = tuple(int(x) for x in np.flatnonzero(np.all(grid == color, axis=0)))
        if len(lines_r) >= 2 and len(lines_c) >= 2:
            candidates.append((color, lines_r, lines_c))
    if len(candidates) != 1:
        raise ValueError("Expected one unambiguous complete divider lattice")
    wall, lines_r, lines_c = candidates[0]
    rb, cb = _bands(grid.shape[0], lines_r), _bands(grid.shape[1], lines_c)
    rooms = np.empty((len(rb), len(cb)), dtype=np.int8)
    for i, (a, b) in enumerate(rb):
        for j, (c, d) in enumerate(cb):
            block = grid[a:b, c:d]
            if not np.all(block == block[0, 0]):
                raise ValueError("Every compartment must be solid")
            rooms[i, j] = block[0, 0]
    bg = int(rooms[1, 1])
    corner_colors = tuple(int(rooms[p]) for p in _corners(*rooms.shape))
    if bg == wall or bg in corner_colors or wall in corner_colors:
        raise ValueError("Background, dividers, and source ink must be distinct")
    if not np.all(rooms[1:-1, 1:-1] == bg):
        raise ValueError("Input interior compartments must be empty")
    if any(int(v) not in {bg, *corner_colors} for v in rooms.ravel()):
        raise ValueError("Edge decorations must use a corner ink color")
    return grid, wall, bg, rb, cb, rooms


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the divider geometry and visible corner ink of the input."""
    grid, _, bg, rb, cb, rooms = _parse(input_grid)
    origins = _corners(*rooms.shape)
    output = grid.copy()
    for i in range(1, len(rb) - 1):
        for j in range(1, len(cb) - 1):
            distances = [max(abs(i - r), abs(j - c)) for r, c in origins]
            distance = min(distances)
            nearest_inks = {int(rooms[p]) for p, d in zip(origins, distances)
                            if d == distance}
            ink = next(iter(nearest_inks)) if distance % 2 == 0 and len(nearest_inks) == 1 else bg
            a, b = rb[i]
            c, d = cb[j]
            output[a:b, c:d] = ink
    return output


def _wave_field(shape, origins, inks, bg):
    """Vectorized label construction, also useful for explicit shortcut audits."""
    rr, cc = np.indices(shape)
    distances = np.stack([np.maximum(abs(rr - r), abs(cc - c)) for r, c in origins])
    nearest = distances.min(axis=0)
    tied = distances == nearest
    low = np.where(tied, np.asarray(inks)[:, None, None], 10).min(axis=0)
    high = np.where(tied, np.asarray(inks)[:, None, None], -1).max(axis=0)
    field = np.where((nearest % 2 == 0) & (low == high), low, bg).astype(np.int8)
    return field, nearest, tied.sum(axis=0), low == high


def _logical_output(rooms, bg):
    origins = _corners(*rooms.shape)
    field, _, _, _ = _wave_field(rooms.shape, origins, [int(rooms[p]) for p in origins], bg)
    result = rooms.copy()
    result[1:-1, 1:-1] = field[1:-1, 1:-1]
    return result


def _expand(rooms, row_sizes, col_sizes, wall):
    h, w = sum(row_sizes) + len(row_sizes) - 1, sum(col_sizes) + len(col_sizes) - 1
    grid = np.full((h, w), wall, dtype=np.int8)
    r = 0
    for i, height in enumerate(row_sizes):
        c = 0
        for j, width in enumerate(col_sizes):
            grid[r:r + height, c:c + width] = rooms[i, j]
            c += width + 1
        r += height + 1
    return grid


def _sizes(rng, count, mode=None):
    """Allow equal, mirrored, alternating, free, and strongly stretched bands."""
    budget = 31 - count
    extra = budget - count
    mode = int(rng.integers(5)) if mode is None else mode
    if mode == 0:
        return (int(rng.integers(1, budget // count + 1)),) * count
    if mode == 1:
        sizes = [1] * count
        for i in rng.permutation((count + 1) // 2):
            i = int(i)
            j = count - 1 - i
            cost = 1 if i == j else 2
            amount = int(rng.integers((budget - sum(sizes)) // cost + 1))
            sizes[i] += amount
            if i != j:
                sizes[j] += amount
        return tuple(sizes)
    if mode == 2:
        choices = [(a, b) for a in range(1, 7) for b in range(1, 7)
                   if sum(a if i % 2 == 0 else b for i in range(count)) <= budget]
        a, b = choices[int(rng.integers(len(choices)))]
        return tuple(a if i % 2 == 0 else b for i in range(count))
    if mode == 3:
        total = int(rng.integers(extra + 1))
        weights = rng.random(count) + .15
        return tuple(int(x) for x in 1 + rng.multinomial(total, weights / weights.sum()))
    sizes = [1] * count
    # A stretched band may be at an edge or inside: size cannot label a role.
    selected = int(rng.integers(count))
    sizes[selected] += int(rng.integers(max(0, extra // 2), extra + 1))
    remaining = budget - sum(sizes)
    if remaining:
        sizes = [x + int(y) for x, y in zip(sizes, rng.multinomial(remaining, [1 / count] * count))]
    return tuple(sizes)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, 6, replace=False))


def _corner_inks(rng, palette, anchor=False):
    a, b, c, d = (int(x) for x in rng.permutation(palette[2:]))
    if anchor:
        return (a, b, a, c)
    mode = int(rng.integers(6))
    choices = ((a, b, c, d), (a, b, a, c), (a, b, a, b),
               (a, a, a, b), (a, a, a, a), (a, b, c, a))
    # Geometric rotations elsewhere make all orientations of these patterns legal.
    return choices[mode]


def _make_pair(rng, shape, palette, rotation, anchor=False, test=False):
    rows, cols = shape
    bg, wall = palette[:2]
    rooms = np.full(shape, bg, dtype=np.int8)
    inks = _corner_inks(rng, palette, anchor)
    for p, ink in zip(_corners(rows, cols), inks):
        rooms[p] = ink
    edge = [(i, j) for i in range(rows) for j in range(cols)
            if (i in (0, rows - 1)) != (j in (0, cols - 1))]
    if anchor:
        # This false emitter changes the all-colored-sources shortcut at (1,2).
        rooms[0, 2] = inks[1]
        # This blank edge room would be painted if the interior-only gate vanished.
        protected = {(0, 2), (rows // 2, 0), (1, 2)}
        edge = [p for p in edge if p not in protected]
        number = int(rng.integers(0, min(4, len(edge)) + 1))
    else:
        number = int(rng.integers(0, min(7, len(edge)) + 1))
    if number:
        for index in rng.choice(len(edge), number, replace=False):
            rooms[edge[int(index)]] = int(rng.choice(inks))
    row_mode = None
    col_mode = None
    if test and rng.random() < .35:
        row_mode, col_mode = 0, 4
    row_sizes, col_sizes = _sizes(rng, rows, row_mode), _sizes(rng, cols, col_mode)
    source = _expand(rooms, row_sizes, col_sizes, wall)
    target = _expand(_logical_output(rooms, bg), row_sizes, col_sizes, wall)
    return {"input": np.rot90(source, rotation).copy(),
            "output": np.rot90(target, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = _palette(rng)
    varying_palette = bool(rng.integers(2))
    orientation = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    seen, train_shapes = set(), set()
    for index in range(num_train + num_test):
        training = index < num_train
        anchor = index < 2
        for attempt in range(128):
            local = _palette(rng) if varying_palette and index else palette
            if varying_palette and index == 1 and local[:2] == palette[:2]:
                local = (palette[1], palette[0], *palette[2:])
            if index == 0:
                shape = (5, int(rng.integers(7, 10)))
                rotation = orientation
            elif index == 1:
                # An even second axis supplies an odd-nearest/farther-even witness.
                shape = (9, int(rng.choice([10, 12])))
                rotation = (orientation + int(rng.choice([1, 3]))) % 4
            elif training:
                shape = tuple(int(x) for x in rng.integers(3, 14, size=2))
                rotation = int(rng.integers(4))
            else:
                mode = int(rng.integers(4))
                if mode == 0:
                    shape = tuple(int(rng.choice([4, 6, 8, 10, 12, 14])) for _ in range(2))
                elif mode == 1:
                    shape = (int(rng.choice([3, 4])), int(rng.integers(7, 16)))
                elif mode == 2:
                    shape = (int(rng.integers(12, 16)), int(rng.integers(5, 12)))
                else:
                    shape = tuple(int(x) for x in rng.integers(5, 14, size=2))
                rotation = int(rng.integers(4))
                visible_shape = shape[::-1] if rotation % 2 else shape
                if visible_shape in train_shapes:
                    continue
            pair = _make_pair(rng, shape, local, rotation, anchor, not training)
            key = _key(pair["input"])
            if key in seen or np.array_equal(pair["input"], pair["output"]):
                continue
            seen.add(key)
            split = "train" if training else "test"
            puzzle[split].append(pair)
            if training:
                train_shapes.add(_parse(pair["input"])[-1].shape)
            break
        else:
            raise RuntimeError("Could not sample a distinct nontrivial lattice in 128 attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit(pair):
    """Semantic invariants and counterfactuals; records are derived from grids."""
    grid, wall, bg, rb, cb, rooms = _parse(pair["input"])
    output = _grid(pair["output"])
    _require(output.shape == grid.shape, "Canvas size changed")
    room_output = np.empty_like(rooms)
    wall_mask = np.ones(grid.shape, dtype=bool)
    for i, (a, b) in enumerate(rb):
        for j, (c, d) in enumerate(cb):
            block = output[a:b, c:d]
            _require(np.all(block == block[0, 0]), "A room was only partially filled")
            room_output[i, j] = block[0, 0]
            wall_mask[a:b, c:d] = False
    _require(np.all(output[wall_mask] == wall), "Dividers changed")
    _require(np.array_equal(room_output[0], rooms[0])
             and np.array_equal(room_output[-1], rooms[-1])
             and np.array_equal(room_output[:, 0], rooms[:, 0])
             and np.array_equal(room_output[:, -1], rooms[:, -1]), "An edge room changed")
    origins = _corners(*rooms.shape)
    inks = [int(rooms[p]) for p in origins]
    field, distance, multiplicity, agreement = _wave_field(rooms.shape, origins, inks, bg)
    inside = np.zeros(rooms.shape, dtype=bool)
    inside[1:-1, 1:-1] = True
    _require(np.array_equal(room_output[inside], field[inside]), "Wrong nearest-wave consensus")
    _require(set(np.unique(multiplicity)).issubset({1, 2, 4}), "Impossible corner tie")
    _require(np.array_equal(solve(grid), output), "Input-only solver disagrees")
    even = distance % 2 == 0
    visible = inside & (room_output != bg)
    _require(np.any(visible), "Generated pair has no visible wave")
    # A ablation: remove alternation, retaining the same source competition.
    solid = inside & ~even & agreement
    rr, cc = np.indices(rooms.shape)
    corner_distances = np.stack([np.maximum(abs(rr - r), abs(cc - c)) for r, c in origins])
    even_candidates = np.where(corner_distances % 2 == 0, corner_distances, 100)
    first_even = even_candidates.min(axis=0)
    even_tied = even_candidates == first_even
    even_low = np.where(even_tied, np.asarray(inks)[:, None, None], 10).min(axis=0)
    even_high = np.where(even_tied, np.asarray(inks)[:, None, None], -1).max(axis=0)
    even_first = np.where((first_even < 100) & (even_low == even_high), even_low, bg)
    # B ablations: allow painting edge rooms, or let colored edge rooms emit.
    edge_gate = ~inside & (rooms == bg) & even & agreement
    colored = [tuple(int(x) for x in p) for p in np.argwhere(rooms != bg)]
    all_sources, _, _, _ = _wave_field(rooms.shape, colored, [int(rooms[p]) for p in colored], bg)
    # A physical-pixel metric is a tempting alternative on stretched lattices.
    centers_r = [(a + b - 1) / 2 for a, b in rb]
    centers_c = [(a + b - 1) / 2 for a, b in cb]
    pixel_difference = False
    for i, j in np.argwhere(inside):
        ds = [max(abs(centers_r[i] - centers_r[r]), abs(centers_c[j] - centers_c[c]))
              for r, c in origins]
        best = min(ds)
        colors = {ink for ink, d in zip(inks, ds) if d == best}
        predicted = next(iter(colors)) if best.is_integer() and int(best) % 2 == 0 and len(colors) == 1 else bg
        pixel_difference |= predicted != room_output[i, j]
    return {"shape": rooms.shape, "sizes": (tuple(b - a for a, b in rb), tuple(d - c for c, d in cb)),
            "palette": (bg, wall), "inks": tuple(inks), "distances": set(int(x) for x in distance[visible]),
            "parity_witness": bool(np.any(solid)), "gate_witness": bool(np.any(edge_gate)),
            "false_emitter_witness": bool(np.any((all_sources != room_output) & inside)),
            "agree_tie": bool(np.any(visible & (multiplicity > 1))),
            "disagree_tie": bool(np.any(inside & even & ~agreement)),
            "unique_even": bool(np.any(visible & (multiplicity == 1))),
            "nearest_before_parity": bool(np.any(inside & (even_first != room_output))),
            "pixel_difference": bool(pixel_difference)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            records[split].append(_audit(pair))
    train = records["train"]
    for record in train[:2]:
        _require(all(record[k] for k in ("parity_witness", "gate_witness", "false_emitter_witness",
                                         "agree_tie", "disagree_tie", "unique_even")),
                 "A foundational demonstration lacks a conceptual witness")
    _require({2, 4}.issubset(set().union(*(x["distances"] for x in train))),
             "Training must show more than the first even shell")
    _require(any(x["pixel_difference"] for x in train), "Training does not disprove pixel distance")
    _require(any(x["nearest_before_parity"] for x in train), "Training does not disprove skipping nearer gaps")
    shapes = {x["shape"] for x in train}
    _require(all(x["shape"] not in shapes for x in records["test"]), "Test lacks a new lattice shape")
    all_local = {x["palette"] for s in records.values() for x in s}
    if len(all_local) > 1:
        _require(len({x["palette"] for x in train}) > 1, "Palette variation was not demonstrated")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed variation study plus legal counts and negative checks."""
    dimensions, shapes, layouts, patterns, fingerprints = set(), set(), set(), set(), set()
    stable = varied = regular = stretched = colored_edges = high_shell = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        local_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _audit(pair)
                dimensions.add(pair["input"].shape)
                shapes.add(info["shape"])
                layouts.add(info["sizes"])
                inks = info["inks"]
                roles = {}
                patterns.add(tuple(roles.setdefault(c, len(roles)) for c in inks))
                local_palettes.add(info["palette"])
                regular += all(len(set(sizes)) == 1 for sizes in info["sizes"])
                stretched += any(max(sizes) >= min(sizes) + 4 for sizes in info["sizes"])
                high_shell += 6 in info["distances"]
                parsed = _parse(pair["input"])
                rooms, bg = parsed[-1], parsed[2]
                colored_edges += np.count_nonzero(rooms != bg) > 4
        stable += len(local_palettes) == 1
        varied += len(local_palettes) > 1
    _require(len(fingerprints) == 42, "Seeds repeat complete puzzles")
    _require(len(dimensions) >= 60 and len(layouts) >= 140 and len(shapes) >= 50,
             "Structural variation is too narrow")
    _require(len(patterns) >= 10 and stable >= 8 and varied >= 8, "Ink/palette variation is too narrow")
    _require(regular >= 5 and stretched >= 15 and colored_edges >= 100 and high_shell >= 1,
             "Missing regular/stretch/decoy/outer-shell variation")
    _require(_puzzle_key(generate_puzzle(419742161)) == _puzzle_key(generate_puzzle(419742161)),
             "Seed reproducibility failed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(81 + a + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A legal room family member need not have been sampled or be nontrivial.
    for shape in ((3, 3), (3, 15), (15, 15)):
        rooms = np.zeros(shape, dtype=np.int8)
        for p, color in zip(_corners(*shape), (1, 2, 1, 3)):
            rooms[p] = color
        grid = _expand(rooms, (1,) * shape[0], (1,) * shape[1], 5)
        answer = solve(grid)
        _require(np.array_equal(answer, _expand(_logical_output(rooms, 0),
                                                (1,) * shape[0], (1,) * shape[1], 5)),
                 "Unsampled boundary family member failed")
    # Hand-specified logical decisions test semantics apart from the label kernel.
    rooms = np.zeros((5, 7), dtype=np.int8)
    for p, color in zip(_corners(5, 7), (1, 2, 1, 3)):
        rooms[p] = color
    rooms[0, 2] = 2
    rs, cs = (2, 1, 3, 1, 2), (1, 2, 1, 4, 1, 2, 1)
    grid = _expand(rooms, rs, cs, 5)
    answer = solve(grid)
    rb = _parse(grid)[3]
    cb = _parse(grid)[4]
    for (i, j), expected in { (1, 1): 0, (1, 2): 1, (1, 4): 2, (2, 1): 1,
                              (2, 5): 0, (3, 4): 3, (2, 0): 0, (0, 2): 2}.items():
        _require(int(answer[rb[i][0], cb[j][0]]) == expected,
                 "A hand-checked parity, consensus, source, or boundary decision failed")
    # Validator must detect output corruption, not just check shapes.
    puzzle = generate_puzzle(14, 2, 1)
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    g, _, bg, rb, cb, _ = _parse(corrupted["test"][0]["input"])
    r, c = rb[1][0], cb[1][0]
    corrupted["test"][0]["output"][r, c] = (int(corrupted["test"][0]["output"][r, c]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Validator missed a corrupted interior room")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations, semantic audits)")
