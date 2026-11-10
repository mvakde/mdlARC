"""Glyph pickup streams: input-local symbol lookup along tick-shifted paths.

Only the standard library and NumPy are used.  No parent module is imported.
"""
from __future__ import annotations

import itertools
import numpy as np


GLYPHS = (
    ("111", "010", "010"), ("110", "010", "011"),
    ("011", "010", "110"), ("100", "100", "111"),
    ("001", "001", "111"), ("101", "101", "111"),
    ("111", "100", "111"), ("111", "001", "111"),
    ("110", "011", "010"), ("011", "110", "010"),
    ("010", "111", "101"), ("101", "111", "010"),
)
HOST_SEEDS = (
    0, 39057738, 90363946, 193113817, 453323332, 497479999, 285016401,
    974213386, 82807738, 375071314, 647561610, 504006711, 805635406,
    189924868, 311029057, 45228673, 718653544, 283793461, 569457198,
    709279743, 120479557, 677731238, 377819991, 432186282, 597323346,
    100872338, 454118003, 96305483, 441614724, 76249722, 2803143,
    410548019, 462379221, 668536071, 491885160, 713067491, 887970577,
    852716518, 253292031, 822340458, 823443868, 928632968,
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or a.dtype.kind not in "iu" or
            not all(1 <= n <= 30 for n in a.shape) or
            np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _mask_key(mask):
    return tuple(int(x) for x in np.asarray(mask).ravel())


def _grid_key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _normalize(grid):
    """The full ink separator and blank five-row legend determine rotation."""
    grid = _array(grid)
    bg = int(grid[0, 0])
    choices = []
    for k in range(4):
        a = np.rot90(grid, k)
        if a.shape[0] < 23 or a.shape[1] < 16:
            continue
        if (np.all(a[0] == bg) and np.all(a[4] == bg) and
                np.all(a[5] == a[5, 0]) and int(a[5, 0]) != bg):
            choices.append((a.copy(), k))
    if len(choices) != 1:
        raise ValueError("input needs one oriented legend separator")
    return choices[0]


def _decode(grid):
    a, rotation = _normalize(grid)
    bg, ink = int(a[0, 0]), int(a[5, 0])
    board = a[6:]
    length, width = board.shape
    if not (17 <= length <= 24 and 16 <= width <= 30):
        raise ValueError("board dimensions outside the family")
    if any(int(a[y, x]) != bg for y, x in
           ((0, 0), (0, width - 1), (len(a) - 1, 0), (len(a) - 1, width - 1))):
        raise ValueError("corners must be background")

    keys, key_masks = {}, {}
    header = a[:5]
    key_colors = sorted(set(int(v) for v in np.unique(header)) - {bg})
    if len(key_colors) not in (2, 3) or ink in key_colors:
        raise ValueError("legend must contain two or three colored glyphs")
    for color in key_colors:
        ys, xs = np.where(header == color)
        y, x = int(ys.min()), int(xs.min())
        if y != 1 or int(ys.max()) != 3 or int(xs.max()) - x != 2:
            raise ValueError("legend glyph must have a full 3 by 3 bounding box")
        mask = header[1:4, x:x + 3] == color
        if np.any((header[1:4, x:x + 3] != bg) & ~mask):
            raise ValueError("mixed legend tile")
        signature = _mask_key(mask)
        if signature in keys:
            raise ValueError("ambiguous glyph colors")
        keys[signature] = color
        key_masks[color] = mask

    seeds = sorted(set(int(v) for v in np.unique(board[0])) - {bg})
    if len(seeds) != 1:
        raise ValueError("the start edge must contain one signal color")
    signal = seeds[0]
    positions = tuple(int(v) for v in np.flatnonzero(board[0] == signal))
    if not (1 <= len(positions) <= 5 and all(1 <= p <= width - 2 for p in positions)):
        raise ValueError("invalid signal seeds")
    sides = []
    for column, delta in ((0, 1), (width - 1, -1)):
        values = set(int(v) for v in np.unique(board[1:, column])) - {bg}
        if values:
            if len(values) != 1:
                raise ValueError("mixed control edge")
            sides.append((column, delta, values.pop()))
    if len(sides) != 1:
        raise ValueError("one adjacent control edge is required")
    control_column, delta, control = sides[0]
    if len({bg, ink, signal, control, *key_colors}) != 4 + len(key_colors):
        raise ValueError("semantic palette roles must be distinct")
    ticks = tuple(int(v) for v in np.flatnonzero(board[:, control_column] == control))
    if not (1 <= len(ticks) <= 7 and all(2 <= t <= length - 2 for t in ticks)):
        raise ValueError("invalid tick schedule")

    # Rebuild the board to reject noise, overlapping stations and unmarked ink.
    expected = np.full_like(board, bg)
    expected[0, list(positions)] = signal
    expected[list(ticks), control_column] = control
    occupied = np.zeros_like(board, dtype=bool)
    stations = []
    for row, col in np.argwhere(board[1:, 1:-1] == control):
        row, col = int(row) + 1, int(col) + 1
        if not (2 <= row <= length - 5 and 2 <= col <= width - 3):
            raise ValueError("invalid station port position")
        if occupied[row:row + 4, col - 1:col + 2].any():
            raise ValueError("overlapping station tiles")
        occupied[row:row + 4, col - 1:col + 2] = True
        glyph = board[row + 1:row + 4, col - 1:col + 2] == ink
        signature = _mask_key(glyph)
        if signature not in keys:
            raise ValueError("station glyph is absent from the visible legend")
        expected[row, col] = control
        expected[row + 1:row + 4, col - 1:col + 2][glyph] = ink
        stations.append((row, col, keys[signature], signature))
    if not 4 <= len(stations) <= 6 or not np.array_equal(expected, board):
        raise ValueError("board must consist of seeds, ticks and 4..6 station tiles")
    sides_clockwise = ("top", "right", "bottom", "left")
    orientation = (sides_clockwise[rotation],
                   sides_clockwise[(rotation + (3 if delta == 1 else 1)) % 4])
    return dict(grid=a, rotation=rotation, board=board, bg=bg, ink=ink,
                signal=signal, control=control, keys=keys, key_masks=key_masks,
                positions=positions, ticks=ticks, delta=delta, stations=stations,
                orientation=orientation)


def _paint(info, *, shifts=True, pickups=True, inclusive=True, forget=False):
    """Sweep recurrence.  Optional ablations are used only by validation."""
    a = info["grid"].copy()
    out = a[6:]
    board = info["board"]
    length, width = board.shape
    ports = {(r, c): color for r, c, color, _ in info["stations"]}
    active = [[p, info["signal"]] for p in info["positions"]]
    ticks = set(info["ticks"])
    for row in range(length):
        if shifts and inclusive and row in ticks:
            for track in active:
                track[0] += info["delta"]
        active = [track for track in active if 0 <= track[0] < width]
        for track in active:
            col, color = track
            if forget:
                color = info["signal"]
            if pickups:
                color = ports.get((row, col), color)
            track[1] = color
            if int(board[row, col]) == info["bg"]:
                out[row, col] = color
        if shifts and not inclusive and row in ticks:
            for track in active:
                track[0] += info["delta"]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all conventions from this input, then propagate colored track states."""
    info = _decode(input_grid)
    return np.rot90(_paint(info), -info["rotation"]).copy()


def _history_check(info):
    """Closed-form positions plus last-hit history, separate from sweep painting."""
    board, bg = info["board"], info["bg"]
    length, width = board.shape
    prefix = np.cumsum([int(r in info["ticks"]) for r in range(length)])
    expected = info["grid"].copy()
    hits, missed = [], set((r, c) for r, c, _, _ in info["stations"])
    for start in info["positions"]:
        visits = sorted((r, color) for r, c, color, _ in info["stations"]
                        if start + info["delta"] * int(prefix[r]) == c)
        hits.append(visits)
        for r, c, _, _ in info["stations"]:
            if start + info["delta"] * int(prefix[r]) == c:
                missed.discard((r, c))
        for row in range(length):
            col = start + info["delta"] * int(prefix[row])
            if not 0 <= col < width or int(board[row, col]) != bg:
                continue
            past = [color for rank, color in visits if rank <= row]
            expected[row + 6, col] = past[-1] if past else info["signal"]
    clipped = sum(not 0 <= p + info["delta"] * len(info["ticks"]) < width
                  for p in info["positions"])
    return expected, hits, missed, clipped


def _transform(grid, code):
    a = np.fliplr(grid) if code % 2 else grid
    return np.rot90(a, code // 2).copy()


def _legend(rng, width, glyphs, palette):
    bg, ink, _, _, colors = palette
    a = np.full((6, width), bg, dtype=np.int8)
    a[5] = ink
    count = len(glyphs)
    max_gap = min(3, (width - 2 - 3 * count) // (count - 1))
    gap = int(rng.integers(1, max_gap + 1))
    span = 3 * count + gap * (count - 1)
    start = int(rng.integers(1, width - span))
    for slot, index in enumerate(rng.permutation(count)):
        x = start + slot * (3 + gap)
        a[1:4, x:x + 3][glyphs[int(index)]] = colors[int(index)]
    return a


def _positions(rng, width, tick_count, count, witness):
    candidates = []
    styles = ("comb", "irregular", "grouped") if witness else (
        "run", "comb", "irregular", "grouped")
    style = styles[int(rng.integers(len(styles)))]
    for _ in range(80):
        if style == "run":
            relative = list(range(count))
        elif style == "comb" and count > 1:
            gap = int(rng.integers(2, max(3, (width - 3) // (count - 1) + 1)))
            relative = [i * gap for i in range(count)]
        elif style == "grouped" and count >= 3:
            split = int(rng.integers(1, count))
            gap = int(rng.integers(3, 7))
            relative = list(range(split)) + [split + gap + i for i in range(count - split)]
        else:
            relative = sorted(int(x) for x in rng.choice(width - 2, count, replace=False))
            relative = [x - relative[0] for x in relative]
        if relative[-1] > width - 3:
            continue
        offset = int(rng.integers(1, width - 1 - relative[-1]))
        p = tuple(x + offset for x in relative)
        safe = [x for x in p if 2 <= x and x + tick_count <= width - 3]
        anchors = (list(itertools.permutations(safe, 2)) if witness else
                   [(x, y) for x in safe for y in safe])
        if witness:
            anchors = [(x, y) for x, y in anchors if abs(x - y) >= 5]
        if anchors:
            candidates.append((p, anchors))
    if not candidates:
        # Complete bounded fallback, including the narrowest witness canvas.
        for p in itertools.combinations(range(1, width - 1), count):
            safe = [x for x in p if 2 <= x and x + tick_count <= width - 3]
            anchors = ([(x, y) for x in safe for y in safe if abs(x - y) >= 5]
                       if witness else [(x, y) for x in safe for y in safe])
            if anchors:
                candidates.append((p, anchors))
                break
    if not candidates:
        raise ValueError("cannot fit safe pickup anchors")
    positions, anchors = candidates[int(rng.integers(len(candidates)))]
    primary, secondary = anchors[int(rng.integers(len(anchors)))]
    return positions, primary, secondary


def _make_world(rng, glyphs, palette, witness=False):
    bg, ink, signal, control, colors = palette
    length = int(rng.integers(17, 25))
    width = int(rng.integers(18 if witness else 16, 31))
    rows = []
    for a in range(2, length - 14):
        for b in range(a + 5, length - 9):
            for c in range(b + 5, length - 4):
                rows.append((a, b, c))
    r0, r1, r2 = rows[int(rng.integers(len(rows)))]
    tick_count = int(rng.integers(2, min(6 if witness else 8, width - 10)))
    # The first station band is also the first shift: old and new lanes differ.
    available = np.arange(r0 + 1, length - 1)
    if rng.random() < 0.3:
        remaining = list(range(r0 + 1, r0 + tick_count))
    else:
        remaining = [int(v) for v in rng.choice(available, tick_count - 1, replace=False)]
    ticks = tuple(sorted([r0] + remaining))
    count = int(rng.integers(2, 5)) if witness else int(rng.integers(1, 6))
    positions, primary, secondary = _positions(rng, width, tick_count, count, witness)
    def lane(start, row):
        return start + sum(t <= row for t in ticks)
    board = np.full((length, width), bg, dtype=np.int8)
    board[0, list(positions)] = signal
    board[list(ticks), 0] = control
    stations = []
    occupied = np.zeros_like(board, dtype=bool)
    def add(row, col, glyph_index):
        if not (2 <= row <= length - 5 and 2 <= col <= width - 3):
            return False
        tile = occupied[row:row + 4, col - 1:col + 2]
        if tile.any():
            return False
        tile[:] = True
        board[row, col] = control
        board[row + 1:row + 4, col - 1:col + 2][glyphs[glyph_index]] = ink
        stations.append((row, col, glyph_index))
        return True
    order = list(range(len(glyphs))) if witness else [int(v) for v in rng.permutation(len(glyphs))]
    _require(add(r0, lane(primary, r0), order[0]), "first station placement")
    _require(add(r1, lane(secondary, r1), order[1]), "middle station placement")
    _require(add(r2, lane(primary, r2), order[-1]), "last station placement")
    # Prefer the old secondary lane.  It is a deliberately missed pickup at a tick.
    misses = [(r0, secondary)]
    misses += [(r, c) for r in range(2, length - 4) for c in range(2, width - 2)]
    # Keep the diagnostic preference, randomize the general fallback.
    fallback = misses[1:]
    rng.shuffle(fallback)
    for row, col in misses[:1] + fallback:
        if all(lane(p, row) != col for p in positions):
            if add(row, col, int(rng.integers(len(glyphs)))):
                break
    else:
        raise ValueError("cannot fit a missed station")
    target_count = 4 if witness else int(rng.integers(4, 7))
    for _ in range(100):
        if len(stations) >= target_count:
            break
        row = int(rng.integers(2, length - 4))
        if rng.random() < 0.55:
            col = lane(int(rng.choice(positions)), row)
        else:
            col = int(rng.integers(2, width - 2))
        add(row, col, int(rng.integers(len(glyphs))))
    return np.vstack((_legend(rng, width, glyphs, palette), board))


def _swap_station_codes(world, colors):
    info = _decode(world)
    a = info["grid"].copy()
    for row, col, color, _ in info["stations"]:
        replacement = colors[(colors.index(color) + 1) % len(colors)]
        tile = a[row + 7:row + 10, col - 1:col + 2]
        tile[:] = info["bg"]
        tile[info["key_masks"][replacement]] = info["ink"]
    return a


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    key_count = 2 if num_train < 4 else int(rng.integers(2, 4))
    # Equal-area codes force a shape lookup rather than a cell-count shortcut.
    weights = [sum(row.count("1") for row in glyph) for glyph in GLYPHS]
    first_index = int(rng.integers(len(GLYPHS)))
    second_index = int(rng.choice([i for i in range(len(GLYPHS))
                                   if i != first_index and weights[i] == weights[first_index]]))
    indices = [first_index, second_index]
    if key_count == 3:
        indices.append(int(rng.choice([i for i in range(len(GLYPHS)) if i not in indices])))
    glyphs = [np.array([[int(c) for c in row] for row in GLYPHS[int(i)]], dtype=bool)
              for i in indices]
    colors = [int(v) for v in rng.permutation(10)]
    palette = (colors[0], colors[1], colors[2], colors[3], colors[4:4 + key_count])
    base_code = int(rng.integers(8))
    # A rotation of the same handedness is always held out, including 2+1 puzzles.
    held_out = int(rng.choice([c for c in range(8) if c % 2 == base_code % 2 and c != base_code]))
    codes = [base_code, base_code]
    if num_train >= 3:
        codes.append(int(rng.choice([c for c in range(8) if c % 2 != base_code % 2])))
    while len(codes) < num_train:
        codes.append(int(rng.choice([c for c in range(8) if c != held_out])))
    codes.append(held_out)
    for _ in range(1, num_test):
        eligible = list(range(8)) if num_train >= 3 else [c for c in range(8) if c % 2 == base_code % 2]
        codes.append(int(rng.choice(eligible)))
    vary_bg = bool(rng.integers(2)) and num_train >= 3
    bg_choices = [v for v in range(10) if v not in {palette[1], palette[2], palette[3], *palette[4]}]
    puzzle, seen = {"train": [], "test": []}, set()
    base = _make_world(rng, glyphs, palette, witness=True)
    worlds = [base, _swap_station_codes(base, palette[4])]
    for index, code in enumerate(codes):
        for attempt in range(64):
            if index < 2:
                world = worlds[index]
            else:
                pair_bg = int(rng.choice(bg_choices)) if vary_bg else palette[0]
                pair_palette = (pair_bg, *palette[1:])
                world = _make_world(rng, glyphs, pair_palette)
            inp = _transform(world, code)
            key = _grid_key(inp)
            if key in seen:
                continue
            output = solve(inp)
            info = _decode(inp)
            # Select examples that visibly expose both mechanisms and tick timing.
            wanted = _paint(info)
            visible = set(int(v) for v in wanted[info["grid"] == info["bg"]])
            if len(visible & set(info["key_masks"])) < 2:
                continue
            if any(np.array_equal(wanted, _paint(info, **ablation)) for ablation in
                   ({"shifts": False}, {"pickups": False}, {"inclusive": False},
                    {"forget": True})):
                continue
            seen.add(key)
            split = "train" if index < num_train else "test"
            puzzle[split].append({"input": inp, "output": output})
            break
        else:
            raise ValueError("failed to construct a distinct informative example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _codebook_signature(info, reflected=False):
    return tuple(sorted((color, _mask_key(np.fliplr(mask) if reflected else mask))
                        for color, mask in info["key_masks"].items()))


def _geometry(info):
    return (info["board"].shape, info["positions"], info["ticks"], info["delta"],
            tuple(sorted((r, c) for r, c, _, _ in info["stations"])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "example counts")
    seen, infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _array(pair["input"]), _array(pair["output"])
            _require(inp.shape == out.shape, "canvas size preservation")
            _require(_grid_key(inp) not in seen, "duplicate input")
            seen.add(_grid_key(inp))
            info = _decode(inp)
            expected, hits, missed, _ = _history_check(info)
            canonical_out = np.rot90(out, info["rotation"])
            _require(np.array_equal(canonical_out, expected), "last-hit history rule")
            _require(np.array_equal(_paint(info), expected), "sweep/history disagreement")
            fixed = info["grid"] != info["bg"]
            _require(np.array_equal(canonical_out[fixed], info["grid"][fixed]), "marker preservation")
            visible = set(int(v) for v in canonical_out[~fixed])
            _require(len(visible & set(info["key_masks"])) >= 2, "fewer than two visible lookup colors")
            _require(bool(missed), "must include a bypassed station")
            _require(any(len({c for _, c in h}) >= 2 for h in hits), "later pickup must replace a color")
            for ablation, name in (({"shifts": False}, "B shifts"),
                                   ({"pickups": False}, "A lookup"),
                                   ({"inclusive": False}, "tick inclusivity"),
                                   ({"forget": True}, "persistent color state")):
                _require(not np.array_equal(expected, _paint(info, **ablation)),
                         f"example fails to expose {name}")
            infos.append(info)
    first = infos[0]
    weights = [int(mask.sum()) for mask in first["key_masks"].values()]
    _require(len(set(weights)) < len(weights), "equal-area glyph counterexample missing")
    valid_books = {_codebook_signature(first), _codebook_signature(first, True)}
    for info in infos:
        _require(_codebook_signature(info) in valid_books, "codebook changes beyond a whole-board reflection")
        _require((info["ink"], info["signal"], info["control"]) ==
                 (first["ink"], first["signal"], first["control"]), "puzzle palette roles")
    _require(_geometry(first) == _geometry(infos[1]), "first two examples must isolate glyph identity")
    a, b = infos[:2]
    shared_blank = (a["grid"] == a["bg"]) & (b["grid"] == b["bg"])
    _require(np.any(_paint(a)[shared_blank] != _paint(b)[shared_blank]), "glyph swap has no trail effect")
    train_orientations = {i["orientation"] for i in infos[:n]}
    _require(infos[n]["orientation"] not in train_orientations, "first test must hold out an orientation")
    if n >= 3:
        _require({i["delta"] for i in infos[:n]} == {-1, 1}, "training lacks opposite control sides")


def self_test():
    """Bounded: 42 prescribed seeds, legal counts, ablations and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    shapes, orientations, structures, books, bgs = set(), set(), set(), set(), set()
    signal_counts, tick_counts, station_counts, clipped_counts = set(), set(), set(), set()
    overwritten = missed_total = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            info = _decode(pair["input"])
            _, hits, missed, clipped = _history_check(info)
            shapes.add(np.asarray(pair["input"]).shape)
            orientations.add(info["orientation"])
            structures.add(_geometry(info))
            books.add(_codebook_signature(info))
            bgs.add(info["bg"])
            signal_counts.add(len(info["positions"]))
            tick_counts.add(len(info["ticks"]))
            station_counts.add(len(info["stations"]))
            clipped_counts.add(clipped)
            overwritten += sum(len({c for _, c in h}) > 1 for h in hits)
            missed_total += len(missed)
    _require(len(shapes) >= 35 and len(structures) >= 120, "insufficient structural variation")
    _require(len(orientations) == 8 and len(books) >= 35 and len(bgs) >= 8, "orientation/code/palette variation")
    _require(signal_counts == set(range(1, 6)), "signal count coverage")
    _require(len(tick_counts) >= 4 and station_counts == {4, 5, 6}, "event count coverage")
    _require(any(c > 0 for c in clipped_counts), "no clipping coverage")
    _require(overwritten >= 210 and missed_total >= 210, "interaction coverage")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(17, train, test)
            _require((len(p["train"]), len(p["test"])) == (train, test), "explicit counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (None, 1), (4, False), (4, 1.0)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    again = generate_puzzle(HOST_SEEDS[17])
    _require(all(_grid_key(p[k]) == _grid_key(q[k])
                 for split in ("train", "test")
                 for p, q in zip(puzzles[17][split], again[split])
                 for k in ("input", "output")), "seed determinism")
    # NumPy/list equivalence and purity: solving must not mutate the caller.
    inp = np.asarray(puzzles[0]["test"][0]["input"])
    before = inp.copy()
    _require(np.array_equal(solve(inp.tolist()), solve(inp)), "list input support")
    _require(np.array_equal(inp, before), "solve mutated its input")
    # Hand-built boundary probes: one tick, and all tracks clipped after pickups.
    probe_rng = np.random.default_rng(2026)
    glyphs = [np.array([[int(c) for c in row] for row in GLYPHS[i]], dtype=bool)
              for i in (1, 2)]
    palette = (0, 5, 8, 2, [1, 3])
    board = np.zeros((24, 30), dtype=np.int8)
    board[0, 24] = 8
    board[[2, 15, 16, 17, 18, 19, 20], 0] = 2
    for r, c, index in ((2, 25, 0), (7, 25, 1), (12, 25, 0), (7, 3, 1)):
        board[r, c] = 2
        board[r + 1:r + 4, c - 1:c + 2][glyphs[index]] = 5
    probe = np.vstack((_legend(probe_rng, 30, glyphs, palette), board))
    probes = [probe.copy()]
    probe[[6 + r for r in (15, 16, 17, 18, 19, 20)], 0] = 0
    probes.append(probe)
    for edge in probes:
        info = _decode(edge)
        expected, _, _, clipped = _history_check(info)
        _require(np.array_equal(solve(edge), expected), "boundary history")
        if len(info["ticks"]) == 7:
            _require(clipped == 1 and np.all(expected[-3:] == 0), "complete clipping")
        else:
            _require(len(info["ticks"]) == 1 and clipped == 0, "one-tick probe")
        for code in range(8):
            _require(np.array_equal(solve(_transform(edge, code)), _transform(expected, code)),
                     "rotation/reflection equivariance")
        permutation = probe_rng.permutation(10)
        _require(np.array_equal(solve(permutation[edge]), permutation[expected]),
                 "palette equivariance")
        # Change only a bypassed glyph; its trail has no effect.
        changed = edge.copy()
        tile = changed[14:17, 2:5]  # board port (7, 3), glyph rows 8..10
        tile[:] = 0
        tile[glyphs[0]] = 5
        blank_in_both = (edge == 0) & (changed == 0)
        _require(np.array_equal(solve(changed)[blank_in_both], expected[blank_in_both]),
                 "a missed port changed a track state")
    return {"seeds": 42, "shapes": len(shapes), "structures": len(structures),
            "orientations": len(orientations), "codebooks": len(books),
            "signal_counts": sorted(signal_counts), "tick_counts": sorted(tick_counts),
            "station_counts": sorted(station_counts), "clipped_counts": sorted(clipped_counts),
            "overwritten_tracks": overwritten, "missed_stations": missed_total}


if __name__ == "__main__":
    print(self_test())
