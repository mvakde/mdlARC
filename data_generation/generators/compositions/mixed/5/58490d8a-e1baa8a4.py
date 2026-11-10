"""Count exterior glyphs, complete repeated key rows, then compress the panel.

Only the visible input is used by solve. Sampling metadata is never a solver
argument. No files, global RNG state, or parent generators are used at runtime.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SINGLE = ((0, 0),)
_MOTIFS = (
    _DIAMOND, _SQUARE, _SINGLE,
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2)),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer)
             and bool(np.all((grid >= 0) & (grid <= 9))),
             "grid colors must be integers in 0..9")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _components(mask, diagonal=True):
    """Deterministic flood fill; different colors are segmented separately."""
    pending = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while pending:
        point = min(pending)
        pending.remove(point)
        stack, cells = [point], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                neighbor = (r + dr, c + dc)
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        result.append(cells)
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _starts(grid, axis):
    # The first line of each maximal consecutive run, never global uniqueness.
    return [i for i in range(grid.shape[axis])
            if i == 0 or not np.array_equal(np.take(grid, i, axis=axis),
                                            np.take(grid, i - 1, axis=axis))]


def _compress(grid):
    return grid[np.ix_(_starts(grid, 0), _starts(grid, 1))].copy()


def _runs(grid, axis):
    edges = _starts(grid, axis) + [grid.shape[axis]]
    return tuple(b - a for a, b in zip(edges, edges[1:]))


def _panel(input_grid):
    """Find the unique field rectangle with blank border and aligned row keys."""
    candidates = []
    for field in map(int, np.unique(input_grid)):
        rr, cc = np.where(input_grid == field)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        panel = input_grid[r0:r1, c0:c1]
        if min(panel.shape) < 3:
            continue
        if not all(np.all(edge == field) for edge in
                   (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        positions = np.argwhere(panel != field)
        if not len(positions):
            continue
        # A key is one cell PER ROW. Consecutive repeated keys may touch.
        if (len(set(map(int, positions[:, 1]))) != 1
                or len(set(map(int, positions[:, 0]))) != len(positions)):
            continue
        outside = np.ones(input_grid.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        values = input_grid[outside]
        background = int(np.argmax(np.bincount(values, minlength=10)))
        colors = tuple(int(panel[r, c]) for r, c in positions)
        if (background == field or background in colors
                or np.mean(values == background) <= .60):
            continue
        ring = np.zeros(input_grid.shape, bool)
        ring[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
        ring[r0:r1, c0:c1] = False
        if not np.all(input_grid[ring] == background):
            continue
        candidates.append(dict(field=field, background=background,
                               bounds=(r0, r1, c0, c1), panel=panel.copy(),
                               positions=positions, keys=colors, outside=outside))
    _require(len(candidates) == 1, "input must have one unambiguous key panel")
    return candidates[0]


def _complete(meta, counts):
    panel = np.full(meta["panel"].shape, meta["field"], np.int8)
    for (r, c), color in zip(meta["positions"], meta["keys"]):
        count = counts.get(color, 0)
        _require(count >= 1, "every key must have exterior objects")
        _require(c + 2 * (count - 1) < panel.shape[1] - 1,
                 "tally must fit before the terminal blank border")
        panel[r, c:c + 2 * count:2] = color
    return panel


def _inspect(input_grid):
    meta = _panel(input_grid)
    outside, bg = meta["outside"], meta["background"]
    objects, counts = {}, {}
    owners = np.full(input_grid.shape, -1, np.int16)
    serial = 0
    for color in sorted(set(map(int, input_grid[outside])) - {bg}):
        pieces = _components((input_grid == color) & outside)
        objects[color] = tuple(_normalize(cells) for cells in pieces)
        counts[color] = len(pieces)
        for cells in pieces:
            for r, c in cells:
                owners[r, c] = serial
            serial += 1
    for r, c in np.argwhere(owners >= 0):
        neighborhood = owners[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        _require(np.all((neighborhood < 0) | (neighborhood == owners[r, c])),
                 "separate exterior objects require a background halo")
    completed = _complete(meta, counts)
    meta.update(counts=counts, objects=objects, completed=completed,
                expected=_compress(completed),
                distractors=tuple(sorted(set(counts) - set(meta["keys"]))))
    return meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count only exterior eight-connected objects, tally, then run-collapse."""
    return _inspect(_grid(input_grid))["expected"].copy()


def _shape(rng):
    if rng.random() < .55:
        cells = _MOTIFS[int(rng.integers(len(_MOTIFS)))]
    else:
        cells = {(1, 1)}
        target = int(rng.integers(1, 10))
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r + dr < 3 and 0 <= c + dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    return _normalize(cells)


def _row_layout(rng, keys, field, test_long_runs=False, witness=False):
    order = list(map(int, rng.permutation(keys)))
    # The first category recurs after the other categories. With one category,
    # a blank row explicitly separates its occurrences.
    sequence = order + [order[0]]
    target = int(rng.integers(len(sequence), min(7, len(keys) + 3) + 1))
    while len(sequence) < target:
        choices = [(i, color) for i in range(len(sequence) + 1) for color in keys
                   if (i == 0 or sequence[i - 1] != color)
                   and (i == len(sequence) or sequence[i] != color)]
        if not choices:
            break
        i, color = choices[int(rng.integers(len(choices)))]
        sequence.insert(i, color)
    logical = [field]
    for i, color in enumerate(sequence):
        logical.append(color)
        if i + 1 < len(sequence) and (color == sequence[i + 1] or rng.random() < .55):
            logical.append(field)
    logical.append(field)
    lengths = [1] * len(logical)
    first_key = next(i for i, color in enumerate(logical) if color != field)
    lengths[first_key] = int(rng.integers(4, 6)) if test_long_runs else 2
    if witness:
        lengths[0] = 2
    # At most 22 physical panel rows; finite allocation, no rejection loop.
    cap = 5 if test_long_runs else 3
    budget = int(rng.integers(0, min(9, 22 - sum(lengths),
                                     cap * len(lengths) - sum(lengths)) + 1))
    for _ in range(budget):
        choices = [i for i, n in enumerate(lengths) if n < cap]
        i = choices[int(rng.integers(len(choices)))]
        lengths[i] += 1
    return np.repeat(np.asarray(logical, np.int8), lengths)


def _place(grid, blocked, color, shape, rng):
    sh = 1 + max(r for r, c in shape)
    sw = 1 + max(c for r, c in shape)
    nh, nw = grid.shape[0] - sh + 1, grid.shape[1] - sw + 1
    allowed = np.ones((nh, nw), bool)
    for r, c in shape:
        allowed &= ~blocked[r:r + nh, c:c + nw]
    choices = np.argwhere(allowed)
    if not len(choices):
        return False
    r0, c0 = choices[int(rng.integers(len(choices)))]
    for dr, dc in shape:
        r, c = int(r0 + dr), int(c0 + dc)
        grid[r, c] = color
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return True


def _make_pair(rng, regime, training, index, role_color):
    first, second = training and index == 0, training and index == 1
    if first:
        k = 2
    elif second:
        k = 3
    elif not training and regime == 1:
        k = 4
    else:
        k = int(rng.integers(1, 4 if regime == 1 else 5))
    structural = [c for c in range(10) if not ((first or second) and c == role_color)]
    bg, field = map(int, rng.choice(structural, 2, replace=False))
    pool = [c for c in range(10) if c not in (bg, field)]
    rng.shuffle(pool)
    if first:
        pool.remove(role_color)
        pool.insert(0, role_color)
    if second:
        pool.remove(role_color)
    keys = pool[:k]
    remaining = pool[k:]
    if second:
        remaining.insert(0, role_color)
    counts = {c: int(rng.integers(1, 5)) for c in keys}
    if first:
        counts[keys[0]], counts[keys[1]] = 3, int(rng.integers(1, 3))
    elif second:
        counts = dict(zip(keys, map(int, rng.permutation((1, 2, 4)))))
    elif not training and regime == 0:
        counts[keys[int(rng.integers(k))]] = int(rng.integers(5, 7))
    elif max(counts.values()) == 1:
        counts[keys[0]] = 2
    nd = int(rng.integers(0, 3))
    if second:
        nd = max(1, nd)
    distractors = remaining[:nd]
    dcounts = {c: 1 for c in distractors}
    if second:
        dcounts[role_color] = int(rng.integers(3, 6))
    spare = 20 - sum(counts.values()) - sum(dcounts.values())
    for _ in range(int(rng.integers(0, min(5, spare) + 1))):
        if distractors:
            dcounts[distractors[int(rng.integers(nd))]] += 1
    rows = _row_layout(rng, keys, field,
                       test_long_runs=(not training and regime == 2), witness=first)
    inset, trailing = int(rng.integers(1, 4)), int(rng.integers(1, 5))
    if inset == trailing == 1:
        trailing = 2
    width = inset + 2 * (max(counts.values()) - 1) + trailing + 1
    blank = np.full((len(rows), width), field, np.int8)
    completed = blank.copy()
    for r, color in enumerate(map(int, rows)):
        if color != field:
            blank[r, inset] = color
            completed[r, inset:inset + 2 * counts[color]:2] = color
    objects = []
    for color, n in {**counts, **dcounts}.items():
        prototype = _shape(rng)
        repeated = rng.random() < .25
        for j in range(n):
            shape = prototype if repeated else _shape(rng)
            if first and color == role_color and j < 3:
                shape = (_DIAMOND, _SQUARE, _SINGLE)[j]
            objects.append((color, shape))
    rng.shuffle(objects)
    objects.sort(key=lambda item: -len(item[1]))
    ph, pw = blank.shape
    _require(ph <= 22 and pw <= 18 and len(objects) <= 20, "packing budget exceeded")
    for attempt in range(5):
        fallback = attempt == 4
        if fallback:
            h, w, r0, c0 = 30, 30, 0, 0
        else:
            h = int(rng.integers(max(14, ph + 2), 31))
            w = int(rng.integers(max(15, pw + 4), 31))
            if attempt >= 2:
                h, w = max(27, h), max(27, w)
            if rng.random() < .5:
                r0 = int(rng.choice((0, h - ph)))
                c0 = int(rng.integers(w - pw + 1))
            else:
                r0 = int(rng.integers(1, h - ph))
                c0 = int(rng.integers(1, w - pw))
        grid = np.full((h, w), bg, np.int8)
        grid[r0:r0 + ph, c0:c0 + pw] = blank
        blocked = np.zeros((h, w), bool)
        blocked[max(0, r0 - 1):r0 + ph + 1, max(0, c0 - 1):c0 + pw + 1] = True
        if fallback:
            # pw <= 18 leaves at least 11 columns: three 3x3 slots per
            # four-row band, seven bands, hence >=21 slots for <=20 objects.
            # No shape, count, panel row, or label is changed by fallback.
            slots = [(r, c) for r in range(0, 28, 4)
                     for c in range(pw + 1, 28, 4)]
            _require(len(slots) >= len(objects), "fallback slot proof failed")
            rng.shuffle(slots)
            for (color, shape), (r, c) in zip(objects, slots):
                for dr, dc in shape:
                    grid[r + dr, c + dc] = color
        elif not all(_place(grid, blocked, color, shape, rng) for color, shape in objects):
            continue
        try:
            observed = _inspect(grid)
            _require(observed["counts"] == {**counts, **dcounts}, "placed object counts changed")
            _require(np.array_equal(observed["completed"], completed), "tally construction disagrees")
        except AssertionError:
            if fallback:
                raise
            continue
        return {"input": grid, "output": _compress(completed)}
    raise RuntimeError("unreachable packing failure")


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # larger counts, more categories, thicker key runs
    role_color = int(rng.integers(10))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, regime, split == "train", index, role_color)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_recurrence(grid, field):
    return any(np.any(grid[a] != field) and np.array_equal(grid[a], grid[b])
               for a in range(grid.shape[0]) for b in range(a + 2, grid.shape[0]))


def _features(meta):
    nonblank_runs = []
    starts = _starts(meta["completed"], 0)
    lengths = _runs(meta["completed"], 0)
    for r, n in zip(starts, lengths):
        if np.any(meta["completed"][r] != meta["field"]):
            nonblank_runs.append(n)
    return (max(meta["counts"][c] for c in meta["keys"]),
            len(set(meta["keys"])), max(nonblank_runs))


def _nonblank_duplicate_run(meta):
    full = meta["completed"]
    return any(n > 1 and np.any(full[r] != meta["field"])
               for r, n in zip(_starts(full, 0), _runs(full, 0)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    try:
        _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, metadata = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            meta = _inspect(inp)
            full = meta["completed"]
            _require(np.array_equal(out, meta["expected"]), "incorrect output")
            _require(out.shape[0] < full.shape[0] and out.shape[1] < full.shape[1],
                     "both compression axes must be exercised")
            _require(_nonblank_duplicate_run(meta),
                     "compression must remove repeated nonblank tally rows")
            _require(_has_recurrence(out, meta["field"]), "separated keyed recurrence missing")
            _require(not np.array_equal(_compress(meta["panel"]), out),
                     "keyed counting must affect the answer")
            rows, columns = _runs(full, 0), _runs(full, 1)
            reconstructed = np.repeat(np.repeat(out, rows, axis=0), columns, axis=1)
            _require(np.array_equal(reconstructed, full), "compression lost nonduplicate information")
            _require(len(_starts(out, 0)) == len(out)
                     and len(_starts(out, 1)) == out.shape[1], "output is not maximally compressed")
            # Physical stride-two tallying implies this width after blank
            # prefix/suffix runs collapse; it is not a sampled output width.
            _require(out.shape[1] == 2 * max(meta["counts"][c] for c in meta["keys"]) + 1,
                     "count-dependent output width is wrong")
            for row in out:
                colors = set(map(int, row)) - {meta["field"]}
                _require(len(colors) <= 1, "tally row mixes categories")
                for color in colors:
                    _require(np.count_nonzero(row == color) == meta["counts"][color],
                             "tally count not preserved by compression")
            metadata[split].append(meta)
    train = metadata["train"]
    _require(any(_DIAMOND in m["objects"].get(c, ())
                 and _SQUARE in m["objects"].get(c, ())
                 and _SINGLE in m["objects"].get(c, ())
                 for m in train for c in set(m["keys"])),
             "training needs the diagonal/unequal-area counting witness")
    _require(any(set(a["keys"]) & set(b["distractors"]) for a in train for b in train),
             "training must show pair-local key membership")
    _require(len({m["expected"].shape for m in train}) >= 2,
             "training must show changing answer dimensions")
    _require(len({m["expected"].shape[1] for m in train}) >= 2,
             "training must show count-dependent compressed widths")
    _require(any(len({m["counts"][c] for c in m["keys"]}) > 1 for m in train),
             "training must distinguish whole-vector and per-row compression")
    # Collapsing the blank panel leaves only blank/key/blank columns. It
    # cannot then hold a physical stride-two tally with count greater than
    # one and a blank terminal border. The counting witness requires three.
    _require(any(_compress(m["panel"]).shape[1]
                 < 2 * max(m["counts"][c] for c in m["keys"]) + 1 for m in train),
             "training must disambiguate compression timing")
    maxima = tuple(max(_features(m)[i] for m in train) for i in range(3))
    _require(all(any(v > maximum for v, maximum in zip(_features(m), maxima))
                 for m in metadata["test"]), "test shift is not present")


def self_test():
    """Stream 42 puzzles, retaining only small diversity sets; write no files."""
    puzzles, shapes, outputs, structures, glyphs = set(), set(), set(), set(), set()
    layouts, palettes, counts, key_counts, shifts = set(), set(), set(), set(), set()
    fields, backgrounds, edge_modes = set(), set(), set()
    pair_count, recolor_checks, row_stretch_checks = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_digest(puzzle))
        train_features = [_features(_inspect(p["input"])) for p in puzzle["train"]]
        maxima = [max(f[i] for f in train_features) for i in range(3)]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                meta = _inspect(inp)
                pair_count += 1
                shapes.add(inp.shape)
                outputs.add(out.shape)
                fields.add(meta["field"])
                backgrounds.add(meta["background"])
                palettes.add(tuple(sorted(map(int, np.unique(inp)))))
                counts.update(meta["counts"].values())
                key_counts.add(len(set(meta["keys"])))
                glyphs.update(shape for group in meta["objects"].values() for shape in group)
                layouts.add((_runs(meta["completed"], 0), _runs(meta["completed"], 1)))
                roles = {meta["field"]: 0}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in out.ravel())
                structures.add((out.shape, normalized))
                r0, r1, c0, c1 = meta["bounds"]
                edge_modes.add(r0 == 0 or c0 == 0 or r1 == len(inp) or c1 == inp.shape[1])
                _require(np.array_equal(solve(inp.copy()), out), "input-only solve failed")
                full = meta["completed"]
                # Removing duplicate rows cannot change which columns match,
                # and vice versa. Check both orders without another solver.
                row_first = full[_starts(full, 0)]
                row_first = row_first[:, _starts(row_first, 1)]
                column_first = full[:, _starts(full, 1)]
                column_first = column_first[_starts(column_first, 0)]
                _require(np.array_equal(row_first, out) and np.array_equal(column_first, out),
                         "axis-order consistency failed")
                if split == "test":
                    shifts.update(i for i, (v, maximum) in enumerate(zip(_features(meta), maxima))
                                  if v > maximum)
        # Input-only metamorphic checks exercise conventions beyond the
        # sampler's glyph catalog, with no saved grids or extra RNG state.
        pair = puzzle["train"][0]
        permutation = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
        _require(np.array_equal(solve(permutation[pair["input"]]),
                                permutation[pair["output"]]), "color-role invariance failed")
        recolor_checks += 1
        for pair in puzzle["train"]:
            inp, out = pair["input"], pair["output"]
            if inp.shape[0] == 30:
                continue
            meta = _inspect(inp)
            row = meta["bounds"][0] + int(meta["positions"][0, 0])
            stretched = np.insert(inp, row, inp[row], axis=0)
            _require(np.array_equal(solve(stretched), out), "redundant key-row invariance failed")
            row_stretch_checks += 1
            break
    for value, minimum, label in ((len(puzzles), 42, "puzzles"),
                                   (len(shapes), 50, "input geometries"),
                                   (len(outputs), 15, "output geometries"),
                                   (len(structures), 90, "color-normalized tally structures"),
                                   (len(glyphs), 40, "glyph geometries"),
                                   (len(layouts), 100, "run schedules"),
                                   (len(palettes), 35, "palettes")):
        _require(value >= minimum, f"insufficient {label}: {value}")
    _require(fields == backgrounds == set(range(10)), "structural color roles too narrow")
    _require(key_counts == {1, 2, 3, 4} and set(range(1, 7)) <= counts, "count variation missing")
    _require(shifts == {0, 1, 2} and edge_modes == {True, False}, "geometry/shift modes missing")
    _require(recolor_checks == 42 and row_stretch_checks >= 30,
             "input-only metamorphic coverage missing")
    _require(_digest(generate_puzzle(731)) == _digest(generate_puzzle(731)), "nondeterminism")
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        p = generate_puzzle(800 + i, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(90, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted answer was accepted")
    return dict(seeds=42, pairs=pair_count, puzzles=len(puzzles), input_shapes=len(shapes),
                output_shapes=len(outputs), normalized_outputs=len(structures),
                glyph_shapes=len(glyphs), run_schedules=len(layouts), palettes=len(palettes),
                shift_axes=sorted(shifts), recolor_checks=recolor_checks,
                row_stretch_checks=row_stretch_checks)
