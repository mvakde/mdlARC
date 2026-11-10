"""Row signatures define the odd panel in each horizontal lattice band.

Output labels are fixed: uniform row -> 5, mixed row -> 0. The lattice
color is visible in the outer frame and is absent from panel interiors.
No parent code, seed, demonstrations, or generation metadata is used by solve.
"""

from collections import Counter
from itertools import combinations

import numpy as np


POSITIVE = 5
BACKGROUND = 0
LATTICE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SHIFTS = ("bands", "width", "height")
GEOMETRIES = tuple(
    (h, w, bands, copies)
    for h in range(2, 14)
    for w in range(3, 9)
    for bands in range(1, 30 // (h + 1) + 1)
    for copies in range(3, 8)
    if bands * (h + 1) + 1 <= 30 and copies * (w + 1) + 1 <= 30
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


def _counts(num_train, num_test):
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= n <= 30 for n in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _signature(panel):
    return tuple(bool(np.all(row == row[0])) for row in panel)


def _parse(input_grid):
    grid = _grid(input_grid)
    lattice = int(grid[0, 0])
    if lattice not in LATTICE_COLORS:
        raise ValueError("lattice must differ from binary labels 0 and 5")
    rs = np.flatnonzero(np.all(grid == lattice, axis=1))
    cs = np.flatnonzero(np.all(grid == lattice, axis=0))
    if (len(rs) < 2 or len(cs) < 4 or rs[0] != 0 or cs[0] != 0
            or rs[-1] != grid.shape[0] - 1 or cs[-1] != grid.shape[1] - 1):
        raise ValueError("missing outer lattice or too few panels")
    if not (np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1])):
        raise ValueError("lattice must have equal, one-cell-separated panels")
    h, w = int(rs[1] - 1), int(cs[1] - 1)
    bands, copies = len(rs) - 1, len(cs) - 1
    if not (2 <= h <= 13 and 3 <= w <= 8 and 3 <= copies <= 7):
        raise ValueError("panel geometry is outside this family")
    panels = []
    for r, s in zip(rs, rs[1:]):
        row = []
        for c, d in zip(cs, cs[1:]):
            panel = grid[r + 1:s, c + 1:d]
            if np.any(panel == lattice):
                raise ValueError("lattice color may not occur inside a panel")
            row.append(panel)
        panels.append(row)
    return dict(lattice=lattice, h=h, w=w, bands=bands, copies=copies,
                panels=panels, rs=rs)


def _band_decision(row):
    masks = [_signature(panel) for panel in row]
    counts = Counter(masks)
    # With at least three panels, n-1 identical signatures and one other
    # signature establish a unique mode without any cross-band inference.
    if len(counts) != 2 or sorted(counts.values()) != [1, len(row) - 1]:
        raise ValueError("each band needs a strict modal mask and one visible exception")
    master = max(counts, key=counts.get)
    column = next(c for c, mask in enumerate(masks) if mask != master)
    return master, masks[column], column


def _binary_output(selected, width, lattice):
    h = len(selected[0])
    output = np.full((len(selected) * (h + 1) + 1, width + 2), lattice, dtype=np.int8)
    for b, mask in enumerate(selected):
        r = 1 + b * (h + 1)
        colors = np.where(np.asarray(mask), POSITIVE, BACKGROUND).astype(np.int8)
        output[r:r + h, 1:-1] = colors[:, None]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select the nonmodal *row signature* independently in every band."""
    info = _parse(input_grid)
    selected = [_band_decision(row)[1] for row in info["panels"]]
    return _binary_output(selected, info["w"], info["lattice"])


def _geometry(rng, split, shift, witness=False):
    axis = {"height": 0, "width": 1, "bands": 2}[shift]
    boundary = {"height": 5, "width": 5, "bands": 3}[shift]
    choices = [g for g in GEOMETRIES
               if (g[axis] <= boundary if split == "train" else g[axis] > boundary)]
    if witness:
        choices = [g for g in choices if 4 <= g[0] <= 6 and g[2] == 3]
    # Give smaller, readable layouts some extra mass without fixing a template.
    weights = np.array([1.6 if g[2] <= 4 and g[3] <= 4 else 1.0 for g in choices])
    return choices[int(rng.choice(len(choices), p=weights / weights.sum()))]


def _witness_masks(rng, h):
    """Three local masters: add a positive, remove one, exchange two positions.

    All six signatures are distinct. In particular no global master explains
    all bands, and a positive-count comparison cannot find the exchange.
    """
    used, result = set(), []
    for kind in ("add", "remove", "exchange"):
        candidates = []
        for on in combinations(range(h), 2):
            master = tuple(r in on for r in range(h))
            if master in used:
                continue
            off = [r for r in range(h) if r not in on]
            edits = ([(r,) for r in off] if kind == "add" else
                     [(r,) for r in on] if kind == "remove" else
                     [(a, b) for a in on for b in off])
            for cells in edits:
                variant = tuple(not value if r in cells else value
                                for r, value in enumerate(master))
                if variant not in used:
                    candidates.append((master, variant))
        if not candidates:
            raise AssertionError("witness construction exhausted its finite mask space")
        master, variant = candidates[int(rng.integers(len(candidates)))]
        used.update((master, variant))
        result.append((master, variant))
    order = rng.permutation(3)
    return [result[int(i)] for i in order]


def _random_masks(rng, h):
    density = float(rng.choice((0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                               p=(0.08, 0.17, 0.25, 0.25, 0.17, 0.08)))
    master = rng.random(h) < density
    variant = master.copy()
    on, off = np.flatnonzero(master), np.flatnonzero(~master)
    mode = str(rng.choice(("add", "remove", "exchange", "flip", "shift")))
    if mode == "add" and len(off):
        variant[rng.choice(off, int(rng.integers(1, len(off) + 1)), replace=False)] = True
    elif mode == "remove" and len(on):
        variant[rng.choice(on, int(rng.integers(1, len(on) + 1)), replace=False)] = False
    elif mode == "exchange" and len(on) and len(off):
        count = int(rng.integers(1, min(len(on), len(off)) + 1))
        variant[rng.choice(on, count, replace=False)] = False
        variant[rng.choice(off, count, replace=False)] = True
    elif mode == "shift":
        variant = np.roll(master, int(rng.integers(1, h)))
    else:
        cells = rng.choice(h, int(rng.integers(1, h + 1)), replace=False)
        variant[cells] = ~variant[cells]
    if np.array_equal(master, variant):
        variant[int(rng.integers(h))] ^= True
    return tuple(map(bool, master)), tuple(map(bool, variant))


def _mixed_row(rng, width, palette, style=None):
    a, b = map(int, rng.choice(palette, 2, replace=False))
    if style is None:
        style = str(rng.choice(("first", "middle", "last", "alternating", "runs", "random")))
    row = np.full(width, a, dtype=np.int8)
    if style in ("first", "middle", "last"):
        index = 0 if style == "first" else width - 1 if style == "last" else int(rng.integers(1, width - 1))
        row[index] = b
    elif style == "alternating":
        row[1::2] = b
    elif style == "runs":
        row[int(rng.integers(1, width)):] = b
    else:
        row = np.asarray(rng.choice(palette, width), dtype=np.int8)
        if np.all(row == row[0]):
            row[-1] = b if row[0] == a else a
    return row


def _render(rng, mask, width, palette, *, column_tag=None, forced_style=None):
    panel = np.empty((len(mask), width), dtype=np.int8)
    forced = False
    for r, positive in enumerate(mask):
        if positive:
            color = (int(rng.choice(palette)) if column_tag is None
                     else int(palette[(r + column_tag) % len(palette)]))
            panel[r] = color
        else:
            style = forced_style if not forced else None
            panel[r] = _mixed_row(rng, width, palette, style)
            forced = True
    return panel


def _assemble(rows, lattice):
    h, w = rows[0][0].shape
    grid = np.full((len(rows) * (h + 1) + 1, len(rows[0]) * (w + 1) + 1),
                   lattice, dtype=np.int8)
    for b, panels in enumerate(rows):
        r = 1 + b * (h + 1)
        for c, panel in enumerate(panels):
            s = 1 + c * (w + 1)
            grid[r:r + h, s:s + w] = panel
    return grid


def _pair(rng, geometry, lattice, witness):
    h, w, bands, copies = geometry
    size = int(rng.integers(copies if witness else 2, 10))
    palette = rng.choice([c for c in range(10) if c != lattice], size, replace=False)
    masks = _witness_masks(rng, h) if witness else [_random_masks(rng, h) for _ in range(bands)]
    columns = rng.integers(copies, size=bands)
    if witness and len(set(map(int, columns))) == 1:
        columns[1] = (columns[0] + int(rng.integers(1, copies))) % copies
    literal_copies = not witness and rng.random() < 0.35
    rows, selected = [], []
    for b, (master, variant) in enumerate(masks):
        row = []
        template = _render(rng, master, w, palette)
        for c in range(copies):
            is_exception = c == columns[b]
            mask = variant if is_exception else master
            if literal_copies and not is_exception and rng.random() < 0.8:
                panel = template.copy()
            else:
                panel = _render(rng, mask, w, palette,
                                column_tag=c if witness else None,
                                forced_style=("first", "middle", "last")[b]
                                if witness and is_exception else None)
            row.append(panel)
        rows.append(row)
        selected.append(variant)
    # Labels come directly from construction, rather than calling solve.
    return {"input": _assemble(rows, lattice),
            "output": _binary_output(selected, w, lattice)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(SHIFTS))
    shared_lattice = bool(rng.random() < 0.25)
    lattice = int(rng.choice(LATTICE_COLORS))
    witness_index = int(rng.integers(nt))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = split == "train" and index == witness_index
            for _ in range(32):
                geometry = _geometry(rng, split, shift, witness)
                color = lattice if shared_lattice else int(rng.choice(LATTICE_COLORS))
                pair = _pair(rng, geometry, color, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _negative_style(row):
    colors, counts = np.unique(row, return_counts=True)
    if len(colors) != 2 or sorted(map(int, counts)) != [1, len(row) - 1]:
        return set()
    outlier = int(colors[int(np.argmin(counts))])
    position = int(np.flatnonzero(row == outlier)[0])
    return {"first" if position == 0 else "last" if position == len(row) - 1 else "middle"}


def _analyze(pair):
    info = _parse(pair["input"])
    output = _grid(pair["output"])
    masters, variants, columns, directions = [], [], [], []
    styles, varied_positive, raw_ambiguous = set(), False, False
    for row in info["panels"]:
        master, variant, column = _band_decision(row)
        masters.append(master)
        variants.append(variant)
        columns.append(column)
        add = sum(not a and b for a, b in zip(master, variant))
        remove = sum(a and not b for a, b in zip(master, variant))
        directions.append((add, remove))
        raw_ambiguous |= len({_key(p) for p in row}) == len(row)
        selected = row[column]
        positive_colors = {int(r[0]) for r, flag in zip(selected, variant) if flag}
        varied_positive |= len(positive_colors) >= 2
        for raw_row, flag in zip(selected, variant):
            if not flag:
                styles.update(_negative_style(raw_row))
    expected = _binary_output(variants, info["w"], info["lattice"])
    _require(np.array_equal(output, expected), "output violates exception selection, binary labels, framing, or order")
    return {k: info[k] for k in ("h", "w", "bands", "copies", "lattice")} | dict(
        masters=tuple(masters), variants=tuple(variants), columns=tuple(columns),
        directions=tuple(directions), styles=styles, varied_positive=varied_positive,
        raw_ambiguous=raw_ambiguous)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            records[split].append(_analyze({"input": grid, "output": output}))
    train = records["train"]
    directions = [d for m in train for d in m["directions"]]
    _require(any(a > 0 and r == 0 for a, r in directions), "training needs a positive-row addition")
    _require(any(r > 0 and a == 0 for a, r in directions), "training needs a positive-row removal")
    _require(any(a == r and a > 0 for a, r in directions), "training must distinguish bitmap from positive count")
    _require(set().union(*(m["styles"] for m in train)) == {"first", "middle", "last"},
             "selected training negatives need first/interior/last mismatch contrasts")
    _require(any(m["varied_positive"] and m["raw_ambiguous"] for m in train),
             "training must distinguish row equality from literal-color panel comparison")
    _require(any(len(set(m["masters"])) > 1 for m in train), "training must establish local masters")
    _require(any(len(set(m["columns"])) > 1 for m in train), "training must reject fixed-column selection")
    _require(any(m["variants"] != m["variants"][::-1] for m in train), "training must establish band order")
    _require(any(min(m[k] for m in records["test"]) > max(m[k] for m in train)
                 for k in ("h", "w", "bands")), "missing intended geometry shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def _copy_puzzle(puzzle):
    return {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzle[s]]
            for s in ("train", "test")}


def self_test():
    """Stream 42 seeds; retain only signatures and small aggregate counters.

    No files, execution traces, training jobs, network calls, or caches are made.
    The returned measurements are descriptive generator-owned checks.
    """
    fingerprints, shapes, geometries, palettes, masks, directions = set(), set(), set(), set(), set(), set()
    lattice_colors, shift_axes, panel_heights, panel_widths, band_counts, copy_counts = set(), set(), set(), set(), set(), set()
    balanced = extremes = raw_ambiguous = repeated_columns = literal_repeats = total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        local = {s: [_analyze(p) for p in puzzle[s]] for s in ("train", "test")}
        for axis in ("h", "w", "bands"):
            if min(m[axis] for m in local["test"]) > max(m[axis] for m in local["train"]):
                shift_axes.add(axis)
        for split in ("train", "test"):
            for pair, m in zip(puzzle[split], local[split]):
                grid = pair["input"]
                before = grid.copy()
                _require(np.array_equal(solve(grid), pair["output"]), "input-only solve disagrees")
                _require(np.array_equal(grid, before), "solve mutated input")
                shapes.add(grid.shape)
                geometries.add((m["h"], m["w"], m["bands"], m["copies"]))
                palettes.add(tuple(map(int, np.unique(grid))))
                masks.update(m["masters"] + m["variants"])
                directions.update(m["directions"])
                panel_heights.add(m["h"])
                panel_widths.add(m["w"])
                band_counts.add(m["bands"])
                copy_counts.add(m["copies"])
                lattice_colors.add(m["lattice"])
                balanced += sum(a == r > 0 for a, r in m["directions"])
                extremes += sum(all(v) or not any(v) for v in m["masters"] + m["variants"])
                raw_ambiguous += m["raw_ambiguous"]
                repeated_columns += len(set(m["columns"])) < m["bands"]
                literal_repeats += any(len({_key(p) for p in row}) < len(row)
                                       for row in _parse(grid)["panels"])
                total_pairs += 1
        if seed in (0, 17, 41):
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterministic seed")
        if seed == 0:
            base = puzzle
    _require(len(fingerprints) == 42 and len(shapes) >= 35 and len(geometries) >= 75,
             "insufficient structural variation")
    _require(len(palettes) >= 40 and len(masks) >= 80, "insufficient palette or decision variation")
    _require(len(panel_heights) >= 10 and len(panel_widths) == 6 and len(band_counts) >= 6
             and len(copy_counts) >= 4, "insufficient geometry axes")
    _require(lattice_colors == set(LATTICE_COLORS) and shift_axes == {"h", "w", "bands"},
             "missing palette role or shift coverage")
    _require(balanced >= 42 and extremes >= 10 and repeated_columns >= 10
             and raw_ambiguous >= 42 and literal_repeats >= 10, "missing family-specific variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "counts changed")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Color identity is irrelevant to the predicate, even for the colors 0/5.
    for pair in base["train"] + base["test"]:
        grid = pair["input"]
        lattice = int(grid[0, 0])
        interior_colors = [c for c in range(10) if c != lattice]
        lookup = np.arange(10, dtype=np.int8)
        lookup[interior_colors] = np.roll(interior_colors, 1)
        _require(np.array_equal(solve(lookup[grid]), pair["output"]), "literal-color dependence")
    # Mutations exercise missing/tied exceptions, framing, labels and order.
    for defect in ("missing", "tie", "border", "output", "order"):
        bad = _copy_puzzle(base)
        pair = next(p for p in bad["train"] if _parse(p["input"])["bands"] == 3
                    and len(set(_analyze(p)["masters"])) >= 3)
        info = _parse(pair["input"])
        if defect == "missing":
            _, _, column = _band_decision(info["panels"][0])
            other = (column + 1) % info["copies"]
            info["panels"][0][column][:] = info["panels"][0][other]
        elif defect == "tie":
            row = info["panels"][0]
            master, variant, column = _band_decision(row)
            normal = row[(column + 1) % info["copies"]].copy()
            exceptional = row[column].copy()
            half = info["copies"] // 2
            for c in range(2 * half):
                row[c][:] = normal if c < half else exceptional
            if info["copies"] % 2:
                third = normal.copy()
                for r in range(info["h"]):
                    candidate = list(master)
                    candidate[r] = not candidate[r]
                    if tuple(candidate) not in (master, variant):
                        if master[r]:
                            color = next(x for x in range(10)
                                         if x not in (info["lattice"], int(third[r, 0])))
                            third[r, 0] = color
                        else:
                            third[r] = third[r, 0]
                        break
                row[-1][:] = third
        elif defect == "border":
            pair["input"][0, 0] = 0
        elif defect == "output":
            pair["output"][1, 1] = (int(pair["output"][1, 1]) + 1) % 10
        else:
            pair["output"] = pair["output"][::-1].copy()
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    return dict(seeds=42, pairs=total_pairs, distinct_puzzles=len(fingerprints),
                input_shapes=len(shapes), geometries=len(geometries), palettes=len(palettes),
                row_masks=len(masks), edit_directions=len(directions),
                panel_heights=sorted(panel_heights), panel_widths=sorted(panel_widths),
                band_counts=sorted(band_counts), copy_counts=sorted(copy_counts),
                lattice_colors=sorted(lattice_colors), shift_axes=sorted(shift_axes),
                balanced_bands=balanced, extreme_masks=extremes,
                raw_ambiguous_pairs=raw_ambiguous, repeated_column_pairs=repeated_columns,
                literal_repeat_pairs=literal_repeats,
                semantic_verification="generator-owned checks only")
