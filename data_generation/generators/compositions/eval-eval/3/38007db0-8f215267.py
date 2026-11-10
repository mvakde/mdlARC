"""Exceptional maps with signed, color-matched component-change receipts.

Only NumPy and the standard library are used. No parent code or examples are
loaded. The solver recovers orientation, lattice, master, and frames from input.
"""
from collections import Counter
from itertools import combinations
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors outside 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(g, color, diagonal=False):
    remaining = set(map(tuple, np.argwhere(g == color)))
    groups = []
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        groups.append(tuple(sorted(cells)))
    return groups


def _count(g, color, diagonal=False):
    return len(_components(g, color, diagonal))


def _frame_cells(r, c, h, w):
    return {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)
            if rr in (r, r + h - 1) or cc in (c, c + w - 1, c + w // 2)}


def _capacity(w):
    return (w // 2 - 2) // 2


def _draw_frame(g, r, c, h, w, color):
    for q in _frame_cells(r, c, h, w):
        g[q] = color


def _draw_tally(g, r, c, h, w, color, delta):
    _require(abs(delta) <= _capacity(w), "receipt overflow")
    for j in range(abs(delta)):
        # Canonical map field is on the left: surplus near, deficit far.
        cc = c + 2 + 2*j if delta > 0 else c + w - 3 - 2*j
        g[r + h // 2, cc] = color


def _lattice(panels, lattice):
    bands, copies = len(panels), len(panels[0])
    ph, pw = panels[0][0].shape
    g = np.full((bands*(ph + 1) + 1, copies*(pw + 1) + 1), lattice, np.int8)
    for b, row in enumerate(panels):
        for j, panel in enumerate(row):
            r, c = 1 + b*(ph + 1), 1 + j*(pw + 1)
            g[r:r + ph, c:c + pw] = panel
    return g


def _parse_canonical(g):
    """Recognize the map lattice on the left and all double-chamber frames."""
    lattice = int(g[0, 0])
    _require(bool(np.all(g[:, 0] == lattice)), "lattice must span the left edge")
    ends = np.flatnonzero(g[0] != lattice)
    _require(len(ends) > 0, "missing receipt bank")
    aw = int(ends[0])
    bg = int(g[0, aw])
    _require(bg != lattice and bool(np.all(g[0, aw:] == bg)) and
             bool(np.all(g[-1, aw:] == bg)), "bank margin")
    a = g[:, :aw]
    rs = np.flatnonzero(np.all(a == lattice, axis=1))
    cs = np.flatnonzero(np.all(a == lattice, axis=0))
    _require(len(rs) >= 2 and len(cs) >= 4, "at least three map copies")
    _require(rs[0] == cs[0] == 0 and rs[-1] == len(g)-1 and cs[-1] == aw-1,
             "closed map lattice")
    _require(bool(np.all(np.diff(rs) == rs[1])) and
             bool(np.all(np.diff(cs) == cs[1])), "regular lattice")
    ph, pw = int(rs[1]-1), int(cs[1]-1)
    _require(ph >= 5 and pw >= 2, "map too small")
    panels = [[a[r+1:s, c+1:d].copy() for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    _require(all(not np.any(p == lattice) for row in panels for p in row),
             "lattice color inside map")
    bands, copies = len(panels), len(panels[0])
    frequency = Counter(_key(p) for row in panels for p in row)
    winners = [key for key, n in frequency.items() if n == max(frequency.values())]
    _require(len(winners) == 1, "tied master bitmaps")
    master = next(p for p in panels[0] if _key(p) == winners[0])
    selected, columns = [], []
    for row in panels:
        different = [i for i, p in enumerate(row) if _key(p) != winners[0]]
        _require(len(different) == 1, "each band needs exactly one visible exception")
        columns.append(different[0])
        selected.append(row[different[0]])

    bank = g[:, aw:]
    _require(not np.any(bank == lattice) and bool(np.all(bank[:, 0] == bg)),
             "receipt bank must have a background gap")
    frames = [[] for _ in range(bands)]
    for color in map(int, np.unique(bank)):
        if color == bg:
            continue
        for group in _components(bank, color):
            r0, r1 = min(r for r, c in group), max(r for r, c in group)
            c0, c1 = min(c for r, c in group) + aw, max(c for r, c in group) + aw
            h, w = r1-r0+1, c1-c0+1
            _require(h in (5, 7) and w in (9, 13, 17), "unrecognized receipt frame")
            actual = {(r, c + aw) for r, c in group}
            _require(actual == _frame_cells(r0, c0, h, w), "broken or filled input frame")
            interior = g[r0+1:r1, c0+1:c1].copy()
            interior[:, w//2-1] = bg
            _require(bool(np.all(interior == bg)), "input chambers must be empty")
            b = (r0-1)//(ph+1)
            _require(0 <= b < bands and r0 >= 1+b*(ph+1) and
                     r1 < 1+b*(ph+1)+ph, "frame crosses a band boundary")
            frames[b].append((color, r0, c0, h, w))
    for row in frames:
        row.sort(key=lambda f: (f[1], f[2]))
        colors = [f[0] for f in row]
        _require(1 <= len(row) <= 3 and len(set(colors)) == len(colors),
                 "one to three distinct frame categories per band")
        # Distinct frames cannot touch, even at a corner.
        for i, f in enumerate(row):
            _, r, c, h, w = f
            for q in row[i+1:]:
                _, rr, cc, hh, ww = q
                _require(r+h < rr or rr+hh < r or c+w < cc or cc+ww < c,
                         "receipt frames touch")
    categories = {f[0] for f in frames[0]}
    _require(all({f[0] for f in row} == categories for row in frames),
             "category set must be shared within this input")
    deltas, area_deltas, diagonal_deltas = [], [], []
    for variant, row in zip(selected, frames):
        ds, ads, d8s = {}, {}, {}
        for color, r, c, h, w in row:
            ds[color] = _count(variant, color) - _count(master, color)
            ads[color] = int(np.count_nonzero(variant == color) - np.count_nonzero(master == color))
            d8s[color] = _count(variant, color, True) - _count(master, color, True)
            _require(abs(ds[color]) <= _capacity(w), "component delta exceeds chamber capacity")
        deltas.append(ds)
        area_deltas.append(ads)
        diagonal_deltas.append(d8s)
    return dict(grid=g, lattice=lattice, background=bg, array_width=aw,
                ph=ph, pw=pw, bands=bands, copies=copies, panels=panels,
                master=master, selected=selected, columns=columns, frames=frames,
                categories=categories, deltas=deltas, area_deltas=area_deltas,
                diagonal_deltas=diagonal_deltas)


def _inspect(input_grid):
    g = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(g, rotation))
        except ValueError:
            continue
        info['rotation'] = rotation
        candidates.append(info)
    _require(len(candidates) == 1, "input must have one recognizable orientation")
    return candidates[0]


def _answer(info):
    g, aw, pw = info['grid'], info['array_width'], info['pw']
    shift = aw - (pw+2)
    out = np.full((g.shape[0], g.shape[1]-shift), info['background'], np.int8)
    out[:, :pw+2] = _lattice([[p] for p in info['selected']], info['lattice'])
    out[:, pw+2:] = g[:, aw:]
    for row, ds in zip(info['frames'], info['deltas']):
        for color, r, c, h, w in row:
            _draw_tally(out, r, c-shift, h, w, color, ds[color])
    return np.rot90(out, -info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only full-bitmap selection and signed component-difference census."""
    return _answer(_inspect(input_grid))


def _teaching_maps(rng, ph, pw, bg, colors, distractor, reverse):
    """Balanced cuts/joins, a diagonal witness, and a zero-delta reshaping."""
    a, b = colors
    offset = int(rng.integers(ph-10))
    lane = int(rng.integers(pw-1))
    def panel(split_a, split_b):
        g = np.full((ph, pw), bg, np.int8)
        aa = [(0, 0), (1, 0), (2, 1), (4, 0)] if split_a else [(i, 0) for i in range(4)]
        bb = [(6, 1), (7, 1), (8, 1), (10, 1)] if split_b else [(i, 1) for i in range(7, 11)]
        for r, c in aa:
            g[r+offset, c+lane] = a
        for r, c in bb:
            g[r+offset, c+lane] = b
        g[offset+5, lane+1] = distractor
        return g
    master = panel(reverse, not reverse)
    balanced = panel(not reverse, reverse)
    resized = master.copy()
    if reverse:
        resized[offset, lane+1] = a  # Grow one piece, still three 4-components.
        resized[offset+7, lane+1] = bg  # Shorten B's connected bar.
    else:
        resized[offset, lane+1] = a  # Grow the connected A bar.
        # Cut B's three-cell bar, then join its lower piece to the singleton.
        # The +1/-1 topology changes cancel; growth also changes area, not count.
        resized[offset+7, lane+1] = bg
        resized[offset+9, lane+1] = b
        resized[offset+6, lane] = b
    panels = [master, balanced, resized]
    if rng.random() < .5:
        panels = [p[::-1].copy() for p in panels]
    if rng.random() < .5:
        panels = [p[:, ::-1].copy() for p in panels]
    return panels[0], panels[1:]


def _shape(rng, ph, pw):
    target = int(rng.integers(1, min(9, ph*pw)+1))
    kind = int(rng.integers(4))
    if kind == 0:
        # Horizontal/vertical bars coexist with grown and compact objects.
        vertical = rng.random() < .5
        target = min(target, ph if vertical else pw)
        return tuple((i, 0) if vertical else (0, i) for i in range(target))
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = sorted({(r+dr, c+dc) for r, c in cells
                           for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
        valid = []
        for q in frontier:
            trial = cells | {q}
            if (max(r for r, c in trial)-min(r for r, c in trial) < ph and
                    max(c for r, c in trial)-min(c for r, c in trial) < pw):
                valid.append(q)
        if not valid:
            break
        # Compact growth and unconstrained growth give different silhouettes.
        if kind == 1:
            best = min(max(r for r, c in cells | {q})-min(r for r, c in cells | {q}) +
                       max(c for r, c in cells | {q})-min(c for r, c in cells | {q}) for q in valid)
            valid = [q for q in valid if
                     max(r for r, c in cells | {q})-min(r for r, c in cells | {q}) +
                     max(c for r, c in cells | {q})-min(c for r, c in cells | {q}) == best]
        cells.add(valid[int(rng.integers(len(valid)))])
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _place_shape(rng, g, bg, color, shape):
    sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
    origins = [(r, c) for r in range(g.shape[0]-sh+1) for c in range(g.shape[1]-sw+1)]
    for index in rng.permutation(len(origins)):
        r0, c0 = origins[int(index)]
        cells = {(r+r0, c+c0) for r, c in shape}
        if any(g[q] != bg for q in cells):
            continue
        # Four-neighbor separation within a color; diagonal contact is allowed.
        if any(0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and
               (rr, cc) not in cells and g[rr, cc] == color
               for r, c in cells for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))):
            continue
        for q in cells:
            g[q] = color
        return True
    return False


def _master_map(rng, ph, pw, bg, colors, distractors):
    g = np.full((ph, pw), bg, np.int8)
    pool = list(colors) + list(distractors)
    objects = int(rng.integers(2, min(13, max(3, ph*pw//3))+1))
    specifications = [int(rng.choice(colors))] + [int(rng.choice(pool)) for _ in range(objects-1)]
    for color in specifications:
        shape = _shape(rng, ph, pw)
        if not _place_shape(rng, g, bg, color, shape):
            _place_shape(rng, g, bg, color, ((0, 0),))
    return g


def _mutate(rng, master, bg, colors, distractors, cap):
    palette = [bg] + list(colors) + list(distractors)
    for attempt in range(80):
        g = master.copy()
        mode = int(rng.integers(5))
        visible = [(color, part) for color in palette if color != bg
                   for part in _components(master, color)]
        if mode == 0 and visible:  # Erasure or recoloring of a whole object.
            color, part = visible[int(rng.integers(len(visible)))]
            target = int(rng.choice([c for c in palette if c != color]))
            for q in part:
                g[q] = target
        elif mode == 1 and visible:  # Move a body, possibly joining other bodies.
            color, part = visible[int(rng.integers(len(visible)))]
            for q in part:
                g[q] = bg
            r0, c0 = min(r for r, c in part), min(c for r, c in part)
            _place_shape(rng, g, bg, color, tuple((r-r0, c-c0) for r, c in part))
        elif mode == 2:  # Add an independently sampled connected body.
            _place_shape(rng, g, bg, int(rng.choice(colors)), _shape(rng, *g.shape))
        else:  # Local cuts, bridges, exchanges and shape changes.
            number = int(rng.integers(1, min(9, max(2, g.size//3))+1))
            for index in rng.choice(g.size, number, replace=False):
                old = int(g.flat[index])
                g.flat[index] = int(rng.choice([c for c in palette if c != old]))
        if np.array_equal(g, master):
            continue
        if all(abs(_count(g, c)-_count(master, c)) <= cap for c in colors):
            return g
    # Guaranteed different, valid fallback: recolor one pixel to an unmatched
    # color. A one-pixel deletion in a width <=5 map changes its count by <=3.
    # Enumerate candidates rather than assume that bound for a narrow chamber.
    for index in rng.permutation(master.size):
        for target in palette:
            if master.flat[index] == target:
                continue
            g = master.copy()
            g.flat[index] = target
            if all(abs(_count(g, c)-_count(master, c)) <= cap for c in colors):
                return g
    raise RuntimeError("no legal one-cell variant")


def _geometry(rng, categories):
    options = [(pw, copies, long) for pw in range(2, 6) for copies in range(3, 7)
               for long in (9, 13, 17) if copies*(pw+1)+1+1+long <= 30]
    pw, copies, long = options[int(rng.integers(len(options)))]
    minimum = 6*categories-1
    bands = int(rng.integers(1, min(4, 29//(minimum+1))+1))
    ph = int(rng.integers(minimum, 29//bands))
    return ph, pw, bands, copies, long


def _extrapolation_maps(rng, categories, bg, colors, distractors):
    """Three cuts split one bar into four pieces, or three bridges undo them."""
    minimum = max(7, 6*categories-1)
    bands = int(rng.integers(1, min(3, 29//(minimum+1))+1))
    ph = int(rng.integers(minimum, 29//bands))
    geometry = (ph, 2, bands, 3, 17)
    primary = colors[0]
    master = _master_map(rng, ph, 2, bg, colors, distractors)
    master[master == primary] = bg
    length = int(rng.integers(7, min(ph, 13)+1))
    r0, column = int(rng.integers(ph-length+1)), int(rng.integers(2))
    cuts = [q for q in combinations(range(1, length-1), 3)
            if q[1]-q[0] >= 2 and q[2]-q[1] >= 2]
    positions = cuts[int(rng.integers(len(cuts)))]
    master[r0:r0+length, column] = primary
    reverse = rng.random() < .5
    if reverse:
        for r in positions:
            master[r0+r, column] = bg
    variants = []
    for b in range(bands):
        g = master.copy()
        if reverse:
            g[r0:r0+length, column] = primary
        else:
            positions = cuts[int(rng.integers(len(cuts)))]
            for r in positions:
                g[r0+r, column] = bg
        variants.append(g)
    return geometry, master, variants


def _build_pair(rng, geometry, master, variants, bg, lattice, colors, rotation, witness=False):
    ph, pw, bands, copies, long = geometry
    columns = list(map(int, rng.integers(copies, size=bands)))
    if witness and len(set(columns)) == 1:
        columns[-1] = (columns[0] + int(rng.integers(1, copies))) % copies
    rows = []
    for b, variant in enumerate(variants):
        row = [master.copy() for _ in range(copies)]
        row[columns[b]] = variant
        rows.append(row)
    a = _lattice(rows, lattice)
    compressed = _lattice([[row[col]] for row, col in zip(rows, columns)], lattice)
    spare = 30-a.shape[1]-long
    gap = int(rng.integers(1, min(3, spare)+1))
    padding = int(rng.integers(min(2, spare-gap)+1))
    inp = np.full((a.shape[0], a.shape[1]+gap+long+padding), bg, np.int8)
    out = np.full((a.shape[0], compressed.shape[1]+gap+long+padding), bg, np.int8)
    inp[:, :a.shape[1]], out[:, :compressed.shape[1]] = a, compressed
    for b, variant in enumerate(variants):
        ds = {c: _count(variant, c)-_count(master, c) for c in colors}
        order = [int(c) for c in rng.permutation(colors)]
        heights = [5]*len(colors)
        gaps = [1]*(len(colors)-1)
        free = ph-sum(heights)-sum(gaps)
        for j in rng.permutation(len(colors)):
            if free >= 2 and rng.random() < .4:
                heights[int(j)] = 7
                free -= 2
        for j in range(len(gaps)):
            if free and rng.random() < .4:
                gaps[j] += 1
                free -= 1
        r = 1+b*(ph+1)+int(rng.integers(free+1))
        for j, (color, h) in enumerate(zip(order, heights)):
            widths = [w for w in (9, 13, 17) if w <= long and _capacity(w) >= abs(ds[color])]
            w = int(rng.choice(widths))
            offset = int(rng.integers(long+padding-w+1))
            ci, co = a.shape[1]+gap+offset, compressed.shape[1]+gap+offset
            _draw_frame(inp, r, ci, h, w, color)
            _draw_frame(out, r, co, h, w, color)
            _draw_tally(out, r, co, h, w, color, ds[color])
            r += h + (gaps[j] if j < len(gaps) else 0)
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal example counts")
    # Nonnegative seeds use default_rng(seed) literally; negative integer seeds
    # are deterministically mapped to NumPy's unsigned seed domain.
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    novel = int(rng.integers(10))
    base_rotation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt+nq):
        train = index < nt
        teaching = index < 2
        categories = 2 if teaching else int(rng.integers(1, 4))
        available = [c for c in range(10) if c != novel]
        if train:
            palette = list(map(int, rng.choice(available, categories+2, replace=False)))
            bg, lattice, colors = palette[0], palette[1], palette[2:]
        else:
            palette = list(map(int, rng.choice(available, categories+1, replace=False)))
            bg, lattice, colors = palette[0], palette[1], [novel]+palette[2:]
        unused = [c for c in range(10) if c not in [bg, lattice]+colors]
        distractors = [novel] if teaching else list(map(int, rng.choice(
            unused, int(rng.integers(0, min(3, len(unused))+1)), replace=False)))
        rotation = (base_rotation + (2*index if teaching else
                    int(rng.choice((0, 2))) if train else int(rng.choice((1, 3))))) % 4
        for attempt in range(32):
            if teaching:
                ph, pw = int(rng.integers(11, 14)), int(rng.integers(3, 5))
                geometry = (ph, pw, 2, 3, 13)
                master, variants = _teaching_maps(rng, ph, pw, bg, colors, novel, index == 1)
            elif not train and rng.random() < .35:
                geometry, master, variants = _extrapolation_maps(
                    rng, categories, bg, colors, distractors)
            else:
                geometry = _geometry(rng, categories)
                ph, pw, bands, copies, long = geometry
                master = _master_map(rng, ph, pw, bg, colors, distractors)
                variants = [_mutate(rng, master, bg, colors, distractors, _capacity(long))
                            for _ in range(bands)]
                if bands > 1 and rng.random() < .18:
                    variants[-1] = variants[0].copy()
            pair = _build_pair(rng, geometry, master, variants, bg, lattice, colors,
                               rotation, teaching)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError("unique input attempts exhausted")
        seen.add(key)
        puzzle['train' if train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Grid-derived rule checks and distinguishing training/shift evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
             len(puzzle['train']) > len(puzzle['test']), "puzzle counts")
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _inspect(inp)
            _require(np.array_equal(_answer(info), out), "incorrect selection, receipt, or compression")
            # Explicitly check selection and frame preservation after compression.
            canonical = np.rot90(out, info['rotation'])
            shift = info['array_width'] - (info['pw']+2)
            _require(canonical.shape[0] == info['grid'].shape[0] and
                     canonical.shape[1] == info['grid'].shape[1]-shift, "compression dimensions")
            for b, selected in enumerate(info['selected']):
                r = 1+b*(info['ph']+1)
                _require(np.array_equal(canonical[r:r+info['ph'], 1:1+info['pw']], selected),
                         "selected map was altered")
                for color, rr, c, h, w in info['frames'][b]:
                    _require(all(canonical[r0, c0-shift] == color
                                 for r0, c0 in _frame_cells(rr, c, h, w)), "frame changed")
            details[split].append(info)
    train, test = details['train'], details['test']
    events = [(d, b, c, ds[c]) for d in train for b, ds in enumerate(d['deltas']) for c in ds]
    _require(any(delta >= 2 for d, b, c, delta in events) and
             any(delta <= -2 for d, b, c, delta in events), "signed unary progression missing")
    _require(any(delta != 0 and d['area_deltas'][b][c] == 0 for d, b, c, delta in events),
             "training permits colored-area shortcut")
    _require(any(delta == 0 and d['area_deltas'][b][c] != 0 for d, b, c, delta in events),
             "training needs a size change without a census change")
    cancellation = False
    for d, b, c, delta in events:
        if delta == 0:
            cut = d['master'].copy()
            cut[(d['master'] == c) & (d['selected'][b] != c)] = d['background']
            cancellation |= _count(cut, c) > _count(d['master'], c)
    _require(cancellation, "training needs a same-color split/join cancellation")
    _require(any(delta != d['diagonal_deltas'][b][c] for d, b, c, delta in events),
             "edge connectivity needs a diagonal-contact witness")
    _require(any(all(v == 0 for v in ds.values()) for d in train for ds in d['deltas']),
             "copy an exceptional bitmap even when every delta is zero")
    _require(any(len(set(d['columns'])) > 1 for d in train), "fixed-column shortcut")
    _require(any(_count(d['master'], c) > 0 for d in train for c in d['categories']),
             "template ablation must alter receipts")
    orientations = {d['rotation'] for d in train}
    _require(any((a+2) % 4 in orientations for a in orientations), "opposite training field sides")
    _require(all(d['rotation'] not in orientations for d in test), "test orientation should be held out")
    trained_categories = set().union(*(d['categories'] for d in train))
    _require(all(d['categories']-trained_categories for d in test), "new local test category missing")
    for d in test:
        _require(any(np.any(t['master'] == color) for color in d['categories']-trained_categories
                     for t in train), "test needs an ignored-to-counted color transition")


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded 42-seed audit, legal counts, semantic ablations and mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed duplication")
    shapes, geometries, masters, masks = set(), set(), set(), set()
    backgrounds, lattices, orientations, category_counts, copy_counts, band_counts = (set() for _ in range(6))
    deltas, areas, frame_widths, frame_heights = set(), set(), set(), set()
    zero_test = repeated_columns = repeated_variants = 0
    template_ablation = area_ablation = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        teaching = [_inspect(p['input']) for p in puzzle['train']]
        no_reference = any(ds[c] != _count(v, c) for d in teaching
                           for v, ds in zip(d['selected'], d['deltas']) for c in ds)
        no_components = any(ds[c] != ads[c] for d in teaching
                            for ds, ads in zip(d['deltas'], d['area_deltas']) for c in ds)
        _require(no_reference and no_components, "parent-concept ablation failed to change an answer")
        template_ablation += no_reference
        area_ablation += no_components
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _inspect(pair['input'])
                shapes.add(pair['input'].shape)
                geometries.add((d['ph'], d['pw'], d['bands'], d['copies']))
                masters.add(_key(d['master']))
                backgrounds.add(d['background'])
                lattices.add(d['lattice'])
                orientations.add(d['rotation'])
                category_counts.add(len(d['categories']))
                copy_counts.add(d['copies'])
                band_counts.add(d['bands'])
                repeated_columns += len(set(d['columns'])) < d['bands']
                repeated_variants += len({_key(v) for v in d['selected']}) < d['bands']
                zero_test += split == 'test' and all(v == 0 for ds in d['deltas'] for v in ds.values())
                for ds in d['deltas']:
                    deltas.update(ds.values())
                for row in d['frames']:
                    frame_widths.update(f[4] for f in row)
                    frame_heights.update(f[3] for f in row)
                for panel in [d['master']]+d['selected']:
                    for color in d['categories']:
                        for group in _components(panel, color):
                            r0, c0 = min(r for r, c in group), min(c for r, c in group)
                            masks.add(tuple((r-r0, c-c0) for r, c in group))
                            areas.add(len(group))
    _require(len(shapes) >= 35 and len(geometries) >= 40 and len(masters) >= 150, "structural diversity")
    _require(backgrounds == lattices == set(range(10)), "palette role diversity")
    _require(orientations == set(range(4)) and category_counts == {1, 2, 3}, "orientation/category diversity")
    _require(copy_counts >= {3, 4, 5} and band_counts >= {1, 2, 3, 4}, "lattice grouping diversity")
    _require(frame_widths == {9, 13, 17} and frame_heights == {5, 7}, "receipt geometry diversity")
    _require(deltas >= {-3, -2, -1, 0, 1, 2, 3}, "signed count diversity")
    _require(len(masks) >= 50 and max(areas) >= 8, "object morphology diversity")
    _require(zero_test > 0 and repeated_columns > 0 and repeated_variants > 0, "compatible edge cases")
    _require(_puzzle_key(generate_puzzle(788137553)) == _puzzle_key(generate_puzzle(788137553)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101+10*nt+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit legal counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Also check net cancellation in a small standalone fixture: one split and
    # one join of the SAME color cancel, and an unmatched edit is only copied.
    master = np.zeros((11, 4), np.int8)
    master[0:3, 0] = 2
    master[6, 0] = master[8, 0] = 2
    changed = master.copy()
    changed[1, 0], changed[7, 0] = 0, 2
    ignored = master.copy()
    ignored[4, 2] = 5
    _require(_count(master, 2) == _count(changed, 2) == 3, "cancellation fixture")
    pair = _build_pair(np.random.default_rng(90210), (11, 4, 2, 3, 13),
                       master, [changed, ignored], 0, 9, [2], 1)
    info = _inspect(pair['input'])
    _require(all(ds[2] == 0 for ds in info['deltas']), "net-change cancellation")
    _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "input-only cancellation solve")
    canonical_output = np.rot90(pair['output'], info['rotation'])
    _require(np.array_equal(canonical_output[:, info['pw']+2:],
                           info['grid'][:, info['array_width']:]), "zero receipt must preserve empty chambers")
    # Corrupt a receipt, a copied bitmap, and the repeated-master invariant.
    for defect in ('tally', 'map', 'missing_exception'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
        p = bad['train'][0]
        d = _inspect(p['input'])
        if defect == 'missing_exception':
            inp = np.rot90(p['input'], d['rotation']).copy()
            c = 1+d['columns'][0]*(d['pw']+1)
            inp[1:1+d['ph'], c:c+d['pw']] = d['master']
            p['input'] = np.rot90(inp, -d['rotation']).copy()
        else:
            out = np.rot90(p['output'], d['rotation']).copy()
            if defect == 'map':
                out[1, 1] = d['lattice']
            else:
                color, r, c, h, w = d['frames'][0][0]
                shift = d['array_width']-d['pw']-2
                out[r+h//2, c-shift+2] = d['background'] if out[r+h//2, c-shift+2] == color else color
            p['output'] = np.rot90(out, -d['rotation']).copy()
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in ('train', 'test')),
            "input_shapes": len(shapes), "map_geometries": len(geometries),
            "distinct_masters": len(masters), "component_masks": len(masks),
            "component_areas": sorted(areas), "signed_deltas": sorted(deltas),
            "zero_delta_tests": zero_test, "repeated_columns": repeated_columns,
            "repeated_exception_maps": repeated_variants,
            "template_ablation_changes_answer": template_ablation,
            "area_ablation_changes_answer": area_ablation,
            "net_cancellation_and_ignored_edit": True}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
