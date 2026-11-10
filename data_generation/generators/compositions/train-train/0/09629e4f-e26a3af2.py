"""Fault-stencil loom: read defects relative to a recoverable striped underlay.

Only the standard library and NumPy are used.  The solver never consults the
sampler, a seed, saved examples, or a puzzle-level color assignment.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _runs(mask):
    result, start = [], None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            result.append((start, i))
            start = None
    return result


def _layout(grid):
    candidates = []
    for sep in np.unique(grid):
        row_runs = _runs(~np.all(grid == sep, axis=1))
        col_runs = _runs(~np.all(grid == sep, axis=0))
        r, c = len(row_runs), len(col_runs)
        if r < 2 or c < 2:
            continue
        if any(b-a != r for a, b in row_runs) or any(b-a != c for a, b in col_runs):
            continue
        if (row_runs[0][0] != 0 or row_runs[-1][1] != grid.shape[0]
                or col_runs[0][0] != 0 or col_runs[-1][1] != grid.shape[1]):
            continue
        rows = np.array([x for a, b in row_runs for x in range(a, b)])
        cols = np.array([x for a, b in col_runs for x in range(a, b)])
        core = grid[np.ix_(rows, cols)]
        if np.any(core == sep):
            continue
        panels = [core[i*r:(i+1)*r, j*c:(j+1)*c]
                  for i in range(r) for j in range(c)]
        incidence = {int(color): sum(bool(np.any(p == color)) for p in panels)
                     for color in np.unique(core)}
        markers = [color for color, count in incidence.items() if count == r*c-1]
        if len(markers) != 1:
            continue
        marker = markers[0]
        missing = [divmod(i, c) for i, p in enumerate(panels) if not np.any(p == marker)]
        if len(missing) != 1:
            continue
        candidates.append(dict(separator=int(sep), marker=marker, r=r, c=c,
                               row_runs=row_runs, col_runs=col_runs,
                               rows=rows, cols=cols, core=core, source=missing[0]))
    _require(len(candidates) == 1, "one separator lattice and missing-marker role required")
    return candidates[0]


def _strict_mode(line):
    counts = np.bincount(line, minlength=10)
    winner = int(np.argmax(counts))
    return winner if 2 * int(counts[winner]) > len(line) else None


def _underlay(core, r, c, marker):
    candidates = []
    for orientation in ("vertical", "horizontal"):
        lines = core.T if orientation == "vertical" else core
        modes = [_strict_mode(line) for line in lines]
        if any(x is None for x in modes) or marker in modes or len(set(modes)) < 2:
            continue
        vector = np.array(modes, dtype=np.int8)
        clean = (np.repeat(vector[None, :], r*r, axis=0) if orientation == "vertical"
                 else np.repeat(vector[:, None], c*c, axis=1))
        # Every addressed panel must visibly distinguish keeping stripes from
        # flooding with one color; monochromatic panels would hide that choice.
        if any(len(np.unique(clean[i*r:(i+1)*r, j*c:(j+1)*c])) < 2
               for i in range(r) for j in range(c)):
            continue
        candidates.append((orientation, clean))
    _require(len(candidates) == 1, "striped underlay must have one strict-majority orientation")
    return candidates[0]


def _inspect_input(grid):
    a = _layout(grid)
    a["orientation"], a["clean"] = _underlay(a["core"], a["r"], a["c"], a["marker"])
    r, c = a["r"], a["c"]
    i, j = a["source"]
    a["tile"] = a["core"][i*r:(i+1)*r, j*c:(j+1)*c].copy()
    a["tile_clean"] = a["clean"][i*r:(i+1)*r, j*c:(j+1)*c].copy()
    a["signals"] = a["tile"] != a["tile_clean"]
    _require(0 < int(a["signals"].sum()) < r*c, "source must have faults and quiet cells")
    return a


def _broadcast(clean, tile, signals, r, c):
    result = clean.copy()
    for i, j in np.argwhere(signals):
        result[i*r:(i+1)*r, j*c:(j+1)*c] = tile[i, j]
    return result


def _restore_lattice(grid, a, core):
    result = grid.copy()
    result[np.ix_(a["rows"], a["cols"])] = core
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover stripes, select by absence, and broadcast only the residuals."""
    grid = _grid(input_grid)
    a = _inspect_input(grid)
    core = _broadcast(a["clean"], a["tile"], a["signals"], a["r"], a["c"])
    return _restore_lattice(grid, a, core)


def _key(grid):
    return grid.shape, grid.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _gaps(rng, n):
    gaps = [1] * (n-1)
    spare = 30 - n*n - (n-1)
    for i in rng.permutation(n-1):
        if spare and rng.random() < .38:
            gaps[int(i)] += 1
            spare -= 1
    return tuple(gaps)


def _axis_runs(n, gaps):
    result, start = [], 0
    for i in range(n):
        result.append((start, start+n))
        start += n + (gaps[i] if i < n-1 else 0)
    return result


def _segment(rng, length, palette):
    bands = int(rng.integers(2, length+1))
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, length), bands-1, replace=False))
    widths = np.diff([0] + cuts + [length])
    colors = []
    for _ in widths:
        choices = [x for x in palette if not colors or x != colors[-1]]
        colors.append(int(rng.choice(choices)))
    return np.repeat(colors, widths).astype(np.int8)


def _make_pair(rng, orientation, geometry, source, source_count, local_trap=False):
    r, c = geometry
    palette = [int(x) for x in rng.permutation(10)]
    sep, marker = palette[:2]
    band_palette = palette[2:2+int(rng.integers(2, 6))]
    extras = palette[7:7+int(rng.integers(0, 3))]
    axis_size, blocks = (c, c) if orientation == "vertical" else (r, r)
    # Periodic underlays are one regime, not an assumption of the solver.
    periodic = bool(rng.random() < .24)
    first = _segment(rng, axis_size, band_palette)
    parts = [first if periodic or i == 0 else _segment(rng, axis_size, band_palette)
             for i in range(blocks)]
    vector = np.concatenate(parts)
    clean = (np.repeat(vector[None, :], r*r, axis=0) if orientation == "vertical"
             else np.repeat(vector[:, None], c*c, axis=1))
    band_colors = [int(x) for x in np.unique(clean)]
    noise_palette = band_colors + extras
    noisy = clean.copy()
    changed = np.zeros(clean.shape, dtype=bool)
    span = r*r if orientation == "vertical" else c*c
    line_load = np.zeros(c*c if orientation == "vertical" else r*r, dtype=int)
    cap = (span-1)//2
    counts = {(i, j): {int(x): int(np.count_nonzero(clean[i*r:(i+1)*r, j*c:(j+1)*c] == x))
                      for x in np.unique(clean[i*r:(i+1)*r, j*c:(j+1)*c])}
              for i in range(r) for j in range(c)}

    def put(pos, replacement=None, forced_cell=None, band_only=False):
        i, j = pos
        cells = [(i*r+x, j*c+y) for x in range(r) for y in range(c)]
        rng.shuffle(cells)
        if forced_cell is not None:
            cells = [(i*r+forced_cell[0], j*c+forced_cell[1])]
        for x, y in cells:
            line = y if orientation == "vertical" else x
            old = int(clean[x, y])
            if changed[x, y] or line_load[line] >= cap or counts[pos][old] <= 1:
                continue
            options = [z for z in (band_colors if band_only else noise_palette) if z != old]
            new = int(rng.choice(options)) if replacement is None else int(replacement)
            if new == old:
                continue
            noisy[x, y], changed[x, y] = new, True
            line_load[line] += 1
            counts[pos][old] -= 1
            return True
        return False

    if local_trap:
        # A majority inside the selected little tile is deliberately wrong.
        # Full-span evidence from other panel rows/columns still repairs it.
        local_span = r if orientation == "vertical" else c
        k = local_span//2 + 1
        lane = int(rng.integers(c if orientation == "vertical" else r))
        old = int(clean[source[0]*r, source[1]*c+lane] if orientation == "vertical"
                  else clean[source[0]*r+lane, source[1]*c])
        replacement = int(rng.choice([z for z in band_colors if z != old]))
        for p in rng.choice(local_span, k, replace=False):
            cell = (int(p), lane) if orientation == "vertical" else (lane, int(p))
            _require(put(source, replacement, cell), "could not place local-majority witness")
        source_count = max(source_count, k)
    while int(changed[source[0]*r:(source[0]+1)*r, source[1]*c:(source[1]+1)*c].sum()) < source_count:
        _require(put(source, band_only=not np.any(changed)), "could not place source fault")

    decoys = [(i, j) for i in range(r) for j in range(c) if (i, j) != source]
    rng.shuffle(decoys)
    for pos in decoys:
        _require(put(pos, marker), "could not mark decoy")
    # One sparse decoy defeats selection by minimum defect count, and its
    # empty non-marker residual also supplies an explicit wrong-source witness.
    sparse_decoy = decoys[0]
    target = max(int(changed.sum()), int(round(clean.size * float(rng.uniform(.09, .24)))))
    candidates = [pos for pos in decoys if pos != sparse_decoy]
    for _ in range(clean.size * 2):
        if int(changed.sum()) >= target:
            break
        pos = candidates[int(rng.integers(len(candidates)))]
        i, j = pos
        tile = noisy[i*r:(i+1)*r, j*c:(j+1)*c]
        if int(changed[i*r:(i+1)*r, j*c:(j+1)*c].sum()) >= max(2, r*c//3):
            continue
        use_marker = np.count_nonzero(tile == marker) < 3 and rng.random() < .26
        put(pos, marker if use_marker else None)

    rr, cc = _axis_runs(r, _gaps(rng, r)), _axis_runs(c, _gaps(rng, c))
    inp = np.full((rr[-1][1], cc[-1][1]), sep, dtype=np.int8)
    out = inp.copy()
    si, sj = source
    tile = noisy[si*r:(si+1)*r, sj*c:(sj+1)*c]
    signals = changed[si*r:(si+1)*r, sj*c:(sj+1)*c]
    expanded = _broadcast(clean, tile, signals, r, c)
    for i, (a, b) in enumerate(rr):
        for j, (x, y) in enumerate(cc):
            inp[a:b, x:y] = noisy[i*r:(i+1)*r, j*c:(j+1)*c]
            out[a:b, x:y] = expanded[i*r:(i+1)*r, j*c:(j+1)*c]
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    orientations = ["vertical", "horizontal"]
    rng.shuffle(orientations)
    orientations += [str(rng.choice(["vertical", "horizontal"])) for _ in range(nt-2)]
    puzzle = {"train": [], "test": []}
    seen, train_positions, train_geometries = set(), set(), []
    for index in range(nt+ns):
        training = index < nt
        split = "train" if training else "test"
        orientation = orientations[index] if training else str(rng.choice(["vertical", "horizontal"]))
        for _ in range(160):
            if not training and rng.random() < .5:
                geometry = train_geometries[int(rng.integers(len(train_geometries)))]
            else:
                geometry = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
            r, c = geometry
            positions = [(i, j) for i in range(r) for j in range(c)
                         if (i, j) not in train_positions or (training and index > 1)]
            if not positions:
                continue
            source = positions[int(rng.integers(len(positions)))]
            source_count = (3 if index == 0 else int(rng.integers(2, 4))) if training else int(rng.integers(4, 7))
            try:
                pair = _make_pair(rng, orientation, geometry, source, source_count, local_trap=index == 0)
                a = _check_pair(pair)
            except AssertionError:
                continue
            key = _key(pair["input"])
            if key in seen:
                continue
            seen.add(key)
            puzzle[split].append(pair)
            if training:
                train_positions.add(source)
                train_geometries.append(geometry)
            break
        else:
            raise RuntimeError("could not sample a uniquely recoverable example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "dimensions must be preserved")
    a = _inspect_input(inp)
    r, c = a["r"], a["c"]
    clean, core, marker = a["clean"], a["core"], a["marker"]
    predicted = _broadcast(clean, a["tile"], a["signals"], r, c)
    _require(np.array_equal(out, _restore_lattice(inp, a, predicted)), "incorrect fault-stencil output")
    changed = core != clean
    _require(int(changed.sum()) <= .30*core.size, "underlay is not sparsely corrupted")
    _require(np.any(a["signals"] & np.isin(a["tile"], np.unique(clean))),
             "source needs a fault colored like a legitimate band")
    _require(not np.any(out == marker), "marker must disappear")
    _require(not np.array_equal(predicted, clean), "omitting A must change the answer")
    literal = np.repeat(np.repeat(a["tile"], r, axis=0), c, axis=1)
    _require(not np.array_equal(predicted, literal), "omitting B must change the answer")
    # A deliberately sparse marked panel makes density-based source selection
    # fail, and leaves at least one zero-signal wrong-source alternative.
    source_faults = int(a["signals"].sum())
    sparse = False
    for i in range(r):
        for j in range(c):
            if (i, j) == a["source"]:
                continue
            raw = core[i*r:(i+1)*r, j*c:(j+1)*c]
            base = clean[i*r:(i+1)*r, j*c:(j+1)*c]
            residual = (raw != base) & (raw != marker)
            if np.count_nonzero(raw != base) < source_faults and not np.any(residual):
                sparse = True
    _require(sparse, "need a marked decoy sparser than the source")
    local_lines = a["tile"].T if a["orientation"] == "vertical" else a["tile"]
    base_lines = a["tile_clean"].T if a["orientation"] == "vertical" else a["tile_clean"]
    a["local_trap"] = any(_strict_mode(x) not in (None, int(y[0]))
                          for x, y in zip(local_lines, base_lines))
    return a


def validate_generated_puzzle(puzzle):
    """Check observable family invariants, teaching witnesses, and labels."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp = _grid(pair["input"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(_check_pair(pair))
    train = analyses["train"]
    _require({a["orientation"] for a in train} == {"vertical", "horizontal"}, "teach both stripe orientations")
    _require(len({a["source"] for a in train}) >= 2, "teach position-independent selection")
    _require(any(a["local_trap"] for a in train), "teach global rather than tile-local repair")
    _require(all(a["source"] not in {t["source"] for t in train} for a in analyses["test"]),
             "test source locations must be held out")
    _require(min(int(a["signals"].sum()) for a in analyses["test"])
             > max(int(a["signals"].sum()) for a in train), "test must combine more signals")


def _signature(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed structural audit, ablations, and solver metamorphisms."""
    signatures, shapes, geometries, layouts, roles, positions = set(), set(), set(), set(), set(), set()
    orientations, counts, output_colors = set(), set(), set()
    variable_stripes = unequal_gaps = local_traps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _check_pair(pair)
                _require(np.array_equal(solve(inp), out), "solver disagreement")
                shapes.add(inp.shape)
                geometries.add((a["r"], a["c"]))
                roles.add((a["separator"], a["marker"]))
                positions.add((a["r"], a["c"], a["source"]))
                orientations.add(a["orientation"])
                counts.add(int(a["signals"].sum()))
                output_colors.update(int(x) for x in np.unique(out))
                v = a["clean"][0] if a["orientation"] == "vertical" else a["clean"][:, 0]
                d = a["c"] if a["orientation"] == "vertical" else a["r"]
                variable_stripes += not np.all(v.reshape(d, d) == v[:d])
                layouts.add((a["orientation"], a["r"], a["c"], tuple(int(x) for x in v)))
                gaps = [runs[i+1][0]-runs[i][1] for runs in (a["row_runs"], a["col_runs"])
                        for i in range(len(runs)-1)]
                unequal_gaps += len(set(gaps)) > 1
                local_traps += a["local_trap"]
        # These are whole-grid symmetries, not latent source transformations.
        pair = puzzle["test"][0]
        for transform in (lambda x: x.T, lambda x: np.rot90(x), lambda x: np.fliplr(x)):
            _require(np.array_equal(solve(transform(pair["input"])), transform(pair["output"])),
                     "geometric equivariance failed")
        recolor = np.random.default_rng(seed+1000).permutation(10)
        _require(np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]]),
                 "color permutation changed the rule")
    _require(len(signatures) == 42, "seeds did not vary puzzles")
    _require(len(shapes) >= 25 and len(geometries) == 9, "geometry coverage too narrow")
    _require(len(layouts) >= 150 and len(roles) >= 50 and len(positions) >= 75, "structural diversity too narrow")
    _require(orientations == {"vertical", "horizontal"} and counts.issuperset({2, 3, 4, 5, 6}), "missing basic modes")
    _require(output_colors == set(range(10)) and min(variable_stripes, unequal_gaps, local_traps) >= 42,
             "missing palette, stripe, gap, or global-evidence variation")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(7919+nt*10+ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_signature(generate_puzzle(12345)) == _signature(generate_puzzle(12345)), "nondeterminism")
    bad = copy.deepcopy(generate_puzzle(9))
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed answer")
    return dict(seeds=42, pairs=210, shapes=len(shapes), geometries=len(geometries),
                stripe_layouts=len(layouts), role_assignments=len(roles), source_positions=len(positions),
                variable_stripes=variable_stripes, unequal_gaps=unequal_gaps, local_majority_traps=local_traps)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
