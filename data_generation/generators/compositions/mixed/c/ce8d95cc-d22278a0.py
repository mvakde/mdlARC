"""Collapse a marked line scaffold, then draw competing corner contours.

Only the input determines the answer.  Generation never imports parent code,
and all checks and diagnostics are bounded and remain in memory.
"""

from __future__ import annotations

import hashlib

import numpy as np


HOST_SEEDS = (
    0, 168642606, 961528459, 53279614, 646906829, 187502105,
    742576478, 946346723, 539291899, 673045079, 769772238, 577655346,
    304623000, 910824255, 61768428, 574969239, 940694306, 280795908,
    408420072, 53589476, 694303357, 185041337, 88441973, 171320045,
    64988978, 115691, 871446090, 800642560, 853322362, 593833209,
    421197887, 964722122, 365344180, 755968249, 52849244, 766266240,
    750340752, 95754069, 284730525, 134391597, 153771100, 969505017,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " out of range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("grid must be an integer rectangle, sides 1..30, colors 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _run_starts(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]


def _collapse(grid):
    # Equality is literal and consecutive. Separated repetitions are retained.
    rows = _run_starts(grid, 0)
    cols = _run_starts(grid, 1)
    return grid[np.ix_(rows, cols)].copy()


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("family input needs a unique most frequent background")
    return int(colors[counts.argmax()])


def _corners(shape):
    h, w = shape
    return ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0))


def _sources(grid, background):
    return [(r, c, int(grid[r, c])) for r, c in _corners(grid.shape)
            if grid[r, c] != background]


def _fields(shape, sources, metric="manhattan"):
    rows, cols = np.indices(shape)
    dy = np.stack([abs(rows - r) for r, _, _ in sources])
    dx = np.stack([abs(cols - c) for _, c, _ in sources])
    distance = dy + dx if metric == "manhattan" else np.maximum(dy, dx)
    nearest = distance == distance.min(axis=0)
    return dy, dx, nearest, distance.argmin(axis=0)


def _paint(shape, background, sources, *, metric="manhattan",
           radius="chebyshev", ties="blank", parity=0):
    dy, dx, nearest, owner = _fields(shape, sources, metric)
    unique = nearest.sum(axis=0) == 1
    if ties == "first":
        unique[:] = True
    elif ties == "same_color":
        # Deliberately wrong alternative: fill ties when colors agree.
        low = np.where(nearest, np.array([s[2] for s in sources])[:, None, None], 10).min(axis=0)
        high = np.where(nearest, np.array([s[2] for s in sources])[:, None, None], -1).max(axis=0)
        unique |= low == high
    answer = np.full(shape, background, dtype=np.int8)
    radii = np.maximum(dy, dx) if radius == "chebyshev" else dy + dx
    for i, (_, _, color) in enumerate(sources):
        answer[unique & (owner == i) & (radii[i] % 2 == parity)] = color
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Literal run collapse must precede source extraction and both distances."""
    grid = _grid(input_grid)
    background = _background(grid)
    canonical = _collapse(grid)
    if canonical.shape[0] != canonical.shape[1] or canonical.shape[0] < 3:
        raise ValueError("collapsed family canvas must be square, side at least 3")
    sources = _sources(canonical, background)
    if not sources:
        raise ValueError("family canvas needs a marked corner")
    return _paint(canonical.shape, background, sources)


def _axis_lines(n, occupied_edges, rng):
    """One-cell gaps, plus optional split margins distinguished by a corner.

    Odd sizes use one-cell margins or two split margins. Even sizes use one
    split margin. Its outer row/column contains a source, so the adjacent
    spacing vector is genuinely different before any expansion.
    """
    if n % 2:
        split = bool(all(occupied_edges) and rng.integers(0, 2))
        margins = (2, 2) if split else (1, 1)
    else:
        choices = [i for i, occupied in enumerate(occupied_edges) if occupied]
        side = int(rng.choice(choices))
        margins = (2, 1) if side == 0 else (1, 2)
    first, last = margins
    return tuple(range(first, n - last, 2))


def _expansion(n, lines, rng, mode):
    runs = np.ones(n, dtype=int)
    gaps = [i for i in range(1, n - 1) if i not in lines]
    if mode == "none":
        return runs
    if mode == "witness":
        # An odd raw offset is introduced on each axis; unequal gaps defeat
        # a uniform-stride sampler even though the canonical coordinates agree.
        first, second = rng.choice(gaps, size=2, replace=False)
        runs[int(first)] += 1
        runs[int(second)] += 2
        extra = int(rng.integers(0, min(5, 30 - int(runs.sum())) + 1))
    else:
        extra = int(rng.integers(1, min(12, 30 - n) + 1))
    if extra:
        style = int(rng.integers(0, 3))
        if style == 0:
            allocation = rng.multinomial(extra, np.ones(len(gaps)) / len(gaps))
            runs[gaps] += allocation
        else:
            target = gaps[0] if style == 1 else int(rng.choice(gaps))
            runs[target] += extra
    return runs


def _mask_options(minimum=1):
    return [tuple(c for c in range(4) if bits & (1 << c))
            for bits in range(1, 16) if bits.bit_count() >= minimum]


def _adjacent(mask):
    return len(mask) == 2 and (mask[1] - mask[0]) % 4 in (1, 3)


def _make_input(rng, n, mask, background, repeated_sources, expansion):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    source_colors = palette[:len(mask)]
    if repeated_sources:
        opposite = next(((a, b) for a in mask for b in mask if b - a == 2), None)
        if opposite is not None:
            source_colors[mask.index(opposite[1])] = source_colors[mask.index(opposite[0])]
        else:
            source_colors[-1] = source_colors[0]
    scaffold_palette = [c for c in palette if c not in source_colors]
    rng.shuffle(scaffold_palette)
    scaffold_palette = scaffold_palette[:int(rng.integers(3, min(6, len(scaffold_palette)) + 1))]
    positions = _corners((n, n))
    row_edges = (any(c in mask for c in (0, 1)), any(c in mask for c in (2, 3)))
    col_edges = (any(c in mask for c in (0, 3)), any(c in mask for c in (1, 2)))
    row_lines = _axis_lines(n, row_edges, rng)
    col_lines = _axis_lines(n, col_edges, rng)
    colors = [scaffold_palette[i % len(scaffold_palette)]
              for i in range(len(row_lines) + len(col_lines))]
    rng.shuffle(colors)
    horizontal, vertical = colors[:len(row_lines)], colors[len(row_lines):]
    canonical = np.full((n, n), background, dtype=np.int8)
    for c, color in zip(col_lines, vertical):
        canonical[:, c] = color
    for r, color in zip(row_lines, horizontal):
        canonical[r, :] = color
    crossing_mode = int(rng.integers(0, 5))
    row_bits = rng.integers(0, 2, size=len(row_lines))
    col_bits = rng.integers(0, 2, size=len(col_lines))
    for i, r in enumerate(row_lines):
        for j, c in enumerate(col_lines):
            bit = (crossing_mode if crossing_mode < 2 else
                   int(row_bits[i]) if crossing_mode == 2 else
                   int(col_bits[j]) if crossing_mode == 3 else int(rng.integers(0, 2)))
            canonical[r, c] = vertical[j] if bit else horizontal[i]
    for corner, color in zip(mask, source_colors):
        canonical[positions[corner]] = color
    row_mode, col_mode = expansion
    row_runs = _expansion(n, row_lines, rng, row_mode)
    col_runs = _expansion(n, col_lines, rng, col_mode)
    raw = np.repeat(np.repeat(canonical, row_runs, axis=0), col_runs, axis=1)
    return raw, canonical


def _choose_geometry(rng, sizes, masks, used):
    choices = [(n, mask) for n in sizes for mask in masks if (n, mask) not in used]
    if not choices:
        raise AssertionError("geometry inventory exhausted")
    result = choices[int(rng.integers(len(choices)))]
    used.add(result)
    return result


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    background = int(rng.integers(0, 10))
    changing_background = bool(rng.integers(0, 2))
    used = set()
    adjacent = [m for m in _mask_options(2) if _adjacent(m)]
    triples = [m for m in _mask_options(3) if len(m) == 3]
    # Even the minimum two demonstrations expose both metrics, contour parity,
    # active ties, same-color ties, unequal spacing, and repeated separated gaps.
    a = _choose_geometry(rng, [7, 9, 11, 13], adjacent, used)
    b = _choose_geometry(rng, [8, 10, 12, 14], triples, used)
    specs = [(a, False, ("witness", "witness")),
             (b, True, ("witness", "witness"))]
    for _ in range(num_train - 2):
        geom = _choose_geometry(rng, list(range(7, 17)), _mask_options(), used)
        mode = int(rng.integers(0, 4))
        expansion = (("none", "random") if mode == 0 else
                     ("random", "none") if mode == 1 else ("random", "random"))
        specs.append((geom, len(geom[1]) > 1 and bool(rng.integers(0, 3) == 0), expansion))
    rng.shuffle(specs)
    train_sizes = {geom[0] for geom, _, _ in specs}
    single_shown = any(len(geom[1]) == 1 for geom, _, _ in specs)
    test_masks = _mask_options(1 if single_shown else 2)
    test_specs = []
    for i in range(num_test):
        shift = int(rng.integers(0, 3))
        sizes = (list(range(max(train_sizes) + 1, 25)) if shift == 0 else
                 sorted(train_sizes) if shift == 1 else list(range(7, 17)))
        geom = _choose_geometry(rng, sizes, test_masks, used)
        test_specs.append((geom, len(geom[1]) > 1 and bool(rng.integers(0, 2)),
                           ("random", "random")))
    puzzle = {"train": [], "test": []}
    for split, planned in (("train", specs), ("test", test_specs)):
        for index, ((n, mask), repeated, expansion) in enumerate(planned):
            bg = ((background + index) % 10 if changing_background else background)
            for _ in range(64):
                raw, canonical = _make_input(rng, n, mask, bg, repeated, expansion)
                try:
                    if _background(raw) != bg:
                        continue
                except ValueError:
                    continue
                sources = _sources(canonical, bg)
                answer = _paint(canonical.shape, bg, sources)
                _require(np.array_equal(_collapse(raw), canonical), "construction lost a vector")
                # Mandatory demonstrations must actually defeat the named shortcuts.
                if expansion[0] == "witness":
                    witnesses = _witnesses(raw, canonical, answer, bg)
                    required = ("collapse_before_distance", "chebyshev_radius", "active_tie")
                    if not all(witnesses[name] for name in required):
                        continue
                    if not repeated and not witnesses["manhattan_ownership"]:
                        continue
                    if repeated and np.array_equal(
                            _paint(canonical.shape, bg, sources, ties="same_color"), answer):
                        continue
                puzzle[split].append({"input": raw, "output": answer})
                break
            else:
                raise RuntimeError("bounded construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(raw, canonical, answer, bg):
    sources = _sources(canonical, bg)
    # Compare raw distances sampled at the ACTUAL run starts. This gives the
    # wrong ordering every possible benefit, including correct final shape.
    raw_answer = _paint(raw.shape, bg, _sources(raw, bg))
    sampled = raw_answer[np.ix_(_run_starts(raw, 0), _run_starts(raw, 1))]
    wrong_metric = _paint(canonical.shape, bg, sources, metric="chebyshev")
    wrong_radius = _paint(canonical.shape, bg, sources, radius="manhattan")
    wrong_tie = _paint(canonical.shape, bg, sources, ties="first")
    return {"collapse_before_distance": not np.array_equal(sampled, answer),
            "manhattan_ownership": not np.array_equal(wrong_metric, answer),
            "chebyshev_radius": not np.array_equal(wrong_radius, answer),
            "active_tie": not np.array_equal(wrong_tie, answer)}


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    raw, answer = _grid(pair["input"]), _grid(pair["output"])
    bg = _background(raw)
    canonical = _collapse(raw)
    n = canonical.shape[0]
    _require(canonical.shape == (n, n) and 7 <= n <= 24, "canonical size")
    _require(answer.shape == canonical.shape and raw.shape != answer.shape, "collapse dimensions")
    sources = _sources(canonical, bg)
    mask = tuple(i for i, (r, c) in enumerate(_corners(canonical.shape)) if canonical[r, c] != bg)
    _require(1 <= len(sources) <= 4, "corner sources")
    source_colors = {s[2] for s in sources}
    # Remove just the corner markers to inspect the orthogonal scaffold.
    scaffold = canonical.copy()
    for r, c in _corners(canonical.shape):
        scaffold[r, c] = bg
    hlines = [r for r in range(n) if np.all(scaffold[r] != bg)]
    vlines = [c for c in range(n) if np.all(scaffold[:, c] != bg)]
    _require(hlines and vlines and 0 not in hlines + vlines and n - 1 not in hlines + vlines,
             "interior orthogonal lines required")
    gaps_r = [r for r in range(n) if r not in hlines]
    gaps_c = [c for c in range(n) if c not in vlines]
    _require(np.all(scaffold[np.ix_(gaps_r, gaps_c)] == bg), "unexplained scaffold mark")
    horizontal = [int(scaffold[r, 0]) for r in hlines]
    vertical = [int(scaffold[0, c]) for c in vlines]
    for i, r in enumerate(hlines):
        _require(np.all(scaffold[r, gaps_c] == horizontal[i]), "horizontal signature")
        for j, c in enumerate(vlines):
            _require(int(scaffold[r, c]) in (horizontal[i], vertical[j]), "crossing value")
    for j, c in enumerate(vlines):
        _require(np.all(scaffold[gaps_r, c] == vertical[j]), "vertical signature")
    _require(not source_colors & set(horizontal + vertical), "source/scaffold palettes overlap")
    rows, cols = _run_starts(raw, 0), _run_starts(raw, 1)
    row_runs = np.diff(np.r_[rows, raw.shape[0]])
    col_runs = np.diff(np.r_[cols, raw.shape[1]])
    _require(np.all(row_runs[hlines] == 1) and np.all(col_runs[vlines] == 1), "thick line")
    _require(row_runs[0] == row_runs[-1] == col_runs[0] == col_runs[-1] == 1, "corner must be singleton")
    reconstructed = np.repeat(np.repeat(canonical, row_runs, axis=0), col_runs, axis=1)
    _require(np.array_equal(reconstructed, raw), "lossless run reconstruction")
    # Direct cellwise check of the specified rule; still generator-owned.
    for r in range(n):
        for c in range(n):
            ds = [abs(r - sr) + abs(c - sc) for sr, sc, _ in sources]
            nearest = min(ds)
            expected = bg
            if ds.count(nearest) == 1:
                sr, sc, color = sources[ds.index(nearest)]
                if max(abs(r - sr), abs(c - sc)) % 2 == 0:
                    expected = color
            _require(int(answer[r, c]) == expected, "territory/contour cell mismatch")
    _require(np.array_equal(solve(raw), answer), "solve mismatch")
    _require(not np.array_equal(canonical, answer), "parent B is inactive")
    _require(set(np.unique(answer)) == source_colors | {bg}, "wrong output palette")
    same_tie = not np.array_equal(_paint(canonical.shape, bg, sources, ties="same_color"), answer)
    return {"n": n, "mask": mask, "background": bg,
            "hlines": len(hlines), "vlines": len(vlines),
            "row_runs": tuple(int(x) for x in row_runs),
            "col_runs": tuple(int(x) for x in col_runs),
            "same_color_tie": same_tie,
            "witnesses": _witnesses(raw, canonical, answer, bg)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    records, seen, geometries = {}, set(), set()
    for split in ("train", "test"):
        records[split] = []
        for pair in puzzle[split]:
            record = _inspect_pair(pair)
            key = _key(np.asarray(pair["input"], dtype=np.int8))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            geometry = (record["n"], record["mask"])
            _require(geometry not in geometries, "recolored repeated source geometry")
            geometries.add(geometry)
            records[split].append(record)
    train = records["train"]
    _require(any(_adjacent(r["mask"]) and r["n"] % 2 and
                 all(r["witnesses"].values()) for r in train), "simple active tie demonstration")
    _require(any(len(r["mask"]) == 3 and r["n"] % 2 == 0 and r["same_color_tie"] and
                 all(r["witnesses"][name] for name in
                     ("collapse_before_distance", "chebyshev_radius", "active_tie"))
                 for r in train), "same-color tie demonstration")
    _require(any(max(r["row_runs"]) > 1 and max(r["col_runs"]) > 1 for r in train),
             "two-axis compression not demonstrated")
    _require(all(len(r["mask"]) > 1 for r in records["test"]) or
             any(len(r["mask"]) == 1 for r in train), "single-source extension not demonstrated")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 specified seeds, all 30 legal count combinations, no file output."""
    shapes, geometries, layouts, backgrounds, masks, hashes = (set() for _ in range(6))
    single_axis = repeated_ties = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                rec = _inspect_pair(pair)
                shapes.add(tuple(pair["input"].shape))
                geometries.add((rec["n"], rec["mask"], rec["hlines"], rec["vlines"]))
                layouts.add((rec["row_runs"], rec["col_runs"]))
                backgrounds.add(rec["background"])
                masks.add(rec["mask"])
                single_axis += (max(rec["row_runs"]) == 1) != (max(rec["col_runs"]) == 1)
                repeated_ties += rec["same_color_tie"]
        # Palette permutations preserve spatial rules, including zero sources.
        perm = np.random.default_rng(seed + 1).permutation(10)
        for pair in puzzle["test"]:
            _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]),
                     "palette equivariance")
        # Quarter-turns preserve collapse and both metrics.
        for pair in puzzle["train"][:1]:
            _require(np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"])),
                     "rotation equivariance")
    _require(len(hashes) == 42 and len(shapes) >= 70, "insufficient input variation")
    _require(len(geometries) >= 75 and len(layouts) >= 150, "insufficient geometry variation")
    _require(len(backgrounds) == 10 and len(masks) >= 12, "insufficient role/mask variation")
    _require(single_axis >= 10 and repeated_ties >= 42, "missing boundary regimes")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123456 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "input_shapes": len(shapes),
            "canonical_geometries": len(geometries), "spacing_layouts": len(layouts),
            "backgrounds": len(backgrounds), "corner_masks": len(masks),
            "one_axis_compression_pairs": single_axis, "same_color_tie_pairs": repeated_ties,
            "legal_count_combinations": 30, "independent_solver": False}


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), separators=(",", ":")))
