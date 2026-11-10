"""Competitive seams: length-ranked ownership in a noisy panel lattice.

The renderer constructs labels from its explicit rectangles.  The public solver
reconstructs the lattice and rectangles from the input alone.  No parent code,
files, example labels, puzzle palette, or random state is used by solve().
"""

from __future__ import annotations

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape), "bad grid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= grid) & (grid <= 9)), "bad ARC color")
    return grid.astype(np.int64, copy=False)


def _key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def _intervals(side, separators):
    edges = [-1, *separators, side]
    return [(a + 1, b) for a, b in zip(edges, edges[1:])]


def _core(block):
    """The unique solid 2D rectangle, recovered without an inset or mode lookup."""
    equal = ((block[:-1, :-1] == block[1:, :-1]) &
             (block[:-1, :-1] == block[:-1, 1:]) &
             (block[:-1, :-1] == block[1:, 1:]))
    rr, cc = np.nonzero(equal)
    _need(len(rr) > 0, "chamber has no solid core")
    values = np.unique(block[rr, cc])
    _need(len(values) == 1, "competing solid cores")
    color = int(values[0])
    top, bottom = int(rr.min()), int(rr.max()) + 2
    left, right = int(cc.min()), int(cc.max()) + 2
    height, width = bottom - top, right - left
    _need(np.all(block[top:bottom, left:right] == color), "nonrectangular core")
    _need(len(rr) == (height - 1) * (width - 1), "fragmented solid region")
    _need(2 <= min(height, width) < max(height, width), "core must be an elongated rectangle")
    _need(top >= 1 and left >= 1 and bottom < block.shape[0] and right < block.shape[1],
          "a core needs camouflage on every side")
    # No same-color protrusion can silently extend a measured core.
    for edge in (block[top - 1, left:right], block[bottom, left:right],
                 block[top:bottom, left - 1], block[top:bottom, right]):
        _need(not np.any(edge == color), "core merges into its camouflage")
    counts = np.bincount(block.ravel(), minlength=10)
    return {"color": color, "box": (top, bottom, left, right),
            "length": max(height, width), "thickness": min(height, width),
            "axis": 0 if height > width else 1,
            "mode": int(counts.argmax()), "mode_margin": int(counts.max() - counts[color]),
            "fixed_sample": int(block[1, 1]), "shape": block.shape}


def _parse(input_grid):
    grid = _grid(input_grid)
    rows = [int(x) for x in np.flatnonzero(np.all(grid == grid[:, :1], axis=1))]
    cols = [int(x) for x in np.flatnonzero(np.all(grid == grid[:1, :], axis=0))]
    _need(1 <= len(rows) <= 2 and 1 <= len(cols) <= 2, "need a complete separator lattice")
    _need((len(rows) + 1) * (len(cols) + 1) <= 6, "too many chambers")
    separator = int(grid[rows[0], cols[0]])
    _need(all(np.all(grid[r, :] == separator) for r in rows) and
          all(np.all(grid[:, c] == separator) for c in cols), "separator colors disagree")
    row_spans, col_spans = _intervals(grid.shape[0], rows), _intervals(grid.shape[1], cols)
    _need(all(b - a >= 4 for a, b in row_spans + col_spans), "chamber too small")
    cores, blocks = [], []
    for ra, rb in row_spans:
        core_row, block_row = [], []
        for ca, cb in col_spans:
            block = grid[ra:rb, ca:cb]
            core_row.append(_core(block))
            block_row.append(block)
        cores.append(core_row)
        blocks.append(block_row)
    flat = [core for row in cores for core in row]
    _need(len({x["axis"] for x in flat}) == 1, "bars must be parallel")
    _need(len({x["length"] for x in flat}) == len(flat), "length ties are outside this family")
    _need(len({x["color"] for x in flat}) == len(flat), "core colors must be distinct")
    _need(separator not in {x["color"] for x in flat}, "separator is also a core color")
    axis = flat[0]["axis"]
    # The bars in each bank share a terminal baseline, as in parent A.
    banks = cores if axis == 0 else list(zip(*cores))
    edge_slots = (0, 1) if axis == 0 else (2, 3)
    for bank in banks:
        _need(any(len({x["box"][side] for x in bank}) == 1 for side in edge_slots),
              "bank has no common terminal baseline")
    lengths = np.asarray([[x["length"] for x in row] for row in cores], dtype=int)
    colors = np.asarray([[x["color"] for x in row] for row in cores], dtype=int)
    return {"grid": grid, "separator": separator, "row_spans": row_spans,
            "col_spans": col_spans, "cores": cores, "blocks": blocks,
            "lengths": lengths, "colors": colors, "axis": axis}


def _owners(lengths, measure=None):
    """At each compressed site, index the longest incident core."""
    m, n = lengths.shape
    score = lengths if measure is None else measure
    owners = np.empty((2 * m - 1, 2 * n - 1), dtype=int)
    for r in range(owners.shape[0]):
        ri = (r // 2,) if r % 2 == 0 else (r // 2, r // 2 + 1)
        for c in range(owners.shape[1]):
            ci = (c // 2,) if c % 2 == 0 else (c // 2, c // 2 + 1)
            candidates = [(i, j) for i in ri for j in ci]
            i, j = max(candidates, key=lambda ij: int(score[ij]))
            owners[r, c] = i * n + j
    return owners


def _from_info(info, measure=None):
    return info["colors"].ravel()[_owners(info["lengths"], measure)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Denoise the bars, then let relative length decide each local seam."""
    return _from_info(_parse(input_grid))


def _direct_label(lengths, colors):
    """Paint from explicit construction data; this never parses an input."""
    m, n = lengths.shape
    output = np.empty((2 * m - 1, 2 * n - 1), dtype=np.int64)
    output[::2, ::2] = colors
    for r in range(m):
        for c in range(n - 1):
            winner = c if lengths[r, c] > lengths[r, c + 1] else c + 1
            output[2 * r, 2 * c + 1] = colors[r, winner]
    for r in range(m - 1):
        for c in range(n):
            winner = r if lengths[r, c] > lengths[r + 1, c] else r + 1
            output[2 * r + 1, 2 * c] = colors[winner, c]
        for c in range(n - 1):
            dr, dc = np.unravel_index(int(np.argmax(lengths[r:r + 2, c:c + 2])), (2, 2))
            output[2 * r + 1, 2 * c + 1] = colors[r + dr, c + dc]
    return output


def _camouflage(rng, shape, box, color, palette, biased):
    """Noise may reuse every signal color, but cannot imitate a solid 2D core."""
    top, bottom, left, right = box
    grid = np.full(shape, -1, dtype=int)
    grid[top:bottom, left:right] = color
    favored = int(rng.choice([x for x in palette if x != color]))
    weights = rng.dirichlet(np.ones(len(palette)))
    if biased:
        weights *= 0.22
        weights[palette.index(favored)] += 0.78
    for r in range(shape[0]):
        for c in range(shape[1]):
            if grid[r, c] != -1:
                continue
            forbidden = set()
            if ((r in (top - 1, bottom) and left <= c < right) or
                    (c in (left - 1, right) and top <= r < bottom)):
                forbidden.add(color)
            for a in (r - 1, r):
                for b in (c - 1, c):
                    if not (0 <= a < shape[0] - 1 and 0 <= b < shape[1] - 1):
                        continue
                    others = [int(grid[i, j]) for i in (a, a + 1) for j in (b, b + 1)
                              if (i, j) != (r, c)]
                    if others[0] >= 0 and len(set(others)) == 1:
                        forbidden.add(others[0])
            indices = [i for i, value in enumerate(palette) if value not in forbidden]
            if indices:
                probabilities = weights[indices]
                probabilities = probabilities / probabilities.sum()
                value = palette[int(rng.choice(indices, p=probabilities))]
            else:
                value = int(rng.choice([x for x in range(10) if x not in forbidden]))
            grid[r, c] = value
    return grid


def _draw(rng, lengths, colors, separator, widths, rotation, witness=False):
    m, n = lengths.shape
    row_min = lengths.max(axis=1) + 2
    col_min = widths.max(axis=0) + 2
    min_h, min_w = int(row_min.sum() + m - 1), int(col_min.sum() + n - 1)
    _need(min_h <= 30 and min_w <= 30, "layout cannot fit")
    # Slack changes band widths, offsets and aspect ratios, rather than scaling
    # every panel by a common factor.  Avoid a huge sea of noise.
    extra_h = int(rng.integers(0, min(30 - min_h, 7) + 1))
    extra_w = int(rng.integers(0, min(30 - min_w, 9) + 1))
    row_sizes = row_min + rng.multinomial(extra_h, np.full(m, 1 / m))
    col_sizes = col_min + rng.multinomial(extra_w, np.full(n, 1 / n))
    height, width = int(row_sizes.sum() + m - 1), int(col_sizes.sum() + n - 1)
    grid = np.full((height, width), separator, dtype=np.int64)
    noise_size = int(rng.integers(3, 11))
    palette = [int(x) for x in rng.choice(10, noise_size, replace=False)]
    row0 = 0
    for r in range(m):
        bank_slack = int(row_sizes[r] - lengths[r].max() - 2)
        bottom_gap = int(rng.integers(1, min(3, bank_slack + 1) + 1))
        bottom = int(row_sizes[r] - bottom_gap)
        col0 = 0
        for c in range(n):
            left = int(rng.integers(1, int(col_sizes[c] - widths[r, c])))
            box = (bottom - int(lengths[r, c]), bottom, left, left + int(widths[r, c]))
            block = _camouflage(rng, (int(row_sizes[r]), int(col_sizes[c])), box,
                                int(colors[r, c]), palette, witness or rng.random() < 0.45)
            grid[row0:row0 + row_sizes[r], col0:col0 + col_sizes[c]] = block
            col0 += int(col_sizes[c]) + 1
        row0 += int(row_sizes[r]) + 1
    output = _direct_label(lengths, colors)
    return {"input": np.rot90(grid, rotation).copy(),
            "output": np.rot90(output, rotation).copy()}


def _rank_matrix(lengths):
    ordered = sorted((int(x) for x in lengths.ravel()), reverse=True)
    return np.asarray([[ordered.index(int(x)) + 1 for x in row] for row in lengths], dtype=int)


def _owner_key(info):
    return _key(_owners(info["lengths"]))


def _area(info):
    return np.asarray([[x["length"] * x["thickness"] for x in row]
                       for row in info["cores"]], dtype=int)


def _global_shortcut(info):
    result = info["colors"].ravel()[_owners(info["lengths"])].copy()
    seam = np.ones(result.shape, dtype=bool)
    seam[::2, ::2] = False
    result[seam] = info["colors"].ravel()[int(info["lengths"].argmax())]
    return result


def _random_geometry(rng, shape):
    m, n = shape
    count = m * n
    for _ in range(200):
        ceiling = 12 if count == 4 else (11 if m == 2 else 9)
        lengths = rng.choice(np.arange(3, ceiling + 1), size=count, replace=False).reshape(shape)
        if int(lengths.max(axis=1).sum() + 3 * m - 1) <= 30:
            break
    else:
        # Feasible even in the three-bank case; this is a construction fallback,
        # not a separate solving convention.
        lengths = np.arange(3, count + 3).reshape(shape)
        lengths = lengths[rng.permutation(m)]
        lengths = lengths[:, rng.permutation(n)]
    widths = np.asarray([[int(rng.integers(2, min(4, int(x) - 1) + 1)) for x in row]
                         for row in lengths], dtype=int)
    return lengths.astype(int), widths


def _make_pair(rng, lengths, colors, separator, widths, rotation, seen,
               witness=False, forbidden_owners=None):
    for _ in range(100):
        pair = _draw(rng, lengths, colors, separator, widths, rotation, witness)
        try:
            info = _parse(pair["input"])
        except ValueError:
            continue  # A chance full-length monochromatic noise line is illegal.
        if _key(pair["input"]) in seen:
            continue
        if forbidden_owners is not None and _owner_key(info) in forbidden_owners:
            return None  # Noise cannot change ownership; resample the geometry.
        if witness:
            flat = [x for row in info["cores"] for x in row]
            if not any(x["mode_margin"] > 0 for x in flat):
                continue
            if not any(x["fixed_sample"] != x["color"] for x in flat):
                continue
        _need(np.array_equal(solve(pair["input"]), pair["output"]), "renderer/solver disagree")
        seen.add(_key(pair["input"]))
        return pair
    raise RuntimeError("could not draw a clear, distinct lattice")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    palette = [int(x) for x in rng.permutation(10)]
    first_rotation = int(rng.integers(4))
    second_rotation = (first_rotation + 1) % 4
    anchor = int(rng.integers(5, 7))
    # A shared length changes rank 2 -> 3; its neighbor changes from shorter
    # to longer.  Every rank also moves in the physical 2x2 panel order.
    while True:
        ranks = rng.permutation(np.arange(1, 5)).reshape(2, 2)
        a = tuple(np.argwhere(ranks == 2)[0])
        b = tuple(np.argwhere(ranks == 3)[0])
        if abs(int(a[0] - b[0])) + abs(int(a[1] - b[1])) == 1:
            break
    first_lengths = np.asarray([anchor + 1, anchor, anchor - 1, anchor - 2])[ranks - 1]
    first_widths = np.full((2, 2), 2, dtype=int)
    first_widths[ranks == 2] = 3  # Shorter bar has greater area than rank 1.
    first_colors = np.asarray(palette[:4], dtype=int).reshape(2, 2)
    physical_old_ranks = np.rot90(ranks, first_rotation)
    physical_second_ranks = np.asarray([4, 3, 1, 2])[physical_old_ranks - 1]
    second_ranks = np.rot90(physical_second_ranks, -second_rotation)
    second_lengths = np.asarray([anchor + 2, anchor + 1, anchor, anchor - 2])[second_ranks - 1]
    physical_colors = np.rot90(first_colors, first_rotation)
    second_colors = np.rot90(physical_colors, -second_rotation)
    second_widths = np.asarray([[int(rng.integers(2, min(4, int(x) - 1) + 1)) for x in row]
                               for row in second_lengths], dtype=int)
    for lengths, colors, sep, widths, rotation, witness in (
            (first_lengths, first_colors, palette[4], first_widths, first_rotation, True),
            (second_lengths, second_colors, palette[5], second_widths, second_rotation, False)):
        puzzle["train"].append(_make_pair(rng, lengths, colors, sep, widths,
                                           rotation, seen, witness))
    for index in range(2, num_train):
        # A third demonstration exposes the six-panel extension whenever the
        # requested training count allows it.  Later examples mix sizes.
        shape = ((2, 3) if rng.integers(2) else (3, 2)) if index == 2 or rng.random() < 0.55 else (2, 2)
        lengths, widths = _random_geometry(rng, shape)
        roles = rng.permutation(10)
        colors = roles[:lengths.size].reshape(shape).astype(int)
        pair = _make_pair(rng, lengths, colors, int(roles[lengths.size]), widths,
                          int(rng.integers(4)), seen)
        puzzle["train"].append(pair)
    train_owners = {_owner_key(_parse(pair["input"])) for pair in puzzle["train"]}
    for _ in range(num_test):
        for attempt in range(200):
            shape = (2, 3) if rng.integers(2) else (3, 2)
            lengths, widths = _random_geometry(rng, shape)
            roles = rng.permutation(10)
            colors = roles[:6].reshape(shape).astype(int)
            pair = _make_pair(rng, lengths, colors, int(roles[6]), widths,
                              int(rng.integers(4)), seen, forbidden_owners=train_owners)
            if pair is not None:
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("could not obtain a new test ownership pattern")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate returned grids, including the teaching witnesses, without metadata."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, info_by_split = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            info_by_split[split].append(info)
            _need(output.shape == tuple(2 * x - 1 for x in info["lengths"].shape), "bad compression")
            _need(np.array_equal(output[::2, ::2], info["colors"]), "core order lost")
            _need(np.array_equal(output, _from_info(info)), "wrong local length winner")
            _need(info["separator"] not in output, "scaffold was not replaced")
            _need(set(int(x) for x in output.ravel()) == set(int(x) for x in info["colors"].ravel()),
                  "a core color was lost or an extraneous color survived")
    train = info_by_split["train"]
    first, second = train[:2]
    _need(first["lengths"].shape == second["lengths"].shape == (2, 2), "missing simple demonstrations")
    _need(first["axis"] != second["axis"], "both longitudinal directions must be demonstrated")
    _need(np.array_equal(first["colors"], second["colors"]), "paired witnesses need comparable colors")
    _need(np.all(_rank_matrix(first["lengths"]) != _rank_matrix(second["lengths"])),
          "fixed-position ranking is not contradicted")
    first_ranks, second_ranks = _rank_matrix(first["lengths"]), _rank_matrix(second["lengths"])
    anchors = [p for p in np.ndindex(2, 2)
               if first["lengths"][p] == second["lengths"][p] and first_ranks[p] != second_ranks[p]]
    _need(anchors, "no relative-rank anchor")
    _need(any(first["lengths"][p] > first["lengths"][q] and
              second["lengths"][p] < second["lengths"][q]
              for p in anchors for q in np.ndindex(2, 2)
              if abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1),
          "same-length anchor must visibly lose a previously owned seam")
    _need(not np.array_equal(_from_info(first), _from_info(second)), "anchor context does not affect ownership")
    _need(not np.array_equal(_from_info(first, _area(first)), _from_info(first)),
          "training does not distinguish area from length")
    _need(any(x["mode_margin"] > 0 for row in first["cores"] for x in row),
          "no denoising-vs-mode witness")
    _need(any(x["fixed_sample"] != x["color"] for row in first["cores"] for x in row),
          "fixed inset sampling was not contradicted")
    _need(any(not np.array_equal(_global_shortcut(x), _from_info(x)) for x in train),
          "locality is not demonstrated")
    train_owners = {_owner_key(x) for x in train}
    for info in info_by_split["test"]:
        _need(info["lengths"].size == 6, "test should recombine six chambers")
        _need(_owner_key(info) not in train_owners, "test repeats a demonstrated ownership pattern")


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded checks over 42 seeds, every legal count, and metamorphic cases."""
    shapes, layouts, normalized, palettes, patterns, lengths, thicknesses = (set() for _ in range(7))
    axes, separators, input_keys = set(), set(), set()
    total = minority = area_conflicts = role_exchanges = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        local_infos = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                info = _parse(pair["input"])
                local_infos.append(info)
                shapes.add(pair["input"].shape)
                layouts.add(info["lengths"].shape)
                axes.add(info["axis"])
                separators.add(info["separator"])
                lengths.update(int(x) for x in info["lengths"].ravel())
                thicknesses.update(x["thickness"] for row in info["cores"] for x in row)
                patterns.add(_owner_key(info))
                palettes.add(tuple(int(x) for x in info["colors"].ravel()))
                normalized.add((pair["input"].shape, info["lengths"].shape, info["axis"],
                                tuple(info["lengths"].ravel()),
                                tuple(x["box"] for row in info["cores"] for x in row)))
                input_keys.add(_key(pair["input"]))
                minority += any(x["mode_margin"] > 0 for row in info["cores"] for x in row)
                area_conflicts += not np.array_equal(_from_info(info, _area(info)), pair["output"])
                _need(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list input failed")
        role_exchanges += any(a["separator"] in b["colors"] for a in local_infos for b in local_infos)
        # Rotation and literal recoloring must commute with the rule.
        pair = puzzle["train"][0]
        for turn in (1, 2, 3):
            _need(np.array_equal(solve(np.rot90(pair["input"], turn)),
                                 np.rot90(pair["output"], turn)), "rotation covariance failed")
        recolor = np.random.default_rng(10000 + seed).permutation(10)
        _need(np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]]),
              "color equivariance failed")
    _need(len(shapes) >= 50 and len(normalized) >= 180, "geometry diversity collapsed")
    _need(layouts == {(2, 2), (2, 3), (3, 2)} and axes == {0, 1}, "lattice variation collapsed")
    _need(len(palettes) >= 150 and separators == set(range(10)), "palette variation collapsed")
    _need(len(patterns) >= 50 and len(input_keys) == total, "ownership or input diversity collapsed")
    _need(len(lengths) >= 9 and thicknesses == {2, 3, 4}, "bar geometry variation collapsed")
    _need(minority >= 42 and area_conflicts >= 42 and role_exchanges >= 20, "witness coverage collapsed")
    # These legitimate family members go beyond the sampler's length and
    # thickness ranges.  Re-rendering with thin cores and different noise must
    # preserve the answer; only longitudinal comparisons govern ownership.
    extra_rng = np.random.default_rng(314159)
    extra_lengths = np.asarray([[16, 13], [5, 3]])
    extra_colors = np.asarray([[0, 7], [4, 2]])
    thick = np.asarray([[5, 6], [3, 2]])
    thin = np.full((2, 2), 2, dtype=int)
    for rotation in range(4):
        broad = _make_pair(extra_rng, extra_lengths, extra_colors, 9, thick, rotation, set())
        narrow = _make_pair(extra_rng, extra_lengths, extra_colors, 9, thin, rotation, set())
        _need(np.array_equal(solve(broad["input"]), solve(narrow["input"])),
              "thickness, noise, or gutter changes altered ownership")
    a, b = generate_puzzle(13579, 7, 3), generate_puzzle(13579, 7, 3)
    _need(_same_puzzle(a, b), "seed is not deterministic")
    _need(not _same_puzzle(a, generate_puzzle(13580, 7, 3)), "seeds collapsed")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(70000 + 10 * train + test, train, test)
            _need(len(puzzle["train"]) == train and len(puzzle["test"]) == test, "counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, rotation and palette checks)")
