"""Modal-tile wavefronts: self-copying controlled by geodesic cyclic runs.

Only NumPy and Python's standard library are used. No parent code is imported.
"""
from collections import deque
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected an integer color grid with dimensions 1..30")
    return a.astype(np.int64, copy=False)


def _orient(grid, code):
    a = np.rot90(grid, code % 4)
    return (np.fliplr(a) if code >= 4 else a).copy()


def _undo(grid, code):
    return np.rot90(np.fliplr(grid) if code >= 4 else grid, -(code % 4)).copy()


def _neighbors(cell, shape):
    r, c = cell
    h, w = shape
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _mode(tile):
    colors, counts = np.unique(tile, return_counts=True)
    if len(colors) < 2 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("The tile needs multiple colors and a strict unique mode")
    return int(colors[np.argmax(counts)])


def _distances(mask, root):
    d = np.full(mask.shape, -1, dtype=np.int64)
    if not mask[root]:
        raise ValueError("The bar must touch a modal cell")
    d[root] = 0
    queue = deque([root])
    while queue:
        cell = queue.popleft()
        for other in _neighbors(cell, mask.shape):
            if mask[other] and d[other] < 0:
                d[other] = d[cell] + 1
                queue.append(other)
    return d


def _render(tile, palette, k, background, root):
    h, w = tile.shape
    mode = _mode(tile)
    d = _distances(tile == mode, root)
    out = np.full((h * h, w * w), background, dtype=np.int64)
    for r, c in np.argwhere(d >= 0):
        block = tile.copy()
        block[tile == mode] = palette[(int(d[r, c]) // k) % len(palette)]
        out[r*h:(r+1)*h, c*w:(c+1)*w] = block
    return out


def _pack(tile, palette, k, background, marker, root, orientation):
    h, w = tile.shape
    inp = np.full((h + 5, w + k + 3), background, dtype=np.int64)
    x = k + 1
    inp[3:3+h, x:x+w] = tile
    inp[1, x:x+len(palette)] = palette
    inp[3+root[0], 1:1+k] = marker
    return {"input": _orient(inp, orientation),
            "output": _orient(_render(tile, palette, k, background, root), orientation)}


def _canonical_candidates(a):
    """Decode the geometric grammar, without accessing any outputs."""
    h = a.shape[0] - 5
    if not 3 <= h <= 5:
        return []
    bg = int(a[0, 0])
    found = []
    for k in (1, 2):
        w = a.shape[1] - k - 3
        if not 3 <= w <= 5:
            continue
        x = k + 1
        tile = a[3:3+h, x:x+w].copy()
        if np.any(tile == bg):
            continue
        palette = []
        for color in a[1, x:]:
            if int(color) == bg:
                break
            palette.append(int(color))
        if len(palette) not in (2, 3) or len(set(palette)) != len(palette):
            continue
        rows = np.flatnonzero(a[3:3+h, 1] != bg)
        if len(rows) != 1:
            continue
        root = (int(rows[0]), 0)
        marker = int(a[3+root[0], 1])
        tile_colors = set(int(c) for c in np.unique(tile))
        if (marker == bg or marker in palette or marker in tile_colors
                or bg in palette or tile_colors.intersection(palette)):
            continue
        expected = np.full(a.shape, bg, dtype=np.int64)
        expected[3:3+h, x:x+w] = tile
        expected[1, x:x+len(palette)] = palette
        expected[3+root[0], 1:1+k] = marker
        if not np.array_equal(a, expected):
            continue
        try:
            mode = _mode(tile)
            d = _distances(tile == mode, root)
        except ValueError:
            continue
        found.append({"tile": tile, "palette": tuple(palette), "k": k,
                      "background": bg, "marker": marker, "root": root,
                      "mode": mode, "distance": d})
    return found


def _decode(input_grid):
    a = _grid(input_grid)
    options = []
    for code in range(8):
        for info in _canonical_candidates(_orient(a, code)):
            info["to_canonical"] = code
            options.append(info)
    if len(options) != 1:
        raise ValueError("Expected one unambiguous tile, attached bar, and palette")
    return options[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the wave and its mosaic from this input alone."""
    info = _decode(input_grid)
    answer = _render(info["tile"], info["palette"], info["k"],
                     info["background"], info["root"])
    return _undo(answer, info["to_canonical"])


def _detour_witness(mask, root, d, k, p):
    rr, cc = np.indices(mask.shape)
    direct = np.abs(rr - root[0]) + cc
    return bool(np.any((d >= 0) & ((d // k) % p != (direct // k) % p)))


def _branch(mask, d):
    return any(sum(bool(mask[n]) for n in _neighbors(tuple(cell), mask.shape)) >= 3
               for cell in np.argwhere(d >= 0))


def _sample_network(rng, shape, k, p, *, forbidden=None, detour=False,
                    branch=False, island=False, attempts=240):
    """Bounded connected growth, optionally followed by a detached modal island."""
    allowed = np.ones(shape, dtype=bool) if forbidden is None else ~forbidden
    roots = np.flatnonzero(allowed[:, 0])
    low = max(6, k * p + 1)
    high = min(13 - int(island), shape[0] * shape[1] - 2 - int(island),
               int(allowed.sum()) - int(island))
    if not len(roots) or high < low:
        return None
    for _ in range(attempts):
        root = (int(rng.choice(roots)), 0)
        target = int(rng.integers(low, high + 1))
        cells = {root}
        frontier = {n for n in _neighbors(root, shape) if allowed[n]}
        while len(cells) < target and frontier:
            # Sorting removes dependence on set/hash iteration order.
            cell = sorted(frontier)[int(rng.integers(len(frontier)))]
            frontier.remove(cell)
            cells.add(cell)
            frontier.update(n for n in _neighbors(cell, shape)
                            if allowed[n] and n not in cells)
        if len(cells) != target:
            continue
        mask = np.zeros(shape, dtype=bool)
        for cell in cells:
            mask[cell] = True
        d = _distances(mask, root)
        if int(d.max()) < k * p:
            continue
        if detour and not _detour_witness(mask, root, d, k, p):
            continue
        if branch and not _branch(mask, d):
            continue
        if island:
            choices = [tuple(cell) for cell in np.argwhere(allowed & ~mask)
                       if not any(mask[n] for n in _neighbors(tuple(cell), shape))]
            if not choices:
                continue
            cell = choices[int(rng.integers(len(choices)))]
            mask[cell] = True
        return mask, root
    return None


def _color_tile(rng, mask, palette, background, marker, exclude_mode=None):
    pool = [c for c in range(10) if c not in (background, marker) and c not in palette]
    modal_pool = [c for c in pool if c != exclude_mode]
    mode = int(rng.choice(modal_pool))
    m = int(mask.sum())
    rest = mask.size - m
    # Every nonmodal count is positive and strictly less than m.
    minimum = max(1, (rest + m - 2) // (m - 1))
    maximum = min(len(pool) - 1, rest)
    if minimum > maximum:
        raise AssertionError("Infeasible tile color partition")
    n = int(rng.integers(minimum, maximum + 1))
    others = [int(c) for c in rng.choice([c for c in pool if c != mode], n, replace=False)]
    counts = [1] * n
    for _ in range(rest - n):
        eligible = [i for i in range(n) if counts[i] < m - 1]
        counts[int(rng.choice(eligible))] += 1
    values = np.array([c for c, count in zip(others, counts) for _ in range(count)])
    rng.shuffle(values)
    tile = np.full(mask.shape, mode, dtype=np.int64)
    tile[~mask] = values
    return tile


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(8))
    background, marker = [int(x) for x in rng.choice(10, 2, replace=False)]
    p = int(rng.integers(2, 4))
    pool = [c for c in range(10) if c not in (background, marker)]
    palette = tuple(int(c) for c in rng.choice(pool, p, replace=False))
    shape = ((4, 5), (5, 4), (5, 5))[int(rng.integers(3))]
    # The first two stencils are disjoint. Neither a fixed tile coordinate nor
    # a fixed literal color can select the mode throughout the demonstrations.
    networks = None
    for _ in range(120):
        first = _sample_network(rng, shape, 2, p, detour=True, branch=True)
        if first is None:
            continue
        second = _sample_network(rng, shape, 2, p, forbidden=first[0], detour=True)
        if second is not None:
            networks = first, second
            break
    if networks is None:
        # A constructive backstop keeps a rare unlucky rejection stream from
        # losing the required disjoint, branching, detouring demonstrations.
        # Both backbones have eccentricity six and detour witnesses for p=2,3.
        shape = (5, 5)
        rows1 = ("00000", "10000", "10001", "11111", "11111")
        rows2 = ("11011", "01110", "01110", "00000", "00000")
        mask1 = np.array([[c == "1" for c in row] for row in rows1])
        mask2 = np.array([[c == "1" for c in row] for row in rows2])
        networks = ((mask1, (2, 0)), (mask2, (0, 0)))
    (mask1, root1), (mask2, root2) = networks
    tile1 = _color_tile(rng, mask1, palette, background, marker)
    tile2 = _color_tile(rng, mask2, palette, background, marker, exclude_mode=_mode(tile1))
    train = [_pack(tile1, palette, 1, background, marker, root1, orientation),
             _pack(tile2, palette, 2, background, marker, root2, orientation)]
    if num_train >= 3:
        train.append(_pack(tile1, palette, 2, background, marker, root1, orientation))
    # With four demonstrations, a second palette size and unreachable modal ink
    # are shown together. The test recombines this size with the two-cell bar.
    other_p = 5 - p
    other_palette = tuple(int(c) for c in rng.choice(pool, other_p, replace=False))
    shown_shapes = [shape]
    if num_train >= 4:
        other_shape = ((3, 4), (4, 3), (3, 5), (5, 3), (4, 4))[int(rng.integers(5))]
        network = _sample_network(rng, other_shape, 1, other_p, branch=True,
                                  island=True, attempts=1600)
        if network is None:
            # A larger tile gives a detached island room without changing the rule.
            other_shape = (5, 5)
            network = _sample_network(rng, other_shape, 1, other_p, branch=True,
                                      island=True, attempts=3200)
        if network is None:
            raise RuntimeError("Could not construct an island demonstration")
        tile = _color_tile(rng, network[0], other_palette, background, marker)
        train.append(_pack(tile, other_palette, 1, background, marker, network[1], orientation))
        shown_shapes.append(other_shape)
    used = {_key(pair["input"]) for pair in train}

    def fresh(k, pal, tile_shape, *, island, branch, detour):
        for _ in range(20):
            net = _sample_network(rng, tile_shape, k, len(pal), island=island,
                                  branch=branch, detour=detour, attempts=400)
            if net is None:
                continue
            tile = _color_tile(rng, net[0], pal, background, marker)
            pair = _pack(tile, pal, k, background, marker, net[1], orientation)
            if _key(pair["input"]) not in used:
                used.add(_key(pair["input"]))
                return pair
        raise RuntimeError("Could not sample a distinct wavefront example")

    while len(train) < num_train:
        # Keep (2, other_p) for the test, while letting spatial/color axes combine.
        size, k = (p, int(rng.integers(1, 3))) if rng.random() < .5 else (other_p, 1)
        pal = tuple(int(c) for c in rng.choice(pool, size, replace=False))
        tile_shape = shown_shapes[int(rng.integers(len(shown_shapes)))]
        train.append(fresh(k, pal, tile_shape, island=bool(rng.integers(2)),
                           branch=False, detour=k == 2))
    test = []
    for i in range(num_test):
        size = other_p if num_train >= 4 else p
        k = 2 if i == 0 else int(rng.integers(1, 3))
        pal = tuple(int(c) for c in rng.choice(pool, size, replace=False))
        # The hardest spatial combination uses a generously sized shown shape.
        test.append(fresh(k, pal, shape, island=num_train >= 4,
                          branch=True, detour=k == 2 or size == 3))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    canonical = _orient(out, info["to_canonical"])
    tile, d = info["tile"], info["distance"]
    h, w = tile.shape
    _require(canonical.shape == (h*h, w*w), "Self-sized block dimensions")
    mode = info["mode"]
    for r in range(h):
        for c in range(w):
            block = canonical[r*h:(r+1)*h, c*w:(c+1)*w]
            if d[r, c] < 0:
                _require(np.all(block == info["background"]), "Unreachable block must be blank")
            else:
                phase = (int(d[r, c]) // info["k"]) % len(info["palette"])
                _require(np.all(block[tile == mode] == info["palette"][phase]), "Wave ink")
                _require(np.array_equal(block[tile != mode], tile[tile != mode]), "Exact nonmodal detail")
    _require(int(d.max()) >= info["k"] * len(info["palette"]), "Visible full cycle and wrap")
    _require(np.array_equal(solve(inp), out), "Input-only solve mismatch")
    return info


def validate_generated_puzzle(puzzle):
    """Validate labels, input grammar, topology, and demonstration counterexamples."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "Splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "Pair counts")
    used = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(pair["input"])
            _require(key not in used, "Duplicate input")
            used.add(key)
            infos[split].append(_check_pair(pair))
    tr, te = infos["train"], infos["test"]
    for role in ("background", "marker", "to_canonical"):
        _require(len({x[role] for x in tr + te}) == 1, f"Puzzle-wide {role}")
    a, b = tr[:2]
    _require(a["tile"].shape == b["tile"].shape, "Comparable first stencil shapes")
    _require(a["mode"] != b["mode"], "Mode must change literal color")
    _require(not np.any((a["tile"] == a["mode"]) & (b["tile"] == b["mode"])), "Fixed-coordinate selector survives")
    _require(a["palette"] == b["palette"] and (a["k"], b["k"]) == (1, 2), "Run-length evidence")
    _require(_branch(a["tile"] == a["mode"], a["distance"]), "Branch demonstration")
    # A node-serial clock differs from the synchronous distance shells. Sorting
    # by BFS layer is an especially plausible version of this simpler clock.
    ordered = sorted((int(a["distance"][r, c]), int(r), int(c))
                     for r, c in np.argwhere(a["distance"] >= 0))
    _require(any((index % len(a["palette"])) != (distance % len(a["palette"]))
                 for index, (distance, _, _) in enumerate(ordered)), "Serial-clock shortcut survives")
    _require(_detour_witness(b["tile"] == b["mode"], b["root"], b["distance"], 2, len(b["palette"])), "Manhattan shortcut survives")
    if len(train) >= 3:
        contrast = tr[2]
        _require(np.array_equal(a["tile"], contrast["tile"]) and a["root"] == contrast["root"]
                 and a["palette"] == contrast["palette"] and contrast["k"] == 2, "Controlled bar contrast")
        _require(not np.array_equal(train[0]["output"], train[2]["output"]), "Bar must change answer")
    has_island = lambda x: bool(np.any((x["tile"] == x["mode"]) & (x["distance"] < 0)))
    if len(train) >= 4:
        _require(has_island(tr[3]), "Detached-ink demonstration")
        _require(len(tr[3]["palette"]) != len(a["palette"]), "Palette-size evidence")
        held = (2, len(tr[3]["palette"]))
        _require(all((x["k"], len(x["palette"])) != held for x in tr), "Held-out factor combination leaked")
        _require((te[0]["k"], len(te[0]["palette"])) == held, "Missing recombination shift")
    for x in te:
        _require(x["tile"].shape in {t["tile"].shape for t in tr}, "Unsupported tile dimensions")
        _require(x["k"] in {t["k"] for t in tr}, "Unsupported run length")
        _require(len(x["palette"]) in {len(t["palette"]) for t in tr}, "Unsupported palette size")
        _require(not has_island(x) or any(has_island(t) for t in tr), "Unshown reachability convention")
    # Explicit parental ablations: mode-only unchanged copies, and a full-rectangle
    # wave with no modal stencil, must both disagree with every demonstration.
    for pair, x in zip(train, tr):
        tile, mode, root = x["tile"], x["mode"], x["root"]
        h, w = tile.shape
        only_a = np.full((h*h, w*w), x["background"], dtype=int)
        no_a = only_a.copy()
        for r in range(h):
            for c in range(w):
                if tile[r, c] == mode:
                    only_a[r*h:(r+1)*h, c*w:(c+1)*w] = tile
                copy = tile.copy()
                phase = ((abs(r-root[0]) + c) // x["k"]) % len(x["palette"])
                copy[tile == mode] = x["palette"][phase]
                no_a[r*h:(r+1)*h, c*w:(c+1)*w] = copy
        answer = _orient(_grid(pair["output"]), x["to_canonical"])
        _require(not np.array_equal(answer, only_a), "B is dispensable")
        _require(not np.array_equal(answer, no_a), "A is dispensable")


def self_test():
    """Bounded 42-seed coverage plus count, mutation, and equivariance checks."""
    shapes, masks, partitions, roots, phases, palettes, roles, structures = (set() for _ in range(8))
    island_count = detour_count = branch_count = loop_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 3:
            again = generate_puzzle(seed)
            _require(all(_key(a[k]) == _key(b[k]) for split in puzzle
                         for a, b in zip(puzzle[split], again[split]) for k in a), "Determinism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _decode(pair["input"])
                tile, d, mode = x["tile"], x["distance"], x["mode"]
                mask = tile == mode
                shapes.add(tile.shape)
                masks.add((tile.shape, tuple(mask.flat)))
                partitions.add(tuple(sorted(np.unique(tile, return_counts=True)[1].tolist(), reverse=True)))
                roots.add((tile.shape, x["root"]))
                phases.add((x["k"], len(x["palette"])))
                palettes.add(x["palette"])
                roles.add((x["background"], x["marker"], mode))
                relabel = {}
                structures.add((tile.shape, tuple(relabel.setdefault(int(c), len(relabel)) for c in tile.flat)))
                island_count += np.any(mask & (d < 0))
                detour_count += _detour_witness(mask, x["root"], d, x["k"], len(x["palette"]))
                branch_count += _branch(mask, d)
                reachable = d >= 0
                edges = sum(sum(bool(reachable[n]) for n in _neighbors(tuple(cell), tile.shape))
                            for cell in np.argwhere(reachable)) // 2
                loop_count += edges >= int(reachable.sum())
        # Geometry is encoded: rotate/reflect a valid input and its answer.
        if seed < 8:
            pair = puzzle["test"][0]
            _require(np.array_equal(solve(_orient(pair["input"], seed)),
                                    _orient(pair["output"], seed)), "Dihedral equivariance")
    _require(len(shapes) >= 7 and len(masks) >= 100 and len(partitions) >= 25, "Structural variation")
    _require(len(roots) >= 20 and len(palettes) >= 80 and len(roles) >= 35, "Role variation")
    _require(phases == {(1, 2), (1, 3), (2, 2), (2, 3)}, "Run/palette factor coverage")
    _require(len(structures) >= 120 and island_count >= 70 and detour_count >= 80
             and branch_count >= 100 and loop_count >= 10, "Topology coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            puzzle = generate_puzzle(718 + 10*a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Deliberately corrupt a reachable block and ensure semantic validation fails.
    puzzle = generate_puzzle(91)
    x = _decode(puzzle["train"][0]["input"])
    out = _orient(puzzle["train"][0]["output"], x["to_canonical"])
    r, c = x["root"]
    h, w = x["tile"].shape
    out[r*h, c*w] = x["background"]
    puzzle["train"][0]["output"] = _undo(out, x["to_canonical"])
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt copy was accepted")
    # Solver coverage beyond generation's pedagogical restrictions: a short
    # path and a detached multi-cell component, with a genuine plurality mode.
    mask = np.zeros((5, 5), dtype=bool)
    mask[:3, 0] = True
    mask[3:, 3:] = True
    tile = np.ones((5, 5), dtype=int)
    tile[~mask] = np.repeat([2, 3, 4], 6)
    for code in range(8):
        pair = _pack(tile, (7, 8, 9), 2, 0, 6, (0, 0), code)
        answer = _undo(solve(pair["input"]), code)
        _require(np.all(answer[:5, :5][mask] == 7), "Source phase on short path")
        _require(np.all(answer[10:15, :5][mask] == 8), "Short path next run")
        _require(np.all(answer[15:25, 15:25] == 0), "Detached component activated")
        _require(np.array_equal(answer[:5, :5][~mask], tile[~mask]), "Short-path copy detail")
    return {"seeds": 42, "tile_shapes": len(shapes), "modal_masks": len(masks),
            "count_partitions": len(partitions), "root_geometries": len(roots),
            "ordered_palettes": len(palettes), "role_assignments": len(roles),
            "color_normalized_tiles": len(structures), "island_pairs": int(island_count),
            "visible_detour_pairs": int(detour_count), "branch_pairs": int(branch_count),
            "loop_pairs": int(loop_count)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
