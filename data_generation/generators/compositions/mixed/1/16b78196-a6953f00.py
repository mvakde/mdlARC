"""Assemble a square at a jagged band; crop its parity-selected top corner.

The solver enumerates every translation into every possible square containing
the fixed band. No sampler profiles, intended sockets, seed, or examples enter
that search. Only the standard library and NumPy are required. No file I/O occurs.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= s <= 30 for s in grid.shape),
             "grids must be two dimensional with sides 1..30")
    _require(np.issubdtype(grid.dtype, np.integer) and
             bool(np.all((0 <= grid) & (grid <= 9))), "invalid grid colors")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(grid, background):
    """Object identities are four-connected monochrome components."""
    remaining = set(map(tuple, np.argwhere(grid != background)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(grid[first])
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining and grid[point] == color:
                    remaining.remove(point)
                    stack.append(point)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _dilate(mask):
    padded = np.pad(mask, 1)
    height, width = mask.shape
    return np.logical_or.reduce([padded[r:r+height, c:c+width]
                                 for r in range(3) for c in range(3)])


def _scene(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(bool(np.all(grid[0] == bg) and np.all(grid[-1] == bg) and
                  np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)),
             "the exterior border must be uniform background")
    components = _components(grid, bg)
    _require(len(components) >= 2, "missing structural band or pieces")
    spans = [cells.max(axis=0)-cells.min(axis=0)+1 for cells, _ in components]
    longest = max(int(span.max()) for span in spans)
    candidates = [i for i, span in enumerate(spans) if span.max() == longest]
    _require(len(candidates) == 1, "the band must have uniquely greatest extent")
    index = candidates[0]
    band_cells, band_color = components[index]
    span = spans[index]
    _require(span[0] != span[1], "ambiguous band axis")
    vertical = bool(span[0] > span[1])
    # Canonical coordinates always have a vertical band. Transposition is an
    # internal coordinate change, never a rotation of a movable piece.
    canonical = grid if vertical else grid.T
    band_cells = band_cells if vertical else band_cells[:, ::-1]
    n = longest
    _require(6 <= n <= 12, "family square side must be 6..12")
    r0 = int(band_cells[:, 0].min())
    mask = canonical == band_color
    _require(np.count_nonzero(mask) == len(band_cells), "band color reused")
    for row in range(r0, r0+n):
        columns = np.flatnonzero(mask[row])
        _require(len(columns) > 0 and np.array_equal(
            columns, np.arange(columns[0], columns[-1]+1)), "band row has holes")
    pieces = []
    occupied = mask.copy()
    for i, (cells, color) in enumerate(components):
        if i == index:
            continue
        cells = cells if vertical else cells[:, ::-1]
        _require(not bool(np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]])),
                 "input components need an eight-neighbor clearance")
        occupied[cells[:, 0], cells[:, 1]] = True
        pieces.append((cells.copy(), color))
    _require(sum(len(cells) for cells, _ in components) == n*n,
             "the band and every piece must together have square area")
    return {"grid": grid, "canonical": canonical, "background": bg,
            "band": band_color, "band_cells": band_cells, "pieces": pieces,
            "vertical": vertical, "n": n, "r0": r0}


def _tilings(band_cells, pieces, n, limit=2, node_limit=None):
    """General exact cover, without a chain, convexity, or socket assumption.

    Each candidate is an untranslated-or-translated copy of an entire component.
    Branch on the uncovered cell with fewest available placements. Stop only
    after two distinct colored canvases, not after two identity permutations.
    """
    full = (1 << (n*n))-1
    band_bits = sum(1 << (int(r)*n+int(c)) for r, c in band_cells)
    target = full ^ band_bits
    options, by_cell = [], [[] for _ in range(n*n)]
    normalized, earlier, groups = [], [], {}
    for i, (cells, color) in enumerate(pieces):
        local = cells-cells.min(axis=0)
        normalized.append(local)
        h, w = map(int, local.max(axis=0)+1)
        identity = (color, tuple(map(tuple, local)))
        earlier.append(groups.get(identity, 0))
        groups[identity] = groups.get(identity, 0) | (1 << i)
        mask = sum(1 << (int(r)*n+int(c)) for r, c in local)
        for row in range(n-h+1):
            for col in range(n-w+1):
                bits = mask << (row*n+col)
                if bits & band_bits:
                    continue
                j = len(options)
                options.append((i, bits, row, col))
                remaining = bits
                while remaining:
                    bit = remaining & -remaining
                    remaining -= bit
                    by_cell[bit.bit_length()-1].append(j)
    solutions, nodes, visited = {}, 0, set()

    def visit(used, occupied, chosen, colors):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise ValueError("construction proposal exceeded search budget")
        if len(solutions) >= limit:
            return
        # Different orders can reach the same colored partial cover. Its entire
        # continuation is identical, including when unlike shapes share colors.
        state = (used, occupied, colors)
        if state in visited:
            return
        visited.add(state)
        if occupied == target:
            if used != (1 << len(pieces))-1:
                return
            canvas = np.full((n, n), -1, dtype=np.int8)
            placements = []
            for j in chosen:
                i, _, r, c = options[j]
                cells = normalized[i]+[r, c]
                canvas[cells[:, 0], cells[:, 1]] = pieces[i][1]
                placements.append((i, cells))
            solutions.setdefault(canvas.tobytes(), (canvas, placements))
            return
        best = None
        remaining = target ^ occupied
        while remaining:
            bit = remaining & -remaining
            remaining -= bit
            choices = []
            for j in by_cell[bit.bit_length()-1]:
                i, bits, _, _ = options[j]
                if not used & (1 << i) and not bits & occupied and not earlier[i] & ~used:
                    choices.append(j)
            if not choices:
                return
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
        for j in best:
            i, bits, _, _ = options[j]
            color = pieces[i][1]
            next_colors = list(colors)
            next_colors[color] |= bits
            visit(used | (1 << i), occupied | bits, chosen+[j], tuple(next_colors))
            if len(solutions) >= limit:
                return

    visit(0, 0, [], (0,)*10)
    return list(solutions.values()), nodes


def _assemble(input_grid, node_limit=None):
    info = _scene(input_grid)
    grid, n, r0 = info["canonical"], info["n"], info["r0"]
    band, pieces = info["band_cells"], info["pieces"]
    lower = max(0, int(band[:, 1].max())-n+1)
    upper = min(int(band[:, 1].min()), grid.shape[1]-n)
    solutions, total_nodes = {}, 0
    # No selected or maximal socket is supplied. Even the square's perpendicular
    # position is unknown until the global packing has been solved.
    for c0 in range(lower, upper+1):
        local_band = band-[r0, c0]
        remaining = None if node_limit is None else node_limit-total_nodes
        results, nodes = _tilings(local_band, pieces, n, node_limit=remaining)
        total_nodes += nodes
        for canvas, placements in results:
            canvas = canvas.copy()
            canvas[local_band[:, 0], local_band[:, 1]] = info["band"]
            solutions[(c0, canvas.tobytes())] = (canvas, placements, c0)
            _require(len(solutions) <= 1, "more than one colored square assembly")
    _require(len(solutions) == 1, "no square assembly")
    canvas, placements, c0 = next(iter(solutions.values()))
    assembled = np.full_like(grid, info["background"])
    assembled[r0:r0+n, c0:c0+n] = canvas
    info.update({"canvas": (canvas if info["vertical"] else canvas.T).copy(),
                 "assembled": (assembled if info["vertical"] else assembled.T).copy(),
                 "placements": [(i, cells+[r0, c0]) for i, cells in placements],
                 "c0": c0, "search_nodes": total_nodes})
    return info


def _crop(canvas):
    return (canvas[:2, :2] if canvas.shape[0] % 2 else canvas[:2, -2:]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the literal parity crop of the unique reconstructed square."""
    return _crop(_assemble(input_grid)["canvas"])


def _partition(rng, n):
    widths = []
    while n:
        choices = [w for w in range(2, 6) if w <= n and n-w != 1]
        width = int(rng.choice(choices))
        widths.append(width)
        n -= width
    return widths


def _square(rng, n, palette):
    """Construct only a source-like strip subset; solve uses arbitrary tilings."""
    _, band_color, colors = palette
    core = int(rng.integers(1, min(3, n-4)+1))
    position = int(rng.integers(2, n-core-1))
    left = position-rng.integers(0, 2, size=n)
    right = position+core+rng.integers(0, 2, size=n)
    canvas = np.full((n, n), band_color, dtype=np.int8)
    pieces = []
    for side, edge in ((0, left), (1, right)):
        start = 0
        for width in _partition(rng, n):
            segment = edge[start:start+width]
            lo = np.zeros(width, dtype=int) if side == 0 else segment.copy()
            hi = segment.copy() if side == 0 else np.full(width, n, dtype=int)
            cuts = [lo]
            # Internal seams are independently jagged, with unequal layer depths.
            if int((hi-lo).min()) >= 3 and rng.random() < .70:
                middle = lo+1+rng.integers(0, int((hi-lo).min())-1, size=width)
                cuts.append(middle)
            cuts.append(hi)
            for lower, upper in zip(cuts, cuts[1:]):
                cells = np.array([(start+r, c) for r in range(width)
                                  for c in range(int(lower[r]), int(upper[r]))], dtype=int)
                test = np.zeros((n, n), dtype=np.int8)
                test[cells[:, 0], cells[:, 1]] = 1
                _require(len(_components(test, 0)) == 1, "disconnected proposal")
                color = int(rng.choice(colors))
                canvas[cells[:, 0], cells[:, 1]] = color
                pieces.append((cells, color))
            start += width
    _require(4 <= len(pieces) <= 12, "scatter piece budget")
    if rng.random() < .5:
        canvas = canvas[::-1].copy()
        pieces = [(np.column_stack((n-1-cells[:, 0], cells[:, 1])), color)
                  for cells, color in pieces]
    return canvas, pieces


def _scatter(rng, base, pieces, bg):
    h, w = base.shape
    for _ in range(16):
        grid = base.copy()
        forbidden = _dilate(grid != bg)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*5)
        for i in order:
            cells, color = pieces[i]
            origin = cells.min(axis=0)
            local = cells-origin
            ph, pw = map(int, local.max(axis=0)+1)
            valid = np.ones((h-ph-1, w-pw-1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[1+r:1+r+valid.shape[0], 1+c:1+c+valid.shape[1]]
            # Actual input objects move; their origins never encode destinations.
            sr, sc = origin-1
            if 0 <= sr < valid.shape[0] and 0 <= sc < valid.shape[1]:
                valid[sr, sc] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            target = local+choices[int(rng.integers(len(choices)))]+1
            grid[target[:, 0], target[:, 1]] = color
            forbidden = _dilate(grid != bg)
        else:
            return grid
    raise ValueError("scatter clearance")


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    payload_size = int(rng.integers(3, 9))
    return colors[0], colors[1], colors[2:2+payload_size]


def _make_pair(rng, n, vertical, palette, witness):
    bg, band, _ = palette
    canvas, pieces = _square(rng, n, palette)
    small = n <= 8
    height = int(rng.integers(18 if small else 24, 31))
    width = int(rng.integers(18 if small else 24, 31))
    r0 = int(rng.integers(2, height-n-1))
    c0 = int(rng.integers(2, width-n-1))
    base = np.full((height, width), bg, dtype=np.int8)
    band_cells = np.argwhere(canvas == band)+[r0, c0]
    base[band_cells[:, 0], band_cells[:, 1]] = band
    pieces = [(cells+[r0, c0], color) for cells, color in pieces]
    inp = _scatter(rng, base, pieces, bg)
    if not vertical:
        inp, canvas = inp.T.copy(), canvas.T.copy()
    info = _assemble(inp, node_limit=20000)
    _require(np.array_equal(info["canvas"], canvas), "construction/search disagreement")
    output = _crop(canvas)
    if witness:
        _require(not np.array_equal(canvas[:2, :2], canvas[:2, -2:]),
                 "training needs distinguishable top corners")
        _require(len(np.unique(output)) >= 2, "training needs an informative seam")
        original_canvas = (inp[r0:r0+n, c0:c0+n] if vertical
                           else inp[c0:c0+n, r0:r0+n])
        _require(not np.array_equal(_crop(original_canvas), output),
                 "packing must affect a selected training corner")
    return {"input": inp.copy(), "output": output}


def _count(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)) and low <= value <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and
             not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    odds = list(map(int, rng.permutation([7, 9])))
    evens = list(map(int, rng.permutation([6, 8, 10])))
    sides = [odds[0], evens[0]]
    if num_train >= 4:
        sides += [odds[1], evens[1]]
    while len(sides) < num_train:
        sides.append(int(rng.integers(6, 11)))
    rng.shuffle(sides)
    # The first test always increases scale. Further tests may reuse a familiar
    # size with new structure, and both parities occur if there are >=2 tests.
    first_parity = int(rng.integers(2))
    test_sides = [11 if first_parity else 12]
    for i in range(1, num_test):
        parity = 1-first_parity if i == 1 else int(rng.integers(2))
        test_sides.append(int(rng.choice([7, 9, 11] if parity else [6, 8, 10, 12])))
    base_vertical = bool(rng.integers(2))
    mixed = num_train >= 4 and rng.random() < .65
    if mixed:
        # Each demonstrated axis gets both parities. Without this balancing,
        # four examples could accidentally make axis predict the crop corner.
        odd_sides = [side for side in sides if side % 2]
        even_sides = [side for side in sides if not side % 2]
        rng.shuffle(odd_sides)
        rng.shuffle(even_sides)
        first = [odd_sides.pop(), even_sides.pop()]
        second = [odd_sides.pop(), even_sides.pop()]
        rng.shuffle(first)
        rng.shuffle(second)
        remaining = odd_sides+even_sides
        rng.shuffle(remaining)
        sides = first+second+remaining
    stable_palette = rng.random() < .3
    shared_palette = _palette(rng)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, lengths in (("train", sides), ("test", test_sides)):
        for index, n in enumerate(lengths):
            vertical = base_vertical
            if mixed and (index >= 2 if split == "train" else rng.random() < .5):
                vertical = not vertical
            palette = shared_palette if stable_palette else _palette(rng)
            for _ in range(320):
                try:
                    pair = _make_pair(rng, n, vertical, palette, witness=split == "train")
                except ValueError:
                    continue
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _assemble(inp)
    _require(out.shape == (2, 2) and np.array_equal(out, _crop(info["canvas"])),
             "incorrect completed-canvas parity crop")
    _require(np.array_equal(inp == info["band"], info["assembled"] == info["band"]),
             "the band changed")
    _require(np.array_equal(np.bincount(inp.ravel(), minlength=10),
                           np.bincount(info["assembled"].ravel(), minlength=10)),
             "assembly must preserve every color count")
    used, occupied = set(), set(map(tuple, info["band_cells"]))
    for i, target in info["placements"]:
        source, color = info["pieces"][i]
        delta = target.min(axis=0)-source.min(axis=0)
        _require(bool(np.any(delta != 0)), "generated components must actually move")
        _require(set(map(tuple, source+delta)) == set(map(tuple, target)),
                 "a component was changed rather than translated")
        cells = set(map(tuple, target))
        _require(i not in used and not cells & occupied, "piece reuse or overlap")
        used.add(i)
        occupied |= cells
        canonical_output = info["assembled"] if info["vertical"] else info["assembled"].T
        _require(bool(np.all(canonical_output[target[:, 0], target[:, 1]] == color)),
                 "piece was recolored")
    _require(used == set(range(len(info["pieces"]))) and len(occupied) == info["n"]**2,
             "incomplete exact cover")
    return info


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, "num_train", 2, 10)
    _count(b, "num_test", 1, 4)
    _require(a > b, "invalid count relation")
    seen, witnesses, training_sizes = set(), set(), {0: set(), 1: set()}
    packing_witnesses = set()
    orientation_parities = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            inp, n, canvas = _grid(pair["input"]), info["n"], info["canvas"]
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training_sizes[n % 2].add(n)
                orientation_parities.setdefault(info["vertical"], set()).add(n % 2)
                if not np.array_equal(canvas[:2, :2], canvas[:2, -2:]):
                    witnesses.add(n % 2)
                canonical = info["canonical"]
                r0, c0 = info["r0"], info["c0"]
                incomplete = canonical[r0:r0+n, c0:c0+n]
                if not info["vertical"]:
                    incomplete = incomplete.T
                if not np.array_equal(_crop(incomplete), _crop(canvas)):
                    packing_witnesses.add(n % 2)
    _require(witnesses == {0, 1}, "training must distinguish both parity branches")
    _require(packing_witnesses == {0, 1}, "packing must affect both training branches")
    if a >= 4:
        _require(all(len(sizes) >= 2 for sizes in training_sizes.values()),
                 "four training examples need two sizes per parity")
    _require(all(parities == {0, 1} for parities in orientation_parities.values()),
             "each demonstrated band axis needs both parity branches")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded, streaming checks: 42 puzzles, no saved collections or traces."""
    structures, puzzles, shapes, roles = set(), set(), set(), set()
    sizes, counts, outputs, orientations, nodes = set(), set(), set(), set(), []
    repeated = raw_parity_disagreement = nonuniform = pair_count = 0
    stable_roles = varied_roles = 0
    split_sizes = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(hashlib.sha256(repr(_puzzle_key(puzzle)).encode()).hexdigest())
        local_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                before = pair["input"].copy()
                info = _assemble(before)
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solver mutated its input"
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in before.flat)
                structures.add((before.shape, hashlib.sha256(normalized).hexdigest()))
                shapes.add(before.shape)
                sizes.add(info["n"])
                split_sizes[split].add(info["n"])
                counts.add(len(info["pieces"]))
                orientations.add(info["vertical"])
                roles.add((info["background"], info["band"]))
                local_roles.add((info["background"], info["band"]))
                colors = [color for _, color in info["pieces"]]
                repeated += len(set(colors)) < len(colors)
                raw_parity_disagreement += before.shape[0] % 2 != info["n"] % 2
                nonuniform += len(np.unique(pair["output"])) > 1
                outputs.add(_key(pair["output"]))
                nodes.append(info["search_nodes"])
        stable_roles += len(local_roles) == 1
        varied_roles += len(local_roles) > 1
    assert len(puzzles) == 42 and len(structures) >= 200
    assert sizes == set(range(6, 13)) and orientations == {False, True}
    assert len(shapes) >= 40 and len(roles) >= 30 and len(counts) >= 5
    assert repeated >= 30 and raw_parity_disagreement >= 40 and nonuniform >= 100
    assert len(outputs) >= 100 and stable_roles >= 1 and varied_roles >= 1
    assert max(nodes) < 100000, "sampled search became too expensive"
    first = generate_puzzle(1234, 2, 1)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(1234, 2, 1))
    for a, b in ((10, 4), (10, 1), (3, 2), (5, 4)):
        puzzle = generate_puzzle(7000+a+b, a, b)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Omitting B gives the other corner, and omitting A leaves incomplete cells.
    for index, pair in enumerate(first["train"]):
        info = _assemble(pair["input"])
        wrong = info["canvas"][:2, -2:] if info["n"] % 2 else info["canvas"][:2, :2]
        assert not np.array_equal(pair["output"], wrong)
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        broken["train"][index]["output"] = wrong.copy()
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect label accepted")
    # A deliberately symmetric two-color cover must be rejected as ambiguous.
    band = np.array([(r, c) for r in range(6) for c in (2, 3)])
    rectangle = np.array([(r, c) for r in range(3) for c in range(2)])
    pieces = [(rectangle.copy(), color) for color in (1, 2, 3, 4)]
    alternatives, _ = _tilings(band, pieces, 6)
    assert len(alternatives) == 2
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(puzzles),
            "color_normalized_inputs": len(structures), "workspace_shapes": len(shapes),
            "square_sides": sorted(sizes), "component_counts": sorted(counts),
            "split_square_sides": {s: sorted(values) for s, values in split_sizes.items()},
            "role_assignments": len(roles), "reused_color_pairs": repeated,
            "raw_height_parity_disagrees_with_square": raw_parity_disagreement,
            "nonuniform_outputs": nonuniform, "distinct_outputs": len(outputs),
            "stable_role_puzzles": stable_roles, "varying_role_puzzles": varied_roles,
            "maximum_search_nodes": max(nodes), "independent_solver": False}


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, exact-cover/parity invariants and API checks")
