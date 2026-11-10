"""Pack complementary marked tiles, then route around the completed material.

The input-only solver searches every full-width profile-chain placement on both
sides of the band, deduplicates complete colored packings, then counts shortest
paths. No parent module, generation mode, seed, or stored example is consulted.
"""
from collections import deque
import hashlib

import numpy as np


BACKGROUND = 0
TERMINAL = 2
ROUTE = 4
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRECTIONS:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _dilate(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _scene(grid):
    _require(not np.any(grid == ROUTE), "route color must be absent from input")
    _require(np.count_nonzero(grid == TERMINAL) == 2, "expected two terminal marks")
    bands = []
    for color in sorted(set(map(int, np.unique(grid))) - {BACKGROUND, TERMINAL}):
        cells = np.argwhere(grid == color)
        if len(_components(grid == color)) != 1:
            continue
        for axis in (0, 1):
            if len(np.unique(cells[:, 1-axis])) == grid.shape[1-axis]:
                bands.append((color, axis))
    _require(len(bands) == 1, "expected one uniquely spanning structural color")
    band, axis = bands[0]
    canonical = grid.copy() if axis == 0 else grid.T.copy()
    fixed = canonical == band
    edges = []
    for c in range(canonical.shape[1]):
        rows = np.flatnonzero(fixed[:, c])
        _require(np.array_equal(rows, np.arange(rows[0], rows[-1]+1)),
                 "band must be column convex")
        edges.append((int(rows[0]), int(rows[-1])+1))
    edges = np.array(edges, dtype=int)
    pieces = []
    for cells in _components((canonical != BACKGROUND) & ~fixed):
        values = canonical[cells[:, 0], cells[:, 1]]
        body = set(map(int, values)) - {TERMINAL}
        _require(len(body) == 1, "each detached tile has one body color plus optional marks")
        _require(not np.any(_dilate(fixed)[cells[:, 0], cells[:, 1]]),
                 "tiles must have clearance from band")
        pieces.append((cells, values.copy()))
    _require(1 <= len(pieces) <= 12, "expected 1..12 detached tiles")
    return canonical, band, axis, edges, pieces


def _description(cells, sign):
    cross = cells[:, 1] - cells[:, 1].min()
    outward = sign*cells[:, 0]
    lower, upper = [], []
    for c in range(int(cross.max())+1):
        values = np.sort(outward[cross == c])
        _require(len(values) > 0 and np.array_equal(values, np.arange(values[0], values[-1]+1)),
                 "tile must be a full-width column-convex layer")
        lower.append(int(values[0]))
        upper.append(int(values[-1])+1)
    origin = min(lower)
    return (tuple(x-origin for x in lower), tuple(x-origin for x in upper),
            np.column_stack((outward-origin, cross)))


def _pack(input_grid):
    """Exhaust all profile chains and global covers; reject a second color grid."""
    grid = _grid(input_grid)
    canonical, band, axis, edges, pieces = _scene(grid)
    h, w = canonical.shape
    base = np.zeros_like(canonical)
    base[canonical == band] = band
    candidates = []
    for sign in (-1, 1):
        descriptions = [_description(cells, sign) for cells, _ in pieces]
        lookup = {}
        for i, (lo, hi, local) in enumerate(descriptions):
            lookup.setdefault(lo, []).append(i)
        boundary = 1-edges[:, 0] if sign < 0 else edges[:, 1]
        limit = 1 if sign < 0 else h
        for width in sorted({len(item[0]) for item in descriptions}):
            for start in range(w-width+1):
                initial = tuple(map(int, boundary[start:start+width]))

                def extend(profile, used, placements):
                    offset = min(profile)
                    normalized = tuple(v-offset for v in profile)
                    if used and len(set(profile)) == 1:
                        occupied = frozenset((int(r), int(c))
                                             for _, cells in placements for r, c in cells)
                        candidates.append({"bits": used, "cells": occupied,
                                           "placements": placements, "side": sign,
                                           "start": start, "width": width,
                                           "end": profile[0], "initial": initial})
                    for i in lookup.get(normalized, ()):
                        if used & (1 << i):
                            continue
                        lo, hi, local = descriptions[i]
                        next_profile = tuple(offset+x for x in hi)
                        if max(next_profile) > limit:
                            continue
                        cells = np.column_stack((sign*(local[:, 0]+offset), local[:, 1]+start))
                        if np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= h):
                            continue
                        if np.any(base[cells[:, 0], cells[:, 1]] != BACKGROUND):
                            continue
                        if np.array_equal(cells, pieces[i][0]):
                            continue
                        extend(next_profile, used | (1 << i), placements+[(i, cells)])

                extend(initial, 0, [])
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate["bits"] & (1 << i):
                by_piece[i].append(candidate)
    solutions = {}
    full = (1 << len(pieces))-1

    def cover(used, occupied, tabs):
        if len(solutions) > 1:
            return
        if used == full:
            answer = base.copy()
            for tab in tabs:
                for i, cells in tab["placements"]:
                    answer[cells[:, 0], cells[:, 1]] = pieces[i][1]
            solutions.setdefault(answer.tobytes(), (answer, tabs))
            return
        options = [(i, [tab for tab in by_piece[i]
                        if not tab["bits"] & used and not tab["cells"] & occupied])
                   for i in range(len(pieces)) if not used & (1 << i)]
        _, available = min(options, key=lambda item: len(item[1]))
        for tab in available:
            cover(used | tab["bits"], occupied | tab["cells"], tabs+[tab])

    cover(0, frozenset(), [])
    _require(len(solutions) == 1, "packing must have one distinct colored answer before routing")
    packed, tabs = next(iter(solutions.values()))
    info = {"band": band, "axis": axis, "pieces": pieces, "tabs": tabs,
            "canonical_input": canonical, "canonical_packed": packed,
            "candidates": len(candidates)}
    return (packed if axis == 0 else packed.T).copy(), info


def _route(grid):
    """BFS with shortest-path multiplicities capped at two."""
    terminals = [tuple(map(int, p)) for p in np.argwhere(grid == TERMINAL)]
    _require(len(terminals) == 2, "expected two packed terminals")
    start, goal = terminals
    distances, ways, parent = {start: 0}, {start: 1}, {}
    queue = deque([start])
    h, w = grid.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRECTIONS:
            p = (r+dr, c+dc)
            if not (0 <= p[0] < h and 0 <= p[1] < w):
                continue
            if grid[p] not in (BACKGROUND, TERMINAL):
                continue
            distance = distances[(r, c)]+1
            if p not in distances:
                distances[p] = distance
                ways[p] = ways[(r, c)]
                parent[p] = (r, c)
                queue.append(p)
            elif distances[p] == distance:
                ways[p] = min(2, ways[p]+ways[(r, c)])
    if goal not in distances:
        return None, 0
    path, current = [goal], goal
    while current != start:
        current = parent[current]
        path.append(current)
    return path[::-1], ways[goal]


def _solve_details(input_grid):
    packed, info = _pack(input_grid)
    path, count = _route(packed)
    _require(path is not None and count == 1 and len(path) > 2,
             "packed terminals must have one nontrivial shortest route")
    result = packed.copy()
    for p in path[1:-1]:
        _require(packed[p] == BACKGROUND, "route may only paint background")
        result[p] = ROUTE
    info.update({"packed": packed, "path": path})
    return result, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from this grid, using the fixed 0/2/4 role convention."""
    return _solve_details(input_grid)[0]


def _cuts(rng, width, count, depth):
    for _ in range(80):
        root = rng.integers(0, int(rng.integers(2, 4)), size=width)
        root -= root.min()
        if not root.max():
            continue
        cuts = [root]
        for _ in range(count-1):
            shape = rng.integers(0, int(rng.integers(2, 4)), size=width)
            shape -= shape.min()
            if not shape.max():
                shape[int(rng.integers(width))] = 1
            lower = cuts[-1]
            offset = max(int(np.max(lower+1-shape)),
                         int(np.max(np.maximum(lower[:-1], lower[1:])+1
                                    -np.minimum(shape[:-1], shape[1:]))))
            cuts.append(shape+offset+int(rng.integers(2)))
        if int(cuts[-1].max()) < depth:
            cuts.append(np.full(width, depth, dtype=int))
            return cuts
    raise ValueError("cannot fit complementary layers")


def _scatter(rng, base, pieces):
    h, w = base.shape
    for _ in range(16):
        result = base.copy()
        forbidden = _dilate(base != BACKGROUND)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*4)
        for i in order:
            cells, values = pieces[i]
            origin = cells.min(axis=0)
            local = cells-origin
            ph, pw = local.max(axis=0)+1
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            if 0 <= origin[0] < valid.shape[0] and 0 <= origin[1] < valid.shape[1]:
                valid[tuple(origin)] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            placed = local+choices[int(rng.integers(len(choices)))]
            result[placed[:, 0], placed[:, 1]] = values
            forbidden = _dilate(result != BACKGROUND)
        else:
            return result
    raise ValueError("scatter clearance exhausted")


def _make_pair(rng, mode, complexity, orientation):
    """Construct marked obstacle geometry, not a precomputed drawing template."""
    band = int(rng.choice([1, 3, 5, 6, 7, 8, 9]))
    palette = [int(c) for c in rng.permutation([c for c in (1, 3, 5, 6, 7, 8, 9) if c != band])]
    if mode == "wrap":
        n = 1
    elif mode == "bridge":
        n = 3 if complexity == 0 else 4
    else:
        n = int(rng.integers(2, 4 if complexity == 0 else 5))
    widths = [int(rng.integers(3, 7 if n < 3 else 6)) for _ in range(n)]
    if n > 1 and rng.random() < .6:
        widths = [widths[0]]*n
    gaps = [1 if mode == "bridge" else int(rng.integers(1, 4)) for _ in range(n-1)]
    occupied_width = sum(widths)+sum(gaps)
    w = int(rng.integers(max(14, occupied_width+2), 31))
    start = int(rng.integers(1, w-occupied_width))
    starts = [start]
    for width, gap in zip(widths, gaps):
        starts.append(starts[-1]+width+gap)
    starts = starts[:n]
    common_depth = int(rng.integers(6 if complexity == 0 else 7, 10))
    depths = [common_depth]*n
    if mode == "bridge":
        depths[0] += int(rng.integers(1, 3))
        depths[-1] += int(rng.integers(1, 3))
    counts = [int(rng.integers(1, 4)) for _ in range(n)]
    if mode == "wrap":
        counts[0] = int(rng.integers(2, 4))
    # At least one genuinely complementary inter-piece match in every example.
    if max(counts) == 1:
        counts[int(rng.integers(n))] = 2
    cuts = [_cuts(rng, width, count, depth)
            for width, count, depth in zip(widths, counts, depths)]
    core = int(rng.integers(2, 4))
    h = int(rng.integers(max(22 if n > 1 else 18, max(depths)+core+10), 31))
    y0 = h-max(depths)-int(rng.integers(1, 4))
    _require(y0-core >= 6, "insufficient scattering space")
    base = np.zeros((h, w), dtype=np.int8)
    base[y0-core:y0, :] = band
    # A little unused structural context, never an instruction to fill all gaps.
    for c in range(w):
        if not any(a <= c < a+width for a, width in zip(starts, widths)):
            extension = int(rng.choice((0, 0, 0, 1, 2)))
            base[y0:y0+extension, c] = band
    for a, width, profiles in zip(starts, widths, cuts):
        for c in range(width):
            base[y0:y0+int(profiles[0][c]), a+c] = band
    packed = base.copy()
    pieces = []
    # Repeated body colors are allowed: separated components are identities.
    palette_size = int(rng.integers(1, min(len(palette), sum(counts))+1))
    for a, width, profiles in zip(starts, widths, cuts):
        for lower, upper in zip(profiles, profiles[1:]):
            cells = np.array([(y0+u, a+c) for c in range(width)
                              for u in range(int(lower[c]), int(upper[c]))], dtype=int)
            color = palette[int(rng.integers(palette_size))]
            packed[cells[:, 0], cells[:, 1]] = color
            pieces.append((cells, np.full(len(cells), color, dtype=np.int8)))
    if mode == "wrap":
        terminal_columns = [starts[0], starts[0]+widths[0]-1]
        terminal_tabs = [0, 0]
    elif mode == "bridge":
        terminal_columns = [starts[0]+widths[0]-1, starts[-1]]
        terminal_tabs = [0, n-1]
    else:
        terminal_columns = [starts[0], starts[-1]+widths[-1]-1]
        terminal_tabs = [0, n-1]
    for c, tab_index in zip(terminal_columns, terminal_tabs):
        local_column = c-starts[tab_index]
        lower = int(cuts[tab_index][0][local_column])
        upper = common_depth-2
        _require(lower <= upper, "terminal needs a side-face cell below obstacle tip")
        r = y0+int(rng.integers(lower, upper+1))
        _require(packed[r, c] not in (BACKGROUND, band), "marker must ride a tile")
        packed[r, c] = TERMINAL
        for cells, values in pieces:
            matches = np.flatnonzero(np.all(cells == (r, c), axis=1))
            if len(matches):
                values[matches[0]] = TERMINAL
                break
    path, count = _route(packed)
    _require(path is not None and count == 1, "construction lacks unique route")
    inp = _scatter(rng, base, pieces)
    output = packed.copy()
    for cell in path[1:-1]:
        output[cell] = ROUTE
    # Whole-scene symmetries preserve each tile's translation within the pair.
    k = orientation+2*int(rng.integers(2))
    inp, output = np.rot90(inp, k).copy(), np.rot90(output, k).copy()
    if rng.random() < .5:
        inp, output = inp[:, ::-1].copy(), output[:, ::-1].copy()
    expected, info = _solve_details(inp)
    _require(np.array_equal(expected, output), "constructed packing and exhaustive search disagree")
    # Omitting A must change the route as well as leaving the wrong tile layout.
    unpacked_path, unpacked_count = _route(inp)
    _require(unpacked_count != 1 or unpacked_path != info["path"],
             "packing would be causally irrelevant to routing")
    return {"input": inp, "output": output}


def _count(value, name, lo, hi):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name+" must be an integer")
    _require(lo <= int(value) <= hi, name+" out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a deterministic puzzle; all output conventions are fixed or visible."""
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(2))
    train_modes = ["wrap", "bridge"]
    train_modes.extend(str(rng.choice(("wrap", "span", "bridge"))) for _ in range(a-2))
    rng.shuffle(train_modes)
    test_modes = ["bridge"]+[str(rng.choice(("span", "bridge"))) for _ in range(b-1)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, modes in (("train", train_modes), ("test", test_modes)):
        for mode in modes:
            for _ in range(160):
                try:
                    pair = _make_pair(rng, mode, int(split == "test"),
                                      orientation if split == "train" else 1-orientation)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded attempts exhausted while making "+mode)
    return puzzle


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape changed")
    expected, info = _solve_details(inp)
    _require(np.array_equal(expected, out), "output does not match packed shortest route")
    band, packed = info["band"], info["packed"]
    _require(np.array_equal(inp == band, out == band), "band changed")
    _require(np.count_nonzero(out == TERMINAL) == 2, "terminals changed")
    # Packing preserves material, including the marker cells, before path paint.
    _require(np.array_equal(np.bincount(inp.ravel(), minlength=10),
                            np.bincount(packed.ravel(), minlength=10)), "packing lost material")
    canonical = info["canonical_packed"]
    used = set()
    for tab in info["tabs"]:
        for i, cells in tab["placements"]:
            _require(i not in used, "tile reused")
            used.add(i)
            source, values = info["pieces"][i]
            delta = cells[0]-source[0]
            _require(np.array_equal(source+delta, cells) and np.any(delta), "tile not translated intact")
            _require(np.array_equal(canonical[cells[:, 0], cells[:, 1]], values), "tile recolored")
        # The fixed inner profile plus every chain layer fills a rectangle.
        sign, end = tab["side"], tab["end"]
        low = min(tab["initial"])
        for c in range(tab["start"], tab["start"]+tab["width"]):
            for u in range(low, end):
                _require(canonical[sign*u, c] != BACKGROUND, "hole in completed tab")
    _require(used == set(range(len(info["pieces"]))), "unused tile")
    path = info["path"]
    distance = abs(path[0][0]-path[-1][0])+abs(path[0][1]-path[-1][1])
    _require(len(path)-1 > distance, "obstacles must force a real detour")
    _require(np.count_nonzero(out == ROUTE) == len(path)-2 > 0, "wrong intermediate cells")
    _require(np.count_nonzero(packed == BACKGROUND) > len(path), "no off-route background")
    # A stronger ablation retains the correctly relocated endpoints but removes
    # every packed payload wall. It must destroy the intended route or uniqueness.
    without_walls = np.zeros_like(packed)
    without_walls[packed == band] = band
    without_walls[packed == TERMINAL] = TERMINAL
    other, count = _route(without_walls)
    _require(count != 1 or other != path, "packed wall geometry has no effect")
    before, count = _route(inp)
    _require(count != 1 or before != path, "routing before packing gives same route")
    _require(not np.array_equal(packed, out), "parent B has no effect")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a = _count(len(puzzle["train"]), "num_train", 2, 10)
    b = _count(len(puzzle["test"]), "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    seen, train_tabs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                train_tabs.add(len(info["tabs"]))
    _require(1 in train_tabs and any(n >= 3 for n in train_tabs),
             "training must demonstrate single-tab and interacting-tab routing")


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
    """42 puzzles streamed, no files/traces or retained corpus, bounded retries."""
    structures, shapes, bands, tabs, widths, counts, depths, paths = (set() for _ in range(8))
    puzzle_hashes, repeated_colors = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_hashes.add(_digest(puzzle))
        if seed == 0:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _validate_pair(pair)
                shapes.add(inp.shape)
                bands.add(info["band"])
                tabs.add(len(info["tabs"]))
                widths.update(tab["width"] for tab in info["tabs"])
                counts.update(len(tab["placements"]) for tab in info["tabs"])
                depths.update(tab["end"]-min(tab["initial"]) for tab in info["tabs"])
                paths.add(len(info["path"])-1)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                colors = [next(int(c) for c in values if c != TERMINAL) for _, values in info["pieces"]]
                repeated_colors += len(set(colors)) < len(colors)
                _require(np.array_equal(solve(inp.tolist()), pair["output"]), "list-input solve failed")
    _require(len(puzzle_hashes) == 42 and len(structures) >= 200, "insufficient structural variation")
    _require(len(shapes) >= 30 and len(bands) >= 5, "insufficient canvas/palette variation")
    _require(tabs >= {1, 2, 3, 4} and widths >= {3, 4, 5, 6} and counts >= {1, 2, 3},
             "missing tab, width, or chain-count variation")
    _require(len(depths) >= 5 and len(paths) >= 12 and repeated_colors >= 60,
             "insufficient interlock/routing variation")
    # Fail-closed checks for ambiguity and unreachable routes.
    ambiguous = np.zeros((3, 3), dtype=np.int8)
    ambiguous[0, 0] = ambiguous[2, 2] = TERMINAL
    _require(_route(ambiguous)[1] == 2, "BFS failed to detect tied shortest routes")
    ambiguous[1, :] = 8
    _require(_route(ambiguous)[1] == 0, "BFS failed to detect disconnection")
    # Equal masks at two identical sockets: interchangeable identical marked
    # tiles give one colored packing; different body colors give two answers.
    fixture = np.zeros((16, 18), dtype=np.int8)
    fixture[7:9, :] = 8
    fixture[9, [4, 11]] = 8
    cap = np.array([(r, c) for c, low in enumerate((0, 1, 0))
                    for r in range(low, 3)], dtype=int)
    for offset in (0, 9):
        fixture[cap[:, 0], cap[:, 1]+offset] = 1
        fixture[1, offset] = TERMINAL
    _pack(fixture)
    fixture[(fixture == 1) & (np.indices(fixture.shape)[1] >= 9)] = 3
    try:
        _pack(fixture)
    except ValueError as error:
        _require("one distinct colored answer" in str(error), "wrong packing rejection")
    else:
        raise AssertionError("ambiguous packing accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = generate_puzzle(0)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid output accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed self-test passed")
