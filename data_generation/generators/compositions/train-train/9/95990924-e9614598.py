"""Corner handshakes: selective corner projection and colored midpoint crosses.

Only the standard library and NumPy are used. No parent code or stored puzzles
are imported. `solve` reads tile geometry and colors directly from its argument.
"""
from collections import defaultdict
import json

import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_PORTS = ((-1, -1), (-1, 2), (2, -1), (2, 2))
_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must have two dimensions of size 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(int, copy=False)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.flat))


def _body(position):
    r, c = position
    return {(r + dr, c + dc) for dr, dc in _CORNERS}


def _geometry(positions):
    """Return ports and nearest mutually facing pairs, without using colors.

    Port index = four times tile index plus corner index (NW, NE, SW, SE).
    All potential handshakes, including mismatches, must be cell-centered and
    unobstructed. This is an input constraint, not a hidden tie-breaking rule.
    """
    bodies = [_body(p) for p in positions]
    occupied = set()
    for body in bodies:
        _require(not occupied & body, "overlapping tiles")
        _require(all((r + dr, c + dc) not in occupied for r, c in body
                     for dr, dc in _CROSS[1:]), "touching tiles")
        occupied.update(body)
    ports = [(r + dr, c + dc) for r, c in positions for dr, dc in _PORTS]
    _require(len(ports) == len(set(ports)), "coincident virtual ports")
    _require(not occupied.intersection(ports), "port falls on a tile")
    by_row, by_col = defaultdict(list), defaultdict(list)
    for i, (r, c) in enumerate(ports):
        by_row[r].append((c, i))
        by_col[c].append((r, i))
    links = []
    for axis, groups in (("H", by_row), ("V", by_col)):
        for line in groups.values():
            line.sort()
            for (x, a), (y, b) in zip(line, line[1:]):
                if a // 4 == b // 4:
                    continue
                ka, kb = a % 4, b % 4
                facing = ((ka in (1, 3) and kb in (0, 2)) if axis == "H"
                          else (ka in (2, 3) and kb in (0, 1)))
                if not facing:
                    continue
                _require(y - x >= 4 and (y - x) % 2 == 0,
                         "nearest facing ports need even separation >=4")
                p, q = ports[a], ports[b]
                segment = ({(p[0], c) for c in range(p[1] + 1, q[1])}
                           if axis == "H" else
                           {(r, p[1]) for r in range(p[0] + 1, q[0])})
                _require(not segment & occupied, "tile obstructs a potential link")
                center = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
                cross = {(center[0] + dr, center[1] + dc) for dr, dc in _CROSS}
                _require(not cross & occupied and not cross.intersection(ports),
                         "potential cross hits a tile or a virtual port")
                links.append((a, b, axis, center))
    return ports, links


def _read_input(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(bool(np.all(grid[0] == bg) and np.all(grid[-1] == bg)
                  and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)),
             "input border must be uniform background")
    remaining = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)}
    positions, colors = [], []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _CROSS[1:]:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        r, c = min(y for y, _ in component), min(x for _, x in component)
        _require(component == _body((r, c)), "each component must be a 2x2 tile")
        tile_colors = [int(grid[r + dr, c + dc]) for dr, dc in _CORNERS]
        _require(len(set(tile_colors)) == 4, "a tile needs four distinct colors")
        positions.append((r, c))
        colors.extend(tile_colors)
    _require(2 <= len(positions), "at least two tiles are required")
    ports, links = _geometry(positions)
    h, w = grid.shape
    _require(all(0 <= r < h and 0 <= c < w for r, c in ports),
             "all virtual ports must fit")
    for _, _, _, (r, c) in links:
        _require(1 <= r < h - 1 and 1 <= c < w - 1,
                 "every potential cross must fit without clipping")
    return grid, bg, positions, colors, ports, links


def _interpret(input_grid):
    grid, bg, positions, colors, ports, links = _read_input(input_grid)
    active = [edge for edge in links if colors[edge[0]] == colors[edge[1]]]
    paint, cross_cells, used_ports = {}, set(), set()
    for a, b, _, center in active:
        color = colors[a]
        footprint = {ports[a], ports[b]}
        cross = {(center[0] + dr, center[1] + dc) for dr, dc in _CROSS}
        cross_cells.update(cross)
        used_ports.update((a, b))
        footprint.update(cross)
        for cell in footprint:
            _require(int(grid[cell]) == bg, "construction overwrites an input cell")
            _require(cell not in paint or paint[cell] == color,
                     "different-colored constructions collide")
            paint[cell] = color
    degree = defaultdict(int)
    for a, b, _, _ in active:
        degree[a] += 1
        degree[b] += 1
    staggered = [edge for edge in active
                 if (positions[edge[0] // 4][0] != positions[edge[1] // 4][0]
                     if edge[2] == "H" else
                     positions[edge[0] // 4][1] != positions[edge[1] // 4][1])]
    return {"grid": grid, "background": bg, "positions": positions,
            "colors": colors, "ports": ports, "links": links, "active": active,
            "paint": paint, "cross_cells": cross_cells, "used_ports": used_ports,
            "branch": any(d == 2 for d in degree.values()),
            "stagger": bool(staggered),
            "active_colors": {colors[a] for a, _, _, _ in active},
            "axes": {axis for _, _, axis, _ in active},
            "spans": {abs(ports[a][0] - ports[b][0]) +
                      abs(ports[a][1] - ports[b][1]) for a, b, _, _ in active}}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Preserve tiles; paint only matched ports and their midpoint crosses."""
    info = _interpret(input_grid)
    output = info["grid"].copy()
    for cell, color in info["paint"].items():
        output[cell] = color
    return output


def _blocked_matches(info):
    """Valid even-span same-color facing pairs excluded by nearer ports."""
    ports, colors = info["ports"], info["colors"]
    nearest = {frozenset((a, b)) for a, b, _, _ in info["links"]}
    result = []
    for a, p in enumerate(ports):
        for b in range(a + 1, len(ports)):
            q = ports[b]
            if a // 4 == b // 4 or colors[a] != colors[b]:
                continue
            if p[0] == q[0]:
                left, right = sorted((a, b), key=lambda i: ports[i][1])
                face = left % 4 in (1, 3) and right % 4 in (0, 2)
            elif p[1] == q[1]:
                left, right = sorted((a, b), key=lambda i: ports[i][0])
                face = left % 4 in (2, 3) and right % 4 in (0, 1)
            else:
                continue
            distance = abs(p[0] - q[0]) + abs(p[1] - q[1])
            if face and distance >= 4 and distance % 2 == 0 and frozenset((a, b)) not in nearest:
                result.append((a, b))
    return result


def _fits(positions):
    return (max(r for r, _ in positions) - min(r for r, _ in positions) <= 24
            and max(c for _, c in positions) - min(c for _, c in positions) <= 24)


def _grow(rng, count, stagger):
    # The SE port of tile zero can meet both the right and the lower tile.
    sr = int(rng.choice((0, 3))) if stagger else 0
    sc = int(rng.choice((0, 3))) if stagger else 0
    if stagger and sr == sc == 0:
        sr = 3
    positions = [(0, 0), (sr, int(rng.choice((7, 9, 11, 13)))),
                 (int(rng.choice((7, 9, 11, 13))), sc)]
    for _ in range(count - 3):
        for attempt in range(180):
            r, c = positions[int(rng.integers(len(positions)))]
            distance = int(rng.choice((7, 9, 11, 13, 15)))
            offset = int(rng.choice((-3, 0, 3))) if stagger else 0
            if rng.random() < .5:
                candidate = (r + offset, c + int(rng.choice((-1, 1))) * distance)
            else:
                candidate = (r + int(rng.choice((-1, 1))) * distance, c + offset)
            proposed = positions + [candidate]
            if not _fits(proposed):
                continue
            try:
                _geometry(proposed)
            except ValueError:
                continue
            positions = proposed
            break
        else:
            raise ValueError("growth exhausted")
    _geometry(positions)
    return positions


def _color_ports(rng, positions, links, palette, chain):
    n = 4 * len(positions)
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[root(a)] = root(b)

    active = [bool(rng.random() < .55) for _ in links]
    if chain:
        # Four tiles in a line plus a staggered fifth tile. Only the middle
        # pair and the staggered spur match. Equal outer ports are blocked.
        for i, (a, b, _, _) in enumerate(links):
            active[i] = ({a // 4, b // 4} == {1, 2} or 4 in (a // 4, b // 4))
        union(1, 12)  # first NE and fourth NW, not consecutive along the row
    else:
        for i, (a, b, _, _) in enumerate(links):
            if 3 in (a, b) and ({a // 4, b // 4} in ({0, 1}, {0, 2})):
                active[i] = True
    for enabled, (a, b, _, _) in zip(active, links):
        if enabled:
            union(a, b)
    different = []
    for t in range(len(positions)):
        different.extend((4 * t + a, 4 * t + b) for a in range(4) for b in range(a + 1, 4))
    different.extend((a, b) for enabled, (a, b, _, _) in zip(active, links) if not enabled)
    graph = {root(i): set() for i in range(n)}
    for a, b in different:
        a, b = root(a), root(b)
        _require(a != b, "inconsistent port equalities")
        graph[a].add(b)
        graph[b].add(a)
    order = list(graph)
    rng.shuffle(order)
    order.sort(key=lambda a: -len(graph[a]))
    assigned = {}
    for node in order:
        options = [c for c in palette if c not in {assigned.get(b) for b in graph[node]}]
        _require(bool(options), "palette coloring exhausted")
        assigned[node] = int(rng.choice(options))
    return [assigned[root(i)] for i in range(n)]


def _make_pair(rng, palette, background, style, used):
    for attempt in range(400):
        try:
            chain = style == "chain"
            if chain:
                gaps = [7, 7, 7]
                if rng.random() < .5:
                    gaps[int(rng.integers(3))] = 9
                columns = [0, gaps[0], sum(gaps[:2]), sum(gaps)]
                positions = [(0, c) for c in columns]
                anchor = int(rng.choice((0, 3)))
                spur_col = columns[anchor] + (3 if anchor == 0 else -3)
                spur_row = int(rng.choice((-1, 1))) * int(rng.choice((7, 9, 11, 13)))
                positions.append((spur_row, spur_col))
            else:
                count = int(rng.integers(4, 9) if style == "combined"
                            else rng.integers(3, 8))
                positions = _grow(rng, count, style == "combined")
            ports, links = _geometry(positions)
            colors = _color_ports(rng, positions, links, palette, chain)
            min_r, min_c = min(r for r, _ in positions), min(c for _, c in positions)
            positions = [(r - min_r + 2, c - min_c + 2) for r, c in positions]
            # Independent random margins, with a bounded 30-cell canvas.
            h0, w0 = max(r for r, _ in positions) + 4, max(c for _, c in positions) + 4
            _require(h0 <= 30 and w0 <= 30, "scene exceeds canvas")
            h = int(rng.integers(h0, min(30, h0 + 5) + 1))
            w = int(rng.integers(w0, min(30, w0 + 5) + 1))
            shift_r, shift_c = int(rng.integers(h - h0 + 1)), int(rng.integers(w - w0 + 1))
            grid = np.full((h, w), background, dtype=int)
            for t, (r, c) in enumerate(positions):
                for k, (dr, dc) in enumerate(_CORNERS):
                    grid[r + shift_r + dr, c + shift_c + dc] = colors[4 * t + k]
            # Rotate/reflect entire scenes; corner roles are always read in
            # the resulting global coordinates, not an unstated local frame.
            grid = np.rot90(grid, int(rng.integers(4)))
            if rng.random() < .5:
                grid = np.fliplr(grid)
            grid = grid.copy()
            info = _interpret(grid)
            _require(bool(info["active"]), "scene has no handshake")
            _require(len(info["active"]) < len(info["links"]), "scene needs a mismatch")
            if chain:
                _require(info["stagger"] and bool(_blocked_matches(info)),
                         "chain needs stagger and a blocked distant match")
            else:
                _require(info["branch"] and len(info["active_colors"]) >= 2,
                         "branch scene needs a branching port and two result colors")
                if style == "combined":
                    _require(info["stagger"], "test must combine branch and stagger")
                else:
                    _require(not info["stagger"], "straight teaching scene cannot stagger")
            _require(_key(grid) not in used, "duplicate input")
            output = solve(grid)
            used.add(_key(grid))
            return {"input": grid, "output": output}
        except ValueError:
            continue
    raise RuntimeError("bounded scene sampler exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and
                 isinstance(value, (int, np.integer)) and lo <= value <= hi,
                 f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and
             isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.choice(10, size=int(rng.integers(5, 8)), replace=False)]
    backgrounds = [c for c in range(10) if c not in palette]
    fixed_background = int(rng.choice(backgrounds))
    vary_background = bool(rng.integers(2))
    used = set()
    train = []
    # Evidence requirements, then shuffle so position never identifies a case.
    styles = ["straight", "chain"] + [str(rng.choice(("straight", "chain")))
                                        for _ in range(num_train - 2)]
    rng.shuffle(styles)
    for style in styles:
        bg = int(rng.choice(backgrounds)) if vary_background else fixed_background
        train.append(_make_pair(rng, palette, bg, style, used))
    test = []
    for _ in range(num_test):
        bg = int(rng.choice(backgrounds)) if vary_background else fixed_background
        test.append(_make_pair(rng, palette, bg, "combined", used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "each pair needs exactly input/output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output shape changed")
    info = _interpret(inp)
    expected = inp.copy()
    for cell, color in info["paint"].items():
        expected[cell] = color
    _require(np.array_equal(expected, out), "incorrect corner handshake output")
    changed = set(map(tuple, np.argwhere(inp != out)))
    _require(changed == set(info["paint"]), "wrong set of modified cells")
    _require(bool(info["active"]) and len(info["active"]) < len(info["links"]),
             "every generated pair needs an active link and a color mismatch")
    _require(len(info["used_ports"]) < len(info["ports"]), "needs unused corners")
    # A-ablation: treating the colored tile cells themselves as endpoints puts
    # these annotations on occupied cells, while the true ports are outside.
    _require(all(inp[info["ports"][i]] == info["background"]
                 for i in info["used_ports"]), "A's projected endpoints are not empty")
    # B-ablation: merely annotating matching corner ports omits every cross.
    _require(bool(info["cross_cells"] - set(info["ports"])), "B has no independent effect")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train/test must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input within puzzle")
            seen.add(key)
            features[split].append(info)
    train, test = features["train"], features["test"]
    _require(any(x["branch"] and not x["stagger"] and len(x["active_colors"]) >= 2
                 for x in train), "training must demonstrate multicolor straight branching")
    _require(any(x["stagger"] and not x["branch"] and _blocked_matches(x)
                 for x in train), "training must demonstrate stagger and nearest-port blocking")
    _require(all(not (x["branch"] and x["stagger"]) for x in train),
             "training reserves the combined topology for testing")
    _require(all(x["branch"] and x["stagger"] and len(x["active_colors"]) >= 2
                 for x in test), "test must combine the demonstrated mechanisms")


def _serialize(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                           for p in puzzle[s]] for s in ("train", "test")},
                      separators=(",", ":"))


def _semantic_fixtures():
    """Hand-calculated cells, not expected outputs produced by solve."""
    cases = [
        ((6, 15), [(2, 2), (2, 11)],
         [[1, 2, 3, 4], [2, 5, 6, 7]],
         [(2, [(1, 4), (1, 10)], (1, 7))]),
        ((9, 15), [(2, 2), (5, 11)],
         [[1, 2, 3, 4], [4, 5, 6, 7]],
         [(4, [(4, 4), (4, 10)], (4, 7))]),
        ((16, 16), [(2, 2), (2, 11), (11, 2)],
         [[1, 2, 3, 4], [5, 6, 4, 7], [8, 4, 5, 6]],
         [(4, [(4, 4), (4, 10)], (4, 7)),
          (4, [(4, 4), (10, 4)], (7, 4))]),
        ((6, 27), [(2, 2), (2, 9), (2, 16), (2, 23)],
         [[1, 2, 3, 4], [5, 6, 7, 8], [6, 9, 8, 1], [2, 3, 4, 5]],
         [(6, [(1, 11), (1, 15)], (1, 13)),
          (8, [(4, 11), (4, 15)], (4, 13))]),
    ]
    for shape, positions, tiles, constructions in cases:
        grid = np.zeros(shape, dtype=int)
        for (r, c), tile in zip(positions, tiles):
            grid[r:r + 2, c:c + 2] = np.array(tile).reshape(2, 2)
        expected = grid.copy()
        for color, endpoints, (r, c) in constructions:
            for cell in endpoints + [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                expected[cell] = color
        _require(np.array_equal(solve(grid), expected), "hand-calculated fixture failed")


def self_test():
    """Bounded 42-seed semantic, ablation, diversity and interface checks."""
    _semantic_fixtures()
    signatures, shapes, layouts, colors, counts, gaps, active_counts = (set() for _ in range(7))
    edge_crosses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_serialize(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _check_pair(pair)
                inp = pair["input"]
                saved = inp.copy()
                _require(np.array_equal(solve(inp), pair["output"]), "solve disagrees")
                _require(np.array_equal(inp, saved), "solve mutates its input")
                shapes.add(inp.shape)
                counts.add(len(info["positions"]))
                active_counts.add(len(info["active"]))
                gaps.update(info["spans"])
                colors.update(info["active_colors"])
                r0 = min(r for r, _ in info["positions"])
                c0 = min(c for _, c in info["positions"])
                layouts.add(tuple((r - r0, c - c0) for r, c in info["positions"]))
                edge_crosses += any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1)
                                    for r, c in info["cross_cells"])
                # Painting all A-style corners would incorrectly activate an
                # unused port. Omitting B leaves the cross cells unchanged.
                annotate_all = inp.copy()
                for port, color in zip(info["ports"], info["colors"]):
                    annotate_all[port] = color
                _require(not np.array_equal(annotate_all, pair["output"]), "A-only shortcut survives")
                endpoints_only = inp.copy()
                for i in info["used_ports"]:
                    endpoints_only[info["ports"][i]] = info["colors"][i]
                _require(not np.array_equal(endpoints_only, pair["output"]), "B ablation survives")
                # Translation plus palette permutations must commute with solve.
                permutation = np.roll(np.arange(10), seed % 9 + 1)
                renamed = permutation[inp]
                _require(np.array_equal(solve(renamed), permutation[pair["output"]]),
                         "literal color identity leaked into solving")
    _require(len(signatures) == 42, "seed diversity failed")
    _require(len(shapes) >= 35 and len(layouts) >= 70, "geometry diversity failed")
    _require(len(colors) == 10 and len(counts) >= 5 and len(gaps) >= 5
             and len(active_counts) >= 5, "structural/palette variation too narrow")
    _require(edge_crosses > 0, "boundary-reaching crosses absent")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12345, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count request ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_serialize(generate_puzzle(812345, 7, 3)) ==
             _serialize(generate_puzzle(812345, 7, 3)), "determinism failed")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, geometry and ablations")
