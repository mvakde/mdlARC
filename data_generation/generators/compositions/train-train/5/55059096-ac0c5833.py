"""Diagonal relay printers: connected cross centers share an oriented stencil.

Only NumPy and the standard library are used. No parent code or puzzle data is
loaded. The solver parses the four visible roles and computes graph reachability
from the input; it never consults the generator or a puzzle seed.
"""

from collections import deque
import numpy as np


# Clockwise from the southeast missing corner. Row coordinates increase down.
_DIRECTIONS = ((1, 1), (1, -1), (-1, -1), (-1, 1))
_HOST_SEEDS = (0, 693952463, 43504034, 702867709, 806097031, 549744972,
               14387924, 581151232, 117463223, 268903178, 768072302,
               973865550, 264758199, 42091079, 775370702, 409654113,
               270951574, 66725525, 208999257, 709364954, 826066361,
               813450359, 650743889, 787815638, 712901715, 45844383,
               709201338, 123213320, 248802351, 824307078, 773165717,
               957315816, 25189577, 145220691, 802467394, 504984355,
               385284811, 988293219, 931269094, 311359314, 440549897,
               577257600)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return grid.astype(np.int8, copy=False)


def _cross(center):
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _turn(r, c, k):
    return ((r, c), (c, -r), (-r, -c), (-c, r))[k % 4]


def _anchor(node):
    r, c, k = node
    dr, dc = _DIRECTIONS[k]
    return r + dr, c + dc


def _place(node, points, orientation=None):
    ar, ac = _anchor(node)
    k = node[2] if orientation is None else orientation
    return {(ar + dr, ac + dc) for i, j in points
            for dr, dc in (_turn(i, j, k),)}


def _zone(node):
    return _place(node, ((i, j) for i in range(3) for j in range(3)))


def _support(node):
    r, c, _ = node
    return _zone(node) | {(r + i, c + j) for i in (-1, 0, 1)
                          for j in (-1, 0, 1)}


def _diagonal(a, b):
    return a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _graph(nodes):
    adjacency = [set() for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if _diagonal(nodes[i], nodes[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
    components, remaining = [], set(range(len(nodes)))
    while remaining:
        root = min(remaining)
        found, queue = {root}, [root]
        while queue:
            u = queue.pop()
            for v in adjacency[u] - found:
                found.add(v)
                queue.append(v)
        remaining -= found
        components.append(tuple(sorted(found)))
    return adjacency, components


def _parse_with_roles(grid, background, cross_color, marker, ink):
    cross_pixels = set(map(tuple, np.argwhere(grid == cross_color).tolist()))
    _require(len(cross_pixels) >= 10 and len(cross_pixels) % 5 == 0,
             "cross role must contain complete unit crosses")
    centers = [p for p in sorted(cross_pixels) if _cross(p) <= cross_pixels]
    used = set()
    for center in centers:
        cells = _cross(center)
        _require(not cells & used, "overlapping cross decomposition")
        used |= cells
    _require(used == cross_pixels, "unexplained cross pixels")
    nodes, expected_markers, occupied = [], set(), set()
    for r, c in centers:
        _require(1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1,
                 "cross touches outside grid")
        corners = [(r + dr, c + dc) for dr, dc in _DIRECTIONS]
        missing = [k for k, p in enumerate(corners) if grid[p] != marker]
        _require(len(missing) == 1, "printer needs exactly three corner markers")
        node = (r, c, missing[0])
        support = _support(node)
        _require(all(0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                     for rr, cc in support), "port extends beyond boundary")
        _require(not occupied & support, "printer tiles or reserved ports overlap")
        occupied |= support
        expected_markers.update(p for k, p in enumerate(corners) if k != node[2])
        nodes.append(node)
    _require(expected_markers == set(map(tuple, np.argwhere(grid == marker).tolist())),
             "unexplained marker pixels")
    ink_pixels = set(map(tuple, np.argwhere(grid == ink).tolist()))
    sources, assigned = {}, set()
    for i, node in enumerate(nodes):
        local = ink_pixels & _zone(node)
        if not local:
            continue
        ar, ac = _anchor(node)
        points = frozenset(_turn(r - ar, c - ac, -node[2]) for r, c in local)
        height = 1 + max(r for r, c in points)
        width = 1 + max(c for r, c in points)
        _require(2 <= height <= 3 and 2 <= width <= 3, "invalid stencil extent")
        _require(min(r for r, c in points) == 0 and min(c for r, c in points) == 0,
                 "stencil must visibly touch all four bounding sides")
        sources[i] = points
        assigned |= local
    _require(assigned == ink_pixels and sources, "ink must belong to clean exemplars")
    adjacency, components = _graph(nodes)
    _require(all(sum(i in sources for i in comp) <= 1 for comp in components),
             "a connected component has conflicting exemplars")
    return {"grid": grid, "palette": (background, cross_color, marker, ink),
            "nodes": nodes, "sources": sources, "adjacency": adjacency,
            "components": components}


def _analyze(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 4, "inputs use exactly four visible roles")
    background = int(values[np.argmax(counts)])
    _require(int(max(counts)) > grid.size // 2, "background must occupy a majority")
    foreground = [int(v) for v in values if v != background]
    interpretations = []
    for cross_color in foreground:
        for marker in foreground:
            if marker == cross_color:
                continue
            ink = next(c for c in foreground if c not in (cross_color, marker))
            try:
                interpretations.append(_parse_with_roles(
                    grid, background, cross_color, marker, ink))
            except AssertionError:
                pass
    _require(len(interpretations) == 1, "input roles do not have a unique interpretation")
    return interpretations[0]


def _payload(info, mode="rule"):
    """Also supports deliberate ablations for evidence checks, not solving."""
    nodes, sources = info["nodes"], info["sources"]
    result = set()
    for comp in info["components"]:
        local = [i for i in comp if i in sources]
        if not local and mode != "global":
            continue
        source = local[0] if local else min(sources)
        points = sources[min(sources)] if mode == "one_stencil" else sources[source]
        for i in comp:
            if mode == "direct" and i != source and i not in info["adjacency"][source]:
                continue
            if mode == "mirror":
                # Reflect the observed exemplar about row/column axes until
                # its quadrant matches the recipient, without swapping axes.
                sr, sc = _DIRECTIONS[nodes[source][2]]
                tr, tc = _DIRECTIONS[nodes[i][2]]
                ar, ac = _anchor(nodes[i])
                for r, c in points:
                    dr, dc = _turn(r, c, nodes[source][2])
                    result.add((ar + dr * sr * tr, ac + dc * sc * tc))
                continue
            orientation = nodes[source][2] if mode == "no_turn" else None
            result |= _place(nodes[i], points, orientation)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy each component's sole exemplar to its printers, by quarter-turn."""
    info = _analyze(input_grid)
    output = info["grid"].copy()
    ink = info["palette"][3]
    for cell in _payload(info):
        output[cell] = ink
    return output


def _stencil(rng, height, width, *, empty_anchor=None, asymmetric=False, solid=False):
    for _ in range(500):
        if solid:
            mask = np.ones((height, width), dtype=bool)
        else:
            density = float(rng.uniform(.35, .85))
            mask = rng.random((height, width)) < density
        if empty_anchor is not None:
            mask[0, 0] = not empty_anchor
        if not all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())):
            continue
        if asymmetric:
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue
            # Explicitly expose rotation versus a mirror-in-each-axis shortcut.
            if height == width and np.array_equal(mask, mask.T):
                continue
        return frozenset(map(tuple, np.argwhere(mask).tolist()))
    raise RuntimeError("bounded stencil sampling failed")


def _layout(rng, height, width, sizes, orientations, near_miss=False, require_relay=False):
    """Grow arbitrary diagonal graphs; reject accidental links between groups.

    Evidence-critical examples start the first group with a bent relay path.
    Otherwise any earlier parent is available, permitting straight groups,
    branches, cycles and extra edges. There is no object-slot lattice.
    """
    for _ in range(180):
        nodes, groups, occupied = [], [], set()
        failed = False
        for group_id, size in enumerate(sizes):
            group = []
            for offset in range(size):
                k = orientations[group_id][offset]
                if offset == 0:
                    choices = [(r, c) for r in range(1, height - 1)
                               for c in range(1, width - 1)]
                else:
                    forced_path = require_relay and group_id == 0 and offset < 3
                    parents = group[-1:] if forced_path else group
                    choices = list({(nodes[p][0] + dr * distance,
                                     nodes[p][1] + dc * distance)
                                    for p in parents for dr, dc in _DIRECTIONS
                                    for distance in range(3, 13)})
                    choices.sort()
                rng.shuffle(choices)
                for r, c in choices:
                    node = (r, c, k)
                    support = _support(node)
                    if not all(0 <= rr < height and 0 <= cc < width for rr, cc in support):
                        continue
                    if occupied & support:
                        continue
                    if any(_diagonal(node, old) for j, old in enumerate(nodes) if j not in group):
                        continue
                    if (require_relay and group_id == 0 and offset == 2
                            and _diagonal(node, nodes[group[0]])):
                        continue
                    if near_miss and group_id == len(sizes) - 1 and offset == 0:
                        if not any(abs(abs(r - old[0]) - abs(c - old[1])) == 1
                                   and max(abs(r - old[0]), abs(c - old[1])) >= 3
                                   for old in nodes):
                            continue
                    group.append(len(nodes))
                    nodes.append(node)
                    occupied |= support
                    break
                else:
                    failed = True
                    break
            if failed:
                break
            groups.append(group)
        if not failed:
            return nodes, groups
    return None


def _make_pair(rng, palette, regime):
    for _ in range(70):
        base = int(rng.integers(4))
        if regime == "relay_demo":
            # An entire linked but seedless component is negative evidence:
            # merely having a diagonal neighbor does not authorize a copy.
            sizes, active = [3, 2], 1
            orientations = [[base, (base + 1) % 4, (base + 2) % 4],
                            rng.integers(4, size=2).tolist()]
            stencils = [_stencil(rng, 3, 3, empty_anchor=True, asymmetric=True)]
            low = 19
        elif regime == "ownership_demo":
            sizes, active = [2, 2, 1], 2
            orientations = [[base, (base + 3) % 4],
                            [(base + 2) % 4, (base + 2) % 4], [int(rng.integers(4))]]
            stencils = [_stencil(rng, 2, 2, solid=True),
                        _stencil(rng, 2, 3, empty_anchor=False, asymmetric=True)]
            low = 21
        elif regime == "combined_test":
            sizes = [int(rng.integers(3, 6)), int(rng.integers(2, 4)),
                     int(rng.integers(1, 3))]
            if sum(sizes) > 9:
                sizes[0] = 3
            active, low = 2, 25
            orientations = [rng.integers(4, size=n).tolist() for n in sizes]
            stencils = []
        else:
            active = int(rng.integers(1, 3))
            sizes = [int(rng.integers(2, 6)) for _ in range(active)] + [int(rng.integers(1, 3))]
            if sum(sizes) > 9:
                sizes[0] = 2
            orientations = [rng.integers(4, size=n).tolist() for n in sizes]
            stencils, low = [], 21
        if not stencils:
            for _ in range(active):
                h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
                points = _stencil(rng, h, w, solid=bool(rng.random() < .18))
                for _retry in range(500):
                    if points not in stencils:
                        break
                    points = _stencil(rng, h, w)
                else:
                    raise RuntimeError("bounded distinct-stencil sampling failed")
                stencils.append(points)
        height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        layout = _layout(rng, height, width, sizes, orientations,
                         near_miss=regime in ("relay_demo", "ownership_demo"),
                         require_relay=regime in ("relay_demo", "combined_test"))
        if layout is None:
            continue
        nodes, groups = layout
        background, cross_color, marker, ink = palette
        grid = np.full((height, width), background, dtype=np.int8)
        for node in nodes:
            for cell in _cross(node[:2]):
                grid[cell] = cross_color
            for k, (dr, dc) in enumerate(_DIRECTIONS):
                if k != node[2]:
                    grid[node[0] + dr, node[1] + dc] = marker
        for group, stencil in zip(groups[:active], stencils):
            for cell in _place(nodes[group[0]], stencil):
                grid[cell] = ink
        # Ground truth uses the construction groups. The solver instead
        # reconstructs all relationships from the visible input.
        output = grid.copy()
        for group, stencil in zip(groups[:active], stencils):
            for index in group:
                for cell in _place(nodes[index], stencil):
                    output[cell] = ink
        try:
            info = _analyze(grid)
            _require({frozenset(nodes[i] for i in g) for g in groups} ==
                     {frozenset(info["nodes"][i] for i in g) for g in info["components"]},
                     "constructed and parsed components differ")
            _require(np.array_equal(solve(grid), output), "constructed graph and parsed graph disagree")
            return {"input": grid, "output": output}
        except AssertionError:
            continue
    raise RuntimeError(f"bounded construction failed for {regime}")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _signature(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples, with the same input-only rule in both splits."""
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(10, 4, replace=False))
    shared_palette = bool(rng.random() < .6)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            regime = ("combined_test" if split == "test" else
                      "relay_demo" if index == 0 else
                      "ownership_demo" if index == 1 else "free")
            for _ in range(20):
                colors = palette if shared_palette else tuple(int(c) for c in rng.choice(10, 4, replace=False))
                pair = _make_pair(rng, colors, regime)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    _require(grid.shape == output.shape, "output size changed")
    info = _analyze(grid)
    background, cross_color, marker, ink = info["palette"]
    expected = _payload(info)
    observed = set(map(tuple, np.argwhere(output == ink).tolist()))
    _require(expected == observed, "incorrect transported stencil pixels")
    _require(np.array_equal(grid[grid != background], output[grid != background]),
             "an original foreground cell changed")
    _require(np.all((grid == output) | ((grid == background) & (output == ink))),
             "only background-to-ink additions are allowed")
    _require(not np.array_equal(grid, output), "identity example")
    active = [c for c in info["components"] if any(i in info["sources"] for i in c)]
    inactive = [c for c in info["components"] if not any(i in info["sources"] for i in c)]
    _require(active and inactive, "generated examples need both active and inactive printers")
    _require(all(len(c) >= 2 for c in active), "exemplar has no recipient")
    _require(len(observed) == sum(len(c) * len(info["sources"][next(i for i in c if i in info["sources"])])
                                 for c in active), "payload conservation per component")
    info["active"], info["inactive"] = active, inactive
    info["turns"] = {(info["nodes"][j][2] - info["nodes"][i][2]) % 4
                     for comp in active for i in comp if i in info["sources"]
                     for j in comp if j != i}
    info["indirect"] = expected != _payload(info, "direct")
    info["rotation_needed"] = expected != _payload(info, "no_turn")
    info["mirror_wrong"] = expected != _payload(info, "mirror")
    info["alignment_needed"] = expected != _payload(info, "global")
    info["ownership_needed"] = expected != _payload(info, "one_stencil")
    return info


def validate_generated_puzzle(puzzle):
    """Validate schema, local geometry, graph ownership and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _signature(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    _require(set().union(*(x["turns"] for x in train)) == {0, 1, 2, 3},
             "training must show all four relative rotations")
    _require(any(x["indirect"] for x in train), "missing indirect relay demonstration")
    _require(any(any(len(c) >= 2 for c in x["inactive"]) for x in train),
             "linked seedless printers must be demonstrated")
    _require(any(x["rotation_needed"] for x in train), "rotation shortcut not distinguished")
    _require(any(x["mirror_wrong"] for x in train), "reflection shortcut not distinguished")
    _require(all(x["alignment_needed"] for x in train), "alignment shortcut not distinguished")
    _require(any(len(x["sources"]) >= 2 and len(set(x["sources"].values())) >= 2 for x in train),
             "missing component-specific stencil demonstration")
    _require(any(x["ownership_needed"] for x in train), "global stencil shortcut not distinguished")
    _require(any((0, 0) not in p for x in train for p in x["sources"].values()),
             "missing empty-anchor demonstration")
    _require(any((0, 0) in p for x in train for p in x["sources"].values()),
             "missing occupied-anchor demonstration")
    slopes = {1 if (x["nodes"][i][0] - x["nodes"][j][0]) *
                    (x["nodes"][i][1] - x["nodes"][j][1]) > 0 else -1
              for x in train for i, neighbors in enumerate(x["adjacency"]) for j in neighbors}
    _require(slopes == {-1, 1}, "both diagonal slopes must be demonstrated")
    for x in infos["test"]:
        _require(len(x["sources"]) == 2 and len(x["nodes"]) >= 6 and x["indirect"],
                 "missing combined test shift")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded 42-seed coverage audit plus determinism/count checks."""
    shapes, layouts, masks, palettes, counts, graph_types = set(), set(), set(), set(), set(), set()
    empty = solid = rectangular = branches = indirect = 0
    max_depth = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                nodes = info["nodes"]
                shapes.add(pair["input"].shape)
                layouts.add(tuple(nodes))
                palettes.add(info["palette"])
                counts.add(len(nodes))
                graph_types.add((tuple(sorted(map(len, info["components"]))),
                                 tuple(sorted(map(len, info["adjacency"])))))
                branches += any(len(n) >= 3 for n in info["adjacency"])
                indirect += info["indirect"]
                for source, points in info["sources"].items():
                    h = 1 + max(r for r, c in points)
                    w = 1 + max(c for r, c in points)
                    masks.add((h, w, tuple(sorted(points))))
                    empty += (0, 0) not in points
                    solid += len(points) == h * w
                    rectangular += h != w
                    distances, queue = {source: 0}, deque([source])
                    while queue:
                        u = queue.popleft()
                        for v in info["adjacency"][u] - distances.keys():
                            distances[v] = distances[u] + 1
                            queue.append(v)
                    max_depth = max(max_depth, max(distances.values()))
    _require(len(shapes) >= 30 and len(layouts) >= 200 and len(masks) >= 50,
             "insufficient structural or stencil variation")
    _require(len(palettes) >= 30 and len(counts) >= 5 and len(graph_types) >= 12,
             "insufficient palette/count/topology variation")
    _require(empty >= 42 and solid >= 42 and rectangular >= 42 and branches > 0 and max_depth >= 3,
             "missing family-specific structural coverage")
    _require(_equal(generate_puzzle(937), generate_puzzle(937)), "nondeterministic default")
    _require(_equal(generate_puzzle(938, 7, 3), generate_puzzle(938, 7, 3)), "nondeterministic counts")
    # Rule-level equivariance and nested-list input, independent of split/order.
    exemplar = generate_puzzle(0, 2, 1)["train"][0]
    permutation = np.arange(9, -1, -1, dtype=np.int8)
    for turns in range(4):
        moved_input = permutation[np.rot90(exemplar["input"], turns)]
        moved_output = permutation[np.rot90(exemplar["output"], turns)]
        _require(np.array_equal(solve(moved_input.tolist()), moved_output),
                 "rotation/color-permutation equivariance failed")
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(2000 + i, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "shapes": len(shapes), "layouts": len(layouts),
            "stencils": len(masks), "palettes": len(palettes),
            "printer_counts": sorted(counts), "graph_types": len(graph_types),
            "empty_anchors": empty, "solid_stencils": solid,
            "rectangular_stencils": rectangular, "branching_pairs": branches,
            "indirect_pairs": indirect, "max_relay_depth": max_depth}
