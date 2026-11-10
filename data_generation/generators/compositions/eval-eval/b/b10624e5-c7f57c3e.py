"""Alternating mirror relays: two scaffold-registered states on a wire tree.

Only standard-library modules and NumPy are used. Generation paints logical
blocks; the solver independently reads square anchors, straight wires, and the
two visible reference motifs. No examples, seeds, or sampled metadata enter solve.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _array(grid):
    a = np.asarray(grid)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points) + 1, max(c for r, c in points) + 1)


def _flip(mapping, tag):
    v, h = tag
    return {((-r if v else r), (-c if h else c)): color
            for (r, c), color in mapping.items()}


def _labels(count, edges, root):
    neighbors = [[] for _ in range(count)]
    for a, b, axis in edges:
        neighbors[a].append((b, axis))
        neighbors[b].append((a, axis))
    tags, depths = {root: (0, 0)}, {root: 0}
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b, axis in neighbors[a]:
            v, h = tags[a]
            tag = (v ^ (axis == 0), h ^ (axis == 1))
            if b in tags:
                _require(tags[b] == tag, "inconsistent wire cycle")
                continue
            tags[b], depths[b] = tag, depths[a] + 1
            queue.append(b)
    _require(len(tags) == count and len(edges) == count - 1, "wires must form one tree")
    return tags, depths


def _wire_edges(grid, wire, node_ids, count):
    components = _components(grid == wire)
    _require(len(components) == count - 1, "wrong number of wire segments")
    edges = []
    height, width = grid.shape
    for group in components:
        r0, c0, r1, c1 = _bounds(group)
        _require(len(group) >= 2 and (r1 - r0 == 1 or c1 - c0 == 1), "wire must be straight and thin")
        if r1 - r0 == 1:
            _require(c0 > 0 and c1 < width, "dangling horizontal wire")
            a, b, axis = int(node_ids[r0, c0 - 1]), int(node_ids[r0, c1]), 1
        else:
            _require(r0 > 0 and r1 < height, "dangling vertical wire")
            a, b, axis = int(node_ids[r0 - 1, c0]), int(node_ids[r1, c0]), 0
        _require(a >= 0 and b >= 0 and a != b, "wire endpoints must touch two anchors")
        touching = set()
        for r, c in group:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                y, x = r + dr, c + dc
                if 0 <= y < height and 0 <= x < width and node_ids[y, x] >= 0:
                    touching.add(int(node_ids[y, x]))
        _require(touching == {a, b}, "wire touches an extra anchor")
        edges.append((a, b, axis))
    _require(len({tuple(sorted((a, b))) for a, b, axis in edges}) == len(edges), "duplicate wire")
    return edges


def _read_roles(grid, background, anchor, wire, squares):
    ids = np.full(grid.shape, -1, dtype=np.int8)
    nodes = []
    for index, group in enumerate(squares):
        r0, c0, r1, c1 = _bounds(group)
        for p in group:
            ids[p] = index
        nodes.append({"top": r0, "left": c0, "side": r1 - r0})
    edges = _wire_edges(grid, wire, ids, len(nodes))
    objects = _components((grid != background) & (grid != wire), diagonal=True)
    _require(len(objects) == len(nodes), "motifs must be isolated after wires are removed")
    assigned = set()
    for group in objects:
        matches = {int(ids[p]) for p in group if ids[p] >= 0}
        _require(len(matches) == 1, "each motif needs exactly one square anchor")
        index = matches.pop()
        _require(index not in assigned, "split motif")
        assigned.add(index)
        node = nodes[index]
        side, top, left = node["side"], node["top"], node["left"]
        blocks = {}
        for r, c in group:
            q = (r - top) // side, (c - left) // side
            blocks.setdefault(q, []).append(int(grid[r, c]))
        raw = {}
        for p, colors in blocks.items():
            _require(len(colors) == side * side and len(set(colors)) == 1, "motif is not anchor-aligned pixel scaling")
            raw[p] = colors[0]
        _require(raw.get((0, 0)) == anchor and sum(c == anchor for c in raw.values()) == 1,
                 "anchor color is reserved for the square")
        _require(all((r != 0 and c != 0) or (r, c) == (0, 0) for r, c in raw),
                 "cardinal port corridors must be free of decorations")
        node["raw"] = raw
    references = [i for i, node in enumerate(nodes) if len(node["raw"]) > 1]
    _require(len(references) == 2, "exactly two complete reference motifs are required")
    root, other = references
    tags, depths = _labels(len(nodes), edges, root)
    _require(depths[other] % 2 == 1, "reference states must be an odd number of bridges apart")
    templates = (nodes[root]["raw"], _flip(nodes[other]["raw"], tags[other]))
    _require(templates[0] != templates[1], "the two normalized states must differ")
    common = {p: c for p, c in templates[0].items() if templates[1].get(p) == c}
    _require(len(common) >= 2 and common.get((0, 0)) == anchor, "missing shared scaffold beyond the anchor")
    union = set(templates[0]) | set(templates[1])
    rectangles = []
    for i, node in enumerate(nodes):
        v, h = tags[i]
        state = v ^ h
        target = _flip(templates[state], tags[i])
        if i in references:
            _require(node["raw"] == target, "reference does not agree with relay phase")
        else:
            _require(node["raw"] == {(0, 0): anchor}, "uncompleted nodes contain extra material")
        support = _flip({p: anchor for p in union}, tags[i])
        y0, x0, y1, x1 = _bounds(support)
        s, y, x = node["side"], node["top"], node["left"]
        rect = y + y0 * s, x + x0 * s, y + y1 * s, x + x1 * s
        _require(0 <= rect[0] < rect[2] <= grid.shape[0] and
                 0 <= rect[1] < rect[3] <= grid.shape[1], "replacement would be clipped")
        for prior in rectangles:
            _require(rect[2] < prior[0] or prior[2] < rect[0] or
                     rect[3] < prior[1] or prior[3] < rect[1], "motif reserves must have a background gap")
        rectangles.append(rect)
        for r, c in support:
            block = grid[y + r*s:y + (r+1)*s, x + c*s:x + (c+1)*s]
            _require(not np.any(block == wire), "decoration collides with a wire")
        node.update(tag=tags[i], depth=depths[i], state=state)
    shared = set(templates[0]) & set(templates[1])
    recolor = any(templates[0][p] != templates[1][p] for p in shared)
    relocate = set(templates[0]) != set(templates[1])
    return {"background": background, "anchor": anchor, "wire": wire, "nodes": nodes,
            "edges": edges, "references": references, "templates": templates,
            "reference_distance": depths[other], "recolor": recolor, "relocate": relocate,
            "change": "combined" if recolor and relocate else "recolor" if recolor else "relocate"}


def _decode(grid):
    background = int(np.bincount(grid.ravel().astype(int), minlength=10).argmax())
    _require(np.count_nonzero(grid == background) * 2 > grid.size, "background must occupy a strict majority")
    colors = [int(c) for c in np.unique(grid) if c != background]
    candidates = []
    for anchor in colors:
        squares = _components(grid == anchor)
        if not 4 <= len(squares) <= 8:
            continue
        valid = True
        for group in squares:
            r0, c0, r1, c1 = _bounds(group)
            if not (r1-r0 == c1-c0 and 1 <= r1-r0 <= 3 and len(group) == (r1-r0)**2):
                valid = False
                break
        if not valid:
            continue
        for wire in colors:
            if wire == anchor:
                continue
            try:
                candidates.append(_read_roles(grid, background, anchor, wire, squares))
            except ValueError:
                pass
    _require(len(candidates) == 1, "network roles or state registration are missing or ambiguous")
    return candidates[0]


def _prediction(grid, info, mode="rule"):
    output = grid.copy()
    root = info["nodes"][info["references"][0]]
    root_center = (root["top"] + (root["side"]-1)//2, root["left"] + (root["side"]-1)//2)
    rows = sorted({n["top"] + (n["side"]-1)//2 for n in info["nodes"]})
    cols = sorted({n["left"] + (n["side"]-1)//2 for n in info["nodes"]})
    for i, node in enumerate(info["nodes"]):
        if i in info["references"]:
            continue
        tag, state = node["tag"], node["state"]
        if mode == "no_exchange":
            state = 0
        elif mode == "no_reflection":
            tag = (0, 0)
        elif mode == "absolute_position":
            center = (node["top"] + (node["side"]-1)//2, node["left"] + (node["side"]-1)//2)
            tag = (int(center[0] != root_center[0]), int(center[1] != root_center[1]))
            state = tag[0] ^ tag[1]
        elif mode == "checkerboard":
            center = (node["top"] + (node["side"]-1)//2, node["left"] + (node["side"]-1)//2)
            tag = ((rows.index(center[0]) - rows.index(root_center[0])) % 2,
                   (cols.index(center[1]) - cols.index(root_center[1])) % 2)
            state = tag[0] ^ tag[1]
        elif mode == "exchange_once":
            state = 1
        template = _flip(info["templates"][state], tag)
        s, y, x = node["side"], node["top"], node["left"]
        for (r, c), color in template.items():
            output[y + r*s:y + (r+1)*s, x + c*s:x + (c+1)*s] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete all bare anchors from this input's two references and wire tree."""
    grid = _array(input_grid)
    return _prediction(grid, _decode(grid))


def _patch(rng, corner, maximum=4):
    sy, sx = corner
    cells = {(sy, sx)}
    count = int(rng.integers(1, maximum + 1))
    while len(cells) < count:
        frontier = {(r + dr, c + dc) for r, c in cells
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 1 <= (r + dr)*sy <= 2 and 1 <= (c + dc)*sx <= 2} - cells
        choices = sorted(frontier)
        cells.add(choices[int(rng.integers(len(choices)))])
    return cells


def _design(rng, change, palette):
    bg, wire, anchor = palette[:3]
    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rng.shuffle(corners)
    stable = _patch(rng, corners[0], maximum=3)
    stable_colors = palette[3:3 + int(rng.integers(1, 3))]
    common = {(0, 0): anchor}
    for j, p in enumerate(sorted(stable)):
        common[p] = stable_colors[j % len(stable_colors)]
    a = _patch(rng, corners[1])
    b = a if change == "recolor" else _patch(rng, corners[2])
    interior = _patch(rng, corners[3], maximum=2) if change == "combined" else set()
    maps = []
    for state, cells in enumerate((a, b)):
        mapping = dict(common)
        accents = palette[5:5 + int(rng.integers(1, 3))] if change == "relocate" else palette[5+2*state:7+2*state]
        if change == "relocate":
            accents = palette[5:6]  # pure support movement, with no internal recoloring
        elif rng.random() < .5:
            accents = accents[:1]
        for j, p in enumerate(sorted(cells | interior)):
            mapping[p] = accents[j % len(accents)]
        maps.append(mapping)
    rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
    transformed = []
    for mapping in maps:
        result = {}
        for (r, c), color in mapping.items():
            for _ in range(rotation):
                r, c = c, -r
            result[r, -c if reflected else c] = color
        transformed.append(result)
    return bg, wire, anchor, tuple(transformed)


def _segment(a, b):
    r, c = a
    dr, dc = int(np.sign(b[0] - r)), int(np.sign(b[1] - c))
    return {(r + k*dr, c + k*dc) for k in range(abs(b[0]-r) + abs(b[1]-c) + 1)}


def _grow_tree(rng, count):
    points, edges, occupied = [(2, 2)], [], {(2, 2)}
    for _ in range(count - 1):
        choices = []
        for i, a in enumerate(points):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for length in range(1, 4):
                    b = a[0] + dr*length, a[1] + dc*length
                    if not (0 <= b[0] <= 4 and 0 <= b[1] <= 4):
                        continue
                    if len({p[0] for p in points} | {b[0]}) > 4 or len({p[1] for p in points} | {b[1]}) > 4:
                        continue
                    path = _segment(a, b)
                    if path & occupied == {a}:
                        choices.append((i, b, path, 0 if dr else 1))
        if not choices:
            return None
        i, b, path, axis = choices[int(rng.integers(len(choices)))]
        edges.append((i, len(points), axis))
        points.append(b)
        occupied |= path
    return points, edges


def _references(rng, points, edges, distant):
    options = []
    rows, cols = sorted({r for r, c in points}), sorted({c for r, c in points})
    for a in range(len(points)):
        tags, depths = _labels(len(points), edges, a)
        if set(tags.values()) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue
        for b in range(len(points)):
            if points[a] >= points[b] or (depths[b] < 3 if distant else depths[b] != 1) or depths[b] % 2 != 1:
                continue
            bare = [i for i in range(len(points)) if i not in (a, b)]
            if {tags[i][0] ^ tags[i][1] for i in bare} != {0, 1}:
                continue
            if not any(tags[i] == (0, 0) for i in bare):
                continue
            # Make topology observable: counting coordinate levels instead of
            # actual bridges must change at least one missing motif's frame.
            if not any(tags[i] != ((rows.index(points[i][0])-rows.index(points[a][0])) % 2,
                                   (cols.index(points[i][1])-cols.index(points[a][1])) % 2)
                       for i in bare):
                continue
            options.append((a, b, tags))
    return options[int(rng.integers(len(options)))] if options else None


def _layout(rng, points, scales, templates, fallback=False):
    rows, cols = sorted({r for r, c in points}), sorted({c for r, c in points})
    union = set(templates[0]) | set(templates[1])
    ry, rx = max(abs(r) for r, c in union), max(abs(c) for r, c in union)
    coordinates = []
    dimensions = []
    for axis, levels, radius in ((0, rows, ry), (1, cols, rx)):
        before, after = [], []
        for level in levels:
            sizes = [s for p, s in zip(points, scales) if p[axis] == level]
            before.append(max(radius*s + (s-1)//2 for s in sizes))
            after.append(max((radius+1)*s - 1 - (s-1)//2 for s in sizes))
        centers = [before[0] + (0 if fallback else int(rng.integers(0, 3)))]
        for j in range(1, len(levels)):
            gap = 2 if fallback else int(rng.integers(2, 5))
            centers.append(centers[-1] + after[j-1] + gap + before[j] + 1)
        extent = centers[-1] + after[-1] + 1 + (0 if fallback else int(rng.integers(0, 4)))
        if extent > 30:
            return None
        dimensions.append(extent)
        coordinates.append(dict(zip(levels, centers)))
    centers = [(coordinates[0][r], coordinates[1][c]) for r, c in points]
    return tuple(dimensions), centers


def _paint(grid, template, center, side):
    top, left = center[0] - (side-1)//2, center[1] - (side-1)//2
    for (r, c), color in template.items():
        y, x = top + r*side, left + c*side
        grid[y:y+side, x:x+side] = color


def _construct_pair(rng, change, index, test):
    palette = [int(c) for c in rng.permutation(10)]
    bg, wire, anchor, templates = _design(rng, change, palette)
    for attempt in range(96):
        fallback = attempt == 95
        if fallback:
            points = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 0), (2, 0)]
            edges = [(0, 1, 1), (1, 2, 1), (2, 3, 0), (3, 4, 1), (4, 5, 0)]
        else:
            tree = _grow_tree(rng, int(rng.integers(5, 9)))
            if tree is None:
                continue
            points, edges = tree
        chosen = _references(rng, points, edges, distant=test)
        if chosen is None:
            continue
        a, b, tags = chosen
        scales = [1] * len(points)
        bare = [i for i in range(len(points)) if i not in (a, b)]
        if not test and index == 0:
            scales[bare[int(rng.integers(len(bare)))]] = 2
        elif not test and index == 1:
            scales[a] = 2
        else:
            for i in rng.permutation(len(points))[:int(rng.integers(0, 3))]:
                scales[int(i)] = int(rng.choice([2, 2, 2, 3]))
        if fallback:
            scales = [1] * len(points)
            scales[a if not test and index == 1 else bare[0]] = 2
        arrangement = _layout(rng, points, scales, templates, fallback)
        if arrangement is None:
            continue
        shape, centers = arrangement
        inp, out = np.full(shape, bg, dtype=np.int8), np.full(shape, bg, dtype=np.int8)
        # Paint actual bridge pixels first; corner-only decorations leave their
        # cardinal corridors free. Anchors overwrite the wire inside their square.
        for u, v, axis in edges:
            y0, x0 = centers[u]
            y1, x1 = centers[v]
            if axis == 1:
                inp[y0, min(x0, x1):max(x0, x1)+1] = wire
            else:
                inp[min(y0, y1):max(y0, y1)+1, x0] = wire
        out[:] = inp
        for i, (center, side) in enumerate(zip(centers, scales)):
            v, h = tags[i]
            motif = _flip(templates[v ^ h], tags[i])
            _paint(inp, motif if i in (a, b) else {(0, 0): anchor}, center, side)
            _paint(out, motif, center, side)
        if np.count_nonzero(inp == bg)*2 <= inp.size or np.count_nonzero(out == bg)*2 <= out.size:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded relay construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            change = ("combined" if split == "test" else
                      "recolor" if index == 0 else "relocate" if index == 1 else
                      str(rng.choice(["recolor", "relocate", "combined"])))
            for _ in range(16):
                pair = _construct_pair(rng, change, index, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
        if split == "train":
            rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    inp, out = _array(pair["input"]), _array(pair["output"])
    _require(inp.shape == out.shape, "output shape changed")
    info = _decode(inp)
    expected = _prediction(inp, info)
    _require(np.array_equal(out, expected), "incorrect relay completion")
    _require(not np.array_equal(inp, out), "no completion")
    _require(np.count_nonzero(out == info["background"])*2 > out.size, "output background majority lost")
    bare = [n for i, n in enumerate(info["nodes"]) if i not in info["references"]]
    _require({n["state"] for n in bare} == {0, 1}, "bare nodes must exercise both states")
    for mode in ("no_exchange", "no_reflection", "absolute_position", "checkerboard", "exchange_once"):
        _require(not np.array_equal(out, _prediction(inp, info, mode)), f"shortcut {mode} is not distinguished")
    info["shape"] = inp.shape
    return info


def validate_generated_puzzle(puzzle):
    """Validate returned pixels, relay invariants, ablations, and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _integer(nt, "num_train", 2, 10)
    _integer(ns, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(_array(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    _require({"recolor", "relocate"} <= {i["change"] for i in infos["train"]},
             "demonstrations must separately expose recoloring and relocation")
    _require(all(i["reference_distance"] == 1 for i in infos["train"]), "training references must be adjacent")
    _require(all(i["change"] == "combined" and i["reference_distance"] >= 3 for i in infos["test"]),
             "test shift must combine changes and separate the references")
    enlarged = reduced = False
    for info in infos["train"]:
        root_side = info["nodes"][info["references"][0]]["side"]
        bare = [n for j, n in enumerate(info["nodes"]) if j not in info["references"]]
        enlarged |= any(n["side"] > root_side for n in bare)
        reduced |= any(n["side"] < root_side for n in bare)
    _require(enlarged and reduced, "training must expose enlarged and reduced anchored transfer")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _array(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _vocabulary(info):
    """Color-, state-order-, and dihedral-normalized pair of logical templates."""
    variants = []
    for rotation in range(4):
        for reflection in (False, True):
            maps = []
            for template in info["templates"]:
                transformed = []
                for (r, c), color in template.items():
                    for _ in range(rotation):
                        r, c = c, -r
                    transformed.append((r, -c if reflection else c, color))
                maps.append(sorted(transformed))
            for order in ((0, 1), (1, 0)):
                roles, encoded = {}, []
                for state in order:
                    encoded.append(tuple((r, c, roles.setdefault(color, len(roles))) for r, c, color in maps[state]))
                variants.append(tuple(encoded))
    return min(variants)


def self_test():
    """42 deterministic seeds, meaningful mutations, and counts; no files written."""
    fingerprints, vocabularies, shapes, scales, node_counts = set(), set(), set(), set(), set()
    graph_shapes, distances, backgrounds, palettes, changed = set(), set(), set(), set(), set()
    branch_pairs = skip_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = _fingerprint(puzzle)
        fingerprints.add(fingerprint)
        if seed < 3:
            _require(fingerprint == _fingerprint(generate_puzzle(seed)), "seed is nondeterministic")
        local_vocab = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                vocabulary = _vocabulary(info)
                vocabularies.add(vocabulary)
                local_vocab.add(vocabulary)
                nodes = info["nodes"]
                shapes.add(info["shape"])
                scales.update(n["side"] for n in nodes)
                node_counts.add(len(nodes))
                backgrounds.add(info["background"])
                palettes.add(len(np.unique(pair["input"])))
                distances.add(info["reference_distance"])
                changed.add((split, info["change"]))
                degree = [0]*len(nodes)
                ys, xs = sorted({n["top"]+(n["side"]-1)//2 for n in nodes}), sorted({n["left"]+(n["side"]-1)//2 for n in nodes})
                locations = [(ys.index(n["top"]+(n["side"]-1)//2), xs.index(n["left"]+(n["side"]-1)//2)) for n in nodes]
                graph_shapes.add((tuple(locations), tuple(sorted(info["edges"]))))
                skipped = False
                for a, b, axis in info["edges"]:
                    degree[a] += 1
                    degree[b] += 1
                    skipped |= abs(locations[a][axis] - locations[b][axis]) > 1
                branch_pairs += max(degree) >= 3
                skip_pairs += skipped
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only solver mismatch")
        _require(len(local_vocab) >= 3, "puzzle lacks structural template variation")
    _require(len(fingerprints) == 42, "seeds repeat puzzles")
    _require(len(vocabularies) >= 100 and len(graph_shapes) >= 100, "insufficient structural diversity")
    _require(len(shapes) >= 50 and scales >= {1, 2, 3} and node_counts >= {5, 6, 7, 8}, "missing geometry/scale/count variation")
    _require(len(backgrounds) == 10 and len(palettes) >= 4, "insufficient palette-role variation")
    _require(branch_pairs >= 40 and skip_pairs >= 25 and max(distances) >= 5, "missing network variation")
    _require(changed >= {("train", "recolor"), ("train", "relocate"), ("test", "combined")}, "missing state-change variation")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(500 + 10*nt + ns, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "incorrect counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(42, 2, 1)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    info = _decode(bad["test"][0]["input"])
    bare_index = next(i for i in range(len(info["nodes"])) if i not in info["references"])
    node = info["nodes"][bare_index]
    bad["test"][0]["output"][node["top"], node["left"]] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged scaffold accepted")
    for mode in ("no_exchange", "no_reflection", "absolute_position", "checkerboard", "exchange_once"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
        g = bad["test"][0]["input"]
        bad["test"][0]["output"] = _prediction(g, _decode(g), mode)
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ablated answer accepted: {mode}")
    # The references identify a relation, not a privileged root: replace them by
    # another odd-distance pair taken from the already completed network.
    original = sample["test"][0]
    info = _decode(original["input"])
    alternate = original["input"].copy()
    for index in info["references"]:
        n = info["nodes"][index]
        s, y, x = n["side"], n["top"], n["left"]
        for r, c in n["raw"]:
            alternate[y+r*s:y+(r+1)*s, x+c*s:x+(c+1)*s] = info["background"]
        alternate[y:y+s, x:x+s] = info["anchor"]
    refs = next((a, b) for a, b, axis in info["edges"] if {a, b} != set(info["references"]))
    for index in refs:
        n = info["nodes"][index]
        s, y, x = n["side"], n["top"], n["left"]
        template = _flip(info["templates"][n["state"]], n["tag"])
        for r, c in template:
            alternate[y+r*s:y+(r+1)*s, x+c*s:x+(c+1)*s] = original["output"][y+r*s:y+(r+1)*s, x+c*s:x+(c+1)*s]
    _require(np.array_equal(solve(alternate), original["output"]), "answer depends on privileged references")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations, mutations)")
