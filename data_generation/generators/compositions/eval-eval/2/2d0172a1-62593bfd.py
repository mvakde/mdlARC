"""Companion reunion across membranes. All semantic decisions are input-only."""
from collections import Counter
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)


def _components(mask, directions=DIR8):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(int(y), int(x))], []
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for dy, dx in directions:
                yy, xx = a + dy, b + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    wall = np.pad(mask, 1)
    outside = np.zeros(wall.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = wall.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in DIR4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not wall[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                stack.append((yy, xx))
    return (~wall & ~outside)[1:-1, 1:-1]


def _mask_key(mask):
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _shape_key(mask, mode="rotation"):
    if mode == "area":
        return int(mask.sum())
    if mode == "identity":
        return _mask_key(mask)
    choices = [_mask_key(np.rot90(mask, k)) for k in range(4)]
    if mode == "reflection":
        choices += [_mask_key(np.rot90(mask[:, ::-1], k)) for k in range(4)]
    return min(choices)


def _order(node):
    y, x, yy, xx = node["box"]
    return x + xx, y + yy, y, x, yy, xx, node["color"]


def _parse(grid, background=None):
    grid = np.asarray(grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= d <= 30 for d in grid.shape)
    assert np.all((grid >= 0) & (grid <= 9))
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)]) if background is None else int(background)
    if background is None:
        assert counts.max() * 2 > grid.size, "background is not a strict majority"
    nodes = []
    labels = np.zeros(grid.shape, int)
    for color in values:
        if color == bg:
            continue
        for cells in _components(grid == color):
            lo, hi = cells.min(0), cells.max(0)
            y, x = map(int, lo)
            yy, xx = map(int, hi)
            local = np.zeros((yy-y+1, xx-x+1), bool)
            local[cells[:, 0]-y, cells[:, 1]-x] = True
            hole = _holes(local)
            assert len(_components(hole, DIR4)) <= 1, "multiple holes"
            full_hole = np.zeros(grid.shape, bool)
            full_hole[y:yy+1, x:xx+1] = hole
            node = {"kind": "C" if hole.any() else "B", "color": int(color),
                    "box": (y, x, yy, xx), "mask": local, "cells": cells,
                    "hole": full_hole, "parent": None, "children": []}
            labels[cells[:, 0], cells[:, 1]] = len(nodes)+1
            nodes.append(node)
    assert nodes
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b)), "components touch"
    for i, node in enumerate(nodes):
        owners = []
        for j, candidate in enumerate(nodes):
            if i == j or candidate["kind"] != "C":
                continue
            interior = candidate["hole"][node["cells"][:, 0], node["cells"][:, 1]]
            assert interior.all() or not interior.any(), "partial containment"
            if interior.all():
                owners.append(j)
        if owners:
            parent = min(owners, key=lambda j: int(nodes[j]["hole"].sum()))
            node["parent"] = parent
            nodes[parent]["children"].append(i)
    roots = [i for i, n in enumerate(nodes) if n["parent"] is None]
    assert len(roots) == 1 and nodes[roots[0]]["kind"] == "C", "one outer contour required"
    for node in nodes:
        assert bool(node["children"]) == (node["kind"] == "C"), "empty contour"
    contours = [n["color"] for n in nodes if n["kind"] == "C"]
    assert len(set(contours)) == 1, "input contours share a geometric drawing role"
    return {"nodes": nodes, "root": roots[0], "bg": bg}


def _ancestors(info, i):
    chain = []
    current = info["nodes"][i]["parent"]
    while current is not None:
        chain.append(current)
        current = info["nodes"][current]["parent"]
    return chain


def _assignment(info, mode="normal", sampled=False):
    nodes = info["nodes"]
    leaves = [i for i, n in enumerate(nodes) if n["kind"] == "B"]
    keys = {}
    for i in leaves:
        if sampled:
            keys[i] = nodes[i]["class"]
        elif mode == "color":
            keys[i] = nodes[i]["color"]
        else:
            keys[i] = _shape_key(nodes[i]["mask"], mode)
    classes = {}
    for i in leaves:
        classes.setdefault(keys[i], []).append(i)
    destinations, repeat = {}, set()
    for members in classes.values():
        partnered = len(members) >= 2 and (mode != "exactly_two" or len(members) == 2)
        if partnered:
            repeat.update(members)
            if mode == "outermost":
                owner = info["root"]
            else:
                common = set(_ancestors(info, members[0]))
                for i in members[1:]:
                    common.intersection_update(_ancestors(info, i))
                owner = next(a for a in _ancestors(info, members[0]) if a in common)
            for i in members:
                destinations[i] = nodes[i]["parent"] if mode == "no_relocation" else owner
        else:
            for i in members:
                destinations[i] = nodes[i]["parent"]
    kept = {info["root"]}
    for owner in destinations.values():
        while owner is not None:
            kept.add(owner)
            owner = nodes[owner]["parent"]
    if mode == "no_pruning":
        kept = {i for i, n in enumerate(nodes) if n["kind"] == "C"}
    return destinations, repeat, kept, classes


def _canonical(info, state):
    destinations, repeat, kept, _ = state
    nodes, bg = info["nodes"], info["bg"]

    def point_row(indices):
        indices = sorted(indices, key=lambda i: _order(nodes[i]))
        row = np.full((1, 2*len(indices)-1), bg, np.int8)
        row[0, ::2] = [nodes[i]["color"] for i in indices]
        return row

    def draw(region):
        upper = [i for i, p in destinations.items() if p == region and i in repeat]
        lower = [i for i, p in destinations.items() if p == region and i not in repeat]
        children = sorted([i for i in kept if nodes[i]["parent"] == region],
                          key=lambda i: _order(nodes[i]))
        strata = []
        if upper:
            strata.append(point_row(upper))
        if children:
            parts = [draw(i) for i in children]
            middle = np.full((max(p.shape[0] for p in parts),
                              sum(p.shape[1] for p in parts)+len(parts)-1), bg, np.int8)
            x = 0
            for p in parts:
                middle[:p.shape[0], x:x+p.shape[1]] = p
                x += p.shape[1]+1
            strata.append(middle)
        if lower:
            strata.append(point_row(lower))
        # Only a counterfactual no-pruning drawing can contain an empty frame.
        if not strata:
            strata = [np.full((1, 1), bg, np.int8)]
        h = sum(p.shape[0] for p in strata)+len(strata)-1
        w = max(p.shape[1] for p in strata)
        result = np.full((h+4, w+4), bg, np.int8)
        result[0, :] = result[-1, :] = nodes[region]["color"]
        result[:, 0] = result[:, -1] = nodes[region]["color"]
        y = 2
        for p in strata:
            x = 2+(w-p.shape[1])//2
            result[y:y+p.shape[0], x:x+p.shape[1]] = p
            y += p.shape[0]+1
        return result

    return draw(info["root"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct original shape classes and topology, then redraw the reunion."""
    info = _parse(np.asarray(input_grid))
    result = _canonical(info, _assignment(info))
    assert all(d <= 30 for d in result.shape), "canonical drawing exceeds ARC bounds"
    return result


def _crop(mask):
    y, x = np.nonzero(mask)
    return mask[y.min():y.max()+1, x.min():x.max()+1].copy()


def _sample_mask(rng, forbidden=(), chiral=False, asymmetric=False, compact=False, diagonal=False):
    for _ in range(300):
        kind = int(rng.integers(5))
        limit = 3 if compact else 4
        h, w = (int(rng.integers(1, limit+1)) for _ in range(2))
        if kind == 0:
            mask = np.ones((h, w), bool)
        elif kind == 1:
            mask = np.zeros((h, w), bool)
            mask[:, 0] = mask[-1, :] = True
        else:
            mask = np.zeros((h, w), bool)
            mask[int(rng.integers(h)), int(rng.integers(w))] = True
            directions = DIR4 if kind == 2 else DIR8
            target = int(rng.integers(2, min(10, h*w)+1)) if h*w > 1 else 1
            for _ in range(target-1):
                candidates = set()
                for y, x in zip(*np.nonzero(mask)):
                    for dy, dx in directions:
                        yy, xx = int(y)+dy, int(x)+dx
                        if 0 <= yy < h and 0 <= xx < w and not mask[yy, xx]:
                            candidates.add((yy, xx))
                if not candidates:
                    break
                options = sorted(candidates)
                mask[options[int(rng.integers(len(options)))]] = True
        mask = _crop(mask)
        key = _shape_key(mask)
        if key in forbidden or _holes(mask).any():
            continue
        if chiral and key == _shape_key(mask[:, ::-1]):
            continue
        if asymmetric and len({_mask_key(np.rot90(mask, k)) for k in range(4)}) < 4:
            continue
        if diagonal and len(_components(mask, DIR4)) == 1:
            continue
        return mask
    # Small deterministic, hole-free vocabulary for bounded sampling failure.
    for bits in range(1, 512):
        mask = _crop(np.array([(bits >> k) & 1 for k in range(9)], bool).reshape(3, 3))
        if len(_components(mask)) != 1 or _holes(mask).any():
            continue
        key = _shape_key(mask)
        if key in forbidden or (chiral and key == _shape_key(mask[:, ::-1])):
            continue
        if asymmetric and len({_mask_key(np.rot90(mask, k)) for k in range(4)}) < 4:
            continue
        if diagonal and len(_components(mask, DIR4)) == 1:
            continue
        return mask
    raise RuntimeError("mask vocabulary exhausted")


def _region(children):
    return {"kind": "C", "children": list(children)}


def _random_tree(rng, leaves):
    root = _region(leaves)
    regions = [(root, 1)]
    desired = int(rng.integers(1, 6))
    for _ in range(desired-1):
        # Move only immediate leaves. Wrapping an existing subtree would also
        # increase all its descendants' depths, not just the new node's depth.
        options = [(r, d) for r, d in regions if d < 3 and
                   any(c["kind"] == "B" for c in r["children"])]
        if not options:
            break
        parent, depth = options[int(rng.integers(len(options)))]
        eligible = [i for i, c in enumerate(parent["children"]) if c["kind"] == "B"]
        count = int(rng.integers(1, min(4, len(eligible))+1))
        positions = set(map(int, rng.choice(eligible, count, replace=False)))
        moved = [v for i, v in enumerate(parent["children"]) if i in positions]
        child = _region(moved)
        parent["children"] = [v for i, v in enumerate(parent["children"]) if i not in positions]+[child]
        regions.append((child, depth+1))
    return root


def _build_scene(rng, profile, palette, compact=False):
    bg, line, color_pool = palette
    if profile == "cross":
        sizes = [2]+[1]*int(rng.integers(1, 4))
    elif profile == "local":
        sizes = [3, 1, 1]+([1] if rng.random() < .3 else [])
    elif profile == "test":
        sizes = [int(rng.choice([2, 3])), 2]+[1]*int(rng.integers(0, 3))
        if sum(sizes) < 5:
            sizes.append(1)
    else:
        mode = int(rng.integers(5))
        sizes = ([1]*int(rng.integers(3, 7)) if mode == 0 else
                 [int(rng.choice([2, 3])) for _ in range(int(rng.integers(1, 3)))])
        if mode != 0:
            sizes += [1]*int(rng.integers(0, 4))
        while sum(sizes) < 3:
            sizes.append(1)
        while sum(sizes) > 8:
            sizes.pop()
    leaves, forbidden, bases = [], set(), []
    for cl, size in enumerate(sizes):
        if profile == "local" and cl == 1:
            base = bases[0][:, ::-1].copy()
        else:
            base = _sample_mask(rng, forbidden, chiral=profile == "local" and cl == 0,
                                asymmetric=cl == 0 and profile in ("cross", "local"), compact=compact,
                                diagonal=profile == "cross" and cl == 0)
        bases.append(base)
        forbidden.add(_shape_key(base))
        # Apart from the intended reflection witness, keep unrelated classes distinct.
        if profile != "local" or cl > 0:
            forbidden.add(_shape_key(base[:, ::-1]))
        turns = [int(rng.integers(4)) for _ in range(size)]
        if cl == 0 and profile in ("cross", "local"):
            turns[1] = (turns[0]+int(rng.choice([1, 3]))) % 4
        for turn in turns:
            leaves.append({"kind": "B", "children": [], "class": cl,
                           "mask": np.rot90(base, turn).copy(),
                           "color": int(rng.choice(color_pool))})
    if profile in ("cross", "local"):
        # Teaching colors contradict class-by-color and preserve point identity.
        leaves[0]["color"], leaves[1]["color"] = color_pool[:2]
        leaves[sizes[0]]["color"] = color_pool[0]
    if profile == "cross":
        empty_branch = _region([leaves[0]])
        anchor_branch = _region([leaves[1], leaves[2]])
        deeper = int(rng.integers(3))
        if deeper == 1:
            anchor_branch["children"][1] = _region([leaves[2]])
        elif deeper == 2:
            empty_branch = _region([empty_branch])
        root = _region([empty_branch, anchor_branch]+leaves[3:])
    elif profile == "local":
        # A local singleton makes the triplet's upper stratum observable;
        # without it, moving a lone row from 'upper' to 'lower' is invisible.
        root = _region([_region(leaves[:3]+[leaves[4]]), leaves[3]]+leaves[5:])
        if rng.random() < .35:
            root["children"][0] = _region([root["children"][0]])
    else:
        root = _random_tree(rng, leaves)

    def color_regions(n):
        if n["kind"] == "C":
            n["color"] = line
            rng.shuffle(n["children"])
            for child in n["children"]:
                color_regions(child)
    color_regions(root)
    return root


def _metric_group(parts, rng, compact):
    """Slicing groups are layout devices, not additional enclosing objects."""
    if len(parts) == 1:
        return parts[0]
    count = len(parts)
    # A recursive slicing layout allows rows, columns, and mixed 2D arrangements.
    cut = int(rng.integers(1, count))
    a = _metric_group(parts[:cut], rng, compact)
    b = _metric_group(parts[cut:], rng, compact)
    axis = int(rng.integers(2))
    # On compact retries prefer the composition with the smaller maximum extent.
    if compact and rng.random() < .85:
        scores = [max(a[0].shape[k]+b[0].shape[k]+1,
                      max(a[0].shape[1-k], b[0].shape[1-k])) for k in (0, 1)]
        axis = int(np.argmin(scores))
    cross = 1-axis
    gap = int(rng.integers(1, 3 if compact else 5))
    dims = [0, 0]
    dims[axis] = a[0].shape[axis]+b[0].shape[axis]+gap
    dims[cross] = max(a[0].shape[cross], b[0].shape[cross])
    canvas = np.full(dims, a[2], np.int8)
    records = []
    start = 0
    for grid, entries, bg in (a, b):
        pos = [0, 0]
        pos[axis] = start
        # Input alignments do not determine strata; offsets remain metric variation.
        spare = dims[cross]-grid.shape[cross]
        pos[cross] = int(rng.integers(spare+1))
        y, x = pos
        canvas[y:y+grid.shape[0], x:x+grid.shape[1]] = grid
        for node, (yy, xx, bottom, right) in entries:
            records.append((node, (yy+y, xx+x, bottom+y, right+x)))
        start += grid.shape[axis]+gap
    return canvas, records, a[2]


def _outline(h, w, pads, rng, compact):
    if rng.random() < .16:
        rectangle = np.zeros((h, w), bool)
        rectangle[0, :] = rectangle[-1, :] = True
        rectangle[:, 0] = rectangle[:, -1] = True
        return rectangle
    inside = np.ones((h, w), bool)
    caps = [max(0, p-2) for p in pads]
    # All deformations lie outside the children's bounding rectangle and moat.
    for corner in range(4):
        if rng.random() < .65:
            py = min(caps[corner // 2], int(rng.integers(1, 3)))
            px = min(caps[2+corner % 2], int(rng.integers(1, 3)))
            ys = slice(0, py) if corner < 2 else slice(h-py, h)
            xs = slice(0, px) if corner % 2 == 0 else slice(w-px, w)
            if py and px:
                inside[ys, xs] = False
    for side, length in enumerate((w, w, h, h)):
        if caps[side] and length >= 7 and rng.random() < .35:
            depth = int(rng.integers(1, caps[side]+1))
            lo = int(rng.integers(2, length-3))
            hi = min(length-2, lo+int(rng.integers(2, 5)))
            if side == 0:
                inside[:depth, lo:hi] = False
            elif side == 1:
                inside[h-depth:, lo:hi] = False
            elif side == 2:
                inside[lo:hi, :depth] = False
            else:
                inside[lo:hi, w-depth:] = False
    padded = np.pad(inside, 1)
    eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
              padded[1:-1, :-2] & padded[1:-1, 2:])
    return inside & ~eroded


def _render(node, bg, rng, compact):
    if node["kind"] == "B":
        mask = node["mask"]
        grid = np.where(mask, node["color"], bg).astype(np.int8)
        return grid, [(node, (0, 0, grid.shape[0]-1, grid.shape[1]-1))], bg
    parts = [_render(child, bg, rng, compact) for child in node["children"]]
    content, entries, _ = _metric_group(parts, rng, compact)
    pads = ([3]*4 if compact else [int(rng.integers(3, 7)) for _ in range(4)])
    top, bottom, left, right = pads
    h, w = content.shape[0]+top+bottom, content.shape[1]+left+right
    outline = _outline(h, w, pads, rng, compact)
    grid = np.full((h, w), bg, np.int8)
    grid[outline] = node["color"]
    grid[top:top+content.shape[0], left:left+content.shape[1]] = content
    records = [(node, (0, 0, h-1, w-1))]
    for obj, (y, x, yy, xx) in entries:
        records.append((obj, (y+top, x+left, yy+top, xx+left)))
    return grid, records, bg


def _model_info(root, records, bg):
    nodes = []

    def visit(n, parent):
        i = len(nodes)
        nodes.append({k: v for k, v in n.items() if k != "children"})
        nodes[i].update(parent=parent, children=[])
        for child in n["children"]:
            nodes[i]["children"].append(visit(child, i))
        return i

    visit(root, None)
    boxes = {id(n): box for n, box in records}

    def attach(n, i):
        nodes[i]["box"] = boxes[id(n)]
        for child, index in zip(n["children"], nodes[i]["children"]):
            attach(child, index)
    attach(root, 0)
    return {"nodes": nodes, "root": 0, "bg": bg}


def _palette(rng, stable):
    if stable is not None:
        return stable
    colors = list(map(int, rng.permutation(10)))
    # At least two leaf colors for the required teaching counterexamples.
    return colors[0], colors[1], colors[1:2+int(rng.integers(2, 6))]


def _pair(rng, profile, stable):
    palette = _palette(rng, stable)
    bg = palette[0]
    for attempt in range(240):
        compact = attempt >= 8
        root = _build_scene(rng, profile, palette, compact)
        core, records, _ = _render(root, bg, rng, compact)
        if max(core.shape) > 30:
            continue
        model = _model_info(root, records, bg)
        state = _assignment(model, sampled=True)
        # Tests recombine companionship at two distinct hierarchical destinations.
        if profile == "test":
            if len({state[0][i] for i in state[1]}) < 2:
                continue
            if not any(state[0][i] != model["nodes"][i]["parent"] for i in state[1]):
                continue
        expected = _canonical(model, state)
        if max(expected.shape) > 30:
            continue
        margins = [int(rng.integers(0, min(4, 30-d)+1)) for d in core.shape]
        h, w = core.shape[0]+margins[0], core.shape[1]+margins[1]
        if rng.random() < .08:
            h = w = 30
        y, x = int(rng.integers(h-core.shape[0]+1)), int(rng.integers(w-core.shape[1]+1))
        grid = np.full((h, w), bg, np.int8)
        grid[y:y+core.shape[0], x:x+core.shape[1]] = core
        if np.count_nonzero(grid == bg)*2 <= grid.size:
            continue
        try:
            actual = solve(grid)
            assert np.array_equal(actual, expected)
            if profile in ("cross", "local"):
                info = _parse(grid)
                # Each witness must reject its relevant counterfactuals visibly.
                checks = (("no_relocation", "no_pruning") if profile == "cross"
                          else ("outermost", "reflection", "exactly_two", "area"))
                assert all(not np.array_equal(expected, _canonical(info, _assignment(info, mode)))
                           for mode in checks)
                assert not np.array_equal(expected, _canonical(info, _assignment(info, "identity")))
                assert not np.array_equal(expected, _canonical(info, _assignment(info, "color")))
        except AssertionError:
            continue
        return {"input": grid, "output": expected}
    return _fallback_pair(rng, profile, palette)


def _fallback_pair(rng, profile, palette):
    """A proven-fit procedural scene if the geometric rejection budget expires.

    This is not a source fixture. Its small masks, two-level ownership, palette,
    ordering, and canvas placement still obey the same reunion rule.
    """
    bg, line, pool = palette
    diagonal = np.array([[1, 0], [1, 0], [0, 1]], bool)
    elbow = np.array([[1, 0], [1, 0], [1, 1]], bool)
    if profile == "random":
        profile = str(rng.choice(["cross", "local"]))

    def leaf(mask, cl, color):
        return {"kind": "B", "children": [], "mask": mask.copy(), "class": cl, "color": color}

    if profile == "cross":
        leaves = [leaf(diagonal, 0, pool[0]), leaf(np.rot90(diagonal), 0, pool[1]),
                  leaf(np.ones((1, 3), bool), 1, pool[0])]
        root = _region([_region([leaves[0]]), _region(leaves[1:])])
    elif profile == "local":
        leaves = [leaf(np.rot90(elbow, k), 0, pool[k % len(pool)]) for k in range(3)]
        leaves += [leaf(elbow[:, ::-1], 1, pool[0]), leaf(np.ones((1, 3), bool), 2, pool[1])]
        root = _region([_region(leaves[:3]+[leaves[4]]), leaves[3]])
    else:
        leaves = [leaf(elbow, 0, pool[0]), leaf(np.rot90(elbow), 0, pool[1]),
                  leaf(np.ones((1, 2), bool), 1, pool[0]),
                  leaf(np.ones((2, 1), bool), 1, pool[1]),
                  leaf(np.ones((1, 1), bool), 2, pool[0])]
        root = _region([_region(leaves[:3]), leaves[3], leaves[4]])

    def draw(n):
        if n["kind"] == "B":
            grid = np.where(n["mask"], n["color"], bg).astype(np.int8)
            return grid, [(n, (0, 0, grid.shape[0]-1, grid.shape[1]-1))]
        n["color"] = line
        children = n["children"]
        parts = [draw(c) for c in children]
        # At most four children: two compact rows guarantee a fit here.
        rows = [parts[i:i+2] for i in range(0, len(parts), 2)]
        heights = [max(p[0].shape[0] for p in row) for row in rows]
        widths = [sum(p[0].shape[1] for p in row)+len(row)-1 for row in rows]
        h, w = sum(heights)+len(rows)-1+6, max(widths)+6
        grid = np.full((h, w), bg, np.int8)
        grid[0, :] = grid[-1, :] = line
        grid[:, 0] = grid[:, -1] = line
        entries, y = [(n, (0, 0, h-1, w-1))], 3
        for row, height in zip(rows, heights):
            x = 3
            for part, records in row:
                grid[y:y+part.shape[0], x:x+part.shape[1]] = part
                for obj, (a, b, aa, bb) in records:
                    entries.append((obj, (a+y, b+x, aa+y, bb+x)))
                x += part.shape[1]+1
            y += height+1
        return grid, entries

    core, records = draw(root)
    model = _model_info(root, records, bg)
    expected = _canonical(model, _assignment(model, sampled=True))
    h, w = (int(rng.integers(d, 31)) for d in core.shape)
    y, x = int(rng.integers(h-core.shape[0]+1)), int(rng.integers(w-core.shape[1]+1))
    grid = np.full((h, w), bg, np.int8)
    grid[y:y+core.shape[0], x:x+core.shape[1]] = core
    assert np.array_equal(solve(grid), expected)
    return {"input": grid, "output": expected}


def _counts(rng, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                 not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(name+" must be an integer in its legal range")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1 if num_train is not None else 4)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    grid = np.asarray(grid)
    return tuple(grid.shape), grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    tr, te = _counts(rng, num_train, num_test)
    stable = _palette(rng, None) if rng.random() < .25 else None
    puzzle, seen = {"train": [], "test": []}, set()
    for split, profiles in (("train", ["cross", "local"]+["random"]*(tr-2)),
                            ("test", ["test"]*te)):
        for profile in profiles:
            for _ in range(24):
                pair = _pair(rng, profile, stable)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input search exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def _tree_signature(info, kept=None, dest=None, repeat=None):
    nodes = info["nodes"]
    if kept is None:
        def original(i):
            n = nodes[i]
            return ("B", n["color"]) if n["kind"] == "B" else (
                "C", n["color"], tuple(sorted(original(j) for j in n["children"])))
        return original(info["root"])

    def relocated(i):
        children = [("B", nodes[j]["color"]) for j, p in dest.items() if p == i]
        children += [relocated(j) for j in kept if nodes[j]["parent"] == i]
        return "C", nodes[i]["color"], tuple(sorted(children))
    return relocated(info["root"])


def _features(pair):
    info = _parse(np.asarray(pair["input"]))
    dest, repeat, kept, classes = _assignment(info)
    nodes = info["nodes"]
    contours = [i for i, n in enumerate(nodes) if n["kind"] == "C"]
    leaves = list(dest)
    depth = max(len(_ancestors(info, i)) for i in leaves)
    return {"shape": tuple(np.asarray(pair["input"]).shape), "bg": info["bg"],
            "blobs": len(leaves), "contours": len(contours), "depth": depth,
            "pruned": len(contours)-len(kept), "lifted": sum(dest[i] != nodes[i]["parent"] for i in leaves),
            "local_repeat": any(dest[i] == nodes[i]["parent"] for i in repeat),
            "reunion_regions": len({dest[i] for i in repeat}),
            "multiplicities": tuple(sorted(len(m) for m in classes.values())),
            "diagonal": sum(len(_components(nodes[i]["mask"], DIR4)) > 1 for i in leaves),
            "nonrect": sum(not (nodes[i]["mask"][0].all() and nodes[i]["mask"][-1].all() and
                               nodes[i]["mask"][:, 0].all() and nodes[i]["mask"][:, -1].all()) for i in contours),
            "areas": tuple(sorted(int(nodes[i]["mask"].sum()) for i in leaves)),
            "structure": _tree_signature(info),
            "output_structure": _tree_signature(info, kept, dest, repeat)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    modes = ("no_relocation", "outermost", "identity", "reflection", "area", "color", "exactly_two", "no_pruning")
    rejected = {mode: False for mode in modes}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for g in (inp, out):
                assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
                assert all(1 <= d <= 30 for d in g.shape) and np.all((g >= 0) & (g <= 9))
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            dest, repeat, kept, classes = _assignment(info)
            nodes = info["nodes"]
            f = _features(pair)
            assert 3 <= f["blobs"] <= 8 and 1 <= f["contours"] <= 5 and 1 <= f["depth"] <= 3
            assert all(max(n["mask"].shape) <= 4 for n in nodes if n["kind"] == "B")
            assert np.array_equal(out, solve(inp))
            # Check the returned diagram itself, including topology and conserved colors.
            output_info = _parse(out, info["bg"])
            assert output_info["bg"] == info["bg"]
            assert _tree_signature(output_info) == _tree_signature(info, kept, dest, repeat)
            original_colors = Counter(n["color"] for n in nodes if n["kind"] == "B")
            output_colors = Counter(n["color"] for n in output_info["nodes"] if n["kind"] == "B")
            assert original_colors == output_colors
            onodes = output_info["nodes"]
            outer = onodes[output_info["root"]]
            assert outer["box"] == (0, 0, out.shape[0]-1, out.shape[1]-1)
            for n in onodes:
                if n["kind"] == "B":
                    assert n["mask"].shape == (1, 1), "blobs must become single points"
                else:
                    m = n["mask"]
                    assert m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all()
                    assert int(m.sum()) == 2*sum(m.shape)-4, "one-cell rectangular perimeter"
                    child_boxes = np.array([onodes[i]["box"] for i in n["children"]])
                    y, x, yy, xx = n["box"]
                    assert tuple(child_boxes[:, :2].min(0)) == (y+2, x+2)
                    assert tuple(child_boxes[:, 2:].max(0)) == (yy-2, xx-2)
            # A destination must contain every member and no deeper region can do so.
            for members in classes.values():
                if len(members) < 2:
                    continue
                owner = dest[members[0]]
                assert all(owner in _ancestors(info, i) and dest[i] == owner for i in members)
                common = set.intersection(*(set(_ancestors(info, i)) for i in members))
                assert owner == max(common, key=lambda j: len(_ancestors(info, j)))
            if split == "train":
                for mode in modes:
                    rejected[mode] |= not np.array_equal(out, _canonical(info, _assignment(info, mode)))
            else:
                assert f["reunion_regions"] >= 2, "test needs simultaneous local and wider reunions"
                assert f["lifted"] > 0, "test must change containment"
                assert max(f["multiplicities"]) <= 3
            features[split].append(f)
    assert all(rejected.values()), "training leaves a shortcut unchallenged: "+str(rejected)
    assert any(f["pruned"] > 0 and f["lifted"] > 0 for f in features["train"])
    assert any(3 in f["multiplicities"] and f["local_repeat"] for f in features["train"])
    if any(f["diagonal"] for f in features["test"]):
        assert any(f["diagonal"] and f["lifted"] for f in features["train"])


def _puzzle_key(puzzle):
    return tuple((s, tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s]))
                 for s in ("train", "test"))


def _uncolored_tree(signature):
    return (("B",) if signature[0] == "B" else
            ("C", tuple(sorted(_uncolored_tree(c) for c in signature[2]))))


def self_test():
    """Bounded structural, semantic-invariant, metamorphic, and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for p in puzzles:
        validate_generated_puzzle(p)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    fs = [_features(pair) for p in puzzles for s in p for pair in p[s]]
    assert len({f["shape"] for f in fs}) >= 25
    assert len({_uncolored_tree(f["structure"]) for f in fs}) >= 30
    assert len({_uncolored_tree(f["output_structure"]) for f in fs}) >= 25
    masks, layouts = set(), set()
    for p in puzzles:
        for split in p:
            for pair in p[split]:
                info = _parse(pair["input"])
                masks.update(_shape_key(n["mask"]) for n in info["nodes"] if n["kind"] == "B")
                layouts.add(_grid_key((pair["output"] != info["bg"]).astype(np.int8)))
    assert len(masks) >= 40 and len(layouts) >= 40
    # Exercise the finite-search exit path even though it is rarely sampled.
    fallback_rng = np.random.default_rng(1932)
    fallback_palette = (0, 1, [1, 2, 3])
    fallback = {"train": [_fallback_pair(fallback_rng, mode, fallback_palette)
                          for mode in ("cross", "local")],
                "test": [_fallback_pair(fallback_rng, "test", fallback_palette)]}
    validate_generated_puzzle(fallback)
    assert {f["bg"] for f in fs} == set(range(10))
    assert {1, 2, 3} <= {f["depth"] for f in fs}
    assert {3, 4, 5, 6, 7} <= {f["blobs"] for f in fs}
    assert any(f["diagonal"] for f in fs) and any(f["nonrect"] for f in fs)
    assert len({f["multiplicities"] for f in fs}) >= 8
    assert len({f["pruned"] for f in fs}) >= 3
    # All explicit legal count combinations, not only boundary cases.
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(2100+10*tr+te, tr, te)
            assert (len(p["train"]), len(p["test"])) == (tr, te)
            validate_generated_puzzle(p)
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        p = generate_puzzle(2137, tr, te)
        validate_generated_puzzle(p)
        assert tr is None or len(p["train"]) == tr
        assert te is None or len(p["test"]) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), ("4", 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    rng = np.random.default_rng(784)
    for p in puzzles[:12]:
        pair = p["test"][0]
        inp, out = pair["input"], pair["output"]
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        bg = _parse(inp)["bg"]
        if max(inp.shape) <= 28:
            assert np.array_equal(solve(np.pad(inp, ((1, 1), (1, 1)), constant_values=bg)), out)
        # Replace only the outer contour by its rectangular bounding perimeter.
        # All original child positions, shapes, ownership, and order remain fixed.
        info = _parse(inp)
        outer = info["nodes"][info["root"]]
        altered = inp.copy()
        altered[outer["cells"][:, 0], outer["cells"][:, 1]] = bg
        y, x, yy, xx = outer["box"]
        altered[y, x:xx+1] = altered[yy, x:xx+1] = outer["color"]
        altered[y:yy+1, x] = altered[y:yy+1, xx] = outer["color"]
        assert np.array_equal(solve(altered), out)
    bad = {s: [{k: np.asarray(g).copy() for k, g in pair.items()} for pair in puzzles[0][s]]
           for s in puzzles[0]}
    out = bad["test"][0]["output"]
    out[0, 1] = _parse(bad["test"][0]["input"])["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("opened output frame accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
