"""A repaired track tree is a spatial program of relative-turn glyphs.

Only standard-library modules and NumPy are used. Generation renders labels
from its constructed tree; solve instead reconstructs and walks the input graph.
"""
from collections import deque
import copy
import json

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_MASKS = {
    "L": np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
    "R": np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [q for d in _DIRS if (q := _add(p, d)) in cells]


def _turn(d, command):
    return (-d[1], d[0]) if command == "L" else (d[1], -d[0])


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc) for i in range(n + 1)]


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    directions = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if (r, c) != (0, 0)] if diagonal else _DIRS)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, todo = {start}, [start]
        while todo:
            p = todo.pop()
            for d in directions:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _read(input_grid):
    """Identify roles by geometry/multiplicity, then reconstruct the track tree.

    This function has no access to the RNG, construction tree, or output.
    """
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > inp.size / 2, "background must occupy a strict majority"
    cells_by_color = {
        int(c): {tuple(map(int, p)) for p in np.argwhere(inp == c)}
        for c in np.unique(inp) if c != bg
    }
    glyphs, payloads = {}, set()
    for color, cells in cells_by_color.items():
        found = []
        for component in _components(cells, diagonal=True):
            top = min(r for r, c in component)
            left = min(c for r, c in component)
            bottom = max(r for r, c in component)
            right = max(c for r, c in component)
            if (bottom - top, right - left) != (2, 2):
                break
            occupancy = np.array([[(top + r, left + c) in component
                                   for c in range(3)] for r in range(3)])
            matches = [name for name, mask in _MASKS.items()
                       if np.array_equal(occupancy, mask)]
            if len(matches) != 1:
                break
            found.append(((top + 1, left + 1), matches[0], color))
        else:
            payloads.add(color)
            for center, command, color in found:
                glyphs[center] = (command, color)
    assert glyphs, "no turn glyphs"
    rest = {c: s for c, s in cells_by_color.items() if c not in payloads}
    seeds = [c for c, s in rest.items() if len(s) == 1]
    repairs = [c for c, s in rest.items() if len(s) >= 2
               and all(len(component) == 1 for component in _components(s, True))]
    assert len(seeds) == len(repairs) == 1, "annotation roles are ambiguous"
    seed_color, repair = seeds[0], repairs[0]
    tracks = set(rest) - {seed_color, repair}
    assert len(tracks) == 1, "one neutral track color is required"
    track = next(iter(tracks))
    start = next(iter(rest[seed_color]))
    blocks = set()
    for r, c in glyphs:
        block = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        assert all(_inside(p, inp.shape) for p in block)
        assert not blocks & block
        blocks.update(block)
    cells = set(rest[track]) | set(rest[repair]) | {start}
    assert not cells & blocks
    # Repair annotations are occupied waypoints, including bends/endpoints.
    # Unmarked damage is one missing cell with opposite occupied witnesses.
    original = set(cells)
    gaps = set()
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            p = (r, c)
            if inp[p] != bg or p in blocks:
                continue
            for d in ((1, 0), (0, 1)):
                if _add(p, d) in original and _add(p, (-d[0], -d[1])) in original:
                    gaps.add(p)
    cells.update(gaps)
    # Rebuild each symbol's hidden junction from its three external ports.
    for center in glyphs:
        ports = [d for d in _DIRS
                 if (center[0] + 2 * d[0], center[1] + 2 * d[1]) in cells]
        assert len(ports) == 3, "a glyph must occupy a three-port fork"
        cells.add(center)
        cells.update(_add(center, d) for d in ports)
    assert len(_components(cells)) == 1, "repaired network is disconnected"
    degrees = {p: len(_neighbors(p, cells)) for p in cells}
    assert degrees[start] == 1, "seed must be a terminal endpoint"
    assert sum(degrees.values()) == 2 * (len(cells) - 1), "network must be a tree"
    assert {p for p, n in degrees.items() if n == 3} == set(glyphs)
    assert all(1 <= n <= 3 for n in degrees.values())
    parent, queue = {start: None}, deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, cells):
            if q not in parent:
                parent[q] = p
                queue.append(q)
    for p in glyphs:
        previous = parent[p]
        d = p[0] - previous[0], p[1] - previous[1]
        children = set(_neighbors(p, cells)) - {previous}
        assert children == {_add(p, _turn(d, "L")), _add(p, _turn(d, "R"))}, \
            "outgoing fork arms must be left and right relative to arrival"
    mask = np.zeros(inp.shape, dtype=bool)
    for p in cells:
        mask[p] = True
    assert not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
    return {"input": inp, "background": bg, "track": track, "repair": repair,
            "seed_color": seed_color, "start": start, "glyphs": glyphs,
            "cells": cells, "gaps": gaps, "markers": set(rest[repair]),
            "parent": parent, "blocks": blocks}


def _walk(info, override=None):
    """Execute the spatial program, optionally replacing all commands to audit it."""
    cells, glyphs = info["cells"], info["glyphs"]
    previous, current, color = None, info["start"], info["track"]
    painted, visited = {}, []
    while True:
        if current == info["start"]:
            painted[current] = info["seed_color"]
        else:
            if current in glyphs:
                command, color = glyphs[current]
                visited.append(current)
            painted[current] = color
        following = [q for q in _neighbors(current, cells) if q != previous]
        if not following:
            break
        if current in glyphs:
            d = current[0] - previous[0], current[1] - previous[1]
            choice = _add(current, _turn(d, override or glyphs[current][0]))
            assert choice in following
        else:
            assert len(following) == 1
            choice = following[0]
        previous, current = current, choice
        assert len(painted) <= len(cells)
    return painted, visited


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair the input-only tree, obey its relative turns, prune, and recolor."""
    info = _read(input_grid)
    painted, _ = _walk(info)
    out = np.full(info["input"].shape, info["background"], dtype=np.int8)
    for p, color in painted.items():
        out[p] = color
    return out


def _child_rect(rect, center, d):
    top, bottom, left, right = rect
    r, c = center
    if d == (-1, 0):
        bottom = r - 3
    elif d == (1, 0):
        top = r + 3
    elif d == (0, -1):
        right = c - 3
    else:
        left = c + 3
    return top, bottom, left, right


def _node_options(rect, entry, d):
    top, bottom, left, right = rect
    if d[0]:
        if not left + 3 <= entry[1] <= right - 3:
            return []
        return [(r, entry[1]) for r in range(top + 3, bottom - 2)
                if (r - entry[0]) * d[0] >= 4]
    if not top + 3 <= entry[0] <= bottom - 3:
        return []
    return [(entry[0], c) for c in range(left + 3, right - 2)
            if (c - entry[1]) * d[1] >= 4]


def _leaf(rng, rect, entry, d):
    top, bottom, left, right = rect
    options = ([(r, entry[1]) for r in range(top, bottom + 1)
                if (r - entry[0]) * d[0] >= 3] if d[0] else
               [(entry[0], c) for c in range(left, right + 1)
                if (c - entry[1]) * d[1] >= 3])
    assert options
    end = options[int(rng.integers(len(options)))]
    path = _line(entry, end)
    if rng.random() < .65:
        bends = ([(end[0], c) for c in range(left, right + 1)
                  if abs(c - end[1]) >= 3] if d[0] else
                 [(r, end[1]) for r in range(top, bottom + 1)
                  if abs(r - end[0]) >= 3])
        if bends:
            bend_end = bends[int(rng.integers(len(bends)))]
            path.extend(_line(end, bend_end)[1:])
    return {"edge": path, "node": None}


def _build_tree(rng, shape, active_pattern, full_witness=False):
    h, w = shape
    depth = len(active_pattern)
    col = int(rng.integers(9, w - 9))
    start = (0, col)
    nodes = {}

    def grow(rect, entry, d, level, active, remaining, force=False):
        options = _node_options(rect, entry, d)
        if not options or remaining <= 0:
            assert not active or level >= depth
            return _leaf(rng, rect, entry, d)
        if not active and not force and rng.random() < .45:
            return _leaf(rng, rect, entry, d)
        if level == 0:
            # A deeper program needs room for grandchildren above AND below.
            margin = 9 if depth == 3 else 4
            options = [p for p in options if margin <= p[0] <= h - 1 - margin]
        assert options
        center = options[int(rng.integers(len(options)))]
        command = active_pattern[level] if active else str(rng.choice(["L", "R"]))
        node = {"center": center, "command": command, "children": {}}
        nodes[center] = node
        for cmd in ("L", "R"):
            child_d = _turn(d, cmd)
            child_region = _child_rect(rect, center, child_d)
            child_active = active and command == cmd
            child_force = level == 0 and full_witness
            child_remaining = remaining - 1
            if child_active and level + 1 >= depth:
                child_remaining = 0
            node["children"][cmd] = grow(child_region, center, child_d, level + 1,
                                          child_active, child_remaining, child_force)
        return {"edge": _line(entry, center), "node": node}

    root = grow((0, h - 1, 0, w - 1), start, (1, 0), 0, True, depth, True)
    skeleton = set()

    def collect(branch):
        skeleton.update(branch["edge"])
        if branch["node"] is not None:
            for child in branch["node"]["children"].values():
                collect(child)

    collect(root)
    # The label is rendered by following construction edges, not _read/_walk.
    active_centers = []
    branch = root
    while branch["node"] is not None:
        node = branch["node"]
        active_centers.append(node["center"])
        branch = node["children"][node["command"]]
    return start, root, nodes, skeleton, active_centers


def _rotate(p, shape, k):
    r, c = p
    h, w = shape
    for _ in range(k):
        r, c, h, w = w - 1 - c, r, w, h
    return r, c


def _make_pair(rng, roles, payloads, pattern, rotation, witness=False,
               occurrence_colors=None):
    bg, track, repair, seed_color = roles
    depth = len(pattern)
    h = int(rng.integers(19 if depth == 3 else 15, 31 if depth == 3 else 27))
    w = int(rng.integers(19, 31))
    shape = h, w
    start, root, nodes, skeleton, active_centers = _build_tree(
        rng, shape, pattern, full_witness=witness)
    for node in nodes.values():
        node["color"] = int(rng.choice(payloads))
    if occurrence_colors is not None:
        for p, color in zip(active_centers, occurrence_colors):
            nodes[p]["color"] = color
    painted = {start: seed_color}
    active = set(root["edge"])
    color, branch = track, root
    while True:
        for p in branch["edge"][1:]:
            painted[p] = color
        node = branch["node"]
        if node is None:
            break
        color = node["color"]
        painted[node["center"]] = color
        branch = node["children"][node["command"]]
        active.update(branch["edge"])
    out_shape = shape if rotation % 2 == 0 else shape[::-1]
    transform = lambda p: _rotate(p, shape, rotation)
    skeleton = {transform(p) for p in skeleton}
    active = {transform(p) for p in active}
    start = transform(start)
    glyphs = {transform(p): (n["command"], n["color"]) for p, n in nodes.items()}
    out = np.full(out_shape, bg, dtype=np.int8)
    for p, color in painted.items():
        out[transform(p)] = color
    inp = np.full(out_shape, bg, dtype=np.int8)
    for p in skeleton:
        inp[p] = track
    blocks = set()
    protected = {start}
    for (r, c), (cmd, color) in glyphs.items():
        inp[r - 1:r + 2, c - 1:c + 2] = bg
        inp[r - 1:r + 2, c - 1:c + 2][_MASKS[cmd]] = color
        blocks.update((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        # External ports must be intact so that junction arity is visible.
        protected.update((r + 2 * dr, c + 2 * dc) for dr, dc in _DIRS)
    inp[start] = seed_color
    eligible = sorted(skeleton - blocks - protected)
    rng.shuffle(eligible)
    straight = [p for p in eligible if len(ns := _neighbors(p, skeleton)) == 2
                and ns[0][0] + ns[1][0] == 2 * p[0]
                and ns[0][1] + ns[1][1] == 2 * p[1]]
    # Every pair has a selected, unmarked gap AFTER an executed instruction.
    selected_gaps = [p for p in straight if p in active and out[p] in payloads]
    if not selected_gaps:
        return None
    gap = selected_gaps[int(rng.integers(len(selected_gaps)))]
    damage = [gap]
    inp[gap] = bg

    def separated(p):
        return all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 3 for q in damage)

    # A kept and a discarded annotation distinguish repair from mere recoloring.
    for selected in (True, False):
        candidates = [p for p in eligible if (p in active) == selected and separated(p)
                      and (not selected or out[p] in payloads)]
        if not candidates:
            return None
        # Prefer bend and terminal sites sometimes, rather than only holes.
        special = [p for p in candidates if p not in straight]
        if special and rng.random() < .65:
            candidates = special
        p = candidates[int(rng.integers(len(candidates)))]
        inp[p] = repair
        damage.append(p)
    for p in eligible:
        if len(damage) >= 7:
            break
        if separated(p) and rng.random() < .12:
            inp[p] = repair if rng.random() < .7 or p not in straight else bg
            damage.append(p)
    pair = {"input": inp, "output": out}
    try:
        info = _read(inp)
        assert info["cells"] == skeleton
        assert np.array_equal(solve(inp), out)
    except (AssertionError, ValueError):
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    roles = tuple(palette[:4])
    payloads = tuple(palette[4:4 + int(rng.integers(3, 5))])
    rotation = int(rng.integers(4))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                pattern = ("L", "R") if index == 0 else ("R", "L")
                k = (rotation + 2 * index) % 4
                # One example exposes a color transition; the other shares a
                # new ink color across opposite commands. Neither depth nor
                # literal color can encode the intended output behavior.
                colors = payloads[:2] if index == 0 else (payloads[2], payloads[2])
                witness = True
            else:
                depth = 3 if split == "test" else int(rng.integers(1, 3))
                pattern = tuple(str(c) for c in rng.choice(["L", "R"], depth))
                k = int(rng.integers(4))
                colors, witness = None, split == "test"
            for _ in range(64):
                pair = _make_pair(rng, roles, payloads, pattern, k, witness, colors)
                if pair is None:
                    continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded tree construction exhausted")
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, lengths = set(), set(), {"train": [], "test": []}
    training_colors = {"L": set(), "R": set()}
    incoming_directions = {"L": set(), "R": set()}
    right_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen
            seen.add(key)
            info = _read(inp)
            conventions.add(tuple(info[k] for k in ("background", "track", "repair", "seed_color")))
            painted, visited = _walk(info)
            lengths[split].append(len(visited))
            assert 1 <= len(info["glyphs"]) <= 7
            assert np.array_equal(solve(inp), out), "full output violates spatial program"
            selected = set(painted)
            # Family-specific evidence, checked from grids rather than sampled flags.
            assert selected & info["gaps"], "selected route must need gap repair"
            assert selected & info["markers"], "a retained marker must be repaired"
            payloads = {color for cmd, color in info["glyphs"].values()}
            assert any(out[p] in payloads for p in selected & info["gaps"])
            assert any(out[p] in payloads for p in selected & info["markers"])
            assert info["markers"] - selected, "a discarded marker must disappear"
            assert info["cells"] - selected, "an unchosen branch must be pruned"
            assert all(out[p] == info["background"] for p in info["cells"] - selected)
            assert out[info["start"]] == info["seed_color"]
            if split == "train":
                for p in visited:
                    cmd, color = info["glyphs"][p]
                    training_colors[cmd].add(color)
                    q = info["parent"][p]
                    incoming_directions[cmd].add((p[0] - q[0], p[1] - q[1]))
                alternate, _ = _walk(info, "L")
                right_witness |= alternate != painted
    assert len(conventions) == 1, "role palette changes within a puzzle"
    assert all(len(v) >= 2 for v in training_colors.values()), "glyph meaning is confounded with color"
    assert training_colors["L"] & training_colors["R"], "shared glyph colors must be taught"
    assert all(len(v) >= 2 for v in incoming_directions.values()), "relative turns need varied arrivals"
    assert right_witness, "ignoring glyph commands must change a demonstrated answer"
    # Rule out assigning a fixed color solely by execution depth.
    colors_at_depth = {}
    transition_witness = False
    for pair in puzzle["train"]:
        info = _read(pair["input"])
        _, visited = _walk(info)
        colors = [info["glyphs"][p][1] for p in visited]
        transition_witness |= any(a != b for a, b in zip(colors, colors[1:]))
        for depth, color in enumerate(colors):
            colors_at_depth.setdefault(depth, set()).add(color)
    assert transition_witness and all(len(colors_at_depth[d]) >= 2 for d in (0, 1))
    assert min(lengths["test"]) > max(lengths["train"]), "longer spatial program shift missing"


def _puzzle_key(puzzle):
    return tuple((split, tuple((np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(),
                               np.asarray(p["output"]).tobytes()) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test():
    """Bounded semantic regressions, mutations, and actual 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    shapes, structures, roles, node_counts, bends, headings = set(), set(), set(), set(), 0, set()
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            info = _read(pair["input"])
            shapes.add(info["input"].shape)
            roles.add(tuple(info[k] for k in ("background", "track", "repair", "seed_color")))
            node_counts.add(len(info["glyphs"]))
            shapes_and_cells = (info["input"].shape, tuple(sorted(info["cells"])),
                                tuple(sorted((p, cmd) for p, (cmd, color) in info["glyphs"].items())))
            structures.add(shapes_and_cells)
            for p in info["cells"] - set(info["glyphs"]):
                ns = _neighbors(p, info["cells"])
                if len(ns) == 2 and (ns[0][0] + ns[1][0], ns[0][1] + ns[1][1]) != (2*p[0], 2*p[1]):
                    bends += 1
            for p in info["glyphs"]:
                q = info["parent"][p]
                headings.add((p[0] - q[0], p[1] - q[1]))
            # Any bijective recoloring leaves the input geometry self-describing.
            remap = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
            assert np.array_equal(solve(remap[pair["input"]]), remap[pair["output"]])
    assert len(shapes) >= 45 and len(structures) >= 190
    assert len(roles) >= 35 and node_counts >= {1, 3, 4, 5, 6, 7}
    assert bends > 100 and headings == set(_DIRS)
    # Every legal explicit count is exercised by verify.py; also test boundaries here.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1234, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Check two different parent omissions with input-derived counterfactuals.
    p = puzzles[0]
    for name in ("gap", "pruned", "color", "glyph"):
        changed = copy.deepcopy(p)
        pair = changed["test"][0]
        info = _read(pair["input"])
        painted, visited = _walk(info)
        if name == "gap":
            q = min(set(painted) & info["gaps"])
            pair["output"][q] = info["background"]
        elif name == "pruned":
            q = min(info["cells"] - set(painted))
            pair["output"][q] = info["track"]
        elif name == "color":
            pair["output"][visited[0]] = info["track"]
        else:
            r, c = visited[0]
            cmd, color = info["glyphs"][(r, c)]
            block = pair["input"][r-1:r+2, c-1:c+2]
            block[:] = info["background"]
            block[_MASKS["R" if cmd == "L" else "L"]] = color
        try:
            validate_generated_puzzle(changed)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"{name} mutation was accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "spatial_structures": len(structures), "role_palettes": len(roles),
            "glyph_counts": sorted(node_counts), "corridor_bends": bends,
            "arrival_directions": sorted(headings),
            "verification": "generator-owned semantic checks"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
