"""Selective sluices: set-qualified plugs change which mapped cavities survive.

No file access, saved samples, or generation state is used by solve().
Construction uses a room graph; prediction reconstructs pixels and floods them.
"""
from collections import deque
import hashlib
import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_HOST_SEEDS = (0, 707690579, 345873797, 611978937, 214076457, 486850142,
               436274229, 564243328, 210505712, 415653885, 419274862,
               446312239, 599981064, 332164393, 789522071, 610165607,
               591244628, 940649743, 509034249, 645143803, 610617693,
               405626381, 960273836, 374119011, 406057323, 70928180,
               388439920, 812444961, 331439633, 141184988, 35699805,
               894493210, 832328929, 405756207, 439663546, 950166123,
               251581259, 146336944, 38917451, 461813810, 209794227,
               894518166)


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = _D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _D4
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells = [(r, c)]
        seen[r, c] = True
        for a, b in cells:
            for dr, dc in steps:
                x, y = a + dr, b + dc
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    cells.append((x, y))
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _corner_view(a, corner):
    if corner & 2:
        a = a[::-1]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _decode(inp):
    """Recover the selector, bounded edge key, vessels and rectangular plugs."""
    assert min(inp.shape) >= 6
    selectors = []
    for corner in range(4):
        v = _corner_view(inp, corner)
        frame = int(v[3, 3])
        if not (np.all(v[3, :4] == frame) and np.all(v[:4, 3] == frame)):
            continue
        colors, counts = np.unique(v[:3, :3], return_counts=True)
        pads = colors[counts == 6]
        if len(pads) != 1 or len(colors) != 4:
            continue
        bg = int(pads[0])
        query = frozenset(map(int, colors[colors != bg]))
        if frame == bg or frame in query or np.count_nonzero(inp == frame) != 7:
            continue
        mask = np.zeros(inp.shape, dtype=bool)
        _corner_view(mask, corner)[:4, :4] = True
        selectors.append((corner, bg, frame, query, mask))
    assert len(selectors) == 1, "unique L selector required"
    corner, bg, frame, query, selector = selectors[0]
    ink = (inp != bg) & ~selector
    candidates = []
    for cells in _components(ink, True):
        r0, r1, c0, c1 = _bbox(cells)
        sh, sw = r1 - r0, c1 - c0
        if len(cells) != sh * sw:
            continue
        lanes = []
        if sh == 2 and 2 <= sw <= 3:
            if r0 == 0 and c0 > 0 and c1 < inp.shape[1]:
                lanes.append((0, inp[r0, c0:c1], inp[r1-1, c0:c1]))
            if r1 == inp.shape[0] and c0 > 0 and c1 < inp.shape[1]:
                lanes.append((2, inp[r1-1, c0:c1], inp[r0, c0:c1]))
        if sw == 2 and 2 <= sh <= 3:
            if c0 == 0 and r0 > 0 and r1 < inp.shape[0]:
                lanes.append((1, inp[r0:r1, c0], inp[r0:r1, c1-1]))
            if c1 == inp.shape[1] and r0 > 0 and r1 < inp.shape[0]:
                lanes.append((3, inp[r0:r1, c1-1], inp[r0:r1, c0]))
        for side, src, dst in lanes:
            src, dst = tuple(map(int, src)), tuple(map(int, dst))
            if len(set(src)) == len(src) and not (set(src) & set(query)):
                candidates.append((side, dict(zip(src, dst)), cells))
    assert len(candidates) == 1, "unique noncorner edge lookup required"
    side, mapping, key_cells = candidates[0]
    key_mask = np.zeros(inp.shape, dtype=bool)
    for p in key_cells:
        key_mask[p] = True
    ink &= ~key_mask
    sources = set(mapping)
    assert not sources & (set(query) | {bg, frame})
    assert all(v not in {bg, frame} for v in mapping.values())
    owners = np.full(inp.shape, -1, dtype=int)
    vessels = []
    for cells in _components(ink, True):
        wall = {int(inp[p]) for p in cells} & sources
        assert len(wall) == 1, "one source identity per vessel"
        wall = next(iter(wall))
        owner = len(vessels)
        for p in cells:
            owners[p] = owner
        vessels.append({"wall": wall, "cells": cells, "bbox": _bbox(cells)})
    assert vessels
    gate_mask = ink & ~np.isin(inp, tuple(sources))
    gates = []
    carriers = set()
    for cells in _components(gate_mask):
        r0, r1, c0, c1 = box = _bbox(cells)
        h, w = r1-r0, c1-c0
        assert len(cells) == h*w and min(h, w) in (1, 2) and max(h, w) >= 5
        patch = inp[r0:r1, c0:c1]
        ends = np.concatenate((patch[:, 0], patch[:, -1])) if w > h else np.concatenate((patch[0], patch[-1]))
        assert np.all(ends == ends[0]), "plug endcaps identify its carrier"
        carrier = int(ends[0])
        assert carrier not in sources | set(query) | {bg, frame}
        carriers.add(carrier)
        own = {int(owners[p]) for p in cells}
        assert len(own) == 1 and -1 not in own
        labels = frozenset(int(inp[p]) for p in cells if inp[p] != carrier)
        assert not labels & (sources | {bg, frame})
        gates.append({"cells": cells, "bbox": box, "carrier": carrier,
                      "labels": labels, "counts": {c: sum(inp[p] == c for p in cells) for c in labels},
                      "open": query <= labels, "owner": next(iter(own))})
    assert gates and len(carriers) == 1
    carrier = next(iter(carriers))
    assert carrier not in mapping.values()
    return {"background": bg, "frame": frame, "query": query, "corner": corner,
            "selector": selector, "key": key_mask, "side": side, "mapping": mapping,
            "owners": owners, "vessels": vessels, "gates": gates, "carrier": carrier}


def _regions(a, bg):
    result = []
    h, w = a.shape
    for cells in _components(a == bg):
        edge = any(r in (0, h-1) or c in (0, w-1) for r, c in cells)
        boundary = set()
        for r, c in cells:
            for dr, dc in _D4:
                x, y = r+dr, c+dc
                if 0 <= x < h and 0 <= y < w and a[x, y] != bg:
                    boundary.add((x, y))
        result.append({"cells": cells, "edge": edge, "boundary": boundary})
    return result


def _apply(inp, info, opening="query", paint=True):
    out = inp.copy()
    for gate in info["gates"]:
        if opening == "all" or (opening == "query" and gate["open"]):
            for p in gate["cells"]:
                out[p] = info["background"]
    opened = out.copy()
    # Control glyphs are not physical vessel walls. Some selector arrangements
    # surround a padding cell; clearing the controls prevents a false cavity.
    opened[info["selector"] | info["key"]] = info["background"]
    if paint:
        for region in _regions(opened, info["background"]):
            if region["edge"]:
                continue
            own = {int(info["owners"][p]) for p in region["boundary"]}
            assert len(own) == 1 and -1 not in own, "enclosure must have one vessel owner"
            wall = info["vessels"][next(iter(own))]["wall"]
            for p in region["cells"]:
                out[p] = info["mapping"][wall]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _apply(inp, _decode(inp))


def _axis(rng, count, thickness, budget):
    sizes = [5] * count
    for i in rng.permutation(count):
        i = int(i)
        spare = budget - thickness*(count+1) - sum(sizes)
        sizes[i] += int(rng.integers(min(5, spare)+1))
    lines = [0]
    for size in sizes:
        lines.append(lines[-1] + thickness + size)
    return lines, lines[-1] + thickness


def _plan(rng, count):
    options = [(r, c) for r in range(1, 4) for c in range(1, 4) if r*c >= count]
    for attempt in range(32):
        nr, nc = options[int(rng.integers(len(options)))]
        cells = {(int(rng.integers(nr)), int(rng.integers(nc)))}
        while len(cells) < count:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                               if 0 <= r+dr < nr and 0 <= c+dc < nc} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        occupied = np.zeros((nr+2, nc+2), dtype=bool)
        for r, c in cells:
            occupied[r+1, c+1] = True
        if len(_components(~occupied)) == 1:
            break
    else:
        # Removing cells from the end of a row-major 3-column rectangle never
        # creates a courtyard. Bounded fallback, with the same transformation.
        cells = {(i//3, i % 3) for i in range(count)}
    r0, _, c0, _ = _bbox(cells)
    return sorted((r-r0, c-c0) for r, c in cells)


def _edges(cells):
    present = set(cells)
    edges = []
    for a in cells:
        for direction, (dr, dc) in enumerate(_D4):
            b = (a[0]+dr, a[1]+dc)
            if b not in present:
                edges.append((a, None, direction))
            elif a < b:
                edges.append((a, b, direction))
    return edges


def _path(cells, edges):
    links = {a: [] for a in cells}
    for a, b, _ in edges:
        if b is not None:
            links[a].append(b)
            links[b].append(a)
    start = min(a for a in cells if len(links[a]) == 1)
    result = [start]
    while len(result) < len(cells):
        result.append(next(b for b in links[result[-1]] if b not in result))
    return result


def _make_pair(rng, package, count, turns, reflect):
    # Canonical roles are permuted together at the very end, independently per pair.
    bg, frame, carrier = 0, 1, 2
    nkey = int(rng.integers(2, 4))
    sources = list(range(3, 3+nkey))
    labels = [6, 7, 8, 9]
    q = list(map(int, rng.choice(labels, 3, replace=False)))
    extra = next(c for c in labels if c not in q)
    wall = int(rng.choice(sources))
    fills = {s: int(rng.choice(sources+labels)) for s in sources}
    if package == "leak":
        other = next(s for s in sources if s != wall)
        fills[wall], fills[other] = other, int(rng.choice(labels))
    elif package == "sealed":
        fills[wall] = int(rng.choice(labels))
    thickness = 1 if package == "leak" else 2 if package == "sealed" else int(rng.integers(1, 3))
    cells = _plan(rng, count)
    nr, nc = max(r for r, c in cells)+1, max(c for r, c in cells)+1
    rows, vh = _axis(rng, nr, thickness, 23)
    cols, vw = _axis(rng, nc, thickness, 28)
    h = vh + 7 + int(rng.integers(min(30-vh-7, 5)+1))
    w = max(vw+2, 10) + int(rng.integers(min(30-max(vw+2, 10), 5)+1))
    rbase = 6 + int(rng.integers(h-vh-6))
    cbase = 1 + int(rng.integers(w-vw-1))
    a = np.full((h, w), bg, dtype=np.int8)
    a[3, :4] = frame
    a[:4, 3] = frame
    for p, color in zip(rng.choice(9, 3, replace=False), rng.permutation(q)):
        a[int(p)//3, int(p)%3] = color
    key_start = int(rng.integers(5, w-nkey))
    for j, s in enumerate(rng.permutation(sources)):
        a[0, key_start+j], a[1, key_start+j] = s, fills[s]
    interiors = {}
    for cell in cells:
        r, c = cell
        r0, r1 = rbase+rows[r], rbase+rows[r+1]+thickness
        c0, c1 = cbase+cols[c], cbase+cols[c+1]+thickness
        a[r0:r1, c0:c1] = wall
        interiors[cell] = (r0+thickness, r1-thickness, c0+thickness, c1-thickness)
    for r0, r1, c0, c1 in interiors.values():
        a[r0:r1, c0:c1] = bg
    edges = _edges(cells)
    gate_specs = {}
    if package in ("leak", "sealed"):
        path = _path(cells, edges)
        def link(x, y):
            return next(i for i, (a0, b0, d) in enumerate(edges) if {a0, b0} == {x, y})
        def outer(cell, used=()):
            choices = [i for i, (a0, b0, d) in enumerate(edges) if a0 == cell and b0 is None and i not in used]
            return int(rng.choice(choices))
        first, second = link(path[0], path[1]), link(path[1], path[2])
        if package == "leak":
            gate_specs[first] = q.copy()
            gate_specs[second] = [q[1], q[2], extra]
            vent = outer(path[0])
            gate_specs[vent] = q.copy()
            gate_specs[outer(path[0], (vent,))] = [q[0], q[2], extra]
            gate_specs[outer(path[2])] = [q[0], q[1], extra]
        else:
            gate_specs[first] = q + [extra, q[0], extra]
            gate_specs[second] = [q[0], q[1], extra, q[0], q[1], extra]
            gate_specs[outer(path[2])] = q.copy()
            gate_specs[outer(path[0])] = []
    else:
        density = float(rng.uniform(.35, .85))
        for i in range(len(edges)):
            if rng.random() < density:
                if rng.random() < .45:
                    marks = q.copy()
                    if rng.random() < .5:
                        marks.append(extra)
                else:
                    keep = int(rng.integers(3))
                    marks = list(map(int, rng.permutation(q)[:keep]))
                    if rng.random() < .65:
                        marks.append(extra)
                gate_specs[i] = marks
        if not gate_specs:
            gate_specs[int(rng.integers(len(edges)))] = q.copy()
    gates = []
    for i, marks in gate_specs.items():
        room, neighbor, direction = edges[i]
        r0, r1, c0, c1 = interiors[room]
        horizontal = direction in (0, 2)
        length_max = c1-c0 if horizontal else r1-r0
        # A five-cell thin plug has room for exactly three labels and its two
        # carrier endcaps. Only an optional trailing extra can exceed this cap.
        capacity = (length_max-2)*thickness
        if len(marks) > capacity:
            assert package == "free" and marks[:3] == q
            marks = marks[:capacity]
        length = int(rng.integers(max(5, 2+(len(marks)+thickness-1)//thickness), length_max+1))
        offset = int(rng.integers(length_max-length+1))
        if horizontal:
            row = r0-thickness if direction == 0 else r1
            col = c0+offset
            box = (row, row+thickness, col, col+length)
        else:
            row = r0+offset
            col = c1 if direction == 1 else c0-thickness
            box = (row, row+length, col, col+thickness)
        gr0, gr1, gc0, gc1 = box
        patch = a[gr0:gr1, gc0:gc1]
        patch[:] = carrier
        available = [(r, c) for r in range(patch.shape[0]) for c in range(patch.shape[1])
                     if (0 < c < patch.shape[1]-1 if horizontal else 0 < r < patch.shape[0]-1)]
        if package == "free" and marks:
            target = int(rng.integers(len(marks), len(available)+1))
            marks = marks + list(map(int, rng.choice(marks, target-len(marks))))
        for j, color in zip(rng.choice(len(available), len(marks), replace=False), rng.permutation(marks)):
            patch[available[int(j)]] = color
        gates.append((room, neighbor, box, set(q) <= set(marks)))
    # Construction truth: escape on the room graph, independent of pixel flooding.
    adjacency = {cell: [] for cell in cells}
    escaping = set()
    for room, neighbor, box, opened in gates:
        if opened:
            if neighbor is None:
                escaping.add(room)
            else:
                adjacency[room].append(neighbor)
                adjacency[neighbor].append(room)
    queue = deque(sorted(escaping))
    while queue:
        room = queue.popleft()
        for neighbor in adjacency[room]:
            if neighbor not in escaping:
                escaping.add(neighbor)
                queue.append(neighbor)
    out = a.copy()
    for cell, (r0, r1, c0, c1) in interiors.items():
        if cell not in escaping:
            out[r0:r1, c0:c1] = fills[wall]
    for room, neighbor, (r0, r1, c0, c1), opened in gates:
        if opened:
            out[r0:r1, c0:c1] = bg if room in escaping else fills[wall]
    palette = rng.permutation(10).astype(np.int8)
    a, out = palette[a], palette[out]
    if reflect:
        a, out = a[:, ::-1], out[:, ::-1]
    return {"input": np.rot90(a, turns).copy(), "output": np.rot90(out, turns).copy()}


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(name + " out of range")
    return value


def _fingerprint(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    base_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            package = ("leak" if i == 0 else "sealed" if i == 1 else "free") if split == "train" else "free"
            count = 3 if package != "free" else int(rng.integers(1, 4) if split == "train" else rng.integers(4, 9))
            turns = (base_turn+i) % 4 if split == "train" and i < 2 else int(rng.integers(4))
            for retry in range(12):
                pair = _make_pair(rng, package, count, turns, bool(rng.integers(2)))
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out, info=None):
    info = _decode(inp) if info is None else info
    bg = info["background"]
    scene = inp.copy()
    scene[info["selector"] | info["key"]] = bg
    original_rooms = [r for r in _regions(scene, bg) if not r["edge"]]
    leaking = sum(all(out[p] == bg for p in r["cells"]) for r in original_rooms)
    painted = len(original_rooms)-leaking
    open_gates = [g for g in info["gates"] if g["open"]]
    painted_gates = sum(all(out[p] != bg for p in g["cells"]) for g in open_gates)
    chain = any(info["mapping"][v["wall"]] in info["mapping"] and
                info["mapping"][info["mapping"][v["wall"]]] != info["mapping"][v["wall"]]
                for v in info["vessels"])
    return {"rooms": len(original_rooms), "leaking": int(leaking), "painted": int(painted),
            "gates": len(info["gates"]), "open": len(open_gates), "painted_gates": int(painted_gates),
            "chain": bool(chain), "side": info["side"], "background": bg,
            "key_length": len(info["mapping"]), "wall": info["vessels"][0]["wall"]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, records, features = set(), [], [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen
            seen.add(key)
            info = _decode(inp)
            expected = _apply(inp, info)
            assert np.array_equal(expected, out), "wrong dissolution or post-opening enclosure"
            changeable = inp == info["background"]
            for gate in info["gates"]:
                if gate["open"]:
                    for p in gate["cells"]:
                        changeable[p] = True
            assert np.array_equal(inp[~changeable], out[~changeable]), "preserved ink changed"
            f = _features(inp, out, info)
            features[split].append(f)
            if split == "train":
                training.append((inp, out, info))
                records.extend((info, g) for g in info["gates"])
    # Reconstruct teaching witnesses from cells; no package metadata is trusted.
    assert any(f["leaking"] >= 2 and f["painted"] >= 1 and f["chain"] for f in features["train"])
    assert any(f["painted_gates"] >= 1 and f["painted"] >= 2 for f in features["train"])
    assert any(g["open"] and len(g["labels"]) == 3 and sum(g["counts"].values()) == 3 for _, g in records)
    assert any(g["open"] and len(g["labels"]) > 3 and any(n > 1 for n in g["counts"].values()) for _, g in records)
    assert any(not g["labels"] for _, g in records)
    assert any(not g["open"] and len(g["labels"]) == 3 and sum(g["counts"].values()) == 3 for _, g in records)
    assert any(not g["open"] and sum(g["counts"].values()) >= 6 for _, g in records)
    assert any(all(any(not g["open"] and g["labels"] & info["query"] == info["query"]-{q}
                       for g in info["gates"]) for q in info["query"]) for _, _, info in training)
    for opening, paint in (("none", True), ("all", True), ("query", False)):
        assert any(not np.array_equal(_apply(inp, info, opening, paint), out) for inp, out, info in training)
    assert len({f["side"] for f in features["train"]}) >= 2
    assert all(f["rooms"] <= 3 for f in features["train"])
    assert all(4 <= f["rooms"] <= 8 for f in features["test"])


def self_test():
    """42 supplied seeds, streaming statistics, counterfactuals and mutations."""
    shapes, normalized, backgrounds, sides, counts, thicknesses = set(), set(), set(), set(), set(), set()
    puzzle_hashes = set()
    leaking = painted_gate = chains = 0
    for index, seed in enumerate(_HOST_SEEDS):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                info = _decode(inp)
                f = _features(inp, out, info)
                shapes.add(inp.shape)
                backgrounds.add(info["background"])
                sides.add(info["side"])
                counts.add(f["rooms"])
                leaking += f["leaking"] > 0
                painted_gate += f["painted_gates"] > 0
                chains += f["chain"] and f["painted"] > 0
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                normalized.add((inp.shape, norm))
                for gate in info["gates"]:
                    r0, r1, c0, c1 = gate["bbox"]
                    thicknesses.add(min(r1-r0, c1-c0))
                # Equivariance: all information needed for a new orientation or palette is visible.
                if index < 3:
                    permutation = np.arange(10, dtype=np.int8)[::-1]
                    assert np.array_equal(solve(np.rot90(permutation[inp])), np.rot90(permutation[out]))
        puzzle_hashes.add(digest.hexdigest())
        if index < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], again[s]) for k in x)
        # Membership counterfactual, retaining the same wall network and lookup.
        for pair in p["train"]:
            inp = pair["input"]
            info = _decode(inp)
            option = next((g for g in info["gates"] if not g["open"] and len(g["labels"]) == 3), None)
            if option:
                changed = inp.copy()
                view = _corner_view(changed, info["corner"])
                positions = list(zip(*np.nonzero(view[:3, :3] != info["background"])))
                for point, color in zip(positions, sorted(option["labels"])):
                    view[point] = color
                altered = solve(changed)
                assert any(altered[point] != inp[point] for point in option["cells"])
                assert not np.array_equal(altered[~info["selector"]], pair["output"][~info["selector"]])
                break
        else:
            raise AssertionError("no membership counterfactual")
    assert len(puzzle_hashes) == 42 and len(normalized) == 210
    assert len(shapes) >= 50 and backgrounds == set(range(10)) and sides == set(range(4))
    assert counts == set(range(1, 9)) and thicknesses == {1, 2}
    assert leaking >= 42 and painted_gate >= 42 and chains >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(19, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(0, 2, 1)
    info = _decode(bad["train"][0]["input"])
    r, c = np.argwhere(info["key"])[0]
    bad["train"][0]["output"][r, c] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("key corruption accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
