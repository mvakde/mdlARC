"""Reassemble printed polyominoes, then grow their one global echo drawing.

The pieces transport a coordinate frame, including the one-pixel echo vector.
No reference files, seeds, training pairs or generation state are used by solve.
All diagnostics are bounded and kept in memory; this module writes no files.
"""
from itertools import combinations
import hashlib
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    left = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in N4:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    todo.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _mask(cells):
    s = _normal(cells)
    a = np.zeros((max(r for r, c in s)+1, max(c for r, c in s)+1), dtype=bool)
    for p in s:
        a[p] = True
    return a


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4)


def _vector(d, code):
    r, c = d
    if code >= 4:
        c = -c
    for _ in range(code % 4):
        r, c = -c, r
    return r, c


def _orientations(shape):
    a = _mask(shape)
    return tuple(sorted({_normal(np.argwhere(_orient(a, k)).tolist()) for k in range(8)}))


def _class(shape):
    return _orientations(shape)[0]


def _aligned(a, b):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _segment(a, b):
    assert _aligned(a, b)
    n = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    dr, dc = (b[0]-a[0]) // n, (b[1]-a[1]) // n
    return {(a[0]+i*dr, a[1]+i*dc) for i in range(n+1)}


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _intersects(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c)*_cross(a, b, d) <= 0 and _cross(c, d, a)*_cross(c, d, b) <= 0


def _topology(points):
    """Unique simple full cycle first; otherwise unique simple full path.

    Small Hamiltonian searches operate on visible landmarks only. Crossings at
    half-pixels are forbidden too. No generation ordering is available here.
    """
    ps, n = list(points), len(points)
    assert 3 <= n <= 8
    edges = [(i, j) for i, j in combinations(range(n), 2) if _aligned(ps[i], ps[j])
             and not (_segment(ps[i], ps[j]) - {ps[i], ps[j]}) & set(ps)]
    adjacency = [[] for _ in ps]
    for e, (i, j) in enumerate(edges):
        adjacency[i].append((j, e))
        adjacency[j].append((i, e))
    assert all(adjacency), "isolated ink landmark"
    conflict = [set() for _ in edges]
    for e, (i, j) in enumerate(edges):
        for f, (u, v) in enumerate(edges[:e]):
            if not {i, j} & {u, v} and _intersects(ps[i], ps[j], ps[u], ps[v]):
                conflict[e].add(f)
                conflict[f].add(e)

    def search(closed):
        solutions, budget = set(), [0]

        def visit(path, used, selected):
            budget[0] += 1
            assert budget[0] <= 16000, "topology budget exceeded"
            if len(solutions) > 1:
                return
            if len(path) == n:
                chosen = selected
                if closed:
                    choices = [e for v, e in adjacency[path[-1]] if v == path[0]]
                    if not choices or conflict[choices[0]] & set(selected):
                        return
                    chosen = selected + [choices[0]]
                solutions.add(tuple(sorted(chosen)))
                return
            for v, e in adjacency[path[-1]]:
                if v not in used and not conflict[e] & set(selected):
                    visit(path+[v], used | {v}, selected+[e])
        for start in ([0] if closed else range(n)):
            visit([start], {start}, [])
            if len(solutions) > 1:
                break
        return solutions

    cycles = search(True)
    assert len(cycles) <= 1, "ambiguous contours"
    chosen = cycles or search(False)
    assert len(chosen) == 1, "ambiguous or disconnected path"
    selected = [edges[k] for k in next(iter(chosen))]
    assert all(max(abs(ps[i][k]-ps[j][k]) for k in (0, 1)) >= 2 for i, j in selected), "unit link"
    pixels = set().union(*(_segment(ps[i], ps[j]) for i, j in selected))
    all_pixels = set().union(*(_segment(ps[i], ps[j]) for i, j in edges))
    return pixels, {"closed": bool(cycles), "chord_cells": len(all_pixels-pixels),
                    "edges": selected, "vertex_count": n,
                    "diagonal": any(ps[i][0] != ps[j][0] and ps[i][1] != ps[j][1] for i, j in selected)}


def _paint(base, source, offset, ink, echo):
    out = base.copy()
    copies = {(r+offset[0], c+offset[1]) for r, c in source}
    assert all(0 <= r < out.shape[0] and 0 <= c < out.shape[1] for r, c in copies), "clipped echo"
    for p in copies-source:
        out[p] = echo
    for p in source:
        out[p] = ink
    return out, copies


def _decode_mask(cells, scale):
    r0, r1, c0, c1 = _bounds(cells)
    if (r1-r0) % scale or (c1-c0) % scale or len(cells) % (scale*scale):
        return None
    counts = {}
    for r, c in cells:
        p = (r-r0)//scale, (c-c0)//scale
        counts[p] = counts.get(p, 0)+1
    if any(n != scale*scale for n in counts.values()):
        return None
    return tuple(sorted(counts))


def _recover(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size//2, "background must dominate"
    boards = []
    for structural in range(10):
        if structural == bg:
            continue
        for component in _components(a == structural):
            r0, r1, c0, c1 = _bounds(component)
            b = a[r0:r1, c0:c1]
            if min(b.shape) < 5 or not np.all((b == bg) | (b == structural)):
                continue
            if not all(np.all(side == structural) for side in (b[0], b[-1], b[:, 0], b[:, -1])):
                continue
            holes = _components(b == bg)
            if 2 <= len(holes) <= 3:
                boards.append((structural, (r0, r1, c0, c1), b, holes))
    accepted = []
    for structural, bounds, board, holes in boards:
        r0, r1, c0, c1 = bounds
        outside = np.ones(a.shape, dtype=bool)
        outside[r0:r1, c0:c1] = False
        external = outside & (a != bg)
        pieces = _components(external)
        if len(pieces) != len(holes) or np.any(a[external] == structural):
            continue
        roles, amounts = np.unique(a[external], return_counts=True)
        if len(roles) != 3:
            continue
        order = np.argsort(amounts)
        echo, ink, material = map(int, roles[order])
        if amounts[order[0]] != 1 or not 3 <= amounts[order[1]] <= 8 or amounts[order[2]] <= 2*amounts[order[1]]:
            continue
        hole_classes = [_class(h) for h in holes]
        if len(set(hole_classes)) != len(holes) or any(len(_orientations(h)) != 8 for h in holes):
            continue
        for scale in (2, 3):
            if max(board.shape)*scale > 30:
                continue
            decoded = [_decode_mask(cells, scale) for cells in pieces]
            if any(s is None for s in decoded) or sorted(_class(s) for s in decoded) != sorted(hole_classes):
                continue
            base = np.repeat(np.repeat(board, scale, 0), scale, 1)
            labels = np.zeros(base.shape, dtype=np.int8)
            transformed_codes, raw_offset, marker_piece = [], None, None
            raw_shapes = []
            for index, (cells, shape) in enumerate(zip(pieces, decoded), 1):
                hr = hole_classes.index(_class(shape))
                dest = holes[hr]
                dh0, dh1, dw0, dw1 = _bounds(dest)
                destination = np.repeat(np.repeat(_mask(dest), scale, 0), scale, 1)
                pr0, pr1, pc0, pc1 = _bounds(cells)
                patch = a[pr0:pr1, pc0:pc1]
                source_mask = patch != bg
                matches = [code for code in range(8) if _orient(source_mask, code).shape == destination.shape
                           and np.array_equal(_orient(source_mask, code), destination)]
                assert len(matches) == 1, "piece orientation is ambiguous"
                code = matches[0]
                transformed_codes.append(code)
                printed = _orient(patch, code)
                view = base[dh0*scale:dh1*scale, dw0*scale:dw1*scale]
                view[destination] = printed[destination]
                labels[dh0*scale:dh1*scale, dw0*scale:dw1*scale][destination] = index
                raw_shapes.append(_class(shape))
                assert np.any(patch == ink), "every piece must contribute a landmark"
                if np.any(patch == echo):
                    mark = tuple(map(int, np.argwhere(patch == echo)[0]))
                    anchors = [tuple(map(int, p)) for p in np.argwhere(patch == ink)
                               if max(abs(int(p[0])-mark[0]), abs(int(p[1])-mark[1])) == 1]
                    assert len(anchors) == 1, "ambiguous carried marker"
                    raw_offset = mark[0]-anchors[0][0], mark[1]-anchors[0][1]
                    marker_piece = index
            ps = [tuple(map(int, p)) for p in np.argwhere(base == ink)]
            mark = tuple(map(int, np.argwhere(base == echo)[0]))
            anchors = [p for p in ps if max(abs(p[0]-mark[0]), abs(p[1]-mark[1])) == 1]
            assert len(anchors) == 1
            offset = mark[0]-anchors[0][0], mark[1]-anchors[0][1]
            assert labels[mark] == labels[anchors[0]] == marker_piece
            assert _vector(raw_offset, transformed_codes[marker_piece-1]) == offset
            source, topo = _topology(ps)
            assert mark not in source and any(p not in ps for p in source)
            expected, copies = _paint(base, source, offset, ink, echo)
            bare = base.copy()
            bare[(bare == ink) | (bare == echo)] = material
            ablated, _ = _paint(base, source, raw_offset, ink, echo)
            cross_links = sum(labels[ps[i]] != labels[ps[j]] for i, j in topo["edges"])
            wall_ink = sum(bare[p] == structural for p in source)
            accepted.append((expected, {**topo, "background": bg, "structural": structural,
                "material": material, "ink": ink, "echo": echo, "scale": scale,
                "piece_count": len(pieces), "piece_classes": tuple(sorted(raw_shapes)),
                "areas": tuple(sorted(len(s) for s in decoded)), "codes": tuple(transformed_codes),
                "reflection": any(code >= 4 for code in transformed_codes),
                "offset": offset, "raw_offset": raw_offset, "cross_links": cross_links,
                "wall_ink": wall_ink, "visible_echo": len(copies-source-{mark}),
                "overlap": len(copies & source), "vector_matters": not np.array_equal(ablated, expected),
                "source_added": len(source-set(ps)), "board_shape": board.shape,
                "input_shape": a.shape, "board_pattern": _key(board == structural),
                "assembled": base, "source_pixels": source, "marker": mark,
                "points": ps, "labels": labels}))
    assert len(accepted) == 1, "expected exactly one blueprint, scale and assembly"
    return accepted[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _recover(input_grid)[0]


def _grow(rng, area):
    cells = {(0, 0)}
    while len(cells) < area:
        options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in N4}-cells)
        options = [p for p in options if max(_mask(cells | {p}).shape) <= 5]
        if not options:
            return None
        cells.add(options[int(rng.integers(len(options)))])
    return _normal(cells)


def _holes(rng, h, w, count):
    used, shapes, result = set(), set(), []
    for _ in range(count):
        for attempt in range(100):
            shape = _grow(rng, int(rng.integers(4, 10)))
            if shape is None or len(_orientations(shape)) != 8 or _class(shape) in shapes:
                continue
            sh, sw = _mask(shape).shape
            if sh > h-2 or sw > w-2:
                continue
            origins = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            rng.shuffle(origins)
            for r0, c0 in origins:
                cells = {(r+r0, c+c0) for r, c in shape}
                if any((r+dr, c+dc) in used for r, c in cells for dr, dc in ((0, 0),)+N4):
                    continue
                result.append(tuple(sorted(cells)))
                used.update(cells)
                shapes.add(_class(shape))
                break
            else:
                continue
            break
        else:
            return None
    return result


def _rectangles(rng, labels, square):
    # Bounded reservoir instead of retaining all O(H^2 W^2) rectangles.
    chosen, seen = [], 0
    rows = [set(np.flatnonzero(labels[r]).tolist()) for r in range(1, labels.shape[0]-1)]
    for r0 in range(1, labels.shape[0]-3):
        for r1 in range(r0+2, labels.shape[0]-1):
            cols = sorted(rows[r0-1] & rows[r1-1])
            for c0, c1 in combinations(cols, 2):
                if c1-c0 < 2 or square and c1-c0 != r1-r0:
                    continue
                ps = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
                if len({int(labels[p]) for p in ps}) != int(labels.max()):
                    continue
                seen += 1
                if len(chosen) < 40:
                    chosen.append(ps)
                else:
                    j = int(rng.integers(seen))
                    if j < 40:
                        chosen[j] = ps
    rng.shuffle(chosen)
    return chosen


def _find_drawing(rng, labels, kind, offsets):
    candidates = []
    if kind in ("square", "cycle", "clipped"):
        boxes = _rectangles(rng, labels, kind == "square")
        for ps in boxes:
            if kind != "clipped":
                candidates.append(ps)
                continue
            # Cut one or both opposite corners, producing five/six vertices.
            r0, c0 = ps[0]
            r1, c1 = ps[2]
            for d in range(2, min(r1-r0, c1-c0)-1):
                for opposite in (False, True):
                    q = [(r0, c0+d), (r0, c1)]
                    q += [(r1-d, c1), (r1, c1-d)] if opposite else [(r1, c1)]
                    q += [(r1, c0), (r0+d, c0)]
                    if all(labels[p] for p in q):
                        candidates.append(q)
    else:
        points = [tuple(map(int, p)) for p in np.argwhere(labels)]
        points = [p for p in points if 0 < p[0] < labels.shape[0]-1 and 0 < p[1] < labels.shape[1]-1]
        # Build short random walks through actual printed supports.
        adjacency = {p: [q for q in points if q != p and _aligned(p, q)
                        and max(abs(p[k]-q[k]) for k in (0, 1)) >= 2] for p in points}
        n = 3 if kind == "open" else int(rng.integers(4, 6))
        for _ in range(450):
            ps = [points[int(rng.integers(len(points)))]]
            while len(ps) < n:
                choices = [p for p in adjacency[ps[-1]] if p not in ps]
                if not choices:
                    break
                ps.append(choices[int(rng.integers(len(choices)))])
            if len(ps) == n:
                candidates.append(ps)
    rng.shuffle(candidates)
    for ps in candidates:
        if len({int(labels[p]) for p in ps}) != int(labels.max()):
            continue
        try:
            source, info = _topology(ps)
        except AssertionError:
            continue
        if kind in ("open", "walk") and info["closed"]:
            continue
        intended = set().union(*(_segment(ps[i], ps[(i+1) % len(ps)]) for i in range(len(ps) if info["closed"] else len(ps)-1)))
        if source != intended or kind == "square" and info["chord_cells"] == 0:
            continue
        options = []
        for d in offsets:
            copies = {(r+d[0], c+d[1]) for r, c in source}
            if not all(0 <= r < labels.shape[0] and 0 <= c < labels.shape[1] for r, c in copies):
                continue
            for anchor in ps:
                mark = anchor[0]+d[0], anchor[1]+d[1]
                if mark in source or labels[mark] != labels[anchor]:
                    continue
                if sum(max(abs(mark[k]-p[k]) for k in (0, 1)) == 1 for p in ps) != 1:
                    continue
                if len(copies-source-{mark}) < 2 or not copies & source:
                    continue
                options.append((d, mark, int(labels[anchor])))
        if options:
            d, mark, marker_piece = options[int(rng.integers(len(options)))]
            return ps, source, info, d, mark, marker_piece
    return None


def _pack(rng, board, patches, bg):
    objects = [board]+patches
    for attempt in range(32):
        h, w = (map(int, rng.integers(23, 31, size=2)) if attempt < 20 else (30, 30))
        canvas = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        # No shelf or fixed board location. Place larger rectangles first.
        order = sorted(range(len(objects)), key=lambda j: -objects[j].size)
        for j in order:
            obj = objects[j]
            oh, ow = obj.shape
            choices = [(r, c) for r in range(h-oh+1) for c in range(w-ow+1)
                       if not blocked[r:r+oh, c:c+ow].any()]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            canvas[r:r+oh, c:c+ow] = obj
            blocked[max(0, r-1):min(h, r+oh+1), max(0, c-1):min(w, c+ow+1)] = True
        else:
            if np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            # Randomly crop some unused margin, preserving clear background.
            r0, r1, c0, c1 = _bounds(np.argwhere(canvas != bg).tolist())
            r0, c0 = max(0, r0-int(rng.integers(4))), max(0, c0-int(rng.integers(4)))
            r1, c1 = min(h, r1+int(rng.integers(4))), min(w, c1+int(rng.integers(4)))
            crop = canvas[r0:r1, c0:c1]
            return crop.copy() if np.count_nonzero(crop == bg) > crop.size//2 else canvas
    return None


def _make_pair(rng, kind, scale, offsets, reflection_witness=False, previous_roles=None):
    for attempt in range(160):
        h, w = map(int, rng.integers(7, 30//scale+1, size=2))
        h, w = min(h, 11), min(w, 11)
        count = 2 if kind == "open" else int(rng.choice([2, 3], p=[.7, .3]))
        holes = _holes(rng, h, w, count)
        if holes is None:
            continue
        labels = np.zeros((h*scale, w*scale), dtype=np.int8)
        for i, cells in enumerate(holes, 1):
            for r, c in cells:
                labels[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = i
        found = _find_drawing(rng, labels, kind, offsets)
        if found is None:
            continue
        ps, source, topo, offset, mark, marker_piece = found
        palette = list(map(int, rng.permutation(10)[:5]))
        if previous_roles is not None:
            for _ in range(24):
                if all(palette[j] != previous_roles[j] for j in range(5)):
                    break
                palette = list(map(int, rng.permutation(10)[:5]))
            else:
                palette = [(c+1) % 10 for c in previous_roles]
        bg, structural, material, ink, echo = palette
        board = np.full((h, w), structural, dtype=np.int8)
        for cells in holes:
            for p in cells:
                board[p] = bg
        assembled = np.repeat(np.repeat(board, scale, 0), scale, 1)
        assembled[labels != 0] = material
        for p in ps:
            assembled[p] = ink
        assembled[mark] = echo
        output, _ = _paint(assembled, source, offset, ink, echo)
        patches = []
        for i, cells in enumerate(holes, 1):
            r0, r1, c0, c1 = _bounds(cells)
            patch = assembled[r0*scale:r1*scale, c0*scale:c1*scale].copy()
            patch[labels[r0*scale:r1*scale, c0*scale:c1*scale] != i] = bg
            codes = list(range(8))
            if i == marker_piece:
                codes = [code for code in codes if _vector(offset, code) != offset]
                if reflection_witness:
                    codes = [code for code in codes if code >= 4]
            code = codes[int(rng.integers(len(codes)))]
            patches.append(_orient(patch, code).copy())
        canvas = _pack(rng, board, patches, bg)
        if canvas is None:
            continue
        predicted, info = _recover(canvas)
        assert np.array_equal(predicted, output), "construction disagrees with transported print"
        if not (info["cross_links"] and info["wall_ink"] and info["vector_matters"]):
            continue
        return {"input": canvas, "output": output}, palette
    raise RuntimeError(f"bounded construction exhausted for {kind}/{scale}")


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # A sampled held-out world-frame direction is the train/test shift.
    held = OFFSETS[int(rng.integers(8))]
    training_offsets = tuple(d for d in OFFSETS if d != held)
    train_k = [2, 3]+[int(rng.choice([2, 3])) for _ in range(nt-2)]
    rng.shuffle(train_k)
    kinds = ["square", "open"]+[str(rng.choice(["cycle", "clipped", "walk", "open"])) for _ in range(nt-2)]
    # Shuffle complete obligations, keeping both witnesses even for 2/1 calls.
    training_plan = list(zip(kinds, train_k))
    rng.shuffle(training_plan)
    plans = {"train": training_plan,
             "test": [(str(rng.choice(["clipped", "walk", "cycle"], p=[.5, .25, .25])),
                       int(rng.choice([2, 3]))) for _ in range(nq)]}
    puzzle, seen, previous = {"train": [], "test": []}, set(), None
    for split in ("train", "test"):
        for kind, k in plans[split]:
            for retry in range(8):
                pair, palette = _make_pair(rng, kind, k, training_offsets if split == "train" else (held,),
                                           reflection_witness=split == "train" and kind == "square",
                                           previous_roles=previous)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input budget exhausted")
            previous = palette
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out)
            assert info["source_added"] >= 1 and info["visible_echo"] >= 2
            assert info["cross_links"] >= 1 and info["wall_ink"] >= 1
            assert info["overlap"] >= 1 and info["vector_matters"]
            assert all(4 <= area <= 9 for area in info["areas"])
            # Source-over-copy, material preservation and transported marker.
            assert all(out[p] == info["ink"] for p in info["source_pixels"])
            assert out[info["marker"]] == info["echo"]
            assert np.array_equal(out[info["assembled"] == info["ink"]], info["assembled"][info["assembled"] == info["ink"]])
            features[split].append(info)
    train = features["train"]
    assert any(i["chord_cells"] for i in train)
    assert any(not i["closed"] for i in train)
    assert any(i["reflection"] for i in train)
    assert {i["scale"] for i in train} == {2, 3}
    assert len({i["piece_classes"] for i in train}) >= 2
    assert len({i["background"] for i in train}) >= 2
    demonstrated = {i["offset"] for i in train}
    asked = {i["offset"] for i in features["test"]}
    assert len(asked) == 1 and asked.isdisjoint(demonstrated)


def self_test():
    """42 seeds, streaming summaries, positive and negative geometry checks."""
    signatures, shapes, classes, offsets, counts, vertices, areas = (set() for _ in range(7))
    diagonals = reflections = chords = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                expected, info = _recover(pair["input"])
                assert np.array_equal(expected, pair["output"])
                digest.update(bytes(pair["input"].shape)+pair["input"].tobytes())
                shapes.add(info["board_shape"])
                classes.update(info["piece_classes"])
                offsets.add(info["offset"])
                counts.add(info["piece_count"])
                vertices.add(info["vertex_count"])
                areas.update(info["areas"])
                diagonals += info["diagonal"]
                reflections += info["reflection"]
                chords += info["chord_cells"] > 0
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 12 and len(classes) >= 25
    assert offsets == set(OFFSETS) and counts == {2, 3}
    assert {3, 4, 5, 6} <= vertices and set(range(4, 10)) <= areas
    assert diagonals and reflections and chords
    a, b = generate_puzzle(713, 2, 1), generate_puzzle(713, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Wrong answers and broken solid macrocells must be rejected.
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    expected, info = _recover(a["train"][0]["input"])
    broken = a["train"][0]["input"].copy()
    loc = tuple(np.argwhere(broken == info["material"])[0])
    broken[loc] = info["background"]
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken macrocell accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
