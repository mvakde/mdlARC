"""Register overlapping views of a command tree, then draw its branching trace.

Only NumPy and the standard library are used. There are no file reads, latent
dictionaries, imported parent generators, or saved examples in the solver.
"""
from __future__ import annotations

from collections import deque
import hashlib
import itertools
import numbers

import numpy as np


# A's four occupancy glyphs are retained; their new, explicitly fixed meanings
# start ADJACENT to the pen, in the indicated direction, rather than below it.
MASKS = (
    ("110", "101", "010"),  # right three
    ("101", "101", "111"),  # left two
    ("101", "010", "010"),  # down two
    ("111", "010", "101"),  # left four
)
STROKES = ((0, 1, 3), (0, -1, 2), (1, 0, 2), (0, -1, 4))
_BITS = tuple(tuple(int(x) for row in mask for x in row) for mask in MASKS)
_ROOT_BITS = (0, 0, 0, 0, 1, 0, 0, 0, 0)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "bad colors"
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _groups(points):
    left = set(points)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        group, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _render(cells, background):
    cells = _normalize(cells)
    h = max(r for r, c in cells)+1
    w = max(c for r, c in cells)+1
    assert max(h, w) <= 30, "drawing exceeds grid limit"
    a = np.full((h, w), background, dtype=np.int8)
    for p, color in cells.items():
        a[p] = color
    return a


def _tile(kind, color, wire):
    """A five-cell square frame is a physical registration key, not output ink."""
    cells = {(r, c): wire for r in range(5) for c in range(5)
             if r in (0, 4) or c in (0, 4)}
    bits = _ROOT_BITS if kind == -1 else _BITS[kind]
    cells.update({(1+r, 1+c): color for r in range(3) for c in range(3)
                  if bits[3*r+c]})
    return cells


def _decode_core(core, background, wire):
    bits = tuple(int(x != background) for x in core.flat)
    colors = set(int(x) for x in core[core != background])
    if len(colors) != 1 or wire in colors:
        return None
    color = next(iter(colors))
    if bits == _ROOT_BITS:
        return -1, color
    if bits in _BITS:
        return _BITS.index(bits), color
    return None


def _read(input_grid):
    """Infer roles, register full keys, and recover the tree from input pixels."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    assert counts.max()*2 > a.size, "background must be a strict majority"
    bg = int(values[counts.argmax()])
    frames = []
    for r in range(a.shape[0]-4):
        for c in range(a.shape[1]-4):
            block = a[r:r+5, c:c+5]
            wire = int(block[0, 0])
            if wire == bg:
                continue
            if not (np.all(block[0] == wire) and np.all(block[4] == wire)
                    and np.all(block[:, 0] == wire) and np.all(block[:, 4] == wire)):
                continue
            key = _decode_core(block[1:4, 1:4], bg, wire)
            if key is not None:
                frames.append((key, (r, c), wire))
    assert frames and len({f[2] for f in frames}) == 1, "unclear frame color"
    wire = frames[0][2]
    wire_groups = _groups(zip(*np.nonzero(a == wire)))
    component_at = {p: i for i, group in enumerate(wire_groups) for p in group}
    parts = [dict() for _ in wire_groups]
    covered = set(component_at)
    for key, (r, c), _ in frames:
        i = component_at[r, c]
        assert key not in parts[i], "a key cannot occur twice in one fragment"
        parts[i][key] = (r, c)
        for (dr, dc), color in _tile(*key, wire).items():
            assert int(a[r+dr, c+dc]) == color
            covered.add((r+dr, c+dc))
    assert len(parts) >= 2 and all(parts), "need separated keyed fragments"
    assert covered == set(zip(*np.nonzero(a != bg))), "unexplained material"

    # A shared COMPLETE key fixes a translation. All available matches must
    # agree; neither overlap maximization nor rectangle area breaks ties.
    offsets = {0: (0, 0)}
    todo = deque([0])
    while todo:
        i = todo.popleft()
        for j in range(len(parts)):
            shared = set(parts[i]) & set(parts[j])
            if not shared:
                continue
            proposals = {(offsets[i][0]+parts[i][k][0]-parts[j][k][0],
                          offsets[i][1]+parts[i][k][1]-parts[j][k][1]) for k in shared}
            assert len(proposals) == 1, "inconsistent shared keys"
            position = next(iter(proposals))
            if j in offsets:
                assert offsets[j] == position, "inconsistent registration cycle"
            else:
                offsets[j] = position
                todo.append(j)
    assert len(offsets) == len(parts), "disconnected overlap graph"

    positions, union, occurrences = {}, {}, {}
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        material = set(wire_groups[i])
        for key, (r, c) in part.items():
            q = r+dr, c+dc
            assert key not in positions or positions[key] == q
            positions[key] = q
            occurrences[key] = occurrences.get(key, 0)+1
            material.update((r+y, c+x) for y, x in _tile(*key, wire))
        for r, c in material:
            q = r+dr, c+dc
            color = int(a[r, c])
            assert q not in union or union[q] == color, "unequal-color collision"
            union[q] = color
    assert len(set(positions.values())) == len(positions), "different keys coincide"
    roots = [k for k in positions if k[0] == -1]
    assert len(roots) == 1, "need one unique root key"
    root = roots[0]
    assert root[1] != wire
    assert all(k[1] != root[1] for k in positions if k != root), "root ink is reserved"

    graph = {k: set() for k in positions}
    expected = {}
    for key, (r, c) in positions.items():
        expected.update({(r+y, c+x): v for (y, x), v in _tile(*key, wire).items()})
        # Background within every key is observed negative space.
        for y in range(5):
            for x in range(5):
                assert union.get((r+y, c+x), bg) == expected.get((r+y, c+x), bg), "damaged key"
    keys = sorted(positions)
    for i, k in enumerate(keys):
        r, c = positions[k]
        for other in keys[i+1:]:
            u, v = positions[other]
            path = []
            if r == u and abs(c-v) >= 6:
                path = [(r+2, x) for x in range(min(c, v)+5, max(c, v))]
            elif c == v and abs(r-u) >= 6:
                path = [(y, c+2) for y in range(min(r, u)+5, max(r, u))]
            if path and all(union.get(p) == wire for p in path):
                graph[k].add(other)
                graph[other].add(k)
                expected.update({p: wire for p in path})
    assert expected == union, "material outside frames and straight wires"
    assert sum(map(len, graph.values())) == 2*(len(graph)-1), "network must be a tree"
    parent, order = {root: None}, [root]
    for k in order:
        for other in sorted(graph[k]):
            if other == parent[k]:
                continue
            assert other not in parent, "cycle"
            parent[other] = k
            order.append(other)
    assert len(order) == len(graph), "disconnected command tree"
    return {"background": bg, "wire": wire, "root": root, "positions": positions,
            "graph": graph, "parent": parent, "order": order, "union": union,
            "occurrences": occurrences, "parts": parts, "offsets": offsets}


def _stroke(start, kind, strokes=STROKES):
    dr, dc, length = strokes[kind]
    return [(start[0]+dr*j, start[1]+dc*j) for j in range(1, length+1)]


def _clear(points, cells, start):
    # Only intended endpoint contact is allowed. This makes output branches
    # legible even when successive commands happen to use the same ink.
    for r, c in points:
        if (r, c) in cells:
            return False
        if any(p in cells and p != start
               for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
            return False
    return True


def _execute(info, strokes=STROKES):
    root = info["root"]
    ends = {root: (0, 0)}
    cells = {(0, 0): root[1]}
    for key in info["order"][1:]:
        start = ends[info["parent"][key]]
        points = _stroke(start, key[0], strokes)
        assert _clear(points, cells, start), "trace crosses or touches another branch"
        cells.update({p: key[1] for p in points})
        ends[key] = points[-1]
    return _render(cells, info["background"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a well-formed family member using this input alone."""
    return _execute(_read(input_grid))


def _trace_tree(rng, n):
    """Build labels/topology from a self-avoiding branching trace, independently
    of the later registration-picture geometry. Each attempt is bounded.
    """
    for _ in range(100):
        kinds = [int(x) for x in rng.permutation(4)]
        kinds += [int(x) for x in rng.integers(4, size=n-4)]
        parents, endpoints = [-1], [(0, 0)]
        cells = {(0, 0): 1}
        failed = False
        for kind in kinds:
            choices = []
            for i, start in enumerate(endpoints):
                points = _stroke(start, kind)
                if _clear(points, cells, start):
                    union = set(cells) | set(points)
                    if (max(r for r, c in union)-min(r for r, c in union) < 25
                            and max(c for r, c in union)-min(c for r, c in union) < 25):
                        choices.append((i, points))
            if not choices:
                failed = True
                break
            parent, points = choices[int(rng.integers(len(choices)))]
            parents.append(parent)
            endpoints.append(points[-1])
            cells.update({p: 1 for p in points})
        children = [parents.count(i) for i in range(n+1)]
        if not failed and max(children) >= 2:
            return parents, kinds
    raise RuntimeError("bounded trace construction failed")


def _embed_tree(rng, parents):
    """An unrelated orthogonal embedding of the same abstract tree."""
    for _ in range(16):
        sites = [(int(rng.integers(4)), int(rng.integers(4)))]
        used = set(sites)
        visits = [0]

        def visit(i):
            visits[0] += 1
            if visits[0] > 4000:
                return False
            if i == len(parents):
                return True
            r, c = sites[parents[i]]
            options = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for j in rng.permutation(4):
                q = options[int(j)]
                if min(q) < 0 or max(q) > 3 or q in used:
                    continue
                used.add(q)
                sites.append(q)
                if visit(i+1):
                    return True
                sites.pop()
                used.remove(q)
            return False

        if visit(1):
            rows = np.r_[0, np.cumsum(rng.integers(6, 9, size=3))]
            cols = np.r_[0, np.cumsum(rng.integers(6, 9, size=3))]
            return [(int(rows[r]), int(cols[c])) for r, c in sites]
    raise RuntimeError("bounded network embedding failed")


def _fragments(rng, parents, kinds, colors, wire, marker, index):
    positions = _embed_tree(rng, parents)
    tiles = [_tile(-1, marker, wire)]+[_tile(k, color, wire) for k, color in zip(kinds, colors)]
    edge_cells = []
    for child in range(1, len(parents)):
        parent = parents[child]
        part = {}
        for v in (parent, child):
            r, c = positions[v]
            part.update({(r+y, c+x): color for (y, x), color in tiles[v].items()})
        r, c = positions[parent]
        u, v = positions[child]
        path = ([(r+2, x) for x in range(min(c, v)+5, max(c, v))] if r == u
                else [(y, c+2) for y in range(min(r, u)+5, max(r, u))])
        part.update({p: wire for p in path})
        edge_cells.append(part)
    groups = [[i] for i in range(len(edge_cells))]
    # First demonstration: individual edges. Second: a three-station view.
    # Later examples mix both sizes. No fragment is the entire network.
    merges = 0 if index == 0 else 1 if index == 1 else int(rng.integers(0, 3))
    for _ in range(merges):
        choices = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if len(groups[i])+len(groups[j]) > 2:
                    continue
                ei, ej = groups[i][0]+1, groups[j][0]+1
                if {ei, parents[ei]} & {ej, parents[ej]}:
                    choices.append((i, j))
        if choices:
            i, j = choices[int(rng.integers(len(choices)))]
            groups[i] += groups.pop(j)
    parts = []
    for group in groups:
        part = {}
        for edge in group:
            part.update(edge_cells[edge])
        parts.append(_normalize(part))
    # This copied observation contributes no new command, even when it is a
    # leaf. Its role is registration, not an extra stroke or a repetition count.
    node_views = [0]*len(tiles)
    for group in groups:
        for v in {v for edge in group for v in (edge+1, parents[edge+1])}:
            node_views[v] += 1
    if index == 0 or max(node_views[1:]) < 2 or rng.random() < .45:
        parts.append(tiles[int(rng.integers(1, len(tiles)))].copy())
    return parts


def _pack(rng, parts, background):
    shapes = [(max(r for r, c in p)+1, max(c for r, c in p)+1) for p in parts]
    # Rectangle packing is deliberately independent of key identity/topology.
    # At most 12 bins, 1,200 search visits per bin, <=8 rectangles, one worker.
    canvases = [(int(rng.integers(max(18, max(h for h, w in shapes)), 31)),
                 int(rng.integers(max(18, max(w for h, w in shapes)), 31))) for _ in range(8)]
    canvases += [(30, 30)]*4
    for height, width in canvases:
        order = sorted(range(len(parts)), key=lambda i: -(shapes[i][0]*shapes[i][1]+float(rng.random())*30))
        placed, visits = {}, [0]

        def visit(depth):
            visits[0] += 1
            if visits[0] > 1200:
                return False
            if depth == len(order):
                return True
            i = order[depth]
            h, w = shapes[i]
            ys = {0, height-h}
            xs = {0, width-w}
            for j, (r, c) in placed.items():
                ph, pw = shapes[j]
                ys.add(r+ph+1)
                xs.add(c+pw+1)
            options = [(r, c) for r in sorted(ys) for c in sorted(xs)
                       if 0 <= r <= height-h and 0 <= c <= width-w
                       and all(r+h < u or u+shapes[j][0] < r
                               or c+w < v or v+shapes[j][1] < c
                               for j, (u, v) in placed.items())]
            for k in rng.permutation(len(options)):
                placed[i] = options[int(k)]
                if visit(depth+1):
                    return True
                del placed[i]
            return False

        if visit(0):
            cells = {(r+placed[i][0], c+placed[i][1]): color
                     for i, part in enumerate(parts) for (r, c), color in part.items()}
            tight = _render(cells, background)
            pads = []
            for d in tight.shape:
                spare = 30-d
                extra = int(rng.integers(min(spare, 5)+1))
                before = int(rng.integers(extra+1))
                pads.append((before, extra-before))
            a = np.pad(tight, pads, constant_values=background)
            if np.count_nonzero(a == background)*2 > a.size:
                return a
    raise RuntimeError("bounded fragment packing failed")


def _make_pair(rng, n, index, first_colors):
    for _ in range(40):
        bg, wire, marker = [int(x) for x in rng.choice(10, 3, replace=False)]
        ink = [c for c in range(10) if c not in (bg, wire, marker)]
        if index == 1:
            ink = [c for c in ink if c not in set(first_colors.values())]
        a, b, c = [int(x) for x in rng.choice(ink, 3, replace=False)]
        parents, kinds = _trace_tree(rng, n)
        colors, used = [], set()
        for kind in kinds:
            preferred = (a, b, c, a)[kind] if index < 2 else int(rng.choice(ink))
            available = [c for c in ink if (kind, c) not in used]
            color = preferred if (kind, preferred) not in used else int(rng.choice(available))
            colors.append(color)
            used.add((kind, color))
        try:
            parts = _fragments(rng, parents, kinds, colors, wire, marker, index)
            inp = _pack(rng, parts, bg)
        except RuntimeError:
            continue
        # Labels are built from the abstract construction, NOT by calling solve.
        cells, endpoints = {(0, 0): marker}, [(0, 0)]
        for i, (kind, color) in enumerate(zip(kinds, colors), 1):
            points = _stroke(endpoints[parents[i]], kind)
            cells.update({p: color for p in points})
            endpoints.append(points[-1])
        out = _render(cells, bg)
        return {"input": inp, "output": out}, dict(zip(kinds, colors))
    raise RuntimeError("bounded pair construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    result, seen, first_colors = {"train": [], "test": []}, set(), {}
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            overall = index if split == "train" else nt+index
            # Two short complete-vocabulary demonstrations, then modest training
            # variation. Every test is a larger tree than every training tree.
            n = 4 if split == "train" and index < 2 else int(rng.integers(4, 6)) if split == "train" else int(rng.integers(6, 8))
            for _ in range(12):
                pair, colors = _make_pair(rng, n, overall, first_colors)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    result[split].append(pair)
                    if overall == 0:
                        first_colors = colors
                    break
            else:
                raise RuntimeError("bounded distinct-input search failed")
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {"train": [], "test": []}
    for split in infos:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read(inp)
            assert np.array_equal(out, _execute(info)), "incorrect branching trace"
            n = len(info["graph"])-1
            assert n in ((4, 5) if split == "train" else (6, 7))
            assert {k[0] for k in info["graph"] if k != info["root"]} == set(range(4))
            child_counts = [sum(p == k for p in info["parent"].values()) for k in info["graph"]]
            assert max(child_counts) >= 2, "no fork"
            expected_ink = 1+sum(STROKES[k[0]][2] for k in info["graph"] if k != info["root"])
            assert np.count_nonzero(out != info["background"]) == expected_ink
            raw_ink = 1+sum(count*STROKES[k[0]][2] for k, count in info["occurrences"].items() if k != info["root"])
            assert raw_ink > expected_ink, "no duplicate-command counterexample"
            assert not np.array_equal(out, _render(info["union"], info["background"])), "B alone suffices"
            assert np.count_nonzero(out != info["background"]) != 1+2*n, "constant-length shortcut survives"
            infos[split].append(info)
    assert len(infos["train"][0]["graph"]) == len(infos["train"][1]["graph"]) == 5
    for kind in range(4):
        first = {k[1] for k in infos["train"][0]["graph"] if k[0] == kind}
        second = {k[1] for k in infos["train"][1]["graph"] if k[0] == kind}
        assert first.isdisjoint(second), "glyph color shortcut not broken"
    for info in infos["train"][:2]:
        by_color = {}
        for kind, color in info["graph"]:
            if kind != -1:
                by_color.setdefault(color, set()).add(kind)
        assert any(len(kinds) > 1 for kinds in by_color.values())
    assert any(len(part) == 1 for part in infos["train"][0]["parts"]), "absorption not demonstrated"
    assert any(len(part) == 3 for part in infos["train"][1]["parts"]), "multi-edge view not demonstrated"
    # A finite, meaningful alternative family: all 24 assignments of these four
    # strokes to the four shapes. The two short demonstrations identify one.
    for permutation in itertools.permutations(STROKES):
        if permutation == STROKES:
            continue
        matches = True
        for pair, info in zip(puzzle["train"][:2], infos["train"][:2]):
            try:
                predicted = _execute(info, permutation)
                matches = matches and np.array_equal(predicted, pair["output"])
            except AssertionError:
                matches = False
            if not matches:
                break
        assert not matches, "ambiguous glyph-to-stroke dictionary"


def self_test():
    """42 seeds, streamed one puzzle at a time; no files or retained traces."""
    signatures, shapes, output_shapes, topologies, counts, palettes = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(repr(inp.shape).encode()+inp.tobytes()+repr(out.shape).encode()+out.tobytes())
                info = _read(inp)
                shapes.add(inp.shape)
                output_shapes.add(out.shape)
                counts.add(len(info["parts"]))
                palettes.add((info["background"], info["wire"], info["root"][1]))
                topologies.add(tuple(sorted((len(v), sum(info["parent"].get(q) == k for q in v))
                                            for k, v in info["graph"].items())))
                # Full color permutation must commute with solving.
                permutation = np.array([7, 4, 9, 1, 6, 0, 8, 3, 5, 2], dtype=np.int8)
                assert np.array_equal(solve(permutation[inp]), permutation[out])
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(output_shapes) >= 15
    assert len(counts) >= 4 and len(topologies) >= 12 and len(palettes) >= 80
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, repeated = generate_puzzle(23), generate_puzzle(23)
    assert all(_key(p[k]) == _key(q[k]) for split in first
               for p, q in zip(first[split], repeated[split]) for k in ("input", "output"))
    def observed_views(a, info):
        wire_groups = _groups(zip(*np.nonzero(a == info["wire"])))
        result = []
        for part in info["parts"]:
            anchor = next(iter(part.values()))
            group = next(group for group in wire_groups if anchor in group)
            cells = {p: info["wire"] for p in group}
            for key, (r, c) in part.items():
                cells.update({(r+y, c+x): v for (y, x), v in _tile(*key, info["wire"]).items()})
            result.append(_normalize(cells))
        return result

    # Repacking preserves the answer: source position and raster order are irrelevant.
    info = _read(first["test"][0]["input"])
    fragments = observed_views(first["test"][0]["input"], info)
    repacked = _pack(np.random.default_rng(193), fragments, info["background"])
    assert np.array_equal(solve(repacked), first["test"][0]["output"])
    # Removing an entirely redundant station view also preserves the answer.
    demo = first["train"][0]
    demo_info = _read(demo["input"])
    views = observed_views(demo["input"], demo_info)
    retained = [view for view, part in zip(views, demo_info["parts"]) if len(part) > 1]
    assert len(retained) < len(views)
    reduced = _pack(np.random.default_rng(194), retained, demo_info["background"])
    assert np.array_equal(solve(reduced), demo["output"])
    # Output mutation must fail semantic validation.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    out = broken["test"][0]["output"]
    r, c = np.argwhere(out != info["background"])[0]
    out[r, c] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, count boundaries, recoloring, repacking, absorption, and output mutation")
