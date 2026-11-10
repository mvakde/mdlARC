"""Route-locked socket assembly; standard library and NumPy only.

The solver parses only its argument. It shares elementary geometry with the
constructor, but never receives the sampled route, scale, colors or seed.
No file I/O, subprocesses, network access, saved traces or unbounded caches.
"""
from functools import lru_cache
from hashlib import sha256
from math import isqrt

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normalise(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


@lru_cache(maxsize=512)
def _orientations(shape, reflect=True):
    result = set()
    for mirror in range(2 if reflect else 1):
        current = [(r, -c if mirror else c) for r, c in shape]
        for _ in range(4):
            result.add(_normalise(current))
            current = [(c, -r) for r, c in current]
    return tuple(sorted(result))


def _canonical(cells):
    return _orientations(_normalise(cells))[0]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRS:
                nxt = r+dr, c+dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        result.append(tuple(sorted(cells)))
    return result


def _checked_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _line(a, b):
    if a[0] == b[0] and a[1] != b[1]:
        return tuple((a[0], c) for c in range(min(a[1], b[1])+1, max(a[1], b[1])))
    if a[1] == b[1] and a[0] != b[0]:
        return tuple((r, a[1]) for r in range(min(a[0], b[0])+1, max(a[0], b[0])))
    return None


def _socket_graph(board, structural, holes, pins):
    owner = np.full(board.shape, -1, dtype=int)
    for i, cells in enumerate(holes):
        for p in cells:
            owner[p] = i
    graph, bridges = [[] for _ in pins], {}
    for i in range(len(pins)):
        for j in range(i+1, len(pins)):
            line = _line(pins[i], pins[j])
            if line is None or any(owner[p] not in (-1, i, j) for p in line):
                continue
            indexes = [k for k, p in enumerate(line) if board[p] == structural]
            if not indexes or indexes != list(range(indexes[0], indexes[-1]+1)):
                continue
            gap = frozenset(line[k] for k in indexes)
            graph[i].append(j)
            graph[j].append(i)
            bridges[i, j] = bridges[j, i] = gap
    return graph, bridges


def _routes(graph, labels=None, key=None, limit=2, bridges=None):
    """Enumerate paths, including the no-crossing condition, with an early cap."""
    n, answers = len(graph), []
    if labels is None:
        candidates = [tuple(range(n))]*n
    else:
        assert len(key) == n
        candidates = [tuple(i for i in range(n) if labels[i] == label) for label in key]

    def visit(path, used, painted):
        if len(answers) >= limit:
            return
        if len(path) == n:
            answers.append(tuple(path))
            return
        for j in candidates[len(path)]:
            if used & (1 << j) or (path and j not in graph[path[-1]]):
                continue
            gap = bridges[path[-1], j] if path and bridges is not None else frozenset()
            if gap & painted:
                continue
            visit(path+[j], used | (1 << j), painted | gap)
    visit([], 0, frozenset())
    return answers


def _assemble(board, holes, route, key, bridges):
    logical = board.copy()
    for i, color in zip(route, key):
        for p in holes[i]:
            logical[p] = color
    painted = set()
    for a, b, color in zip(route, route[1:], key):
        gap = bridges[a, b]
        assert not painted.intersection(gap), "crossing or overlapping bridges"
        painted.update(gap)
        for p in gap:
            logical[p] = color
    return logical


def _decode_piece(cells, k):
    r0, r1, c0, c1 = _bbox(cells)
    assert (r1-r0) % k == (c1-c0) % k == 0, "partial macrocell extent"
    blocks = {}
    for r, c in cells:
        p = (r-r0)//k, (c-c0)//k
        blocks[p] = blocks.get(p, 0)+1
    assert all(n == k*k for n in blocks.values()), "broken macrocell"
    return tuple(sorted(blocks))


def _recover(value):
    inp = _checked_grid(value)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    boards = []
    # The enclosing structural component locates the entire board, including
    # possible internal structural islands. A solid loose rectangle has no pins.
    for structural in range(10):
        if structural == bg:
            continue
        for cells in _components(inp == structural):
            r0, r1, c0, c1 = _bbox(cells)
            board = inp[r0:r1, c0:c1]
            if min(board.shape) < 4:
                continue
            if not all(np.all(edge == structural) for edge in
                       (board[0], board[-1], board[:, 0], board[:, -1])):
                continue
            extra = set(map(int, np.unique(board)))-{bg, structural}
            if len(extra) != 1:
                continue
            marker = extra.pop()
            holes = _components(board != structural)
            pins = []
            if not 3 <= len(holes) <= 7:
                continue
            for hole in holes:
                pin = [p for p in hole if board[p] == marker]
                if len(pin) != 1 or any(board[p] not in (bg, marker) for p in hole):
                    break
                pins.append(pin[0])
            else:
                boards.append((structural, marker, (r0, r1, c0, c1), board.copy(), holes, pins))
    assert len(boards) == 1, "expected a unique pinned blueprint"
    structural, marker, bounds, board, holes, pins = boards[0]
    r0, r1, c0, c1 = bounds
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    moat = np.zeros(inp.shape, dtype=bool)
    moat[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
    assert not np.any(outside & moat & (inp != bg)), "board touches an object"
    tokens, raw_pieces = [], []
    for cells in _components((inp != bg) & outside):
        colors = {int(inp[p]) for p in cells}
        assert len(colors) == 1, "loose objects touch"
        color = colors.pop()
        assert color not in (structural, marker)
        if len(cells) == 1:
            tokens.append(cells[0])
        else:
            assert len(cells) >= 4, "unrecognized external marks"
            raw_pieces.append((color, cells))
    n = len(holes)
    assert len(tokens) == len(raw_pieces) == n
    tokens.sort(key=lambda p: p[1])
    assert len({r for r, c in tokens}) == 1
    assert all(b[1]-a[1] == 2 for a, b in zip(tokens, tokens[1:]))
    last_object_row = max([r1-1]+[max(r for r, c in cells) for _, cells in raw_pieces])
    assert tokens[0][0] >= last_object_row+2
    key = tuple(int(inp[p]) for p in tokens)
    assert len(set(key)) == n and {c for c, cells in raw_pieces} == set(key)
    assert len({c for c, cells in raw_pieces}) == n
    area = sum(map(len, holes))
    piece_area = sum(len(cells) for _, cells in raw_pieces)
    assert piece_area % area == 0
    k = isqrt(piece_area//area)
    assert k >= 2 and k*k*area == piece_area, "no common square scale"
    assert all(side*k <= 30 for side in board.shape)
    pieces = {c: {"shape": _decode_piece(cells, k), "cells": cells}
              for c, cells in raw_pieces}
    labels = [_canonical(cells) for cells in holes]
    sequence = [_canonical(pieces[c]["shape"]) for c in key]
    assert sorted(labels) == sorted(sequence), "piece/socket shape inventory differs"
    graph, bridges = _socket_graph(board, structural, holes, pins)
    routes = _routes(graph, labels, sequence, bridges=bridges)
    assert len(routes) == 1, "key and silhouettes must specify exactly one route"
    route = routes[0]
    logical = _assemble(board, holes, route, key, bridges)
    output = np.repeat(np.repeat(logical, k, 0), k, 1)
    reflection, rotation = 0, 0
    for node, color in zip(route, key):
        destination, source = _normalise(holes[node]), pieces[color]["shape"]
        reflection += destination not in _orientations(source, False)
        rotation += destination != source and destination in _orientations(source, False)
    info = dict(background=bg, structural=structural, marker=marker, bounds=bounds,
                board=board, holes=holes, pins=pins, pieces=pieces, tokens=tokens,
                scale=k, key=key, classes=labels, graph=graph, bridges=bridges,
                route=route, logical=logical, reflection=reflection, rotation=rotation)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover scale, shape assignments and route from this input alone."""
    return _recover(input_grid)[0]


def _abstract_graph(cells):
    pins = [(4*r, 4*c) for r, c in cells]
    graph, bridges = [[] for _ in cells], {}
    for i, a in enumerate(pins):
        for j in range(i+1, len(pins)):
            line = _line(a, pins[j])
            if line is None or any(p in line for h, p in enumerate(pins) if h not in (i, j)):
                continue
            graph[i].append(j)
            graph[j].append(i)
            bridges[i, j] = bridges[j, i] = frozenset(line)
    return graph, bridges


def _sample_layout(rng, n, k, dense, rectangle):
    for _ in range(128):
        if dense:
            nr, nc = 3, 3
        elif n <= 4:
            nr, nc = (1, 3) if n == 3 and k == 2 and rng.random() < .3 else (2, 2)
            if n == 4 and not rectangle and k == 2 and rng.random() < .25:
                nr, nc = 2, 3
        else:
            nr, nc = 2, 3
        universe = [(r, c) for r in range(nr) for c in range(nc)]
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        if len({r for r, c in cells}) != nr or len({c for r, c in cells}) != nc:
            continue
        if rng.random() < .5:
            cells = sorted((c, r) for r, c in cells)
        graph, bridges = _abstract_graph(cells)
        paths = _routes(graph, limit=512, bridges=bridges)
        if paths:
            return cells, graph, bridges, paths[int(rng.integers(len(paths)))]
    # A constructive path spanning every required row and column.
    if dense:
        cells = {5: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
                 6: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)],
                 7: [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0)]}[n]
    else:
        cells = [(0, 0), (0, 1), (1, 1)] if n == 3 else [
            (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)][:n]
        if n == 4:
            cells = [(0, 0), (0, 1), (1, 1), (1, 0)]
    graph, bridges = _abstract_graph(cells)
    paths = _routes(graph, limit=512, bridges=bridges)
    assert paths
    return cells, graph, bridges, paths[int(rng.integers(len(paths)))]


def _type_sequence(rng, graph, abstract_bridges, route, first):
    n = len(route)
    if first:
        seq = [0, 0, 1]
        labels = [None]*n
        for i, t in zip(route, seq):
            labels[i] = t
        assert _routes(graph, labels, seq, bridges=abstract_bridges) == [route]
        return labels, seq, 2
    for attempt in range(100):
        count = int(rng.integers(2 if n == 3 else 3, min(n-1, 6)+1))
        seq = list(range(count))+[int(rng.integers(count)) for _ in range(n-count)]
        seq = list(map(int, rng.permutation(seq)))
        if attempt == 99:
            count, seq = n-1, [0, 0]+list(range(1, n-1))
        labels = [None]*n
        for i, t in zip(route, seq):
            labels[i] = t
        if _routes(graph, labels, seq, bridges=abstract_bridges) == [route]:
            # Make the duplicated class index 0, for compact packing at k=3.
            repeat = next(t for t in seq if seq.count(t) > 1)
            order = [repeat]+[t for t in range(count) if t != repeat]
            remap = {t: j for j, t in enumerate(order)}
            return [remap[t] for t in labels], [remap[t] for t in seq], count
    raise AssertionError("could not distinguish a route")


def _grow_shape(rng, area, compact=False):
    domain = {(r, c) for r in (range(2) if compact else range(-1, 2))
              for c in (range(2) if compact else range(-1, 2))}
    cells = {(0, 0)}
    while len(cells) < area:
        options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS}
                         & domain - cells)
        cells.add(options[int(rng.integers(len(options)))])
    return _canonical(tuple(cells))


def _class_pool(rng, count, k, first, small=False):
    if first:
        a = ((0, 0), (1, 0), (2, 0), (2, 1))
        choices = [((0, 0), (0, 1), (0, 2), (1, 1)),
                   ((0, 0), (0, 1), (1, 0), (1, 1)),
                   ((0, 1), (0, 2), (1, 0), (1, 1))]
        return [_canonical(a), _canonical(choices[int(rng.integers(len(choices)))])]
    result = []
    if k == 3:
        result.append(_canonical(((0, 0),) if rng.random() < .65 else ((0, 0), (0, 1))))
    max_area = 4 if k == 4 or small else 6 if k == 3 else 8
    for _ in range(256):
        if len(result) == count:
            return result
        shape = _grow_shape(rng, int(rng.integers(1, max_area+1)), k == 4)
        if shape not in result:
            result.append(shape)
    # Six small classes suffice for every feasible seven-socket construction.
    fallback = [((0, 0),), ((0, 0), (0, 1)), ((0, 0), (0, 1), (1, 0)),
                ((0, 0), (0, 1), (1, 0), (1, 1)),
                ((0, 0), (0, 1), (0, 2)),
                ((0, 0), (0, 1), (0, 2), (1, 1))]
    for shape in fallback:
        shape = _canonical(shape)
        if shape not in result:
            result.append(shape)
        if len(result) == count:
            return result
    raise AssertionError("insufficient silhouette classes")


def _attach_shape(rng, shape, compact):
    options = []
    for oriented in _orientations(shape):
        if compact:
            if (0, 0) in oriented:
                options.append(oriented)
        else:
            for ar, ac in oriented:
                shifted = tuple((r-ar, c-ac) for r, c in oriented)
                if all(-1 <= r <= 1 and -1 <= c <= 1 for r, c in shifted):
                    options.append(shifted)
    assert options
    return options[int(rng.integers(len(options)))]


def _coordinates(rng, levels, limit, compact, tight):
    base, first, tail = (3, 1, 3) if compact else (4, 2, 3)
    minimum = first+base*(levels-1)+tail
    assert minimum <= limit
    slack = 0 if tight else int(rng.integers(limit-minimum+1))
    # Extra cells may be in margins or any particular inter-pin gap.
    allocation = rng.multinomial(slack, np.ones(levels+1)/(levels+1))
    coords = [first+int(allocation[0])]
    for i in range(1, levels):
        coords.append(coords[-1]+base+int(allocation[i]))
    return coords, coords[-1]+tail+int(allocation[-1])


def _mask(shape, k):
    shape = _normalise(shape)
    _, h, _, w = _bbox(shape)
    mask = np.zeros((h, w), dtype=bool)
    for p in shape:
        mask[p] = True
    return np.repeat(np.repeat(mask, k, 0), k, 1)


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return []
    prefix = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = prefix[h:, w:]-prefix[:-h, w:]-prefix[h:, :-w]+prefix[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, loose, k, key, bg):
    objects = [(np.ones(board.shape, dtype=bool), None, board)]
    objects += [(_mask(shape, k), color, None) for color, shape in loose]
    for _ in range(48):
        canvas = np.full((26, 28), bg, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        order = [0]+sorted(range(1, len(objects)), key=lambda i: (-objects[i][0].size, float(rng.random())))
        for i in order:
            mask, color, data = objects[i]
            h, w = mask.shape
            positions = _free_origins(blocked, h, w)
            if len(positions) == 0:
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            if i == 0:
                canvas[r:r+h, c:c+w] = data
            else:
                canvas[r:r+h, c:c+w][mask] = color
            blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
        else:
            r0, r1, c0, c1 = _bbox(tuple(zip(*np.nonzero(canvas != bg))))
            content = canvas[r0:r1, c0:c1]
            h, w = content.shape
            width = max(w, 2*len(key)-1)+2
            result = np.full((h+4, width), bg, dtype=np.int8)
            left = int(rng.integers(1, width-w))
            result[1:h+1, left:left+w] = content
            key_left = int(rng.integers(1, width-(2*len(key)-1)))
            result[-2, key_left:key_left+2*len(key):2] = key
            return result
    return None


def _make_pair(rng, n, k, dense=False, first=False, rectangle=False):
    for attempt in range(96):
        cells, graph, abstract_bridges, route = _sample_layout(rng, n, k, dense, rectangle)
        labels, seq, count = _type_sequence(rng, graph, abstract_bridges, route, first)
        pool = _class_pool(rng, count, k, first, small=attempt >= 32)
        bg, structural, marker, *colors = map(int, rng.permutation(10))
        key = tuple(colors[:n])
        ys, bh = _coordinates(rng, max(r for r, c in cells)+1, 30//k, k == 4, attempt >= 32)
        xs, bw = _coordinates(rng, max(c for r, c in cells)+1, 30//k, k == 4, attempt >= 32)
        pins = [(ys[r], xs[c]) for r, c in cells]
        board = np.full((bh, bw), structural, dtype=np.int8)
        holes = []
        for node, pin in enumerate(pins):
            shape = _attach_shape(rng, pool[labels[node]], k == 4)
            hole = tuple((r+pin[0], c+pin[1]) for r, c in shape)
            holes.append(hole)
            for p in hole:
                board[p] = bg
            board[pin] = marker
        actual_graph, bridges = _socket_graph(board, structural, holes, pins)
        assert actual_graph == graph
        # Graph construction and decoding must agree, not just chosen edges.
        assert _routes(actual_graph, labels, seq, bridges=bridges) == [route]
        loose = []
        for index, (node, color) in enumerate(zip(route, key)):
            destination = _normalise(holes[node])
            options = _orientations(destination)
            if first and index == 0:
                options = tuple(s for s in options if s not in _orientations(destination, False))
            loose.append((color, options[int(rng.integers(len(options)))]))
        rng.shuffle(loose)
        inp = _pack(rng, board, loose, k, key, bg)
        if inp is None:
            continue
        logical = _assemble(board, holes, route, key, bridges)
        out = np.repeat(np.repeat(logical, k, 0), k, 1)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded packing search exhausted")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    resolution_shift = bool(rng.random() < .25)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            first = split == "train" and index == 0
            rectangle = split == "train" and index == 1
            if first:
                n, k = 3, 2
            elif rectangle:
                n, k = 4, 3
            elif split == "test":
                n, k = (int(rng.integers(3, 5)), 4) if resolution_shift else (int(rng.integers(5, 8)), 2)
            else:
                n = int(rng.integers(3, 7))
                k = int(rng.choice((2, 3))) if n <= 4 else 2
            for _ in range(8):
                pair = _make_pair(rng, n, k, dense=split == "test" and not resolution_shift,
                                  first=first, rectangle=rectangle)
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _checked_grid(pair["input"]), _checked_grid(pair["output"])
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            expected, info = _recover(inp)
            assert np.array_equal(out, expected), "incorrect assembly, bridge, scale or preserved structure"
            assert not np.array_equal(inp, out)
            assert info["scale"] in (2, 3, 4)
            assert all(1 <= len(h) <= 8 for h in info["holes"])
            assert len(set(info["classes"])) < len(info["holes"]), "no shape-assignment ambiguity"
            assert len(_routes(info["graph"], bridges=info["bridges"])) == 2, "no geometry-only ambiguity"
            assert set(map(int, np.unique(out))) == {info["structural"], *info["key"]}
            # Reconstruct the logical resolution from every macrocell, not a
            # strided sample that would miss a one-pixel output corruption.
            k = info["scale"]
            blocks = out.reshape(info["board"].shape[0], k, info["board"].shape[1], k)
            assert np.all(blocks == info["logical"][:, None, :, None])
            if split == "train":
                train_info.append(info)
    a, b = train_info[:2]
    assert len(a["holes"]) == 3 and a["scale"] == 2 and a["reflection"] > 0
    assert len({len(h) for h in a["holes"]}) == 1 and len(set(a["classes"])) == 2
    assert len(_routes(a["graph"], [len(h) for h in a["holes"]],
                       [len(a["pieces"][c]["shape"]) for c in a["key"]], bridges=a["bridges"])) == 2
    assert len(b["holes"]) == 4 and b["scale"] == 3
    assert len(_routes(b["graph"], limit=20, bridges=b["bridges"])) == 8
    return None


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _must_reject(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError("invalid diagnostic accepted")


def self_test():
    """Stream 42 seeds; no artifacts, all-puzzle retention or retry logs."""
    digests, dimensions, boards, patterns, shapes, roles = set(), set(), set(), set(), set(), set()
    counts, scales, reflection, gaps, placements = set(), set(), 0, set(), set()
    branchy, repeats_non_single, unequal = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                expected, a = _recover(pair["input"])
                assert np.array_equal(expected, pair["output"])
                dimensions.add(tuple(pair["input"].shape))
                boards.add(tuple(a["board"].shape))
                patterns.add((tuple(a["board"].shape), (a["board"] == a["structural"]).tobytes()))
                shapes.update(a["classes"])
                counts.add(len(a["holes"]))
                scales.add(a["scale"])
                roles.add((a["background"], a["structural"], a["marker"]))
                reflection += a["reflection"]
                branchy += len(_routes(a["graph"], limit=4, bridges=a["bridges"])) >= 4
                repeats_non_single += any(len(s) > 1 and a["classes"].count(s) > 1 for s in a["classes"])
                lengths = [len(a["bridges"][u, v]) for u, v in zip(a["route"], a["route"][1:])]
                gaps.update(lengths)
                unequal += len(set(lengths)) > 1
                r0, r1, c0, c1 = a["bounds"]
                for piece in a["pieces"].values():
                    pr0, pr1, pc0, pc1 = _bbox(piece["cells"])
                    if pr0 < r0:
                        placements.add("above")
                    if pr1 > r1:
                        placements.add("below")
                    if pc0 < c0:
                        placements.add("left")
                    if pc1 > c1:
                        placements.add("right")
        first = puzzle["train"][0]
        _, a = _recover(first["input"])
        # A resolves the actual repeated-shape color assignment, not just the
        # later bridge coloring: exchange the first two congruent key entries.
        duplicate_key_swap = first["input"].copy()
        duplicate_key_swap[a["tokens"][0]] = a["key"][1]
        duplicate_key_swap[a["tokens"][1]] = a["key"][0]
        reassigned, reassigned_info = _recover(duplicate_key_swap)
        assert reassigned_info["route"] == a["route"]
        assert not np.array_equal(reassigned, first["output"])
        assert all(reassigned_info["logical"][p] == a["key"][1]
                   for p in a["holes"][a["route"][0]])
        # A intervention: reversing only the key preserves socket filling while
        # reversing departure colors along every bridge.
        reversed_key = first["input"].copy()
        for p, color in zip(a["tokens"], a["key"][::-1]):
            reversed_key[p] = color
        reverse_output, reverse_info = _recover(reversed_key)
        assert reverse_info["route"] == a["route"][::-1]
        assert not np.array_equal(reverse_output, first["output"])
        for hole in a["holes"]:
            assert all(reverse_info["logical"][p] == a["logical"][p] for p in hole)
        # B intervention: end shapes have equal area, but exchanging ONLY their
        # loose colors makes the compatible socket route run in reverse.
        shape_swap = first["input"].copy()
        c, d = a["key"][0], a["key"][-1]
        for p in a["pieces"][c]["cells"]:
            shape_swap[p] = d
        for p in a["pieces"][d]["cells"]:
            shape_swap[p] = c
        swapped, swapped_info = _recover(shape_swap)
        assert swapped_info["route"] == a["route"][::-1]
        assert not np.array_equal(swapped, first["output"])
        # Color identities are local roles, including nonzero backgrounds.
        remap = (3*np.arange(10)+7) % 10
        assert np.array_equal(solve(remap[first["input"]]), remap[first["output"]])
    assert len(digests) == 42 and len(dimensions) >= 35 and len(boards) >= 15
    assert len(patterns) >= 140 and len(shapes) >= 25 and len(roles) >= 100
    assert counts == set(range(3, 8)) and scales == {2, 3, 4}
    assert reflection >= 42 and branchy >= 42 and repeats_non_single >= 42 and unequal >= 42
    assert len(gaps) >= 5 and placements == {"above", "below", "left", "right"}
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(nt-1, 4)+1):
            p = generate_puzzle(5000+10*nt+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1), (4, None)):
        _must_reject(lambda nt=nt, nq=nq: generate_puzzle(0, nt, nq))
    p = generate_puzzle(0, 2, 1)
    _, a = _recover(p["train"][0]["input"])
    broken = p["train"][0]["input"].copy()
    broken[a["pieces"][a["key"][0]]["cells"][0]] = a["background"]
    _must_reject(lambda: solve(broken))
    bad_key = p["train"][0]["input"].copy()
    bad_key[a["tokens"][1]] = a["key"][0]
    _must_reject(lambda: solve(bad_key))
    # A truly ambiguous assembly with valid distinct key colors and intact
    # macrocells: four congruent corner sockets admit eight different routes.
    ambiguous = np.zeros((12, 19), dtype=np.int8)
    ambiguous[1:8, 1:8] = 8
    for r, c in ((2, 2), (2, 6), (6, 2), (6, 6)):
        ambiguous[r, c] = 9
    for color, (r, c) in zip((1, 2, 3, 4), ((1, 11), (1, 15), (5, 11), (5, 15))):
        ambiguous[r:r+2, c:c+2] = color
    ambiguous[10, 4:11:2] = (1, 2, 3, 4)
    _must_reject(lambda: solve(ambiguous))
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    _must_reject(lambda: validate_generated_puzzle(p))
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "input_dimensions": len(dimensions), "board_dimensions": len(boards),
            "socket_patterns": len(patterns), "shape_classes": len(shapes),
            "local_role_triples": len(roles), "counts": sorted(counts), "scales": sorted(scales),
            "reflection_assignments": reflection, "branching_scenes": branchy,
            "repeated_non_singleton_scenes": repeats_non_single, "unequal_gap_scenes": unequal,
            "gap_lengths": sorted(gaps), "piece_sides": sorted(placements)}
